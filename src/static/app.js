document.addEventListener('DOMContentLoaded', () => {
    // Parâmetros da UI
    const gsdInput = document.getElementById('gsd');
    const gsdVal = document.getElementById('gsd-val');
    const etaInput = document.getElementById('eta');
    const etaVal = document.getElementById('eta-val');
    const iLocalInput = document.getElementById('i_local');
    const iLocalVal = document.getElementById('i_local-val');
    const thresholdInput = document.getElementById('threshold');
    const thresholdVal = document.getElementById('threshold-val');
    const minAreaInput = document.getElementById('min_area');
    const minAreaVal = document.getElementById('min_area-val');
    const overlapInput = document.getElementById('overlap');
    
    // Preset Buttons
    const presetBtns = document.querySelectorAll('.preset-btn');
    
    // File Upload
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadPreviewContainer = document.getElementById('upload-preview-container');
    const uploadPreview = document.getElementById('upload-preview');
    const previewFilename = document.getElementById('preview-filename');
    const removeImgBtn = document.getElementById('remove-img-btn');
    const runBtn = document.getElementById('run-btn');
    
    // Pipeline Progress
    const pipelinePanel = document.getElementById('pipeline-panel');
    const pipelineStatusText = document.getElementById('pipeline-status-text');
    const progressBar = document.getElementById('progress-bar');
    const steps = {
        upload: document.getElementById('step-upload'),
        tiling: document.getElementById('step-tiling'),
        unet: document.getElementById('step-unet'),
        post: document.getElementById('step-post'),
        calc: document.getElementById('step-calc')
    };

    // Results Dashboard
    const resultsPanel = document.getElementById('results-panel');
    const metricArea = document.getElementById('metric-area');
    const metricGsdRef = document.getElementById('metric-gsd-ref');
    const metricPower = document.getElementById('metric-power');
    const metricDaily = document.getElementById('metric-daily');
    const metricAnnual = document.getElementById('metric-annual');
    const metricGroups = document.getElementById('metric-groups');
    
    // Visualizer Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const visualizerCard = document.querySelector('.visualizer-card');
    
    // Display Imgs
    const resultOverlay = document.getElementById('result-overlay');
    const resultMask = document.getElementById('result-mask');
    const resultOriginal = document.getElementById('result-original');
    
    // Gallery
    const tilesGalleryContainer = document.getElementById('tiles-gallery-container');
    
    // Metadata
    const metaFilename = document.getElementById('meta-filename');
    const metaResolution = document.getElementById('meta-resolution');
    const metaDevice = document.getElementById('meta-device');
    const metaTotalTiles = document.getElementById('meta-total-tiles');
    const metaActiveTiles = document.getElementById('meta-active-tiles');
    const metaPixels = document.getElementById('meta-pixels');

    let selectedFile = null;

    // --- 1. Sincronização dos Sliders e Valores ---
    gsdInput.addEventListener('input', () => {
        gsdVal.textContent = parseFloat(gsdInput.value).toFixed(4) + ' m';
        // Remove active preset if not matching
        presetBtns.forEach(btn => {
            if (Math.abs(parseFloat(btn.dataset.value) - parseFloat(gsdInput.value)) < 0.001) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    });

    etaInput.addEventListener('input', () => {
        etaVal.textContent = (parseFloat(etaInput.value) * 100).toFixed(1) + ' %';
    });

    iLocalInput.addEventListener('input', () => {
        iLocalVal.textContent = parseFloat(iLocalInput.value).toFixed(1) + ' kWh/m²';
    });

    thresholdInput.addEventListener('input', () => {
        thresholdVal.textContent = parseFloat(thresholdInput.value).toFixed(2);
    });

    minAreaInput.addEventListener('input', () => {
        minAreaVal.textContent = minAreaInput.value + ' px';
    });

    // Preset click listeners
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            gsdInput.value = btn.dataset.value;
            gsdInput.dispatchEvent(new Event('input'));
        });
    });

    // Ativa preset ortofoto por padrão
    presetBtns[0].classList.add('active');

    // --- 2. Upload Drag & Drop ---
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeImgBtn && !removeImgBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        if (!file.type.startsWith('image/')) {
            alert('Por favor, envie apenas arquivos de imagem.');
            return;
        }

        selectedFile = file;
        previewFilename.textContent = file.name;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreview.src = e.target.result;
            uploadPreviewContainer.style.display = 'flex';
            runBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        uploadPreview.src = '';
        uploadPreviewContainer.style.display = 'none';
        runBtn.disabled = true;
        resultsPanel.style.display = 'none';
        pipelinePanel.style.display = 'none';
    }

    // --- 3. Controle Visual do Pipeline ---
    function resetPipeline() {
        Object.values(steps).forEach(step => {
            step.classList.remove('active', 'completed');
        });
        progressBar.style.width = '0%';
        pipelineStatusText.textContent = 'Aguardando inicialização...';
    }

    function setStepState(stepName, state) {
        const step = steps[stepName];
        if (!step) return;

        if (state === 'active') {
            step.classList.remove('completed');
            step.classList.add('active');
        } else if (state === 'completed') {
            step.classList.remove('active');
            step.classList.add('completed');
        } else {
            step.classList.remove('active', 'completed');
        }
    }

    // Delay helper para simular eixos do pipeline de forma suave e bonita
    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    // --- 4. Submissão do Pipeline ---
    runBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Limpa estado anterior
        resultsPanel.style.display = 'none';
        pipelinePanel.style.display = 'block';
        pipelinePanel.scrollIntoView({ behavior: 'smooth' });
        resetPipeline();

        // 1. Upload
        setStepState('upload', 'active');
        pipelineStatusText.textContent = 'Enviando imagem aérea e configurando parâmetros no servidor...';
        progressBar.style.width = '10%';
        await sleep(600);
        
        setStepState('upload', 'completed');
        
        // 2. Tiling
        setStepState('tiling', 'active');
        pipelineStatusText.textContent = 'Fragmentando ortofoto de entrada em blocos de 512x512 pixels...';
        progressBar.style.width = '30%';
        
        // Prepara dados de envio
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('gsd', gsdInput.value);
        formData.append('eta', etaInput.value);
        formData.append('i_local', iLocalInput.value);
        formData.append('threshold', thresholdInput.value);
        formData.append('min_area', minAreaInput.value);
        formData.append('tile_size', '512');
        formData.append('overlap', overlapInput.checked);

        const startTime = Date.now();

        try {
            // Chamada à API FastAPI
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Ocorreu um erro no servidor durante o processamento.');
            }

            const data = await response.json();
            const elapsed = Date.now() - startTime;
            
            // Simula o progresso dos passos seguintes dependendo do tempo decorrido
            pipelineStatusText.textContent = `Fragmentação concluída (${data.metadata.total_tiles} tiles gerados). Iniciando predição pela Rede Neural U-Net...`;
            setStepState('tiling', 'completed');
            progressBar.style.width = '50%';
            await sleep(800);

            // 3. U-Net
            setStepState('unet', 'active');
            pipelineStatusText.textContent = `Processando inferência profunda pixel-a-pixel no dispositivo ${data.metadata.device_used}...`;
            progressBar.style.width = '70%';
            await sleep(900);
            
            setStepState('unet', 'completed');

            // 4. Pós-Processamento
            setStepState('post', 'active');
            pipelineStatusText.textContent = 'Executando pós-processamento: fechamento morfológico e filtro de área conectado...';
            progressBar.style.width = '85%';
            await sleep(700);
            
            setStepState('post', 'completed');

            // 5. Estimativa Final
            setStepState('calc', 'active');
            pipelineStatusText.textContent = 'Calculando modelo matemático fotovoltaico e compilando estatísticas...';
            progressBar.style.width = '100%';
            await sleep(600);
            
            setStepState('calc', 'completed');
            pipelineStatusText.textContent = 'Pipeline executado com sucesso!';

            // Renderiza resultados
            renderResults(data);

        } catch (error) {
            console.error(error);
            pipelineStatusText.textContent = `Erro no Pipeline: ${error.message}`;
            progressBar.style.background = '#ef4444';
            progressBar.style.width = '100%';
        }
    });

    // --- 5. Renderização dos Resultados ---
    function renderResults(data) {
        // Exibe o painel de resultados
        resultsPanel.style.display = 'flex';
        
        // Rolagem suave para o painel
        setTimeout(() => {
            resultsPanel.scrollIntoView({ behavior: 'smooth' });
        }, 100);

        // 1. KPIs
        metricArea.textContent = data.results.area_total_m2.toLocaleString('pt-BR', { minimumFractionDigits: 2 });
        metricGsdRef.textContent = data.results.gsd.toFixed(4);
        metricPower.textContent = data.results.potencia_estimada_kw.toLocaleString('pt-BR', { minimumFractionDigits: 2 });
        metricDaily.textContent = data.results.geracao_diaria_kwh.toLocaleString('pt-BR', { minimumFractionDigits: 2 });
        metricAnnual.textContent = data.results.geracao_anual_kwh.toLocaleString('pt-BR', { minimumFractionDigits: 0 });
        metricGroups.textContent = data.results.detected_groups;

        // 2. Imagens
        resultOverlay.src = `data:image/jpeg;base64,${data.images.overlay_b64}`;
        resultMask.src = `data:image/png;base64,${data.images.mask_b64}`;
        resultOriginal.src = `data:image/jpeg;base64,${data.images.original_b64}`;

        // 3. Metadados do Sistema
        metaFilename.textContent = data.metadata.filename;
        metaResolution.textContent = `${data.metadata.width} x ${data.metadata.height} px`;
        metaDevice.textContent = data.metadata.device_used.toUpperCase();
        metaTotalTiles.textContent = data.metadata.total_tiles;
        metaActiveTiles.textContent = data.metadata.tiles_with_detection;
        metaPixels.textContent = data.results.total_solar_pixels.toLocaleString('pt-BR');

        // 4. Galeria de Tiles
        tilesGalleryContainer.innerHTML = '';
        if (data.tiles_gallery.length === 0) {
            tilesGalleryContainer.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; color: var(--text-secondary); padding: 2rem;">
                    <i class="fa-solid fa-circle-exclamation" style="font-size: 2rem; margin-bottom: 0.5rem; color: var(--text-secondary);"></i>
                    <p>Nenhum tile apresentou área de painel solar superior a 1.5% da área total.</p>
                </div>
            `;
        } else {
            data.tiles_gallery.forEach(tile => {
                const card = document.createElement('div');
                card.className = 'tile-container';
                card.innerHTML = `
                    <div class="tile-img-box">
                        <img src="data:image/jpeg;base64,${tile.orig_b64}" alt="Tile ${tile.id} Original">
                        <img class="tile-overlay-img" src="data:image/jpeg;base64,${tile.overlay_b64}" alt="Tile ${tile.id} Overlay">
                    </div>
                    <div class="tile-info">
                        <span class="tile-coords"><i class="fa-solid fa-crosshairs"></i> X:${tile.x} Y:${tile.y}</span>
                        <span class="tile-pct">${tile.detection_ratio}%</span>
                    </div>
                `;
                tilesGalleryContainer.appendChild(card);
            });
        }
    }

    // --- 6. Abas e Tela Cheia do Visualizador ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const tabId = btn.dataset.tab;
            document.getElementById(tabId).classList.add('active');
        });
    });

    fullscreenBtn.addEventListener('click', () => {
        visualizerCard.classList.toggle('fullscreen');
        if (visualizerCard.classList.contains('fullscreen')) {
            fullscreenBtn.innerHTML = '<i class="fa-solid fa-compress"></i> Fechar';
        } else {
            fullscreenBtn.innerHTML = '<i class="fa-solid fa-expand"></i> Tela Cheia';
        }
    });

    // Fecha tela cheia com ESC
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && visualizerCard.classList.contains('fullscreen')) {
            visualizerCard.classList.remove('fullscreen');
            fullscreenBtn.innerHTML = '<i class="fa-solid fa-expand"></i> Tela Cheia';
        }
    });
});
