document.addEventListener('DOMContentLoaded', () => {
    // Parâmetros da UI
    const gsdSlider = document.getElementById('gsd');
    const gsdInput = document.getElementById('gsd-val-input');
    const tiltSlider = document.getElementById('tilt_angle');
    const tiltVal = document.getElementById('tilt_angle-val');
    const latInput = document.getElementById('latitude');
    const useTiltCheck = document.getElementById('use_tilt_correction');
    const etaInput = document.getElementById('eta');
    const etaVal = document.getElementById('eta-val');
    const iLocalInput = document.getElementById('i_local');
    const iLocalVal = document.getElementById('i_local-val');
    const thresholdInput = document.getElementById('threshold');
    const thresholdVal = document.getElementById('threshold-val');
    const minAreaInput = document.getElementById('min_area');
    const minAreaVal = document.getElementById('min_area-val');
    const overlapInput = document.getElementById('overlap');
    const prInput = document.getElementById('pr');
    const prVal = document.getElementById('pr-val');
    
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
    const metricProjArea = document.getElementById('metric-proj-area');
    const metricTiltFactor = document.getElementById('metric-tilt-factor');
    const metricPower = document.getElementById('metric-power');
    const metricDaily = document.getElementById('metric-daily');
    const metricAnnual = document.getElementById('metric-annual');
    const metricModules = document.getElementById('metric-modules');
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
    const metaTiltInfo = document.getElementById('meta-tilt-info');
    const metaTotalTiles = document.getElementById('meta-total-tiles');
    const metaActiveTiles = document.getElementById('meta-active-tiles');
    const metaPixels = document.getElementById('meta-pixels');
    const metaTiltFactorVal = document.getElementById('meta-tilt-factor-val');
    const moduleAreaSlider = document.getElementById('module_area_m2');
    const moduleAreaInput = document.getElementById('module_area-val-input');
    const packingFactorInput = document.getElementById('packing_factor');
    const packingFactorVal = document.getElementById('packing_factor-val');

    let selectedFile = null;

    // --- 1. Sincronização dos Sliders e Valores ---
    function updateMinAreaDisplay() {
        const px = parseInt(minAreaInput.value) || 0;
        const gsd = parseFloat(gsdInput.value) || 0.0389;
        const m2 = px * (gsd * gsd);
        minAreaVal.textContent = `${px} px (~${m2 < 0.1 ? m2.toFixed(3) : m2.toFixed(2)} m²)`;
    }

    gsdSlider.addEventListener('input', () => {
        gsdInput.value = parseFloat(gsdSlider.value).toFixed(4);
        updatePresetActive('gsd', gsdSlider.value);
        updateMinAreaDisplay();
    });

    gsdInput.addEventListener('input', () => {
        const val = parseFloat(gsdInput.value);
        if (!isNaN(val) && val >= 0.0001 && val <= 5.0) {
            gsdSlider.value = val;
            updatePresetActive('gsd', val);
            updateMinAreaDisplay();
        }
    });

    tiltSlider.addEventListener('input', () => {
        tiltVal.textContent = tiltSlider.value + ' °';
        updatePresetActive('tilt', tiltSlider.value);
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
        updateMinAreaDisplay();
    });

    // Inicializa texto da área mínima
    updateMinAreaDisplay();

    prInput.addEventListener('input', () => {
        prVal.textContent = (parseFloat(prInput.value) * 100).toFixed(1) + ' %';
    });

    if (moduleAreaSlider && moduleAreaInput) {
        moduleAreaSlider.addEventListener('input', () => {
            moduleAreaInput.value = parseFloat(moduleAreaSlider.value).toFixed(2);
            updatePresetActive('module_area', moduleAreaSlider.value);
        });
        moduleAreaInput.addEventListener('input', () => {
            const val = parseFloat(moduleAreaInput.value);
            if (!isNaN(val) && val >= 1.0 && val <= 4.0) {
                moduleAreaSlider.value = val;
                updatePresetActive('module_area', val);
            }
        });
    }

    if (packingFactorInput && packingFactorVal) {
        packingFactorInput.addEventListener('input', () => {
            packingFactorVal.textContent = (parseFloat(packingFactorInput.value) * 100).toFixed(0) + ' %';
        });
    }

    function updatePresetActive(type, value) {
        presetBtns.forEach(btn => {
            if (btn.dataset.type === type) {
                if (Math.abs(parseFloat(btn.dataset.value) - parseFloat(value)) < 0.0001) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            }
        });
    }

    // Preset click listeners
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const type = btn.dataset.type;
            const val = btn.dataset.value;
            
            presetBtns.forEach(b => {
                if (b.dataset.type === type) b.classList.remove('active');
            });
            btn.classList.add('active');
            
            if (type === 'gsd') {
                gsdInput.value = val;
                gsdSlider.value = val;
                
                // Ajusta automaticamente a escala de parâmetros recomendada para o município
                const threshSlider = document.getElementById('threshold');
                const threshVal = document.getElementById('threshold-val');
                if (Math.abs(parseFloat(val) - 0.10) < 0.001) {
                    // Preset Maracaju-MS (GSD 10 cm/px)
                    latInput.value = '-21.61';
                    minAreaInput.value = '40'; // 40 px * (0.10)^2 = 0.40 m² (equivalente físico ideal)
                    if (threshSlider) { threshSlider.value = '0.40'; }
                    if (threshVal) { threshVal.textContent = '0.40'; }
                } else if (Math.abs(parseFloat(val) - 0.0389) < 0.001) {
                    // Preset Dourados-MS (GSD 3.89 cm/px)
                    latInput.value = '-22.22';
                    minAreaInput.value = '250'; // 250 px * (0.0389)^2 = 0.38 m²
                    if (threshSlider) { threshSlider.value = '0.65'; }
                    if (threshVal) { threshVal.textContent = '0.65'; }
                } else if (Math.abs(parseFloat(val) - 0.2986) < 0.001) {
                    // Preset Satélite Bing Z19 (GSD ~30 cm/px)
                    minAreaInput.value = '10';
                    if (threshSlider) { threshSlider.value = '0.50'; }
                    if (threshVal) { threshVal.textContent = '0.50'; }
                }
                
                updateMinAreaDisplay();
            } else if (type === 'tilt') {
                tiltSlider.value = val;
                tiltSlider.dispatchEvent(new Event('input'));
            } else if (type === 'module_area' && moduleAreaInput) {
                moduleAreaInput.value = val;
                moduleAreaInput.dispatchEvent(new Event('input'));
            }
        });
    });

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
        
        const validExtensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.geotiff', '.jp2'];
        const fileNameLower = file.name.toLowerCase();
        const hasValidExt = validExtensions.some(ext => fileNameLower.endsWith(ext));
        
        if (!file.type.startsWith('image/') && !hasValidExt) {
            alert('Por favor, selecione um arquivo de imagem válido (GeoTIFF .tif/.tiff, JP2, PNG, JPG ou JPEG).');
            return;
        }

        selectedFile = file;
        previewFilename.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
        
        const isTiffOrJp2 = fileNameLower.endsWith('.tif') || fileNameLower.endsWith('.tiff') || fileNameLower.endsWith('.jp2');
        
        if (isTiffOrJp2) {
            // Browsers don't support native TIFF/JP2 rendering in <img>, so show SVG placeholder
            uploadPreview.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200"><rect width="100%" height="100%" fill="%231e293b"/><text x="50%" y="45%" fill="%23fbbf24" font-family="sans-serif" font-size="20" font-weight="bold" text-anchor="middle">GeoTIFF / Raster de Alta Precisão</text><text x="50%" y="65%" fill="%2394a3b8" font-family="sans-serif" font-size="14" text-anchor="middle">Pronto para inferência U-Net</text></svg>';
            uploadPreviewContainer.style.display = 'flex';
            runBtn.disabled = false;
        } else {
            const reader = new FileReader();
            reader.onload = (e) => {
                uploadPreview.src = e.target.result;
                uploadPreviewContainer.style.display = 'flex';
                runBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
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
        progressBar.style.background = 'linear-gradient(90deg, var(--solar), var(--cyan))';
        pipelineStatusText.textContent = 'Aguardando inicialização...';
    }

    function setStepState(stepName, state) {
        const step = steps[stepName];
        if (!step) return;

        if (state === 'active') {
            step.classList.add('active');
            step.classList.remove('completed');
        } else if (state === 'completed') {
            step.classList.remove('active');
            step.classList.add('completed');
        }
    }

    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    // --- 4. Execução do Pipeline ---
    runBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        runBtn.disabled = true;
        resultsPanel.style.display = 'none';
        pipelinePanel.style.display = 'block';
        pipelinePanel.scrollIntoView({ behavior: 'smooth' });
        resetPipeline();

        // 1. Upload
        setStepState('upload', 'active');
        pipelineStatusText.textContent = 'Enviando imagem aérea e configurando parâmetros no servidor FastAPI...';
        progressBar.style.width = '12%';
        await sleep(500);
        setStepState('upload', 'completed');
        
        // 2. Sliding Window (Tiling)
        setStepState('tiling', 'active');
        pipelineStatusText.textContent = 'Executando Sliding Window com sobreposição e ponderação suave (Hann 2D)...';
        progressBar.style.width = '30%';
        
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('gsd', gsdInput.value);
        formData.append('tilt_angle', tiltSlider.value);
        formData.append('latitude', latInput.value);
        formData.append('use_tilt_correction', useTiltCheck.checked);
        formData.append('eta', etaInput.value);
        formData.append('i_local', iLocalInput.value);
        formData.append('threshold', thresholdInput.value);
        formData.append('min_area', minAreaInput.value);
        formData.append('tile_size', '512');
        formData.append('overlap', overlapInput.checked);
        formData.append('pr', prInput.value);
        formData.append('module_power_w', '550.0');
        formData.append('module_area_m2', moduleAreaInput ? moduleAreaInput.value : '2.30');
        formData.append('packing_factor', packingFactorInput ? packingFactorInput.value : '0.90');

        const startTime = Date.now();

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Ocorreu um erro durante a inferência no servidor.');
            }

            const data = await response.json();
            
            pipelineStatusText.textContent = `Fragmentação concluída (${data.metadata.total_tiles} tiles). Processando inferência na U-Net...`;
            setStepState('tiling', 'completed');
            progressBar.style.width = '55%';
            await sleep(600);

            // 3. U-Net
            setStepState('unet', 'active');
            pipelineStatusText.textContent = `Segmentação convolucional profunda concluída no dispositivo ${data.metadata.device_used}...`;
            progressBar.style.width = '75%';
            await sleep(600);
            setStepState('unet', 'completed');

            // 4. Pós-Processamento
            setStepState('post', 'active');
            pipelineStatusText.textContent = 'Aplicando fechamento morfológico e filtro de área mínima conectada (OpenCV)...';
            progressBar.style.width = '88%';
            await sleep(500);
            setStepState('post', 'completed');

            // 5. Estimativa & Correção
            setStepState('calc', 'active');
            pipelineStatusText.textContent = 'Calculando correção geométrica de inclinação (Jiao et al.) e estimativas de potência...';
            progressBar.style.width = '100%';
            await sleep(400);
            setStepState('calc', 'completed');
            pipelineStatusText.textContent = 'Pipeline executado com sucesso!';

            renderResults(data);

        } catch (error) {
            console.error(error);
            pipelineStatusText.textContent = `Erro: ${error.message}`;
            progressBar.style.background = '#ef4444';
            progressBar.style.width = '100%';
        }
    });

    // --- 5. Renderização dos Resultados ---
    function renderResults(data) {
        resultsPanel.style.display = 'flex';
        
        setTimeout(() => {
            resultsPanel.scrollIntoView({ behavior: 'smooth' });
        }, 100);

        // 1. KPIs
        metricArea.textContent = data.results.area_total_m2.toLocaleString('pt-BR', { minimumFractionDigits: 2 });
        metricProjArea.textContent = data.results.area_projetada_m2.toLocaleString('pt-BR', { minimumFractionDigits: 2 });
        metricTiltFactor.textContent = data.results.tilt_factor.toFixed(2);
        
        metricPower.textContent = data.results.potencia_pico_kwp.toLocaleString('pt-BR', { minimumFractionDigits: 2 });
        metricDaily.textContent = data.results.geracao_diaria_kwh.toLocaleString('pt-BR', { minimumFractionDigits: 2 });
        metricAnnual.textContent = data.results.geracao_anual_kwh.toLocaleString('pt-BR', { minimumFractionDigits: 0 });
        
        metricModules.textContent = data.results.estimated_modules.toLocaleString('pt-BR');
        metricGroups.textContent = data.results.detected_groups;
        document.getElementById('metric-pr-val-ref').textContent = (parseFloat(prInput.value) * 100).toFixed(0);

        // 2. Imagens
        resultOverlay.src = `data:image/jpeg;base64,${data.images.overlay_b64}`;
        resultMask.src = `data:image/png;base64,${data.images.mask_b64}`;
        resultOriginal.src = `data:image/jpeg;base64,${data.images.original_b64}`;

        // 3. Metadados do Sistema
        metaFilename.textContent = data.metadata.filename;
        const gsdCm = (data.results.gsd * 100).toFixed(1);
        metaResolution.textContent = `${data.metadata.width} × ${data.metadata.height} px (${gsdCm} cm/px)`;
        metaDevice.textContent = data.metadata.device_used.toUpperCase();
        metaTiltInfo.textContent = `β: ${data.metadata.tilt_angle}° | φ: ${data.metadata.latitude}°`;
        metaTotalTiles.textContent = data.metadata.total_tiles;
        metaActiveTiles.textContent = data.metadata.tiles_with_detection;
        metaPixels.textContent = data.results.total_solar_pixels.toLocaleString('pt-BR');
        metaTiltFactorVal.textContent = `x${data.results.tilt_factor.toFixed(3)} (${data.results.tilt_angle > 0 && data.metadata.use_tilt_correction ? 'Jiao et al.' : 'Horizontal'})`;

        // 4. Tabela Comparativa de Tipologias de Painel
        const breakdownBody = document.getElementById('panel-breakdown-body');
        const breakdownAreaRef = document.getElementById('breakdown-area-ref');
        if (breakdownAreaRef) breakdownAreaRef.textContent = data.results.area_total_m2.toFixed(2);
        
        if (breakdownBody && data.panel_types_breakdown) {
            breakdownBody.innerHTML = '';
            const currentSelectedArea = parseFloat(moduleAreaInput ? moduleAreaInput.value : 2.30);
            
            data.panel_types_breakdown.forEach(pt => {
                const isSelected = Math.abs(pt.area_m2 - currentSelectedArea) < 0.08;
                const tr = document.createElement('tr');
                if (isSelected) tr.className = 'active-scenario';
                
                tr.innerHTML = `
                    <td>
                        <div class="panel-type-cell">
                            <span class="panel-type-name">${pt.category}</span>
                            <span class="panel-type-tech">${pt.tech} &bull; ${pt.desc}</span>
                        </div>
                    </td>
                    <td>
                        <span class="badge-power">${pt.power_w} W</span>
                    </td>
                    <td>
                        <span class="metric-highlight">${pt.area_m2.toFixed(2)} m²</span>
                    </td>
                    <td>
                        <span class="badge-modules">
                            ${pt.modules_estimated} <small>placas</small>
                        </span>
                    </td>
                    <td>
                        <span class="metric-highlight">${pt.installed_kwp.toFixed(2)} kWp</span>
                    </td>
                    <td>
                        <span class="metric-highlight">${pt.monthly_kwh.toFixed(1)} kWh/mês</span>
                    </td>
                    <td>
                        <button type="button" class="btn-select-preset ${isSelected ? 'selected' : ''}" 
                                data-power="${pt.power_w}" data-area="${pt.area_m2}">
                            ${isSelected ? '<i class="fa-solid fa-check"></i> Ativo' : 'Aplicar'}
                        </button>
                    </td>
                `;
                
                const selectBtn = tr.querySelector('.btn-select-preset');
                selectBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (moduleAreaInput) {
                        moduleAreaInput.value = pt.area_m2.toFixed(2);
                        moduleAreaInput.dispatchEvent(new Event('input'));
                    }
                    if (moduleAreaSlider) {
                        moduleAreaSlider.value = pt.area_m2;
                    }
                    
                    document.querySelectorAll('#panel-breakdown-body tr').forEach(r => r.classList.remove('active-scenario'));
                    document.querySelectorAll('#panel-breakdown-body .btn-select-preset').forEach(b => {
                        b.classList.remove('selected');
                        b.innerHTML = 'Aplicar';
                    });
                    tr.classList.add('active-scenario');
                    selectBtn.classList.add('selected');
                    selectBtn.innerHTML = '<i class="fa-solid fa-check"></i> Ativo';
                    
                    metricModules.textContent = pt.modules_estimated;
                    const purpleTitle = document.querySelector('.metric-card.purple-glow .metric-title');
                    if (purpleTitle) purpleTitle.innerHTML = `Módulos Estimados (~${pt.power_w}W)`;
                });
                
                breakdownBody.appendChild(tr);
            });
        }

        // 5. Galeria de Tiles
        tilesGalleryContainer.innerHTML = '';
        if (!data.tiles_gallery || data.tiles_gallery.length === 0) {
            tilesGalleryContainer.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; color: var(--text-secondary); padding: 2rem;">
                    <i class="fa-solid fa-circle-exclamation" style="font-size: 2rem; margin-bottom: 0.5rem; color: var(--text-secondary);"></i>
                    <p>Nenhum tile apresentou área de painel solar superior ao limiar de detecção.</p>
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
            const targetPane = document.getElementById(tabId);
            if (targetPane) targetPane.classList.add('active');
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

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && visualizerCard.classList.contains('fullscreen')) {
            visualizerCard.classList.remove('fullscreen');
            fullscreenBtn.innerHTML = '<i class="fa-solid fa-expand"></i> Tela Cheia';
        }
    });
});
