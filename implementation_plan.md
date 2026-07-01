# Plano de Implementação: Pipeline Completo do TCC II

Com base no texto da Metodologia do seu TCC, estruturei o projeto completo em **4 Fases Principais**. O objetivo é que o código siga rigorosamente o que está escrito no seu trabalho, garantindo aderência acadêmica e técnica.

## ⚠️ User Review Required

> [!WARNING]
> **Sobre o GDAL, Rasterio, Numpy e PyTorch**
> O seu texto cita explicitamente: *"...com o suporte das bibliotecas geoespaciais GDAL e Rasterio"* e menciona modelos supervisionados (*U-Net*).
> Notei no seu `requirements.txt` que bibliotecas C-based (como `numpy` e `rasterio`) falharam na instalação devido a acentos no diretório (`Implementação`). 
> **Precisaremos resolver isso.** Você prefere:
> 1. Que eu crie um script alternativo sem Rasterio (usando apenas Pillow e matemática para simular o georreferenciamento)? (Isso difere ligeiramente do seu texto).
> 2. Que renomeemos sua pasta de `Implementação` para `Implementacao` (sem cedilha e til) para instalar o Rasterio via `pip` ou `conda`?

## ❓ Open Questions

1. **Máscaras Manuais:** O texto diz *"máscaras binárias anotadas de forma manual"*. Você já criou essas máscaras no Photoshop/QGIS, ou gostaria de alguma ferramenta/script em Python simples que ajude a desenhar e pintar de branco por cima dos telhados nas imagens geradas?
2. **Framework de Deep Learning:** Podemos utilizar **PyTorch** para construir e treinar a U-Net? É o padrão na literatura de visão computacional.
3. **Índice de Irradiação:** Para $I_{local}$ de Dourados-MS, você tem um valor de referência específico que pretende usar (ex: 5.4 kWh/m²/dia), ou deixo configurável no `.env`?

## 🚀 Proposed Changes

A arquitetura do código será dividida em scripts sequenciais, refletindo suas seções do texto:

### Fase 1: Ajuste na Aquisição (Georreferenciamento)
#### [MODIFY] [src/1_data_acquisition.py](file:///c:/Users/vitor/Documents/Faculdade/TCC%20II/Implementa%C3%A7%C3%A3o/src/1_data_acquisition.py)
- Atualmente salvamos a imagem como `.jpeg`. Vamos gerar um "World File" (`.jgw` ou `.tfw`) junto com a imagem, contendo a resolução espacial exata (GSD) e as coordenadas geográficas. Isso tornará a imagem "georreferenciada", satisfazendo o primeiro parágrafo do seu TCC.

### Fase 2: Fragmentação (Seção 3.1)
#### [NEW] [src/2_fragmentation.py](file:///c:/Users/vitor/Documents/Faculdade/TCC%20II/Implementa%C3%A7%C3%A3o/src/2_fragmentation.py)
- Script que lerá a imagem georreferenciada grande e a subdividirá em *tiles* exatos de **512x512 pixels**.
- Os tiles serão separados em pastas: `data/processed/train`, `data/processed/val`, `data/processed/test`.

### Fase 3: Pipeline de Treinamento U-Net (Seção 3.2)
#### [NEW] [src/3_train_unet.py](file:///c:/Users/vitor/Documents/Faculdade/TCC%20II/Implementa%C3%A7%C3%A3o/src/3_train_unet.py)
- Definição da arquitetura **U-Net** de segmentação pixel-a-pixel.
- Pipeline de Data Augmentation integrado (translações, rotações, contraste).
- Loop de treinamento com otimização supervisionada (lendo as imagens e máscaras `ground truth`).
- Salvamento dos pesos treinados (`models/unet_solar.pth`).

### Fase 4: Inferência e Estimativa Matemática (Seções 3.2 e 3.3)
#### [NEW] [src/4_inference_estimation.py](file:///c:/Users/vitor/Documents/Faculdade/TCC%20II/Implementa%C3%A7%C3%A3o/src/4_inference_estimation.py)
- Passará o conjunto de testes pelo modelo treinado, gerando as matrizes binárias de saída (0 ou 1 para painel solar).
- Sumarização dos pixels preditos como "painel solar".
- Conversão da soma de pixels em Metros Quadrados ($A_{total}$), extraindo o tamanho do pixel (GSD).
- Aplicação do Modelo Matemático: $P_{est} = A_{total} \times \eta \times I_{local}$, resultando no valor final em `kW`.

## 🧪 Verification Plan
- Renomear/Configurar ambiente para aceitar bibliotecas geoespaciais.
- Validar se o cálculo de tamanho do pixel georreferenciado bate com a área real aproximada do Google Earth.
- Rodar todo o pipeline com algumas imagens mockadas (1 imagem de treino e 1 máscara).
