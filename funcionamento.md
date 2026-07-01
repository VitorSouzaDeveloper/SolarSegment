# Plano de Implementação: Interface Web Interactiva para Análise Fotovoltaica (TCC II)

Este plano descreve como criar uma interface web moderna, rápida e visualmente impressionante para o seu TCC. O usuário poderá fazer o upload de uma imagem aérea, configurar os parâmetros (GSD, Eficiência, Irradiação Local, Threshold) e visualizar em tempo real o pipeline de fragmentação em tiles, a predição da U-Net, o pós-processamento e os cálculos matemáticos finais de potência estimada.

---

## ⚠️ User Review Required

> [!IMPORTANT]
> **Compatibilidade de Hardware (GPU vs CPU)**
> O script de inferência rodará por padrão na GPU (`cuda`) se disponível, caindo de volta para a CPU se necessário. Como faremos inferência de imagens potencialmente grandes de forma fragmentada (tile por tile), o consumo de memória será controlado e seguro mesmo em CPUs normais.
> 
> A interface será executada localmente usando **FastAPI** (servidor leve em Python) e uma página web única rica em **HTML5, CSS3 avançado e JavaScript Vanilla**.

---

## ❓ Open Questions

*Nenhuma pergunta aberta no momento. O plano segue exatamente a arquitetura já desenvolvida de fragmentação e inferência, acoplando-a a uma API FastAPI robusta.*

---

## 🚀 Proposed Changes

Dividiremos o desenvolvimento em duas partes principais: o Backend (API em FastAPI) e o Frontend (Dashboard responsivo e elegante com animações).

### Componente 1: Backend (FastAPI API)

#### [NEW] [src/web_interface.py](file:///c:/Users/vitor/Documents/Faculdade/TCC%20II/Implementa%C3%A7%C3%A3o/src/web_interface.py)
Criar um servidor FastAPI que conterá:
1. **Endpoint `GET /`**: Serve o dashboard em HTML.
2. **Endpoint `POST /analyze`**:
   - Recebe a imagem enviada, o valor de GSD, eficiência, irradiação, threshold do modelo e tamanho do filtro de área.
   - Executa a fragmentação em memória (usando Pillow) para evitar salvar centenas de arquivos temporários em disco.
   - Carrega o modelo `unet_solar.pth` (apenas uma vez, em cache) e executa a inferência em cada tile.
   - Reconstrói a máscara inteira combinando as predições de cada tile.
   - Aplica operações morfológicas (fechamento) e filtro de tamanho mínimo (remoção de ruído) como na fase de inferência.
   - Calcula a área total de painéis e a potência estimada ($P_{est} = A_{total} \times \eta \times I_{local}$).
   - Retorna as imagens codificadas em Base64 (original, máscara e overlay de predição) junto com os metadados e estatísticas de tiles.

---

### Componente 2: Frontend (Dashboard Premium)

#### [NEW] [src/static/index.html](file:///c:/Users/vitor/Documents/Faculdade/TCC%20II/Implementa%C3%A7%C3%A3o/src/static/index.html)
Criar a página HTML que define a estrutura do dashboard:
- **Painel Lateral de Configurações**: Sliders interativos para GSD (m/pixel), Eficiência do painel ($\eta$), Irradiação Solar ($I_{local}$), Threshold de Confiança do Modelo, e Área Mínima de Painel (em pixels).
- **Área de Upload Interativa**: Drag & drop com pré-visualização.
- **Painel de Progresso do Pipeline**: Feedback visual animado de cada estágio do processamento (Upload -> Tiling -> Inferência U-Net -> Pós-processamento -> Cálculos).
- **Cards de Métricas (KPIs)**: Cards modernos com efeitos luminosos para Área Total ($m^2$), Potência Média ($kW$), Geração Diária ($kWh/dia$) e Quantidade de Painéis Detectados (estimada pelos contornos).
- **Visualizador Comparativo**: Exibição lado a lado com abas para Alternar entre Imagem Original, Máscara da U-Net e Overlay (imagem original com realce translúcido vermelho/dourado sobre as placas).
- **Galeria de Tiles**: Seção mostrando amostras de blocos 512x512 onde placas foram encontradas, ilustrando a fragmentação interna do sistema.

#### [NEW] [src/static/style.css](file:///c:/Users/vitor/Documents/Faculdade/TCC%20II/Implementa%C3%A7%C3%A3o/src/static/style.css)
CSS customizado com estética premium:
- **Tema Escuro Profundo** (Deep Blue/Carbon) com detalhes em Dourado Solar (`#fbbf24`) e Verde Esmeralda (`#10b981`).
- **Glassmorphism**: Efeitos de transparência blur usando `backdrop-filter`.
- **Animações Fluidas**: Transições de hover em botões, pulsações no progresso do pipeline e efeitos de fade-in para os resultados.

#### [NEW] [src/static/app.js](file:///c:/Users/vitor/Documents/Faculdade/TCC%20II/Implementa%C3%A7%C3%A3o/src/static/app.js)
JavaScript para controle da interface:
- Envio assíncrono (AJAX / Fetch) do formulário de upload e parâmetros.
- Simulação/sincronização visual dos passos do pipeline durante o carregamento.
- Renderização dinâmica das estatísticas e imagens Base64 retornadas.
- Criação dos efeitos visuais da galeria de tiles fragmentados.

---

## 🧪 Verification Plan

### Automated/Manual Tests
1. **Inicialização do Servidor**: Rodar `python src/web_interface.py` e verificar se o Uvicorn inicia sem erros na porta `http://127.0.0.1:8000`.
2. **Upload de Teste**: Carregar uma imagem da pasta `data/raw/` ou de teste e verificar se:
   - A fragmentação é executada corretamente.
   - O modelo U-Net faz inferência e gera a predição.
   - O dashboard renderiza com os KPIs calculados de acordo com os parâmetros de GSD e Eficiência.
3. **Validação dos Sliders**: Mudar a Eficiência ou GSD nos controles e reenviar a imagem para ver o impacto direto nos cálculos de área e potência.
