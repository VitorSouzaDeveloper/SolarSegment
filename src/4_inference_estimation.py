import os
import math
import sys
import importlib
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.transforms import v2

# Adiciona a pasta atual ao sys.path para permitir importações
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importa a U-Net de 3_train_unet
train_module = importlib.import_module("3_train_unet")
UNet = train_module.UNet

def create_hann_window_2d(size):
    """
    Cria uma janela de ponderação 2D (Hann Window) para suavizar as bordas na fusão de tiles.
    Pixels no centro do tile recebem peso 1.0, decaindo suavemente nas extremidades
    para eliminar descontinuidades e o 'efeito de borda'.
    """
    hann_1d = np.hanning(size)
    hann_2d = np.outer(hann_1d, hann_1d)
    # Evita pesos estritamente zero nas bordas absolutas
    hann_2d = np.clip(hann_2d, 1e-4, 1.0)
    return hann_2d.astype(np.float32)

def calculate_power_estimation(
    total_pixels,
    gsd,
    tilt_deg=20.0,
    lat_deg=-22.22,
    eta=0.185,
    i_local=5.4,
    pr=0.75,
    module_power_w=550.0,
    module_area_m2=2.2,
    use_tilt_correction=True
):
    """
    Modelo matemático com correção geométrica de inclinação e estimativa fotovoltaica (TCC II).
    
    Parâmetros:
    - total_pixels: Quantidade de pixels classificados como painel solar.
    - gsd: Ground Sample Distance em metros/pixel.
    - tilt_deg (β): Ângulo de inclinação do telhado/módulo em graus (padrão 20.0°).
    - lat_deg (φ): Latitude local em graus (padrão -22.22° para Dourados - MS).
    - eta (η): Eficiência média comercial dos módulos (padrão 18.5%).
    - i_local: Irradiação solar diária média (kWh/m²/dia).
    - pr: Performance Ratio do sistema (padrão 75%).
    - module_power_w: Potência nominal do módulo padrão em Watts (550 W).
    - module_area_m2: Área física do módulo padrão em m² (~2.2 m²).
    - use_tilt_correction: Ativa a correção geométrica de área real inclinada.
    """
    # 1. Área Projetada na Ortofoto (A_proj) em m²
    area_per_pixel = gsd * gsd
    a_proj = total_pixels * area_per_pixel
    
    # 2. Correção Geométrica de Inclinação para Área Real do Módulo (S_pv)
    # Em ortofotos aéreas verticais, a câmera captura a projeção ortogonal no plano horizontal (A_proj).
    # Como os módulos solares são instalados com inclinação β (tilt angle), a área real dos módulos é:
    # S_pv = A_proj / cos(β)
    if use_tilt_correction and tilt_deg > 0:
        beta_rad = math.radians(tilt_deg)
        # Fator geométrico exato de inclinação de telhado: 1 / cos(β)
        cos_beta = math.cos(beta_rad)
        if cos_beta > 0.1:
            correction_factor = 1.0 / cos_beta
        else:
            correction_factor = 1.0
            
        correction_factor = max(1.0, float(correction_factor))
        s_pv = a_proj * correction_factor
    else:
        correction_factor = 1.0
        s_pv = a_proj
        
    # 3. Potência Pico Instalada (kWp) baseada na área real dos módulos
    # P_pico = S_pv (m²) * eta * Irradiância STC (1 kW/m²)
    p_pico = s_pv * eta
    
    # 4. Geração Diária e Anual de Energia (kWh)
    e_diaria = p_pico * i_local * pr
    e_anual = e_diaria * 365.0
    
    # 5. Estimativa de Módulos Físicos Discretos (~550 Wp)
    estimated_modules = int(round(s_pv / module_area_m2)) if module_area_m2 > 0 else 0
    
    return {
        "a_proj_m2": a_proj,
        "s_pv_m2": s_pv,
        "tilt_factor": correction_factor,
        "p_pico_kwp": p_pico,
        "e_diaria_kwh": e_diaria,
        "e_anual_kwh": e_anual,
        "estimated_modules": estimated_modules
    }

def calculate_estimation_errors(estimated, true_val):
    """
    Calcula as métricas de erro para a estimativa de área e potência:
    MAE (Erro Absoluto Médio), MAPE (Erro Percentual Médio) e RMSE.
    """
    if true_val is None or true_val <= 0:
        return None, None, None
        
    est = float(estimated)
    true_v = float(true_val)
    
    mae = abs(est - true_v)
    mape = (abs(true_v - est) / true_v) * 100.0
    rmse = math.sqrt((est - true_v) ** 2)
    
    return mae, mape, rmse

def predict_sliding_window(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 512,
    overlap_ratio: float = 0.25,
    threshold: float = 0.55
):
    """
    Realiza inferência via Janela Deslizante (Sliding Window) com sobreposição e
    janelamento suave Hann para eliminação completa do efeito de borda.
    """
    width, height = image.size
    stride = int(tile_size * (1.0 - overlap_ratio))
    stride = max(1, min(stride, tile_size))
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    # Matrizes de probabilidade e pesos acumulados
    prob_map = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    hann_weight = create_hann_window_2d(tile_size)
    
    x_coords = list(range(0, width, stride))
    y_coords = list(range(0, height, stride))
    
    # Garante cobertura total até a última borda
    if len(x_coords) == 0 or x_coords[-1] + tile_size < width:
        x_coords.append(max(0, width - tile_size))
    if len(y_coords) == 0 or y_coords[-1] + tile_size < height:
        y_coords.append(max(0, height - tile_size))
        
    x_coords = sorted(list(set(x_coords)))
    y_coords = sorted(list(set(y_coords)))
    
    model.eval()
    with torch.no_grad():
        for y in y_coords:
            for x in x_coords:
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                
                tile = image.crop((x, y, x_end, y_end))
                active_w = x_end - x
                active_h = y_end - y
                
                if active_w < tile_size or active_h < tile_size:
                    padded_tile = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    padded_tile.paste(tile, (0, 0))
                    input_tensor = transform(padded_tile).unsqueeze(0).to(device)
                else:
                    input_tensor = transform(tile).unsqueeze(0).to(device)
                    
                output = model(input_tensor)
                prob = torch.sigmoid(output).squeeze().cpu().numpy() # shape (tile_size, tile_size)
                
                # Extrai apenas a porção ativa
                prob_active = prob[0:active_h, 0:active_w]
                weight_active = hann_weight[0:active_h, 0:active_w]
                
                # Pondera e acumula
                prob_map[y:y_end, x:x_end] += prob_active * weight_active
                weight_map[y:y_end, x:x_end] += weight_active

    # Normalização pela soma dos pesos
    weight_map[weight_map == 0] = 1.0
    prob_map /= weight_map
    
    # Binarização com threshold
    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
    return prob_map, binary_mask

def inference_and_estimation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Inferência e Estimativa no dispositivo: {device}")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_img_dir = os.path.join(base_dir, "data", "processed", "test", "Placas")
    
    # Fallback se não existir subpasta Placas
    if not os.path.exists(test_img_dir):
        test_img_dir = os.path.join(base_dir, "data", "processed", "test", "images")
        
    preds_dir = os.path.join(base_dir, "data", "processed", "test", "preds_placas")
    os.makedirs(preds_dir, exist_ok=True)
    
    model_path = os.path.join(base_dir, "models", "unet_solar_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "models", "unet_solar.pth")
        
    if not os.path.exists(model_path):
        print(f"Modelo não encontrado em {model_path}. Execute o treinamento primeiro.")
        return
        
    # Carrega a rede
    model = UNet(in_channels=3, out_channels=1)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Pesos do modelo carregados com sucesso de: {model_path}")
    except Exception as e:
        print(f"Aviso ao carregar pesos: {e}")
        
    model.to(device)
    model.eval()
    
    # Determinação do GSD
    jp2_files = [f for f in os.listdir(os.path.join(base_dir, "data", "raw")) if f.endswith(".jp2")] if os.path.exists(os.path.join(base_dir, "data", "raw")) else []
    
    if jp2_files:
        gsd = 0.0389 # GSD real de ortofoto de alta precisão (3.89 cm)
        print(f"Detectada ortofoto .jp2 ({jp2_files[0]}). Usando GSD: {gsd:.4f} m/pixel.")
    else:
        jgw_path = os.path.join(base_dir, "data", "raw", "esri_highres_stitched.jgw")
        gsd = 0.2986 # Fallback GSD Zoom 19
        if os.path.exists(jgw_path):
            with open(jgw_path, 'r') as f:
                gsd = float(f.readline().strip())
        print(f"GSD identificado: {gsd:.4f} m/pixel")
        
    if not os.path.exists(test_img_dir):
        print(f"Pasta de teste {test_img_dir} não encontrada.")
        return
        
    test_images = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
    if len(test_images) == 0:
        print(f"Nenhuma imagem encontrada em {test_img_dir}.")
        return
        
    total_solar_pixels = 0
    
    print(f"\nIniciando inferência com Janela Deslizante (Overlap 25% + Hann Window) em {len(test_images)} imagens...")
    
    for img_name in test_images:
        img_path = os.path.join(test_img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # 1. Inferência com Janela Deslizante e Fusão Suave
        prob_map, pred_mask_uint8 = predict_sliding_window(
            image, model, device, tile_size=512, overlap_ratio=0.25, threshold=0.55
        )
        
        # 2. Pós-processamento Morfológico (Fechamento)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        pred_mask_uint8 = cv2.morphologyEx(pred_mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # 3. Filtro de Área Mínima (Connected Components)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask_uint8, connectivity=8)
        min_area_pixels = 250
        filtered_mask = np.zeros_like(pred_mask_uint8)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_pixels:
                filtered_mask[labels == i] = 255
                
        solar_pixels_in_img = np.sum(filtered_mask > 0)
        total_solar_pixels += solar_pixels_in_img
        
        # Salva o mapa de predição
        pred_img = Image.fromarray(filtered_mask, mode="L")
        pred_img.save(os.path.join(preds_dir, f"pred_{img_name}"))
        
    print(f"Inferência concluída em {len(test_images)} amostras.")
    print(f"Total de Pixels Identificados como Painéis: {total_solar_pixels}")
    
    # 4. Cálculo do Modelo Matemático Fotovoltaico Corrigido
    res = calculate_power_estimation(
        total_pixels=total_solar_pixels,
        gsd=gsd,
        tilt_deg=20.0,      # Inclinação típica no MS
        lat_deg=-22.22,     # Dourados - MS
        eta=0.185,
        i_local=5.4,
        pr=0.75,
        module_power_w=550.0,
        module_area_m2=2.2,
        use_tilt_correction=True
    )
    
    print("\n" + "=" * 65)
    print("RELATÓRIO DE ESTIMATIVA DE GERAÇÃO FOTOVOLTAICA (TCC II)")
    print("=" * 65)
    print(f"Área Projetada na Ortofoto (A_proj): {res['a_proj_m2']:.2f} m²")
    print(f"Área Real Corrigida do Módulo (S_pv): {res['s_pv_m2']:.2f} m² (Fator: x{res['tilt_factor']:.3f})")
    print(f"Inclinação do Telhado (β): 20.0° | Latitude (φ): -22.22°")
    print(f"Eficiência (η): 18.5% | Irradiação: 5.4 kWh/m²/dia | PR: 75.0%")
    print("-" * 65)
    print(f"Módulos Estimados (~550 Wp): ~{res['estimated_modules']} painéis")
    print(f"POTÊNCIA PICO INSTALADA (P_pico): {res['p_pico_kwp']:.2f} kWp")
    print(f"GERAÇÃO DIÁRIA ESTIMADA (E_diaria): {res['e_diaria_kwh']:.2f} kWh/dia")
    print(f"GERAÇÃO ANUAL ESTIMADA (E_anual): {res['e_anual_kwh']:.2f} kWh/ano")
    print("=" * 65)

if __name__ == "__main__":
    inference_and_estimation()
