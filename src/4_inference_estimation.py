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

STANDARD_PANEL_TYPES = [
    {"category": "Residencial Compacto", "tech": "Policristalino / Mono 60 céls", "power_w": 340, "area_m2": 1.70, "desc": "Residencial compacto"},
    {"category": "Residencial Moderno", "tech": "Half-Cell 108 céls M10", "power_w": 415, "area_m2": 1.95, "desc": "Padrão residencial atual"},
    {"category": "Comercial / Médio Porte", "tech": "Half-Cell 120/144 céls", "power_w": 460, "area_m2": 2.15, "desc": "Comercial e residencial amplo"},
    {"category": "Comercial Padrão Mercado", "tech": "Half-Cell 144 céls M10", "power_w": 550, "area_m2": 2.30, "desc": "Padrão mais vendido no Brasil"},
    {"category": "Alta Potência Industrial", "tech": "Half-Cell 120/132 céls G12", "power_w": 600, "area_m2": 2.60, "desc": "Galpões e usinas de solo"},
    {"category": "Ultra Potência / N-Type TOPCon", "tech": "Bifacial 132 céls G12 TOPCon", "power_w": 680, "area_m2": 2.85, "desc": "Última geração TOPCon"}
]

def calculate_panel_types_breakdown(s_pv: float, i_local: float = 5.4, pr: float = 0.75, packing_factor: float = 0.90):
    breakdown = []
    effective_area = s_pv * packing_factor
    for p in STANDARD_PANEL_TYPES:
        modules_dense = int(round(effective_area / p["area_m2"])) if p["area_m2"] > 0 else 0
        modules_pure = int(round(s_pv / p["area_m2"])) if p["area_m2"] > 0 else 0
        kwp = (modules_dense * p["power_w"]) / 1000.0
        daily_kwh = kwp * i_local * pr
        monthly_kwh = daily_kwh * 30.0
        annual_kwh = daily_kwh * 365.0
        breakdown.append({
            "category": p["category"],
            "tech": p["tech"],
            "power_w": p["power_w"],
            "area_m2": p["area_m2"],
            "desc": p["desc"],
            "modules_estimated": modules_dense,
            "modules_pure": modules_pure,
            "installed_kwp": round(kwp, 2),
            "daily_kwh": round(daily_kwh, 2),
            "monthly_kwh": round(monthly_kwh, 2),
            "annual_kwh": round(annual_kwh, 2)
        })
    return breakdown

def calculate_power_estimation(
    total_pixels,
    gsd,
    tilt_deg=25.0,
    lat_deg=-22.22,
    eta=0.185,
    i_local=5.4,
    pr=0.75,
    module_power_w=550.0,
    module_area_m2=2.3,
    use_tilt_correction=True
):
    """
    Modelo matemático com correção geométrica de inclinação e estimativa fotovoltaica (TCC II).
    """
    # 1. Área Projetada na Ortofoto (A_proj) em m²
    area_per_pixel = gsd * gsd
    a_proj = total_pixels * area_per_pixel
    
    # 2. Correção Geométrica de Inclinação para Área Real do Módulo (S_pv)
    if use_tilt_correction and tilt_deg > 0:
        beta_rad = math.radians(tilt_deg)
        cos_beta = math.cos(beta_rad)
        correction_factor = 1.0 / cos_beta if cos_beta > 0.1 else 1.0
        correction_factor = max(1.0, float(correction_factor))
        s_pv = a_proj * correction_factor
    else:
        correction_factor = 1.0
        s_pv = a_proj
        
    # 3. Potência Pico Instalada (kWp)
    p_pico = s_pv * eta
    
    # 4. Geração Diária e Anual de Energia (kWh)
    e_diaria = p_pico * i_local * pr
    e_anual = e_diaria * 365.0
    
    # 5. Estimativa de Módulos
    estimated_modules = int(round((s_pv * 0.90) / module_area_m2)) if module_area_m2 > 0 else 0
    breakdown = calculate_panel_types_breakdown(s_pv, i_local, pr, packing_factor=0.90)
    
    return {
        "a_proj_m2": a_proj,
        "s_pv_m2": s_pv,
        "tilt_factor": correction_factor,
        "p_pico_kwp": p_pico,
        "e_diaria_kwh": e_diaria,
        "e_anual_kwh": e_anual,
        "estimated_modules": estimated_modules,
        "panel_types_breakdown": breakdown
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
    threshold: float = 0.65
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
    
    def get_axis_coords(dim_size, t_size, st):
        if dim_size <= t_size:
            return [0]
        coords = []
        pos = 0
        while pos + t_size < dim_size:
            coords.append(pos)
            pos += st
        coords.append(max(0, dim_size - t_size))
        return sorted(list(set(coords)))

    x_coords = get_axis_coords(width, tile_size, stride)
    y_coords = get_axis_coords(height, tile_size, stride)
    
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
            image, model, device, tile_size=512, overlap_ratio=0.25, threshold=0.65
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
        tilt_deg=25.0,      # Inclinação típica no MS (25°)
        lat_deg=-22.22,     # Dourados - MS
        eta=0.185,
        i_local=5.4,
        pr=0.75,
        module_power_w=550.0,
        module_area_m2=2.2,
        use_tilt_correction=True
    )
    
    print("\n" + "=" * 85)
    print("RELATÓRIO DE ESTIMATIVA DE GERAÇÃO FOTOVOLTAICA (TCC II)")
    print("=" * 85)
    print(f"Área Projetada na Ortofoto (A_proj): {res['a_proj_m2']:.2f} m²")
    print(f"Área Real Corrigida do Módulo (S_pv): {res['s_pv_m2']:.2f} m² (Fator: x{res['tilt_factor']:.3f})")
    print(f"Inclinação do Telhado (β): 25.0° | Latitude (φ): -22.22°")
    print(f"Eficiência (η): 18.5% | Irradiação: 5.4 kWh/m²/dia | PR: 75.0%")
    print("-" * 85)
    print("ESTIMATIVA COMPARATIVA POR TIPOLOGIA DE PAINEL FOTOVOLTAICO:")
    print(f"{'Categoria / Tecnologia':<28} | {'Potência':<8} | {'Área':<7} | {'Qtd Placas':<10} | {'Potência (kWp)':<14} | {'Geração/Mês':<12}")
    print("-" * 85)
    for p in res["panel_types_breakdown"]:
        print(f"{p['category']:<28} | {p['power_w']:>4} W   | {p['area_m2']:>4.2f} m² | {p['modules_estimated']:>6} un  | {p['installed_kwp']:>8.2f} kWp   | {p['monthly_kwh']:>7.1f} kWh")
    print("=" * 85)

if __name__ == "__main__":
    inference_and_estimation()
