import os
import io
import sys
import math
import base64
import importlib
import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from torchvision.transforms import v2

app = FastAPI(title="SolarSegment - Visão Computacional & Estimativa Fotovoltaica (TCC II)")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

os.makedirs(os.path.join(current_dir, "static"), exist_ok=True)

# Cache global para o modelo U-Net e timestamp do arquivo
model_cache = None
model_mtime = 0

def get_model():
    global model_cache, model_mtime
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    model_path = os.path.join(base_dir, "models", "unet_solar_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "models", "unet_solar.pth")
        
    current_mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0
    
    # Se o modelo já está carregado e o arquivo de pesos não mudou, reutiliza
    if model_cache is not None and current_mtime == model_mtime:
        return model_cache
        
    try:
        train_module = importlib.import_module("3_train_unet")
        UNet = train_module.UNet
    except Exception as e:
        raise ImportError(f"Erro ao importar a U-Net de 3_train_unet: {e}")
        
    model = UNet(in_channels=3, out_channels=1)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modelo U-Net carregado/atualizado com sucesso ({model_path}) no dispositivo: {device}")
        model_mtime = current_mtime
    else:
        print(f"[AVISO] Pesos não encontrados em {model_path}. Inicializando U-Net em modo de demonstração. Execute 3_train_unet.py para treinar os pesos ideais.")
        
    model.to(device)
    model.eval()
    
    model_cache = model
    return model_cache

def create_hann_window_2d(size):
    """Cria janela 2D Hann para fusão suave sem artefatos de borda."""
    hann_1d = np.hanning(size)
    hann_2d = np.outer(hann_1d, hann_1d)
    return np.clip(hann_2d, 1e-4, 1.0).astype(np.float32)

def resize_for_display(img, max_size=1920):
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    
    if w > h:
        new_w = max_size
        new_h = int(h * (max_size / w))
    else:
        new_h = max_size
        new_w = int(w * (max_size / h))
        
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def pil_to_base64(img, format="JPEG", quality=85):
    buffered = io.BytesIO()
    img.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(current_dir, "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="""
    <html>
        <head><title>Erro</title></head>
        <body style="font-family: sans-serif; background: #0b0f19; color: #f8fafc; text-align: center; padding-top: 10%;">
            <h1>static/index.html não encontrado</h1>
            <p>Por favor, certifique-se de que os arquivos estáticos estejam em src/static.</p>
        </body>
    </html>
    """)

def open_geospatial_image(contents: bytes) -> Image.Image:
    """Abre qualquer imagem padrão ou formato geoespacial (.tif, .tiff, .jp2, multibanda/16-bit)."""
    # 1. Tentativa padrão com PIL
    try:
        img = Image.open(io.BytesIO(contents))
        return img.convert("RGB")
    except Exception:
        pass

    # 2. Tentativa com Rasterio para GeoTIFFs complexos e JP2
    try:
        import rasterio
        with rasterio.open(io.BytesIO(contents)) as src:
            if src.count >= 3:
                data = src.read([1, 2, 3])
                arr = np.transpose(data, (1, 2, 0))
            elif src.count == 1:
                data = src.read(1)
                arr = np.stack([data, data, data], axis=-1)
            else:
                data = src.read()
                arr = np.transpose(data[:3], (1, 2, 0))
                
            if arr.dtype != np.uint8:
                mi, ma = float(arr.min()), float(arr.max())
                if ma > mi:
                    arr = ((arr - mi) / (ma - mi) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
                    
            return Image.fromarray(arr).convert("RGB")
    except Exception as e:
        raise ValueError(f"Formato de imagem não suportado ou arquivo corrompido: {e}")

STANDARD_PANEL_TYPES = [
    {
        "category": "Residencial Compacto",
        "tech": "Policristalino / Mono 60 céls",
        "power_w": 340,
        "area_m2": 1.70,
        "desc": "Instalações compactas ou telhados residenciais antigos"
    },
    {
        "category": "Residencial Moderno",
        "tech": "Half-Cell 108 céls M10",
        "power_w": 415,
        "area_m2": 1.95,
        "desc": "Padrão residencial atual (alta densidade em espaço reduzido)"
    },
    {
        "category": "Comercial / Médio Porte",
        "tech": "Half-Cell 120/144 céls",
        "power_w": 460,
        "area_m2": 2.15,
        "desc": "Telhados comerciais e residenciais amplos"
    },
    {
        "category": "Comercial Padrão Mercado",
        "tech": "Half-Cell 144 céls M10",
        "power_w": 550,
        "area_m2": 2.30,
        "desc": "Módulo mais comercializado no Brasil para telhados e usinas"
    },
    {
        "category": "Alta Potência Industrial",
        "tech": "Half-Cell 120/132 céls G12",
        "power_w": 600,
        "area_m2": 2.60,
        "desc": "Galpões industriais, agronegócio e usinas de solo"
    },
    {
        "category": "Ultra Potência / N-Type TOPCon",
        "tech": "Bifacial 132 céls G12 TOPCon",
        "power_w": 680,
        "area_m2": 2.85,
        "desc": "Módulos de última geração com máxima potência de saída"
    }
]

def calculate_panel_types_breakdown(s_pv: float, i_local: float = 5.4, pr: float = 0.75, packing_factor: float = 0.90):
    """Calcula a estimativa comparativa de placas, potência e geração para todas as categorias de mercado."""
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

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    gsd: float = Form(0.0389),
    eta: float = Form(0.185),
    i_local: float = Form(5.4),
    threshold: float = Form(0.65),
    min_area: int = Form(250),
    tile_size: int = Form(512),
    overlap: bool = Form(True),
    pr: float = Form(0.75),
    tilt_angle: float = Form(25.0),
    latitude: float = Form(-22.22),
    use_tilt_correction: bool = Form(True),
    module_power_w: float = Form(550.0),
    module_area_m2: float = Form(2.30),
    packing_factor: float = Form(0.90)
):
    try:
        net = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar o modelo de rede neural: {e}")
        
    try:
        contents = await file.read()
        image = open_geospatial_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar arquivo GeoTIFF/Imagem: {e}")
        
    width, height = image.size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalização de Escala (Invariância de Resolução GSD):
    # A U-Net foi calibrada para a escala nominal de ~0.0389 m/pixel.
    # Ajusta a escala da imagem durante o tiling para que a rede neural sempre observe os painéis
    # com o campo receptivo ideal, independentemente do DPI (96, 163, 300, etc.) exportado no QGIS.
    nominal_gsd = 0.0389
    scale_factor = max(0.2, min(5.0, gsd / nominal_gsd))
    
    if abs(scale_factor - 1.0) > 0.08:
        infer_w = max(tile_size, int(round(width * scale_factor)))
        infer_h = max(tile_size, int(round(height * scale_factor)))
        infer_image = image.resize((infer_w, infer_h), Image.Resampling.BILINEAR)
    else:
        infer_w, infer_h = width, height
        infer_image = image
        
    # Configuração de sobreposição (overlap)
    overlap_ratio = 0.25 if overlap else 0.0
    stride = int(tile_size * (1.0 - overlap_ratio)) if overlap else tile_size
    stride = max(1, min(stride, tile_size))
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    hann_weight = create_hann_window_2d(tile_size) if overlap else np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Matrizes de acumulação na resolução de inferência
    full_prob = np.zeros((infer_h, infer_w), dtype=np.float32)
    full_weight = np.zeros((infer_h, infer_w), dtype=np.float32)
    
    # Função auxiliar para gerar coordenadas uniformes sem passos redundantes
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

    x_coords = get_axis_coords(infer_w, tile_size, stride)
    y_coords = get_axis_coords(infer_h, tile_size, stride)
    
    total_tiles = 0
    candidate_gallery_tiles = []
    max_gallery_tiles = 8
    
    for y in y_coords:
        for x in x_coords:
            box_x = min(x, max(0, infer_w - tile_size))
            box_y = min(y, max(0, infer_h - tile_size))
            
            tile = infer_image.crop((box_x, box_y, box_x + tile_size, box_y + tile_size))
            active_w, active_h = tile.size
            
            if active_w < tile_size or active_h < tile_size:
                padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile, (0, 0))
                tile_tensor = transform(padded).unsqueeze(0).to(device)
            else:
                tile_tensor = transform(tile).unsqueeze(0).to(device)
                
            total_tiles += 1
            
            with torch.no_grad():
                output = net(tile_tensor)
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
                
            prob_crop = prob[:active_h, :active_w]
            weight_crop = hann_weight[:active_h, :active_w]
            
            full_prob[box_y:box_y + active_h, box_x:box_x + active_w] += prob_crop * weight_crop
            full_weight[box_y:box_y + active_h, box_x:box_x + active_w] += weight_crop
            
            tile_mask = prob_crop > threshold
            ratio = np.mean(tile_mask)
            
            if ratio > 0.005:
                tile_np = np.array(tile)
                tile_overlay = tile_np.copy()
                tile_overlay[tile_mask] = [251, 191, 36] # Dourado solar
                tile_overlay_img = Image.blend(tile, Image.fromarray(tile_overlay), alpha=0.65)
                
                candidate_gallery_tiles.append({
                    "box_x": box_x,
                    "box_y": box_y,
                    "width": active_w,
                    "height": active_h,
                    "ratio": ratio,
                    "tile": tile,
                    "overlay": tile_overlay_img
                })
                
    # Deduplicação Espacial da Galeria de Tiles (NMS Espacial)
    # Evita recortes duplicados/redundantes da mesma edificação em imagens pequenas ou com sobreposição de janela
    candidate_gallery_tiles.sort(key=lambda c: c["ratio"], reverse=True)
    selected_gallery_tiles = []
    
    for cand in candidate_gallery_tiles:
        cx = cand["box_x"] + cand["width"] / 2.0
        cy = cand["box_y"] + cand["height"] / 2.0
        
        is_duplicate = False
        for sel in selected_gallery_tiles:
            sel_cx = sel["box_x"] + sel["width"] / 2.0
            sel_cy = sel["box_y"] + sel["height"] / 2.0
            dist = math.hypot(cx - sel_cx, cy - sel_cy)
            
            # Sobreposição de caixas (IoU)
            ix1 = max(cand["box_x"], sel["box_x"])
            iy1 = max(cand["box_y"], sel["box_y"])
            ix2 = min(cand["box_x"] + cand["width"], sel["box_x"] + sel["width"])
            iy2 = min(cand["box_y"] + cand["height"], sel["box_y"] + sel["height"])
            
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = iw * ih
            union_area = (cand["width"] * cand["height"]) + (sel["width"] * sel["height"]) - inter_area
            iou = inter_area / max(1.0, union_area)
            
            if dist < (tile_size * 0.45) or iou > 0.40:
                is_duplicate = True
                break
                
        if not is_duplicate:
            selected_gallery_tiles.append(cand)
            if len(selected_gallery_tiles) >= max_gallery_tiles:
                break
                
    detected_tiles_gallery = []
    for idx, item in enumerate(selected_gallery_tiles):
        orig_tile_b64 = pil_to_base64(item["tile"], format="JPEG", quality=85)
        overlay_tile_b64 = pil_to_base64(item["overlay"], format="JPEG", quality=85)
        
        detected_tiles_gallery.append({
            "id": idx + 1,
            "x": item["box_x"],
            "y": item["box_y"],
            "width": item["width"],
            "height": item["height"],
            "orig_b64": orig_tile_b64,
            "overlay_b64": overlay_tile_b64,
            "detection_ratio": round(item["ratio"] * 100, 2)
        })
        
    tiles_with_detection = len(selected_gallery_tiles)
                    
    # Normalização dos pesos de sobreposição
    full_weight[full_weight == 0] = 1.0
    full_prob /= full_weight
    
    # Se houve re-escala de inferência, mapeia o mapa de probabilidade de volta à resolução original da imagem
    if (infer_w, infer_h) != (width, height):
        full_prob = cv2.resize(full_prob, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # 1. Limiarização
    pred_mask = (full_prob > threshold).astype(np.uint8) * 255
    
    # 2. Pós-Processamento Morfológico (Fechamento)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Filtro de Área Mínima (Connected Components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    filtered_mask = np.zeros_like(pred_mask)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255
            
    pred_mask = filtered_mask
    
    # Contagem de pixels e agrupamentos de painéis
    total_solar_pixels = int(np.sum(pred_mask > 0))
    final_num_labels, _, _, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    detected_panel_groups = max(0, final_num_labels - 1)
    
    # --- MODELO MATEMÁTICO COM CORREÇÃO DE INCLINAÇÃO ---
    # 1. Área Projetada Ortogonal (A_proj)
    area_per_pixel = gsd * gsd
    a_proj = total_solar_pixels * area_per_pixel
    
    # 2. Área Real Corrigida (S_pv = A_proj / cos(β))
    # Ortofotos capturam a projeção horizontal plana; os painéis reais nos telhados têm inclinação β.
    if use_tilt_correction and tilt_angle > 0:
        beta_rad = math.radians(tilt_angle)
        cos_beta = math.cos(beta_rad)
        if cos_beta > 0.1:
            tilt_factor = 1.0 / cos_beta
        else:
            tilt_factor = 1.0
            
        tilt_factor = max(1.0, float(tilt_factor))
        s_pv = a_proj * tilt_factor
    else:
        tilt_factor = 1.0
        s_pv = a_proj
        
    # 3. Potência de Pico Instalada (kWp)
    p_pico = s_pv * eta
    
    # 4. Geração Diária e Anual (kWh)
    generation_daily_kwh = p_pico * i_local * pr
    generation_annual_kwh = generation_daily_kwh * 365.0
    
    # 5. Estimativa de Módulos Físicos Discretos (Considerando Fator de Ocupação/Frestas)
    effective_pv_area = s_pv * packing_factor
    estimated_modules = int(round(effective_pv_area / module_area_m2)) if module_area_m2 > 0 else 0
    
    # Criação do Overlay Visual
    img_np = np.array(image)
    overlay_mask = np.zeros_like(img_np)
    overlay_mask[pred_mask > 0] = [251, 191, 36] # Dourado Solar
    overlay_img = Image.blend(image, Image.fromarray(overlay_mask), alpha=0.65)
    
    display_orig = resize_for_display(image)
    display_mask = resize_for_display(Image.fromarray(pred_mask, mode="L"))
    display_overlay = resize_for_display(overlay_img)
    
    orig_b64 = pil_to_base64(display_orig, format="JPEG", quality=80)
    mask_b64 = pil_to_base64(display_mask, format="PNG")
    overlay_b64 = pil_to_base64(display_overlay, format="JPEG", quality=80)
    
    return {
        "metadata": {
            "filename": file.filename,
            "width": width,
            "height": height,
            "total_tiles": total_tiles,
            "tiles_with_detection": tiles_with_detection,
            "device_used": str(device),
            "tilt_angle": tilt_angle,
            "latitude": latitude,
            "tilt_factor": round(tilt_factor, 3),
            "use_tilt_correction": use_tilt_correction
        },
        "results": {
            "total_solar_pixels": total_solar_pixels,
            "gsd": gsd,
            "area_projetada_m2": round(a_proj, 2),
            "area_total_m2": round(s_pv, 2),
            "s_pv_m2": round(s_pv, 2),
            "tilt_factor": round(tilt_factor, 3),
            "tilt_angle": tilt_angle,
            "latitude": latitude,
            "potencia_pico_kwp": round(p_pico, 2),
            "geracao_diaria_kwh": round(generation_daily_kwh, 2),
            "geracao_anual_kwh": round(generation_annual_kwh, 2),
            "detected_groups": detected_panel_groups,
            "estimated_modules": estimated_modules,
            "pr": pr,
            "eta": eta,
            "i_local": i_local
        },
        "images": {
            "original_b64": orig_b64,
            "mask_b64": mask_b64,
            "overlay_b64": overlay_b64
        },
        "results_text": f"Área Real: {s_pv:.2f} m² (Projetada: {a_proj:.2f} m²) | Potência Pico: {p_pico:.2f} kWp | Módulos: ~{estimated_modules} un",
        "tiles_gallery": detected_tiles_gallery,
        "panel_types_breakdown": calculate_panel_types_breakdown(s_pv, i_local, pr, packing_factor)
    }

app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
