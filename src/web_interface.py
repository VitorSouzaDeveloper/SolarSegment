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

# Cache global para o modelo U-Net
model_cache = None

def get_model():
    global model_cache
    if model_cache is not None:
        return model_cache
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    model_path = os.path.join(base_dir, "models", "unet_solar_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "models", "unet_solar.pth")
        
    try:
        train_module = importlib.import_module("3_train_unet")
        UNet = train_module.UNet
    except Exception as e:
        raise ImportError(f"Erro ao importar a U-Net de 3_train_unet: {e}")
        
    model = UNet(in_channels=3, out_channels=1)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modelo U-Net carregado com sucesso ({model_path}) no dispositivo: {device}")
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

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    gsd: float = Form(0.0389),
    eta: float = Form(0.185),
    i_local: float = Form(5.4),
    threshold: float = Form(0.55),
    min_area: int = Form(250),
    tile_size: int = Form(512),
    overlap: bool = Form(True),
    pr: float = Form(0.75),
    tilt_angle: float = Form(20.0),
    latitude: float = Form(-22.22),
    use_tilt_correction: bool = Form(True),
    module_power_w: float = Form(550.0),
    module_area_m2: float = Form(2.20)
):
    try:
        net = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar o modelo de rede neural: {e}")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler imagem enviada: {e}")
        
    width, height = image.size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuração de sobreposição (overlap)
    overlap_ratio = 0.25 if overlap else 0.0
    stride = int(tile_size * (1.0 - overlap_ratio)) if overlap else tile_size
    stride = max(1, min(stride, tile_size))
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    hann_weight = create_hann_window_2d(tile_size) if overlap else np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Matrizes de acumulação
    full_prob = np.zeros((height, width), dtype=np.float32)
    full_weight = np.zeros((height, width), dtype=np.float32)
    
    x_coords = list(range(0, width, stride))
    y_coords = list(range(0, height, stride))
    
    if len(x_coords) == 0 or x_coords[-1] + tile_size < width:
        x_coords.append(max(0, width - tile_size))
    if len(y_coords) == 0 or y_coords[-1] + tile_size < height:
        y_coords.append(max(0, height - tile_size))
        
    x_coords = sorted(list(set(x_coords)))
    y_coords = sorted(list(set(y_coords)))
    
    total_tiles = 0
    tiles_with_detection = 0
    detected_tiles_gallery = []
    max_gallery_tiles = 8
    
    for y in y_coords:
        for x in x_coords:
            total_tiles += 1
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            tile = image.crop((x, y, x_end, y_end))
            active_h = y_end - y
            active_w = x_end - x
            
            if active_w < tile_size or active_h < tile_size:
                padded_tile = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                padded_tile.paste(tile, (0, 0))
                input_tensor = transform(padded_tile).unsqueeze(0).to(device)
            else:
                input_tensor = transform(tile).unsqueeze(0).to(device)
                
            with torch.no_grad():
                output = net(input_tensor)
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
                
            prob_active = prob[0:active_h, 0:active_w]
            weight_active = hann_weight[0:active_h, 0:active_w]
            
            full_prob[y:y_end, x:x_end] += prob_active * weight_active
            full_weight[y:y_end, x:x_end] += weight_active
            
            tile_mask = prob_active > threshold
            ratio = float(np.mean(tile_mask))
            
            if ratio > 0.015:
                tiles_with_detection += 1
                if len(detected_tiles_gallery) < max_gallery_tiles:
                    tile_np = np.array(tile)
                    tile_overlay = tile_np.copy()
                    tile_overlay[tile_mask] = [251, 191, 36] # Dourado solar
                    tile_overlay_img = Image.blend(tile, Image.fromarray(tile_overlay), alpha=0.55)
                    
                    orig_tile_b64 = pil_to_base64(tile, format="JPEG", quality=85)
                    overlay_tile_b64 = pil_to_base64(tile_overlay_img, format="JPEG", quality=85)
                    
                    detected_tiles_gallery.append({
                        "id": len(detected_tiles_gallery) + 1,
                        "x": x,
                        "y": y,
                        "width": active_w,
                        "height": active_h,
                        "orig_b64": orig_tile_b64,
                        "overlay_b64": overlay_tile_b64,
                        "detection_ratio": round(ratio * 100, 2)
                    })
                    
    # Normalização dos pesos de sobreposição
    full_weight[full_weight == 0] = 1.0
    full_prob /= full_weight
    
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
    
    # 5. Estimativa de Módulos Físicos Discretos
    estimated_modules = int(round(s_pv / module_area_m2)) if module_area_m2 > 0 else 0
    
    # Criação do Overlay Visual
    img_np = np.array(image)
    overlay_mask = np.zeros_like(img_np)
    overlay_mask[pred_mask > 0] = [251, 191, 36] # Dourado Solar
    overlay_img = Image.blend(image, Image.fromarray(overlay_mask), alpha=0.55)
    
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
        "tiles_gallery": detected_tiles_gallery
    }

app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
