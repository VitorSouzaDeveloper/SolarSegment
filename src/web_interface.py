import os
import io
import sys
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

app = FastAPI(title="TCC II Solar Panel Detection & Power Estimation")

# Adiciona a pasta atual ao Python path para permitir a importação de 3_train_unet
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Certifica-se de que a pasta static existe
os.makedirs(os.path.join(current_dir, "static"), exist_ok=True)

# Cache global para o modelo U-Net
model_cache = None

def get_model():
    global model_cache
    if model_cache is not None:
        return model_cache
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    model_path = os.path.join(base_dir, "models", "unet_solar.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo de pesos do modelo não encontrado em {model_path}")
        
    try:
        # Importação dinâmica para contornar o nome do arquivo começando com número
        train_module = importlib.import_module("3_train_unet")
        UNet = train_module.UNet
    except Exception as e:
        raise ImportError(f"Erro ao importar a U-Net de 3_train_unet: {e}")
        
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    model_cache = model
    print(f"Modelo U-Net carregado com sucesso no dispositivo: {device}")
    return model_cache

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
        <body style="font-family: sans-serif; background: #0f172a; color: #f8fafc; text-align: center; padding-top: 10%;">
            <h1>static/index.html não encontrado</h1>
            <p>Por favor, crie o arquivo index.html no diretório src/static.</p>
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
    overlap: bool = Form(False)
):
    try:
        # Carrega o modelo
        net = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar o modelo de rede neural: {e}")
        
    try:
        # Lê os bytes da imagem enviada
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler imagem enviada: {e}")
        
    width, height = image.size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determina o stride baseado na sobreposição
    stride = tile_size // 2 if overlap else tile_size
    
    # Transformador
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    # Matrizes de acumulação para reconstruir a imagem
    full_prob = np.zeros((height, width), dtype=np.float32)
    full_weight = np.zeros((height, width), dtype=np.float32)
    
    x_coords = list(range(0, width, stride))
    y_coords = list(range(0, height, stride))
    
    # Adiciona as coordenadas finais caso a divisão não seja exata
    if len(x_coords) == 0 or x_coords[-1] + tile_size < width:
        x_coords.append(width - tile_size)
    if len(y_coords) == 0 or y_coords[-1] + tile_size < height:
        y_coords.append(height - tile_size)
        
    # Garante que as coordenadas sejam não-negativas
    x_coords = [max(0, x) for x in x_coords]
    y_coords = [max(0, y) for y in y_coords]
    
    # Remove duplicadas mantendo a ordem
    x_coords = sorted(list(set(x_coords)))
    y_coords = sorted(list(set(y_coords)))
    
    total_tiles = 0
    tiles_with_detection = 0
    detected_tiles_gallery = []
    max_gallery_tiles = 8
    
    # Loop de fragmentação e inferência tile-por-tile
    for y in y_coords:
        for x in x_coords:
            total_tiles += 1
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            # Recorta o tile correspondente
            tile = image.crop((x, y, x_end, y_end))
            
            # Pad com preto se o tile for menor que tile_size (nos limites direito/inferior)
            active_h = y_end - y
            active_w = x_end - x
            
            if active_w < tile_size or active_h < tile_size:
                padded_tile = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                padded_tile.paste(tile, (0, 0))
                input_tensor = transform(padded_tile).unsqueeze(0).to(device)
            else:
                input_tensor = transform(tile).unsqueeze(0).to(device)
                
            # Roda inferência
            with torch.no_grad():
                output = net(input_tensor)
                prob = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy() # shape (512, 512)
                
            # Extrai apenas a parte ativa correspondente à imagem original
            prob_active = prob[0:active_h, 0:active_w]
            
            # Acumula na matriz geral
            full_prob[y:y_end, x:x_end] += prob_active
            full_weight[y:y_end, x:x_end] += 1.0
            
            # Analisa se o tile possui detecção significativa
            tile_mask = prob_active > threshold
            ratio = float(np.mean(tile_mask))
            
            if ratio > 0.015:
                tiles_with_detection += 1
                # Se ainda houver vaga na galeria, prepara o tile e seu overlay
                if len(detected_tiles_gallery) < max_gallery_tiles:
                    tile_np = np.array(tile)
                    tile_overlay = tile_np.copy()
                    # Desenha overlay amarelo dourado translúcido nas coordenadas preditas
                    tile_overlay[tile_mask] = [251, 191, 36]
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
                    
    # Média das áreas sobrepostas
    full_weight[full_weight == 0] = 1.0
    full_prob /= full_weight
    
    # 1. Aplicação do Threshold
    pred_mask = (full_prob > threshold).astype(np.uint8) * 255
    
    # 2. Pós-Processamento Morfológico (Fechamento)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Filtro de Área Mínima (OpenCV Connected Components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    filtered_mask = np.zeros_like(pred_mask)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255
            
    pred_mask = filtered_mask
    
    # Contagem de pixels classificados como Painel Solar
    total_solar_pixels = int(np.sum(pred_mask > 0))
    
    # Contagem final de grupos de painéis conectados (após o filtro)
    final_num_labels, _, _, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    detected_panel_groups = max(0, final_num_labels - 1)
    
    # Cálculos Matemáticos de Estimativa
    area_per_pixel = gsd * gsd
    a_total = total_solar_pixels * area_per_pixel
    p_est = a_total * eta * i_local
    
    # Geração diária em kWh/dia e geração anual em kWh/ano
    generation_daily_kwh = a_total * eta * i_local
    generation_annual_kwh = generation_daily_kwh * 365
    
    # Cria a imagem de overlay final
    img_np = np.array(image)
    overlay_mask = np.zeros_like(img_np)
    # Cor amarela dourada para o overlay
    overlay_mask[pred_mask > 0] = [251, 191, 36]
    overlay_img = Image.blend(image, Image.fromarray(overlay_mask), alpha=0.55)
    
    # Redimensiona para exibição web para manter as base64 leves
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
            "device_used": str(device)
        },
        "results": {
            "total_solar_pixels": total_solar_pixels,
            "gsd": gsd,
            "area_total_m2": round(a_total, 2),
            "potencia_estimada_kw": round(p_est, 2),
            "geracao_diaria_kwh": round(generation_daily_kwh, 2),
            "geracao_anual_kwh": round(generation_annual_kwh, 2),
            "detected_groups": detected_panel_groups
        },
        "images": {
            "original_b64": orig_b64,
            "mask_b64": mask_b64,
            "overlay_b64": overlay_b64
        },
        "results_text": f"Área de Painéis: {a_total:.2f} m² | Potência Média: {p_est:.2f} kW | Grupos Detectados: {detected_panel_groups}",
        "tiles_gallery": detected_tiles_gallery
    }

# Monta arquivos estáticos
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    # Inicializa na porta 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
