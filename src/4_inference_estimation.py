import os
import torch
import numpy as np
import cv2
from PIL import Image
import importlib
import sys

# Adiciona a pasta atual (src) e a pasta raiz ao Python path para permitir a importação
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importamos a U-Net do script anterior contornando o erro de sintaxe
train_module = importlib.import_module("3_train_unet")
UNet = train_module.UNet

from torchvision.transforms import v2

def calculate_power_estimation(total_pixels, gsd):
    """
    Realiza o cálculo do modelo matemático da Seção 3.3.
    """
    # 1. Área Total (A_total) em m²
    # Cada pixel representa (gsd * gsd) metros quadrados
    area_per_pixel = gsd * gsd
    a_total = total_pixels * area_per_pixel
    
    # Parâmetros validados na literatura / base climática
    # Eficiência (η) média comercial (18.5%)
    eta = 0.185
    
    # Índice de Irradiação Solar (I_local) para Dourados - MS
    # Média já existente e calculada (aprox 5.4 kWh/m²/dia)
    i_local = 5.4
    
    # Modelo Matemático: Pest = Atotal * η * Ilocal
    p_est = a_total * eta * i_local
    
    return a_total, p_est

def calculate_estimation_errors(estimated, true_val):
    """
    Calcula as métricas de erro para a estimativa de área e energia.
    MAE (Erro Absoluto Médio), MAPE (Erro Percentual Médio) e RMSE.
    """
    if true_val is None or true_val <= 0:
        return None, None, None
        
    est = float(estimated)
    true_v = float(true_val)
    
    mae = abs(est - true_v)
    mape = (abs(true_v - est) / true_v) * 100
    rmse = np.sqrt((est - true_v)**2) # Para 1 amostra, RMSE = MAE. Se houvessem várias amostras, seria a média.
    
    return mae, mape, rmse

def inference_and_estimation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Inferência usando: {device}")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_img_dir = os.path.join(base_dir, "data", "processed", "test", "Placas")
    preds_dir = os.path.join(base_dir, "data", "processed", "test", "preds_placas")
    os.makedirs(preds_dir, exist_ok=True)
    
    model_path = os.path.join(base_dir, "models", "unet_solar.pth")
    if not os.path.exists(model_path):
        print(f"Modelo não encontrado em {model_path}. Treine o modelo primeiro.")
        return
        
    # Carrega a rede
    model = UNet(in_channels=3, out_channels=1)
    # Ignora erro estrito se rodar em CPU sem treinar antes
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Aviso ao carregar pesos: {e}")
        
    model.to(device)
    model.eval()
    
    # Tenta achar uma imagem .jp2 para definir o GSD fixo
    jp2_files = [f for f in os.listdir(os.path.join(base_dir, "data", "raw")) if f.endswith(".jp2")]
    
    if jp2_files:
        gsd = 0.0389 # GSD extraído do QGIS para a imagem INOCENCIA - MS.jp2
        print(f"Detectada ortofoto .jp2. Usando GSD atualizado: {gsd:.4f} m/pixel (aprox 3.89cm).")
    else:
        # Tenta ler GSD do World File (.jgw) das imagens de API
        jgw_path = os.path.join(base_dir, "data", "raw", "esri_highres_stitched.jgw")
        gsd = 0.2986 # Fallback GSD padrão para Zoom 19 (Bing)
        if os.path.exists(jgw_path):
            with open(jgw_path, 'r') as f:
                gsd = float(f.readline().strip())
    
    print(f"GSD identificado: {gsd:.4f} m/pixel")
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    
    total_solar_pixels = 0
    
    with torch.no_grad():
        for img_name in test_images:
            img_path = os.path.join(test_img_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            
            # Prepara a imagem (Add batch dimension)
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Inferência
            output = model(input_tensor)
            
            # 1. Ajuste do Threshold (meio-termo ideal: 0.65)
            prob = torch.sigmoid(output)
            pred_mask = (prob > 0.55).float().squeeze().cpu().numpy()
            
            pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)

            # 2. Operação Morfológica (Fechamento/Dilatação): Junta painéis que estão "colados"
            # O modelo às vezes detecta o painel todo fragmentado. O 'Fechamento' preenche esses buracos e gruda eles.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            pred_mask_uint8 = cv2.morphologyEx(pred_mask_uint8, cv2.MORPH_CLOSE, kernel)
            
            # 3. Filtro de Área Mínima (Remove ruídos pequenos que restaram)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask_uint8, connectivity=8)
            
            min_area_pixels = 250 # Blobs menores que ~0.4 m² são deletados
            filtered_mask = np.zeros_like(pred_mask_uint8)
            
            for i in range(1, num_labels): # Ignora o background (0)
                if stats[i, cv2.CC_STAT_AREA] >= min_area_pixels:
                    filtered_mask[labels == i] = 1
                    
            pred_mask = filtered_mask.astype(np.float32)
            
            # Conta pixels solares
            solar_pixels_in_image = np.sum(pred_mask)
            total_solar_pixels += solar_pixels_in_image
            
            # Salva o mapa de predição
            pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L")
            pred_img.save(os.path.join(preds_dir, f"pred_{img_name}"))
            
    print(f"\nInferência concluída em {len(test_images)} imagens de teste.")
    print(f"Total de Pixels Classificados como 'Painel Solar': {total_solar_pixels}")
    
    # Cálculos Matemáticos (Seção 3.3)
    a_total, p_est = calculate_power_estimation(total_solar_pixels, gsd)
    
    print("-" * 50)
    print("RELATÓRIO DE ESTIMATIVA DE POTÊNCIA (Seção 3.3)")
    print("-" * 50)
    print(f"Área Total Identificada (A_total): {a_total:.2f} m²")
    print(f"Eficiência Adotada (eta): 18.5%")
    print(f"Índice de Irradiação (I_local): 5.4 kWh/m²/dia")
    print("-" * 50)
    print(f"POTÊNCIA MÉDIA DE GERAÇÃO ESTIMADA (P_est): {p_est:.2f} kW")
    print("-" * 50)

    # =========================================================
    # MÉTRICAS DE ESTIMATIVA DE ÁREA E ENERGIA (Para o TCC)
    # Insira os valores reais abaixo para calcular os erros.
    # Exemplo: dados reais da ANEEL, fatura da concessionária.
    # =========================================================
    real_area = None  # Substitua pelo valor real da área em m² (se souber)
    real_power = None # Substitua pelo valor real de potência em kW (se souber)
    
    if real_area is not None or real_power is not None:
        print("\n" + "=" * 50)
        print("MÉTRICAS DE AVALIAÇÃO (ERROS DE ESTIMATIVA)")
        print("=" * 50)
        
        if real_area is not None:
            mae_area, mape_area, rmse_area = calculate_estimation_errors(a_total, real_area)
            print(f"--- Área ---")
            print(f"Estimado: {a_total:.2f} m² | Real: {real_area:.2f} m²")
            print(f"MAE (Erro Absoluto): {mae_area:.2f} m²")
            print(f"MAPE (Erro Percentual): {mape_area:.2f}%")
            print(f"RMSE: {rmse_area:.2f} m²\n")
            
        if real_power is not None:
            mae_pwr, mape_pwr, rmse_pwr = calculate_estimation_errors(p_est, real_power)
            print(f"--- Potência / Energia ---")
            print(f"Estimado: {p_est:.2f} kW | Real: {real_power:.2f} kW")
            print(f"MAE (Erro Absoluto): {mae_pwr:.2f} kW")
            print(f"MAPE (Erro Percentual): {mape_pwr:.2f}%")
            print(f"RMSE: {rmse_pwr:.2f} kW")
        print("=" * 50)

if __name__ == "__main__":
    inference_and_estimation()
