import os
import torch
from PIL import Image
import numpy as np
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import importlib
train_module = importlib.import_module("3_train_unet")
UNet = train_module.UNet
from torchvision.transforms import v2

def generate_pseudo_labels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    train_img_dir = os.path.join(base_dir, "data", "processed", "train", "images")
    train_mask_dir = os.path.join(base_dir, "data", "processed", "train", "masks")
    model_path = os.path.join(base_dir, "models", "unet_solar.pth")
    
    os.makedirs(train_mask_dir, exist_ok=True)
    
    print("Carregando modelo atual...")
    model = UNet(in_channels=3, out_channels=1)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return
        
    model.to(device)
    model.eval()
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    all_images = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    
    print(f"Encontradas {len(all_images)} imagens em {train_img_dir}")
    
    generated_count = 0
    with torch.no_grad():
        for img_name in all_images:
            mask_name = img_name.replace(".jpeg", ".png").replace(".jpg", ".png")
            mask_path = os.path.join(train_mask_dir, mask_name)
            
            # Evita sobrescrever máscaras que você talvez já tenha feito
            if os.path.exists(mask_path):
                continue
                
            img_path = os.path.join(train_img_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            
            # Usando threshold 0.5 para ser equilibrado
            prob = torch.sigmoid(output)
            pred_mask = (prob > 0.5).float().squeeze().cpu().numpy()
            
            pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
            pred_img = Image.fromarray(pred_mask_uint8, mode="L")
            pred_img.save(mask_path)
            generated_count += 1
            
    print("-" * 50)
    print(f"SUCESSO! {generated_count} novas máscaras (Pseudo-Labels) foram geradas.")
    print(f"Elas estão salvas em: {train_mask_dir}")
    print("-" * 50)

if __name__ == "__main__":
    generate_pseudo_labels()
