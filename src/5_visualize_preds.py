import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_random_predictions(num_samples=10):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_dir = os.path.join(base_dir, "data", "processed", "test", "Placas")
    preds_dir = os.path.join(base_dir, "data", "processed", "test", "preds_placas")
    output_dir = os.path.join(base_dir, "data", "processed", "test", "visualizations_placas")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(img_dir) or not os.path.exists(preds_dir):
        print("Diretórios de imagens ou predições não encontrados.")
        return

    # Lista todas as imagens originais
    all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
    
    if len(all_images) == 0:
        print("Nenhuma imagem de teste encontrada.")
        return

    # Filtra imagens que realmente possuem máscara predita (para evitar erros)
    valid_images = [img for img in all_images if os.path.exists(os.path.join(preds_dir, f"pred_{img}"))]
    
    # Tenta focar em imagens que o modelo detectou algo (se possível), para não ver apenas fundos pretos
    # Vamos ler um subconjunto e tentar achar as que tem pixel branco
    interesting_images = []
    print("Buscando imagens onde o modelo detectou painéis solares...")
    random.shuffle(valid_images)
    for img in valid_images:
        pred_arr = np.array(Image.open(os.path.join(preds_dir, f"pred_{img}")).convert("L"))
        if np.sum(pred_arr) > 0: # Tem pelo menos 1 pixel predito como painel
            interesting_images.append(img)
        if len(interesting_images) >= num_samples:
            break
            
    # Se não achou painéis suficientes, completa com outras normais
    if len(interesting_images) < num_samples:
        remaining = [img for img in valid_images if img not in interesting_images]
        interesting_images.extend(remaining[:num_samples - len(interesting_images)])

    for img_name in interesting_images:
        orig_path = os.path.join(img_dir, img_name)
        pred_path = os.path.join(preds_dir, f"pred_{img_name}")

        orig_img = Image.open(orig_path).convert("RGB")
        pred_mask = Image.open(pred_path).convert("L")
        
        orig_arr = np.array(orig_img)
        pred_arr = np.array(pred_mask)
        
        # Cria um overlay: destaca de vermelho onde o modelo achou painel
        overlay = orig_arr.copy()
        overlay[pred_arr > 127] = [255, 0, 0] 
        overlay_img = Image.blend(orig_img, Image.fromarray(overlay), alpha=0.6)

        # Plota a comparação
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Original: {img_name}")
        axes[0].axis("off")
        
        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("Máscara do Modelo (U-Net)")
        axes[1].axis("off")
        
        axes[2].imshow(overlay_img)
        axes[2].set_title("Sobreposição (Vermelho = Painel)")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # Salva o arquivo de visualização
        save_path = os.path.join(output_dir, f"viz_{img_name}")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
    print(f"Salvas {len(interesting_images)} imagens de visualização na pasta:")
    print(output_dir)

if __name__ == "__main__":
    visualize_random_predictions(num_samples=15)
