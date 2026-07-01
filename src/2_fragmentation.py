import os
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image

def fragment_image():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    jp2_files = [f for f in os.listdir(os.path.join(base_dir, "data", "raw")) if f.endswith(".jp2")]
    
    if jp2_files:
        raw_img_path = os.path.join(base_dir, "data", "raw", jp2_files[0])
        print(f"Ortofoto de alta qualidade (.jp2) encontrada: {jp2_files[0]}")
    else:
        raw_img_path = os.path.join(base_dir, "data", "raw", "esri_highres_stitched.jpeg")
        print("Usando imagem gerada por API (Bing/ESRI).")

    if not os.path.exists(raw_img_path):
        print(f"Erro: Nenhuma imagem encontrada em {raw_img_path}.")
        return

    processed_dir = os.path.join(base_dir, "data", "processed")
    train_dir = os.path.join(processed_dir, "train", "images")
    val_dir = os.path.join(processed_dir, "val", "images")
    test_dir = os.path.join(processed_dir, "test", "images")
    
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
        
    TILE_SIZE = 512
    stride = 256
    count = 0
    
    print("Iniciando fragmentação Otimizada (Rasterio) para evitar MemoryError...")
    
    with rasterio.open(raw_img_path) as src:
        img_w = src.width
        img_h = src.height
        
        print(f"Dimensões totais da imagem: {img_w}x{img_h}")
        
        print("Procurando áreas URBANAS na imagem (ignorando pastos, matos e bordas)...")
        
        # Filtro de Variância (Desvio Padrão):
        # Áreas rurais (pastos lisos) têm cores muito uniformes (baixo desvio padrão).
        # Áreas urbanas (casas, ruas, sombras) têm alto contraste e desvio padrão.
        
        import random
        max_attempts = 50000
        attempts = 0
        
        while count < 800 and attempts < max_attempts:
            attempts += 1
            
            # Sorteia uma coordenada aleatória na imagem gigante
            x = random.randint(0, img_w - TILE_SIZE)
            y = random.randint(0, img_h - TILE_SIZE)
            
            window = Window(x, y, TILE_SIZE, TILE_SIZE)
            tile_data = src.read(window=window)
            
            # 1. Ignora tiles que caíram no preenchimento preto (borda)
            if tile_data.max() == 0:
                continue
                
            # 2. Ignora áreas rurais muito planas (baixo contraste)
            # Um desvio padrão abaixo de 35 a 45 geralmente indica pastagem ou floresta densa uniforme
            if tile_data.std() < 35:
                continue
            
            if tile_data.shape[0] >= 3:
                tile_data = tile_data[:3, :, :]
                tile_data = np.transpose(tile_data, (1, 2, 0))
            elif tile_data.shape[0] == 1:
                tile_data = tile_data[0]
            
            tile_img = Image.fromarray(tile_data)
            
            if count % 10 == 8:
                out_dir = val_dir
            elif count % 10 == 9:
                out_dir = test_dir
            else:
                out_dir = train_dir
                
            tile_filename = f"tile_urbano_{count:03d}_x{x}_y{y}.jpeg"
            tile_filepath = os.path.join(out_dir, tile_filename)
            tile_img.save(tile_filepath, quality=95)
            
            count += 1
            
            if count % 50 == 0:
                print(f"Encontrados e extraídos {count} tiles urbanos...")

    print(f"Fragmentação concluída! {count} tiles de {TILE_SIZE}x{TILE_SIZE} gerados.")
    print(f"Treino: ~{int(count * 0.8)} | Validação: ~{int(count * 0.1)} | Teste: ~{int(count * 0.1)}")

if __name__ == "__main__":
    fragment_image()
