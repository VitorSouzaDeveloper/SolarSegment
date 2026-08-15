import os
import random
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

def fragment_image(mode="systematic", tile_size=512, overlap_ratio=0.25, max_tiles=800):
    """
    Fragmentador de ortofotos e imagens de satélite de alta resolução (TCC II).
    
    Modos:
    - 'systematic': Grade deslizante estruturada com sobreposição (overlap) para mitigar
                    o corte de painéis solares nas bordas e garantir cobertura uniforme.
    - 'random': Sorteio estocástico de coordenadas com filtro de contraste urbano.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    raw_dir = os.path.join(base_dir, "data", "raw")
    jp2_files = [f for f in os.listdir(raw_dir) if f.endswith(".jp2")] if os.path.exists(raw_dir) else []
    
    if jp2_files:
        raw_img_path = os.path.join(raw_dir, jp2_files[0])
        print(f"Ortofoto de alta qualidade (.jp2) encontrada: {jp2_files[0]}")
    else:
        raw_img_path = os.path.join(raw_dir, "esri_highres_stitched.jpeg")
        print("Usando imagem consolidada (Bing/ESRI).")

    if not os.path.exists(raw_img_path):
        print(f"Erro: Nenhuma imagem encontrada em {raw_img_path}.")
        return

    processed_dir = os.path.join(base_dir, "data", "processed")
    train_dir = os.path.join(processed_dir, "train", "images")
    val_dir = os.path.join(processed_dir, "val", "images")
    test_dir = os.path.join(processed_dir, "test", "images")
    
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
        
    stride = int(tile_size * (1.0 - overlap_ratio))
    count = 0
    
    print(f"Iniciando fragmentação (Modo: {mode.upper()} | Tile: {tile_size}x{tile_size} | Stride: {stride})...")
    
    with rasterio.open(raw_img_path) as src:
        img_w = src.width
        img_h = src.height
        print(f"Dimensões totais da imagem de entrada: {img_w}x{img_h} pixels")
        
        if mode == "systematic":
            x_coords = list(range(0, img_w - tile_size + 1, stride))
            y_coords = list(range(0, img_h - tile_size + 1, stride))
            
            # Embaralha os pares de coordenadas para distribuir homogeneamente entre treino/val/teste
            coord_pairs = [(x, y) for y in y_coords for x in x_coords]
            random.seed(42)
            random.shuffle(coord_pairs)
            
            for x, y in coord_pairs:
                if count >= max_tiles:
                    break
                    
                window = Window(x, y, tile_size, tile_size)
                tile_data = src.read(window=window)
                
                # 1. Ignora tiles que caíram no preenchimento preto (borda vazia)
                if tile_data.max() == 0 or np.mean(tile_data == 0) > 0.3:
                    continue
                    
                # 2. Filtro de Variância Urbana (ignora pastos e campos uniformes)
                if tile_data.std() < 35.0:
                    continue
                    
                if tile_data.shape[0] >= 3:
                    tile_data = tile_data[:3, :, :]
                    tile_data = np.transpose(tile_data, (1, 2, 0))
                elif tile_data.shape[0] == 1:
                    tile_data = tile_data[0]
                    
                tile_img = Image.fromarray(tile_data)
                
                # Divisão 80% Treino, 10% Validação, 10% Teste
                if count % 10 == 8:
                    out_dir = val_dir
                elif count % 10 == 9:
                    out_dir = test_dir
                else:
                    out_dir = train_dir
                    
                tile_filename = f"tile_urbano_{count:04d}_x{x}_y{y}.jpeg"
                tile_filepath = os.path.join(out_dir, tile_filename)
                tile_img.save(tile_filepath, quality=95)
                
                count += 1
                if count % 50 == 0:
                    print(f"Extraídos {count} tiles urbanos válidos...")
                    
        else: # Modo aleatório (fallback)
            max_attempts = 50000
            attempts = 0
            while count < max_tiles and attempts < max_attempts:
                attempts += 1
                x = random.randint(0, img_w - tile_size)
                y = random.randint(0, img_h - tile_size)
                
                window = Window(x, y, tile_size, tile_size)
                tile_data = src.read(window=window)
                
                if tile_data.max() == 0 or tile_data.std() < 35:
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
                    
                tile_filename = f"tile_urbano_{count:04d}_x{x}_y{y}.jpeg"
                tile_img.save(os.path.join(out_dir, tile_filename), quality=95)
                count += 1

    print(f"\nFragmentação concluída com sucesso! Total de {count} tiles gerados.")
    print(f"Divisão estimada: Treino ~{int(count * 0.8)} | Validação ~{int(count * 0.1)} | Teste ~{int(count * 0.1)}")

if __name__ == "__main__":
    fragment_image(mode="systematic", tile_size=512, overlap_ratio=0.25, max_tiles=800)
