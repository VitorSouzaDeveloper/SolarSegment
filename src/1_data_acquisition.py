import os
import math
import requests
from PIL import Image

def deg2num(lat_deg, lon_deg, zoom):
    """Converte Latitude/Longitude para coordenadas de tile (x, y) do Slippy Map."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def tile_to_quadkey(x, y, z):
    """Converte tile X e Y para o sistema QuadKey usado pelo Bing Maps."""
    quadkey = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0: digit += 1
        if (y & mask) != 0: digit += 2
        quadkey += str(digit)
    return quadkey

# Coordenadas do Parque Alvorada, Dourados - MS
LAT_CENTER = -22.219808
LNG_CENTER = -54.832729
ZOOM = 19 # Zoom 19 suportado pelo Bing para altíssima qualidade (~30cm)

# Um grid 7x7 no zoom 19 gerará uma imagem de 1792x1792, cobrindo uma área excelente
GRID_SIZE = 7

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
tiles_dir = os.path.join(output_dir, "tiles")
os.makedirs(tiles_dir, exist_ok=True)

print("Iniciando download de satélite de ALTÍSSIMA RESOLUÇÃO (Bing Maps Aerial)...")
print(f"Diretório de saída: {output_dir}\n")

# Calcular o tile central
center_x, center_y = deg2num(LAT_CENTER, LNG_CENTER, ZOOM)

half_grid = GRID_SIZE // 2
count = 1
total_images = GRID_SIZE * GRID_SIZE

downloaded_tiles = {}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# Baixar os tiles
for i in range(-half_grid, half_grid + 1):
    for j in range(-half_grid, half_grid + 1):
        x = center_x + i
        y = center_y + j
        
        # Converter para QuadKey do Bing
        qk = tile_to_quadkey(x, y, ZOOM)
        url = f"http://ecn.t3.tiles.virtualearth.net/tiles/a{qk}.jpeg?g=1"
        
        filename = f"bing_tile_z{ZOOM}_x{x}_y{y}.jpeg"
        filepath = os.path.join(tiles_dir, filename)
        
        print(f"[{count}/{total_images}] Baixando Tile ({x}, {y})...")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            try:
                img = Image.open(filepath)
                downloaded_tiles[(i, j)] = img
            except Exception as e:
                print(f"  -> Erro ao abrir a imagem {filename}: {e}")
        else:
            print(f"  -> Erro ao baixar tile: {response.status_code}")
            
        count += 1

# Costurar (Stitch) as imagens
if len(downloaded_tiles) > 0:
    print("\nIniciando costura das imagens...")
    tile_w, tile_h = 256, 256
    
    final_img_w = tile_w * GRID_SIZE
    final_img_h = tile_h * GRID_SIZE
    final_image = Image.new('RGB', (final_img_w, final_img_h))
    
    for i in range(-half_grid, half_grid + 1):
        for j in range(-half_grid, half_grid + 1):
            if (i, j) in downloaded_tiles:
                img = downloaded_tiles[(i, j)]
                paste_x = (i + half_grid) * tile_w
                paste_y = (j + half_grid) * tile_h
                final_image.paste(img, (paste_x, paste_y))
    
    # Substituí o arquivo para ser lido pelo próximo script
    final_filepath = os.path.join(output_dir, "esri_highres_stitched.jpeg")
    final_image.save(final_filepath, quality=100)
    print(f"Imagem final costurada salva em: {final_filepath} ({final_img_w}x{final_img_h} px)")
    
    # GERAÇÃO DO WORLD FILE (.JGW) PARA GEORREFERENCIAMENTO
    R = 20037508.342789244
    n_tiles = 2.0 ** ZOOM
    tile_size_meters = (2 * R) / n_tiles
    resolution = tile_size_meters / 256.0
    
    top_left_x_tile = center_x - half_grid
    top_left_y_tile = center_y - half_grid
    
    x_left_edge = -R + (top_left_x_tile * tile_size_meters)
    y_top_edge = R - (top_left_y_tile * tile_size_meters)
    
    pixel_center_x = x_left_edge + (resolution / 2.0)
    pixel_center_y = y_top_edge - (resolution / 2.0)
    
    jgw_filepath = os.path.join(output_dir, "esri_highres_stitched.jgw")
    with open(jgw_filepath, "w") as f:
        f.write(f"{resolution}\n")
        f.write("0.0\n")
        f.write("0.0\n")
        f.write(f"{-resolution}\n")
        f.write(f"{pixel_center_x}\n")
        f.write(f"{pixel_center_y}\n")
        
    print(f"World File de georreferenciamento salvo em: {jgw_filepath}")
    print(f"Resolução Espacial (GSD): ~{resolution:.4f} metros/pixel")
else:
    print("Nenhum tile foi baixado.")

print("\nProcesso finalizado com sucesso!")
