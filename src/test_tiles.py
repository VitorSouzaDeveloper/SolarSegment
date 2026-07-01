import requests
import math

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def tile_to_quadkey(x, y, z):
    quadkey = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0: digit += 1
        if (y & mask) != 0: digit += 2
        quadkey += str(digit)
    return quadkey

LAT = -22.219808
LNG = -54.832729
headers = {'User-Agent': 'Mozilla/5.0'}

print("Testando resolução real de satélites para Dourados...")

for z in [19, 18, 17]:
    x, y = deg2num(LAT, LNG, z)
    
    # ESRI
    url_esri = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    r_esri = requests.get(url_esri, headers=headers)
    
    # BING
    qk = tile_to_quadkey(x, y, z)
    url_bing = f"http://ecn.t3.tiles.virtualearth.net/tiles/a{qk}.jpeg?g=1"
    r_bing = requests.get(url_bing, headers=headers)
    
    print(f"Zoom {z}:")
    print(f"  ESRI: {len(r_esri.content)} bytes")
    print(f"  BING: {len(r_bing.content)} bytes")
