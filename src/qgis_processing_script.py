# -*- coding: utf-8 -*-

"""
Script de Processamento QGIS para o projeto SolarSegment.
Detecta painéis solares em ortofotos (TIF, JP2) utilizando a arquitetura U-Net.
"""

import os
import sys
import math

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterExtent,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsField,
    QgsFields,
    QgsWkbTypes,
    QgsRasterBlock
)

class SolarSegmentAlgorithm(QgsProcessingAlgorithm):
    INPUT = 'INPUT'
    EXTENT = 'EXTENT'
    MODEL_PATH = 'MODEL_PATH'
    THRESHOLD = 'THRESHOLD'
    MIN_AREA = 'MIN_AREA'
    TILT_ANGLE = 'TILT_ANGLE'
    OUTPUT = 'OUTPUT'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return SolarSegmentAlgorithm()

    def name(self):
        return 'solar_segmentation'

    def displayName(self):
        return self.tr('Detecção de Painéis Solares (U-Net)')

    def group(self):
        return self.tr('SolarSegment')

    def groupId(self):
        return 'solarsegment'

    def shortHelpString(self):
        return self.tr(
            "Detecta painéis solares na ortofoto usando PyTorch (U-Net) e vetoriza os contornos em camada vetorial (Shapefile).\n\n"
            "DICA: Para imagens grandes de municípios inteiros (com bilhões de pixels), utilize o parâmetro "
            "'Extensão de Recorte' selecionando 'Usar a extensão da tela do mapa' para analisar o bairro visível com rapidez e sem estourar a memória RAM."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, self.tr('Ortofoto de Entrada (Raster)')))
        
        # Parâmetro de Extensão (Permite recortar pela tela visível do QGIS)
        self.addParameter(QgsProcessingParameterExtent(
            self.EXTENT, 
            self.tr('Extensão do Recorte / Área de Interesse (Opcional - Ex: Tela do Mapa)'), 
            optional=True
        ))
        
        possible_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'unet_solar_best.pth')),
            os.path.abspath(r'C:\Users\vitor\Documents\Faculdade\TCC II\Implementação\models\unet_solar_best.pth')
        ]
        default_model_path = ""
        for p in possible_paths:
            if os.path.exists(p):
                default_model_path = p
                break
            
        self.addParameter(QgsProcessingParameterFile(
            self.MODEL_PATH, 
            self.tr('Pesos do Modelo (.pth)'), 
            behavior=QgsProcessingParameterFile.File, 
            extension='pth', 
            defaultValue=default_model_path, 
            optional=True
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.THRESHOLD, 
            self.tr('Limiar de Confiança (Threshold)'), 
            type=QgsProcessingParameterNumber.Double, 
            defaultValue=0.40, 
            minValue=0.1, 
            maxValue=0.99
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_AREA, 
            self.tr('Área Mínima (pixels)'), 
            type=QgsProcessingParameterNumber.Integer, 
            defaultValue=40, 
            minValue=5
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.TILT_ANGLE, 
            self.tr('Inclinação do Telhado (graus)'), 
            type=QgsProcessingParameterNumber.Double, 
            defaultValue=25.0, 
            minValue=0.0, 
            maxValue=90.0
        ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, 
            self.tr('Painéis Detectados (Vetor / .shp)'), 
            type=QgsProcessing.TypeVectorPolygon
        ))

    def processAlgorithm(self, parameters, context, feedback):
        # 1. Resolução de DLLs do PyTorch no Windows / OSGeo4W
        if sys.platform == "win32":
            try:
                torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
                if os.path.exists(torch_lib):
                    os.add_dll_directory(torch_lib)
                osgeo_bin = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "..", "bin"))
                if os.path.exists(osgeo_bin):
                    os.add_dll_directory(osgeo_bin)
            except Exception:
                pass

        # 2. Importações protegidas de IA
        try:
            import numpy as np
            import cv2
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torchvision.transforms import v2
        except Exception as e:
            raise Exception(
                "\n======================================================\n"
                "ERRO DE DEPENDÊNCIA DO PYTHON!\n"
                f"Detalhe técnico: {e}\n"
                "======================================================"
            )

        # 3. Arquitetura U-Net Embutida (independente de arquivos externos)
        class DoubleConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            def forward(self, x):
                return self.conv(x)

        class UNet(nn.Module):
            def __init__(self, in_channels=3, out_channels=1):
                super().__init__()
                self.downs = nn.ModuleList([
                    DoubleConv(in_channels, 16),
                    DoubleConv(16, 32),
                    DoubleConv(32, 64),
                    DoubleConv(64, 128)
                ])
                self.pool = nn.MaxPool2d(2)
                self.bottleneck = DoubleConv(128, 256)
                self.ups = nn.ModuleList([
                    nn.ConvTranspose2d(256, 128, 2, stride=2), DoubleConv(256, 128),
                    nn.ConvTranspose2d(128, 64, 2, stride=2), DoubleConv(128, 64),
                    nn.ConvTranspose2d(64, 32, 2, stride=2), DoubleConv(64, 32),
                    nn.ConvTranspose2d(32, 16, 2, stride=2), DoubleConv(32, 16)
                ])
                self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

            def forward(self, x):
                skip_connections = []
                for down in self.downs:
                    x = down(x)
                    skip_connections.append(x)
                    x = self.pool(x)
                x = self.bottleneck(x)
                skip_connections = skip_connections[::-1]
                for i in range(0, len(self.ups), 2):
                    x = self.ups[i](x)
                    skip = skip_connections[i//2]
                    if x.shape != skip.shape:
                        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                    concat_x = torch.cat((skip, x), dim=1)
                    x = self.ups[i+1](concat_x)
                return self.final_conv(x)

        # 4. Leitura dos Parâmetros
        input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        model_path = self.parameterAsFile(parameters, self.MODEL_PATH, context)
        threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)
        min_area_px = self.parameterAsInt(parameters, self.MIN_AREA, context)
        tilt_angle = self.parameterAsDouble(parameters, self.TILT_ANGLE, context)

        if input_layer is None:
            raise Exception("Camada de entrada inválida!")

        provider = input_layer.dataProvider()
        crs = input_layer.crs()

        # Resolução espacial nominal do raster original
        full_extent = input_layer.extent()
        full_cols = provider.xSize()
        full_rows = provider.ySize()
        pixel_width = (full_extent.xMaximum() - full_extent.xMinimum()) / full_cols
        pixel_height = (full_extent.yMaximum() - full_extent.yMinimum()) / full_rows
        gsd_x = pixel_width
        gsd_y = abs(pixel_height)
        area_per_pixel = gsd_x * gsd_y

        # 5. Ajuste da Extensão de Análise (Recorte pela Tela ou Camada)
        target_extent = self.parameterAsExtent(parameters, self.EXTENT, context, crs)
        if not target_extent.isNull() and not target_extent.isEmpty():
            extent = target_extent.intersect(full_extent)
            cols = max(1, int(round((extent.xMaximum() - extent.xMinimum()) / pixel_width)))
            rows = max(1, int(round((extent.yMaximum() - extent.yMinimum()) / gsd_y)))
            feedback.pushInfo(f"Recorte de interesse selecionado: {cols}x{rows} pixels. GSD: {gsd_x:.4f} m/px.")
        else:
            extent = full_extent
            cols = full_cols
            rows = full_rows
            feedback.pushInfo(f"Resolução total da imagem: {cols}x{rows} pixels. GSD estimado: {gsd_x:.4f} m/px.")

        # 6. Proteção de Memória RAM para Ortofotos Gigantescas (> 80 Milhões de pixels)
        total_pixels = cols * rows
        if total_pixels > 80_000_000:
            raise Exception(
                f"\n======================================================\n"
                f"ATENÇÃO: ÁREA SELECIONADA É GIGANTESCA ({cols}x{rows} = {total_pixels/1e6:.1f} Megapixels)!\n"
                "Tentar processar bilhões de pixels de uma só vez esgotaria a memória RAM do computador.\n\n"
                "COMO RESOLVER EM SEGUNDOS:\n"
                "1. Feche esta janela e dê zoom na área urbana ou no quarteirão que você deseja analisar no mapa.\n"
                "2. Abra o algoritmo novamente.\n"
                "3. No campo 'Extensão do Recorte / Área de Interesse', clique no botão '...' e selecione 'Usar a extensão da tela do mapa'.\n"
                "4. Clique em Executar! A IA analisará a região em poucos segundos.\n"
                "======================================================"
            )

        fields = QgsFields()
        fields.append(QgsField('Area_Proj', QVariant.Double, 'double', 10, 2))
        fields.append(QgsField('Area_Real', QVariant.Double, 'double', 10, 2))
        fields.append(QgsField('Pot_kWp', QVariant.Double, 'double', 10, 2))
        fields.append(QgsField('Ger_kWh_an', QVariant.Double, 'double', 10, 2))
        fields.append(QgsField('Modulos_un', QVariant.Int))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, fields, QgsWkbTypes.Polygon, crs)

        feedback.pushInfo("Carregando modelo U-Net...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(in_channels=3, out_channels=1)

        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            feedback.pushInfo(f"Pesos carregados com sucesso: {os.path.basename(model_path)} ({device})")
        else:
            feedback.pushWarning("AVISO: Nenhum arquivo .pth selecionado. Usando U-Net com pesos não-treinados apenas para teste.")

        model.to(device)
        model.eval()

        feedback.pushInfo("Lendo bandas RGB do raster...")
        r_block = provider.block(1, extent, cols, rows)
        g_block = provider.block(2, extent, cols, rows)
        b_block = provider.block(3, extent, cols, rows)

        r_array = self._block_to_array(r_block, cols, rows, np)
        g_array = self._block_to_array(g_block, cols, rows, np)
        b_array = self._block_to_array(b_block, cols, rows, np)
        image_np = np.stack((r_array, g_array, b_array), axis=-1).astype(np.uint8)

        # 7. Invariância de Escala Inteligente e Multi-Escala (Sliding Window Robusta)
        # Calcula o tamanho do recorte nativo equivalente a 512x512 no GSD nominal de treinamento (~0.0389 m/px)
        nominal_gsd = 0.0389
        scale_factor = max(0.2, min(5.0, gsd_x / nominal_gsd))
        tile_size = 512
        crop_size_fine = max(64, int(round(tile_size / scale_factor)))

        # Se a escala for comprimida (GSD > 0.06m/px, como Maracaju 0.10m), usamos 2 escalas complementares:
        # Escala 1 (Fina): foca em strings individuais e painéis residenciais pequenos.
        # Escala 2 (Contexto): foca em grandes usinas e telhados comerciais/industriais contínuos.
        scales = [('Detalhe Fino', crop_size_fine)]
        if scale_factor > 1.2:
            crop_size_context = min(max(cols, rows), int(round(crop_size_fine * 1.85)))
            if crop_size_context > crop_size_fine + 30:
                scales.append(('Contexto Urbano/Comercial', crop_size_context))

        scale_desc = ", ".join([f"{name} ({sz}px)" for name, sz in scales])
        feedback.pushInfo(f"Fator de Escala: {scale_factor:.2f}x | Varredura Multi-Escala: {scale_desc}")

        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        hann_1d = np.hanning(tile_size)
        hann_weight_512 = np.outer(hann_1d, hann_1d).astype(np.float32)
        hann_weight_512 = np.clip(hann_weight_512, 1e-4, 1.0)

        # Planejamento do grid de janelas deslizantes para cálculo de progresso
        total_steps = 0
        scale_plans = []
        for name, crop_sz in scales:
            stride = max(32, int(crop_sz * 0.70))
            y_steps = list(range(0, rows, stride))
            x_steps = list(range(0, cols, stride))
            total_steps += len(y_steps) * len(x_steps)
            scale_plans.append((name, crop_sz, stride, y_steps, x_steps))

        feedback.pushInfo(f"Iniciando inferência por IA ({total_steps} tiles no total)...")

        prob_maps = []
        current_step = 0

        for name, crop_sz, stride, y_steps, x_steps in scale_plans:
            if feedback and feedback.isCanceled():
                break
            feedback.pushInfo(f"Executando passe: {name} ({crop_sz}x{crop_sz} px)...")
            scale_prob = np.zeros((rows, cols), dtype=np.float32)
            scale_weight = np.zeros((rows, cols), dtype=np.float32)

            for y in y_steps:
                if feedback and feedback.isCanceled():
                    break
                for x in x_steps:
                    box_x = min(x, max(0, cols - crop_sz))
                    box_y = min(y, max(0, rows - crop_sz))

                    active_w = min(crop_sz, cols - box_x)
                    active_h = min(crop_sz, rows - box_y)

                    tile_np = image_np[box_y:box_y + active_h, box_x:box_x + active_w]

                    # Redimensiona para a dimensão esperada pela U-Net
                    if (active_w, active_h) != (tile_size, tile_size):
                        tile_resized = cv2.resize(tile_np, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                    else:
                        tile_resized = tile_np

                    tile_tensor = transform(tile_resized).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(tile_tensor)
                        prob_512 = torch.sigmoid(output).squeeze().cpu().numpy()

                    # Remapeia a predição para o tamanho original da janela
                    if (active_w, active_h) != (tile_size, tile_size):
                        prob_crop = cv2.resize(prob_512, (active_w, active_h), interpolation=cv2.INTER_LINEAR)
                        weight_crop = cv2.resize(hann_weight_512, (active_w, active_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        prob_crop = prob_512
                        weight_crop = hann_weight_512

                    scale_prob[box_y:box_y + active_h, box_x:box_x + active_w] += prob_crop * weight_crop
                    scale_weight[box_y:box_y + active_h, box_x:box_x + active_w] += weight_crop

                    current_step += 1
                    if total_steps > 0:
                        feedback.setProgress(int(current_step * 50 / total_steps))

            scale_weight[scale_weight == 0] = 1.0
            scale_prob /= scale_weight
            prob_maps.append(scale_prob)

        if feedback and feedback.isCanceled():
            return {}

        # Fusão das escalas (máximo de probabilidade para detectar desde painéis compactos até usinas inteiras)
        if len(prob_maps) == 1:
            full_prob = prob_maps[0]
        else:
            full_prob = np.maximum.reduce(prob_maps)

        feedback.pushInfo("Vetorização dos Painéis e Cálculos Fotovoltaicos...")
        pred_mask = (full_prob > threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        eta = 0.185     
        i_local = 5.4   
        pr = 0.75       
        beta_rad = math.radians(tilt_angle)
        cos_beta = max(0.1, math.cos(beta_rad))
        tilt_factor = 1.0 / cos_beta

        total_features = len(contours)
        feedback.pushInfo(f"Polígonos candidatos encontrados: {total_features}. Filtrando por área mínima ({min_area_px} px)...")

        inserted_count = 0
        for i, contour in enumerate(contours):
            if feedback and feedback.isCanceled():
                break

            area_px = cv2.contourArea(contour)
            if area_px < min_area_px:
                continue

            points = []
            for point in contour:
                px, py = point[0]
                geo_x = extent.xMinimum() + (px * pixel_width)
                geo_y = extent.yMaximum() - (py * gsd_y)
                points.append(QgsPointXY(geo_x, geo_y))

            if len(points) >= 3:
                points.append(points[0]) 
                geom = QgsGeometry.fromPolygonXY([points])

                a_proj = area_px * area_per_pixel
                s_pv = a_proj * tilt_factor
                p_pico = (s_pv * eta)
                geracao_kwh = (p_pico * i_local * pr) * 365.0
                modulos = int(round((s_pv * 0.90) / 2.30)) 

                feat = QgsFeature()
                feat.setGeometry(geom)
                feat.setAttributes([round(a_proj, 2), round(s_pv, 2), round(p_pico, 2), round(geracao_kwh, 2), modulos])

                sink.addFeature(feat)
                inserted_count += 1

            if total_features > 0:
                feedback.setProgress(50 + int(i * 50 / total_features))

        feedback.pushInfo(f"Concluído com sucesso! {inserted_count} agrupamentos de painéis vetorizados na camada.")
        return {self.OUTPUT: dest_id}

    def _block_to_array(self, block, cols, rows, np):
        from qgis.core import Qgis
        try:
            block.convert(Qgis.DataType.Float32)
        except Exception:
            try:
                block.convert(Qgis.DataType.Byte)
            except Exception:
                pass

        data = block.data()
        if block.dataType() == Qgis.DataType.Byte:
            array = np.frombuffer(data, dtype=np.uint8).reshape((rows, cols)).astype(np.float32)
        else:
            array = np.frombuffer(data, dtype=np.float32).reshape((rows, cols))

        if block.hasNoDataValue():
            nodata = block.noDataValue()
            array[array == nodata] = 0

        max_val = float(np.max(array)) if array.size > 0 else 0
        if max_val > 255.0:
            array = (array / max_val) * 255.0

        return np.clip(array, 0, 255).astype(np.uint8)
