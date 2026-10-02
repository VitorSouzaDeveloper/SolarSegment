# -*- coding: utf-8 -*-

"""
Script de Processamento QGIS para o projeto SolarSegment.
"""

import os
import sys
import math
import importlib

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
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
        return self.tr("Detecta painéis solares na ortofoto usando PyTorch e vetoriza em Shapefile.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT, self.tr('Ortofoto de Entrada (Raster)')))
        
        default_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'unet_solar_best.pth'))
        if not os.path.exists(default_model_path):
            default_model_path = ""
            
        self.addParameter(QgsProcessingParameterFile(self.MODEL_PATH, self.tr('Pesos do Modelo (.pth)'), behavior=QgsProcessingParameterFile.File, extension='pth', defaultValue=default_model_path, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.THRESHOLD, self.tr('Limiar de Confiança (Threshold)'), type=QgsProcessingParameterNumber.Double, defaultValue=0.65, minValue=0.1, maxValue=0.99))
        self.addParameter(QgsProcessingParameterNumber(self.MIN_AREA, self.tr('Área Mínima (pixels)'), type=QgsProcessingParameterNumber.Integer, defaultValue=250, minValue=10))
        self.addParameter(QgsProcessingParameterNumber(self.TILT_ANGLE, self.tr('Inclinação do Telhado (graus)'), type=QgsProcessingParameterNumber.Double, defaultValue=25.0, minValue=0.0, maxValue=90.0))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, self.tr('Painéis Detectados (Vetor / .shp)'), type=QgsWkbTypes.Polygon))

    def processAlgorithm(self, parameters, context, feedback):
        # Importações feitas internamente para garantir que o script apareça na interface do QGIS
        # mesmo que as dependências ainda não estejam instaladas.
        try:
            import numpy as np
            import cv2
            import torch
            from torchvision.transforms import v2
        except ImportError as e:
            raise Exception(
                "\n======================================================\n"
                "ERRO DE DEPENDÊNCIA DO PYTHON!\n"
                "As bibliotecas de Inteligência Artificial não estão instaladas no QGIS.\n\n"
                "Para resolver, vá no menu superior do QGIS em:\n"
                "Plugins (Complementos) -> Python Console (Console Python).\n\n"
                "Lá embaixo na aba de código, cole o seguinte comando e aperte Enter:\n\n"
                "import subprocess; import sys; subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'opencv-python'])\n\n"
                "Após instalar, tente rodar essa ferramenta novamente.\n"
                "======================================================\n"
                f"Detalhe técnico: {e}"
            )

        input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        model_path = self.parameterAsFile(parameters, self.MODEL_PATH, context)
        threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)
        min_area_px = self.parameterAsInt(parameters, self.MIN_AREA, context)
        tilt_angle = self.parameterAsDouble(parameters, self.TILT_ANGLE, context)

        if input_layer is None:
            raise Exception("Camada de entrada inválida!")

        provider = input_layer.dataProvider()
        extent = provider.extent()
        crs = input_layer.crs()
        
        cols = provider.xSize()
        rows = provider.ySize()
        
        pixel_width = (extent.xMaximum() - extent.xMinimum()) / cols
        pixel_height = (extent.yMaximum() - extent.yMinimum()) / rows
        
        gsd_x = pixel_width
        gsd_y = abs(pixel_height)
        area_per_pixel = gsd_x * gsd_y

        feedback.pushInfo(f"Resolução da imagem lida: {cols}x{rows} pixels. GSD estimado: {gsd_x:.4f} m/px.")

        fields = QgsFields()
        fields.append(QgsField('Area_Proj', QVariant.Double, 'double', 10, 2))
        fields.append(QgsField('Area_Real', QVariant.Double, 'double', 10, 2))
        fields.append(QgsField('Pot_kWp', QVariant.Double, 'double', 10, 2))
        fields.append(QgsField('Ger_kWh_an', QVariant.Double, 'double', 10, 2))
        fields.append(QgsField('Modulos_un', QVariant.Int))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, fields, QgsWkbTypes.Polygon, crs)

        feedback.pushInfo("Carregando modelo U-Net...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.append(script_dir)
        
        try:
            train_module = importlib.import_module("3_train_unet")
            UNet = train_module.UNet
        except Exception as e:
            raise Exception(f"Erro ao importar a U-Net do arquivo 3_train_unet.py. Certifique-se de que ele está na mesma pasta. Erro: {e}")

        model = UNet(in_channels=3, out_channels=1)
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            feedback.pushInfo("AVISO: Nenhum arquivo .pth selecionado. Usando U-Net com pesos não-treinados (aleatórios) apenas para demonstração.")
        
        model.to(device)
        model.eval()

        feedback.pushInfo("Lendo os dados do Raster e convertendo matriz...")
        
        r_block = provider.block(1, extent, cols, rows)
        g_block = provider.block(2, extent, cols, rows)
        b_block = provider.block(3, extent, cols, rows)
        
        r_array = self._block_to_array(r_block, cols, rows, np)
        g_array = self._block_to_array(g_block, cols, rows, np)
        b_array = self._block_to_array(b_block, cols, rows, np)
        
        image_np = np.stack((r_array, g_array, b_array), axis=-1).astype(np.uint8)
        
        feedback.pushInfo("Iniciando varredura por IA (Sliding Window)...")
        
        tile_size = 512
        stride = int(tile_size * 0.75) 
        
        full_prob = np.zeros((rows, cols), dtype=np.float32)
        full_weight = np.zeros((rows, cols), dtype=np.float32)
        
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
        hann_1d = np.hanning(tile_size)
        hann_weight = np.outer(hann_1d, hann_1d).astype(np.float32)
        hann_weight = np.clip(hann_weight, 1e-4, 1.0)
        
        y_steps = list(range(0, rows, stride))
        x_steps = list(range(0, cols, stride))
        total_steps = len(y_steps) * len(x_steps)
        current_step = 0

        for y in y_steps:
            if context.feedback().isCanceled():
                break
            for x in x_steps:
                box_x = min(x, max(0, cols - tile_size))
                box_y = min(y, max(0, rows - tile_size))
                
                active_w = min(tile_size, cols - box_x)
                active_h = min(tile_size, rows - box_y)
                
                tile_np = image_np[box_y:box_y+active_h, box_x:box_x+active_w]
                
                if active_w < tile_size or active_h < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[:active_h, :active_w] = tile_np
                    tile_tensor = transform(padded).unsqueeze(0).to(device)
                else:
                    tile_tensor = transform(tile_np).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(tile_tensor)
                    prob = torch.sigmoid(output).squeeze().cpu().numpy()
                    
                prob_crop = prob[:active_h, :active_w]
                weight_crop = hann_weight[:active_h, :active_w]
                
                full_prob[box_y:box_y+active_h, box_x:box_x+active_w] += prob_crop * weight_crop
                full_weight[box_y:box_y+active_h, box_x:box_x+active_w] += weight_crop
                
                current_step += 1
                feedback.setProgress(int(current_step * 50 / total_steps))

        if context.feedback().isCanceled():
            return {}

        feedback.pushInfo("Vetorização dos Painéis e Cálculos de Engenharia...")
        
        full_weight[full_weight == 0] = 1.0
        full_prob /= full_weight
        
        pred_mask = (full_prob > threshold).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        eta = 0.185     
        i_local = 5.4   
        pr = 0.75       
        
        beta_rad = math.radians(tilt_angle)
        cos_beta = max(0.1, math.cos(beta_rad))
        tilt_factor = 1.0 / cos_beta

        total_features = len(contours)
        
        for i, contour in enumerate(contours):
            if context.feedback().isCanceled():
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
                p_pico = s_pv * eta
                geracao_kwh = (p_pico * i_local * pr) * 365.0
                modulos = int(round((s_pv * 0.9) / 2.30)) 
                
                feat = QgsFeature()
                feat.setGeometry(geom)
                feat.setAttributes([a_proj, s_pv, p_pico, geracao_kwh, modulos])
                
                sink.addFeature(feat, QgsFeature.FastInsert)
                
            feedback.setProgress(50 + int(i * 50 / total_features))

        feedback.pushInfo("Concluído! A camada vetorial (.shp) foi gerada no mapa.")
        return {self.OUTPUT: dest_id}

    def _block_to_array(self, block, cols, rows, np):
        block.convert(5) 
        data = block.data()
        array = np.frombuffer(data, dtype=np.float32).reshape((rows, cols))
        
        if block.hasNoDataValue():
            nodata = block.noDataValue()
            array[array == nodata] = 0
            
        return np.clip(array, 0, 255).astype(np.uint8)
