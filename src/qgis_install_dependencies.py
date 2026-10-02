# -*- coding: utf-8 -*-

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import QgsProcessingAlgorithm
import subprocess
import sys

class InstallDependenciesAlgorithm(QgsProcessingAlgorithm):
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return InstallDependenciesAlgorithm()

    def name(self):
        return 'install_solar_deps'

    def displayName(self):
        return self.tr('1. Instalar Dependências da IA (Rodar Primeiro)')

    def group(self):
        return self.tr('SolarSegment')

    def groupId(self):
        return 'solarsegment'

    def shortHelpString(self):
        return self.tr("Rode este script apenas uma vez. Ele irá baixar e instalar automaticamente as bibliotecas PyTorch e OpenCV dentro do QGIS.")

    def initAlgorithm(self, config=None):
        pass # Não precisa de nenhum parâmetro de entrada

    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo("Iniciando a instalação de: torch, torchvision, opencv-python...")
        feedback.pushInfo("Isko pode demorar 1 ou 2 minutos dependendo da sua internet. Por favor, aguarde...")
        
        try:
            # Comando mágico para instalar as bibliotecas no Python interno do QGIS
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', 'opencv-python'
            ])
            feedback.pushInfo("=============================================")
            feedback.pushInfo("SUCESSO! Bibliotecas instaladas com perfeição.")
            feedback.pushInfo("Você já pode usar a ferramenta 'Detecção de Painéis Solares'.")
            feedback.pushInfo("=============================================")
        except Exception as e:
            raise Exception(f"Erro ao instalar as bibliotecas: {e}")

        return {}
