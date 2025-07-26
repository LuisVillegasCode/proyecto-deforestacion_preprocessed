# src/producto.py

"""
Módulo para la definición de clases que manejan productos de datos satelitales.

Contiene una clase base abstracta (Producto_Satelital_Base) que define una interfaz
común para todos los tipos de satélite, y clases concretas (ej. ProductoSPOT6) que
implementan la lógica específica para cada satélite agregado.
"""

# --- Importaciones ---
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import math
from scipy.ndimage import binary_dilation

# --- Clase Base Abstracta ---
class Producto_Satelital_Base(ABC):
    """
    Define la interfaz que todos los productos satelitales deben cumplir.
    """
    def __init__(self, ruta_producto: Path, ruta_salida: Path):
        """
        Inicializa el objeto del producto satelital.

        Args:
            ruta_producto (Path): Ruta a la carpeta principal del producto crudo.
            ruta_salida (Path): Ruta a la carpeta donde se guardarán los resultados.
        """
        if not ruta_producto.is_dir():
            raise FileNotFoundError(f"La ruta del producto no existe: {ruta_producto}")
        self.ruta_base = ruta_producto
        self.ruta_salida = ruta_salida
        self.metadatos = {}
        self.ruta_salida_producto = self.ruta_salida / self.ruta_base.name
        self.ruta_salida_producto.mkdir(parents=True, exist_ok=True)
        self._leer_metadatos()

    @abstractmethod
    def _leer_metadatos(self):
        """Método abstracto para leer los metadatos específicos del satélite."""
        pass

    @abstractmethod
    def ejecutar_preprocesamiento(self):
        """Método abstracto que ejecuta el pipeline de preprocesamiento completo."""
        pass

# --- Clase Concreta para SPOT 6 ---
class ProductoSPOT6(Producto_Satelital_Base):
    """
    Implementación específica para productos SPOT 6. Contiene toda la lógica
    para leer, procesar y guardar los datos de este satélite.
    """
    def _leer_metadatos(self):
        """
        Parsea el archivo DIM.XML para extraer todos los parámetros necesarios.
        """
        xml_files = list(self.ruta_base.rglob('DIM_*.XML'))
        if not xml_files:
            raise FileNotFoundError(f"No se encontró el archivo DIM XML principal en {self.ruta_base} o sus subcarpetas.")
        
        xml_path = xml_files[0]
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        self.metadatos['elevacion_solar'] = float(root.find(".//Located_Geometric_Values[LOCATION_TYPE='Center']//SUN_ELEVATION").text)
        self.metadatos['ganancias'] = {b.find('BAND_ID').text: float(b.find('GAIN').text) for b in root.findall(".//Band_Radiance")}
        self.metadatos['bias'] = {b.find('BAND_ID').text: float(b.find('BIAS').text) for b in root.findall(".//Band_Radiance")}
        self.metadatos['irradiancias'] = {b.find('BAND_ID').text: float(b.find('VALUE').text) for b in root.findall(".//Band_Solar_Irradiance")}
        cloud_mask_node = root.find(".//MEASURE_NAME[.='Cloud_Cotation (CLD)']/../Quality_Mask/Component/COMPONENT_PATH")
        if cloud_mask_node is not None:
             self.metadatos['mascara_nubes_gml'] = xml_path.parent / cloud_mask_node.attrib['href']
        self.metadatos['band_order'] = [b.find('BAND_ID').text for b in root.findall(".//Raster_Index")]
        self.metadatos['rutas_tifs'] = [xml_path.parent / f.attrib['href'] for f in root.findall(".//Data_File/DATA_FILE_PATH")]
        print(f"Metadatos de {self.ruta_base.name} cargados exitosamente.")

    def _calibrar_bloque(self, block_dn: np.ndarray) -> np.ndarray:
        """Realiza la calibración radiométrica en un bloque de datos (DN a Radiancia).

        Este método aplica la fórmula de conversión oficial (Radiancia = DN / Ganancia + Bias)
        proporcionada por el proveedor del satélite en el archivo de metadatos XML.
        Transforma los Niveles Digitales (DN) crudos, que son adimensionales, en valores 
        de Radiancia en el Techo de la Atmósfera (TOA), una unidad de energía física.

        Args:
            block_dn (np.ndarray): Bloque de datos de entrada en formato de
                                Niveles Digitales (enteros).

        Returns:
            np.ndarray: Bloque de datos convertido a Radiancia TOA, en formato
                        de punto flotante (float32).
        """
        block_rad = np.zeros_like(block_dn, dtype=np.float32)
        for i in range(block_dn.shape[0]):
            band_id = self.metadatos['band_order'][i]
            gain, bias = self.metadatos['ganancias'][band_id], self.metadatos['bias'][band_id]
            block_rad[i] = block_dn[i].astype(np.float32) / gain + bias
        return block_rad

    def _corregir_atmosfericamente_bloque(self, block_rad: np.ndarray) -> np.ndarray:
        """
        Realiza la corrección atmosférica (Radiancia a Reflectancia) con DOS1.

        Args:
            block_rad (np.ndarray): Bloque de datos en Radiancia TOA.

        Returns:
            np.ndarray: Bloque de datos convertido a Reflectancia de Superficie.
        """
        sun_zenith = 90.0 - self.metadatos['elevacion_solar']
        block_ref = np.zeros_like(block_rad, dtype=np.float32)
        for i in range(block_rad.shape[0]):
            band_id = self.metadatos['band_order'][i]
            esun = self.metadatos['irradiancias'][band_id]
            dark_object = np.percentile(block_rad[i][block_rad[i] > 0], 1) if np.any(block_rad[i] > 0) else 0
            path_radiance = dark_object - 0.01 * (esun * math.cos(math.radians(sun_zenith))**2) / (math.pi * 1.0**2)
            numerator = math.pi * (block_rad[i] - path_radiance) * 1.0**2
            denominator = esun * math.cos(math.radians(sun_zenith))
            block_ref[i] = numerator / denominator
        return block_ref

    def _finalizar_bloque(self, block_ref: np.ndarray, block_mask: np.ndarray) -> np.ndarray:
        """
        Aplica la máscara de nubes y escala el bloque para el guardado final (UInt16).

        Args:
            block_ref (np.ndarray): Bloque de datos en Reflectancia de Superficie.
            block_mask (np.ndarray): Máscara booleana de nubes para el bloque actual.

        Returns:
            np.ndarray: Bloque de datos final, enmascarado y escalado a UInt16.
        """
        block_ref[np.stack([block_mask]*block_ref.shape[0])] = 0.0
        block_scaled = (np.clip(block_ref, 0, 1) * 10000).astype(rasterio.uint16)
        return block_scaled

    def _procesar_bloque(self, block_dn: np.ndarray, block_mask: np.ndarray) -> np.ndarray:
        """
        Orquesta el procesamiento completo para un único bloque de datos.

        Args:
            block_dn (np.ndarray): Bloque de datos crudos en DN.
            block_mask (np.ndarray): Máscara booleana de nubes para el bloque.

        Returns:
            np.ndarray: Bloque de datos final y procesado.
        """
        block_rad = self._calibrar_bloque(block_dn)
        block_ref = self._corregir_atmosfericamente_bloque(block_rad)
        block_final = self._finalizar_bloque(block_ref, block_mask)
        return block_final

    def procesar_bloque(self, block_dn: np.ndarray, block_mask: np.ndarray) -> np.ndarray:
        """
        Orquesta el procesamiento completo para un único bloque de datos.

        Este método actúa como un pipeline secuencial para un bloque individual,
        llamando a los métodos especializados de calibración, corrección
        atmosférica y finalización en el orden correcto.

        Args:
            block_dn (np.ndarray): Bloque de datos crudos en Niveles Digitales (DN).
            block_mask (np.ndarray): Máscara booleana de nubes para el bloque.

        Returns:
            np.ndarray: Bloque de datos final y procesado, listo para ser escrito.

        Notes on Optimization and Development:
            La versión actual en la rama 'main' procesa los bloques de forma
            secuencial. Para acelerar el tiempo de ejecución de una sola imagen,
            se está desarrollando una optimización en la rama 'feat/parallel-processing'.
            El objetivo de dicha rama es introducir 'multiprocessing' para que esta
            función 'procesar_bloque' sea ejecutada de forma simultánea en
            múltiples núcleos de CPU, reduciendo significativamente el tiempo total.
        """
        block_rad = self._calibrar_bloque(block_dn)
        block_ref = self._corregir_atmosfericamente_bloque(block_rad)
        block_final = self._finalizar_bloque(block_ref, block_mask)
        return block_final