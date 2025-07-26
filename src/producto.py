# src/producto.py

"""
Módulo para la definición de clases que manejan productos de datos satelitales.
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
    def _leer_metadatos(self): pass

    @abstractmethod
    def ejecutar_preprocesamiento(self): pass

# --- Clase Concreta para SPOT 6 ---
class ProductoSPOT6(Producto_Satelital_Base):
    """
    Implementación específica para productos SPOT 6.
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
        """Realiza la calibración radiométrica en un bloque de datos."""
        block_rad = np.zeros_like(block_dn, dtype=np.float32)
        for i in range(block_dn.shape[0]):
            band_id = self.metadatos['band_order'][i]
            gain, bias = self.metadatos['ganancias'][band_id], self.metadatos['bias'][band_id]
            block_rad[i] = block_dn[i].astype(np.float32) / gain + bias
        return block_rad

    def _corregir_atmosfericamente_bloque(self, block_rad: np.ndarray) -> np.ndarray:
        """Realiza la corrección atmosférica en un bloque de datos."""
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
        """Aplica la máscara de nubes y escala el bloque para el guardado final."""
        block_ref[np.stack([block_mask]*block_ref.shape[0])] = 0.0
        block_scaled = (np.clip(block_ref, 0, 1) * 10000).astype(rasterio.uint16)
        return block_scaled

    def _procesar_bloque(self, block_dn: np.ndarray, block_mask: np.ndarray) -> np.ndarray:
        """
        Orquesta el procesamiento completo para un único bloque de datos.
        """
        block_rad = self._calibrar_bloque(block_dn)
        block_ref = self._corregir_atmosfericamente_bloque(block_rad)
        block_final = self._finalizar_bloque(block_ref, block_mask)
        return block_final

    def ejecutar_preprocesamiento(self):
        """
        Método principal que gestiona el flujo de E/S y el bucle de procesamiento por bloques.
        """
        print(f"--- Iniciando Pipeline de Preprocesamiento por Bloques para: {self.ruta_base.name} ---")
        
        vrt_path = self.ruta_salida_producto / 'source.vrt'
        src_files_to_mosaic = [rasterio.open(fp) for fp in self.metadatos['rutas_tifs']]
        merge(src_files_to_mosaic, dst_path=str(vrt_path))
        
        with rasterio.open(vrt_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.uint16, nodata=0, compress=None, tiled=True, blockxsize=512, blockysize=512)
            path_final = self.ruta_salida_producto / 'analysis_ready_data.tif'

            print("  -> Preparando máscara de nubes...")
            ruta_mascara_gml = self.metadatos.get('mascara_nubes_gml')
            if ruta_mascara_gml and ruta_mascara_gml.exists():
                gdf_clouds = gpd.read_file(ruta_mascara_gml)
                cloud_mask_base = rasterize(
                    shapes=gdf_clouds.geometry, out_shape=src.shape, transform=src.transform,
                    fill=0, all_touched=True, dtype=np.uint8
                ).astype(bool)
                cloud_mask_refinada = binary_dilation(cloud_mask_base, iterations=5)
            else:
                cloud_mask_refinada = np.zeros(src.shape, dtype=bool)
                print("     Advertencia: No se encontró máscara de nubes.")

            with rasterio.open(path_final, 'w', **profile) as dst:
                for ji, window in src.block_windows(1):
                    print(f"     Procesando bloque {ji[0]+1}/{len(list(src.block_windows(1)))}")
                    
                    block_dn = src.read(window=window)
                    block_mask = cloud_mask_refinada[window.row_off:window.row_off+window.height, window.col_off:window.col_off+window.width]
                    
                    # Llama al orquestador para que procese el bloque
                    block_procesado = self._procesar_bloque(block_dn, block_mask)
                    
                    dst.write(block_procesado, window=window)

        print(f"--- Pipeline Finalizado. Producto final guardado en: {path_final} ---")
        return path_final