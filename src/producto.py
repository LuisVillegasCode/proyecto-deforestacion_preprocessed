# src/producto.py

from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import math

# --- LA CLASE BASE ABSTRACTA (El "Contrato") ---
class Producto_Satelital_Base(ABC):
    def __init__(self, ruta_producto: Path, ruta_salida: Path):
        if not ruta_producto.is_dir():
            raise FileNotFoundError(f"La ruta del producto no existe: {ruta_producto}")
        self.ruta_base = ruta_producto
        self.ruta_salida = ruta_salida
        self.metadatos = {}
        
        self.ruta_salida_producto = self.ruta_salida / self.ruta_base.name
        self.ruta_salida_producto.mkdir(parents=True, exist_ok=True)
        
        self._leer_metadatos()

    # Métodos abstractos que cada clase hija debe implementar
    @abstractmethod
    def _leer_metadatos(self): pass
    @abstractmethod
    def calibracion_radiometrica(self): pass
    @abstractmethod
    def correccion_atmosferica(self, rad_data, profile): pass
    @abstractmethod
    def aplicar_mascara_nubes(self, ref_data, profile): pass

    def ejecutar_preprocesamiento(self):
        """
        Método principal que ejecuta todo el pipeline en memoria y guarda solo el resultado final.
        """
        print(f"--- Iniciando Pipeline para: {self.ruta_base.name} ---")
        # Paso 1: Calibración
        rad_data, profile = self.calibracion_radiometrica()
        # Paso 2: Corrección Atmosférica
        ref_data = self.correccion_atmosferica(rad_data, profile)
        # Paso 3: Enmascaramiento y Guardado Final
        path_final = self.aplicar_mascara_nubes(ref_data, profile)
        print(f"--- Pipeline Finalizado. Producto final guardado en: {path_final} ---")
        return path_final

# --- CLASE CONCRETA PARA SPOT 6 ---
class ProductoSPOT6(Producto_Satelital_Base):
    def _leer_metadatos(self):
        xml_files = list(self.ruta_base.rglob('DIM_*.XML'))
        if not xml_files:
            raise FileNotFoundError(f"No se encontró el archivo DIM XML principal en {self.ruta_base} o sus subcarpetas.")
        
        xml_path = xml_files[0]
        self.xml_parent_dir = xml_path.parent
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        self.metadatos['nivel_procesamiento'] = root.find('.//PROCESSING_LEVEL').text
        center_located_values = root.find(".//Located_Geometric_Values[LOCATION_TYPE='Center']")
        self.metadatos['elevacion_solar'] = float(center_located_values.find(".//SUN_ELEVATION").text)
        self.metadatos['ganancias'] = {b.find('BAND_ID').text: float(b.find('GAIN').text) for b in root.findall(".//Band_Radiance")}
        self.metadatos['bias'] = {b.find('BAND_ID').text: float(b.find('BIAS').text) for b in root.findall(".//Band_Radiance")}
        self.metadatos['irradiancias'] = {b.find('BAND_ID').text: float(b.find('VALUE').text) for b in root.findall(".//Band_Solar_Irradiance")}
        cloud_mask_node = root.find(".//MEASURE_NAME[.='Cloud_Cotation (CLD)']/../Quality_Mask/Component/COMPONENT_PATH")
        if cloud_mask_node is not None:
             self.metadatos['mascara_nubes'] = self.xml_parent_dir / cloud_mask_node.attrib['href']
        self.metadatos['band_order'] = [b.find('BAND_ID').text for b in root.findall(".//Raster_Index")]
        self.metadatos['rutas_tifs'] = [self.xml_parent_dir / f.attrib['href'] for f in root.findall(".//Data_File/DATA_FILE_PATH")]
        print(f"Metadatos de {self.ruta_base.name} cargados exitosamente.")

    def calibracion_radiometrica(self):
        print("  -> Paso 1: Realizando Calibración Radiométrica en memoria...")
        src_files_to_mosaic = [rasterio.open(fp) for fp in self.metadatos['rutas_tifs']]
        
        # Merge lee los datos de los tiles y los devuelve como un array de NumPy
        merged_data, out_transform = merge(src_files_to_mosaic)
        
        # Obtener el perfil de la imagen para usarlo después
        profile = src_files_to_mosaic[0].profile
        profile.update(height=merged_data.shape[1], width=merged_data.shape[2], transform=out_transform)

        # Se realiza la calibración banda por banda
        radiance_data = np.zeros_like(merged_data, dtype=np.float32)
        for i in range(merged_data.shape[0]):
            band_id = self.metadatos['band_order'][i]
            gain = self.metadatos['ganancias'][band_id]
            bias = self.metadatos['bias'][band_id]
            radiance_data[i] = merged_data[i].astype(np.float32) / gain + bias
            
        return radiance_data, profile

    def correccion_atmosferica(self, rad_data, profile):
        print("  -> Paso 2: Realizando Corrección Atmosférica en memoria...")
        sun_elevation = self.metadatos['elevacion_solar']
        sun_zenith = 90.0 - sun_elevation
        earth_sun_distance = 1.0 
        
        reflectance_data = np.zeros_like(rad_data, dtype=np.float32)
        for i in range(rad_data.shape[0]):
            band_radiance = rad_data[i]
            band_id = self.metadatos['band_order'][i]
            esun = self.metadatos['irradiancias'][band_id]
            
            dark_object = np.percentile(band_radiance[band_radiance > 0], 1)
            path_radiance = dark_object - 0.01 * (esun * math.cos(math.radians(sun_zenith))**2) / (math.pi * earth_sun_distance**2)

            numerator = math.pi * (band_radiance - path_radiance) * earth_sun_distance**2
            denominator = esun * math.cos(math.radians(sun_zenith))
            reflectance_data[i] = numerator / denominator
            
        return reflectance_data
        
    def aplicar_mascara_nubes(self, ref_data, profile):
        print("  -> Paso 3: Aplicando Máscara de Nubes y guardando archivo final...")
        
        ruta_mascara_gml = self.metadatos.get('mascara_nubes')
        
        if not ruta_mascara_gml or not ruta_mascara_gml.exists():
            print("     Advertencia: No se encontró máscara de nubes.")
            masked_data = ref_data
        else:
            gdf_clouds = gpd.read_file(ruta_mascara_gml)
            # Para usar rasterio.mask, necesitamos guardar temporalmente el dato en memoria
            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(**profile, dtype=rasterio.float32) as dataset:
                    dataset.write(ref_data.astype(rasterio.float32))
                    masked_data, _ = mask(dataset=dataset, shapes=gdf_clouds.geometry, invert=True, nodata=np.nan, filled=True)

        # Escalado y conversión a entero para el guardado final
        masked_data_no_nan = np.nan_to_num(masked_data, nan=0.0)
        scaled_data = (masked_data_no_nan * 10000).astype(rasterio.uint16)
        nodata_value_uint16 = 0
        scaled_data[np.isnan(masked_data)] = nodata_value_uint16
        
        # Actualizar el perfil para el archivo final
        profile.update(dtype=rasterio.uint16, nodata=nodata_value_uint16, compress=None, tiled=False)
        
        path_final = self.ruta_salida_producto / 'analysis_ready_data.tif'
        with rasterio.open(path_final, 'w', **profile) as dst:
            dst.write(scaled_data)
        
        print(f"     Archivo final guardado en: {path_final.name}")
        return path_final