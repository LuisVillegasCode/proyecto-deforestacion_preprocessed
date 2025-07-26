# /src/main.py

from pathlib import Path
import multiprocessing
import os
import rasterio
from rasterio.merge import merge
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from .producto import ProductoSPOT6

# Esta es la función que ejecutarán los procesos paralelos.
# Necesita estar fuera de la clase para que 'multiprocessing' pueda manejarla.
def worker_task(args):
    """
    Función de trabajo que lee un bloque de datos y llama al método de procesamiento.
    """
    producto, vrt_path, window, cloud_mask = args
    with rasterio.open(vrt_path) as src:
        block_dn = src.read(window=window)
    
    block_mask_subset = cloud_mask[window.row_off:window.row_off+window.height, window.col_off:window.col_off+window.width]
    
    # Llama al método de la instancia del producto para procesar el bloque
    block_procesado = producto.procesar_bloque(block_dn, block_mask_subset)
    
    return block_procesado, window

def main():
    """
    Función principal que orquesta el pipeline de preprocesamiento en paralelo.
    """
    # --- CONFIGURACIÓN ---
    RUTA_BASE = Path(__file__).resolve().parent.parent
    RUTA_DATOS_RAW = RUTA_BASE / "data" / "raw"
    RUTA_DATOS_PROCESADOS = RUTA_BASE / "data" / "processed"
    
    nombre_carpeta_imagen = "CO_2507090419315"
    ruta_completa_imagen = RUTA_DATOS_RAW / nombre_carpeta_imagen
    
    print(f"Buscando producto en: {ruta_completa_imagen}")

    # --- EJECUCIÓN DEL PIPELINE ---
    try:
        # 1. Crear el objeto del producto para acceder a sus metadatos y métodos
        producto_spot = ProductoSPOT6(
            ruta_producto=ruta_completa_imagen,
            ruta_salida=RUTA_DATOS_PROCESADOS
        )
        
        # 2. Preparar las entradas: VRT y máscara de nubes (tareas que se hacen una sola vez)
        print("  -> Preparando archivos de entrada (VRT y máscara)...")
        vrt_path = producto_spot.ruta_salida_producto / 'source.vrt'
        src_files_to_mosaic = [rasterio.open(fp) for fp in producto_spot.metadatos['rutas_tifs']]
        merge(src_files_to_mosaic, dst_path=str(vrt_path))
        
        with rasterio.open(vrt_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.uint16, nodata=0, compress=None, tiled=True, blockxsize=512, blockysize=512)
            
            ruta_mascara_gml = producto_spot.metadatos.get('mascara_nubes_gml')
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
            
            # 3. Crear la lista de tareas a repartir
            tasks = [(producto_spot, vrt_path, window, cloud_mask_refinada) for _, window in src.block_windows(1)]

        # 4. Configurar y ejecutar el pool de procesos paralelos
        num_procesos = max(1, os.cpu_count() - 1)
        print(f"\n--- Iniciando procesamiento paralelo con {num_procesos} núcleos ---")
        path_final = producto_spot.ruta_salida_producto / 'analysis_ready_data.tif'
        
        with rasterio.open(path_final, 'w', **profile) as dst:
            with multiprocessing.Pool(processes=num_procesos) as pool:
                for block_procesado, window in tqdm(pool.imap_unordered(worker_task, tasks), total=len(tasks)):
                    dst.write(block_procesado, window=window)

        print(f"\n--- Pipeline Finalizado. Producto final guardado en: {path_final} ---")

    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()