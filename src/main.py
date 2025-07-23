# /src/main.py

from pathlib import Path

# Usa un punto (.) para indicar una importación relativa desde el MISMO paquete (src).
from .producto import ProductoSPOT6
# En el futuro, podrías añadir: from .producto import ProductoSentinel2

def main():
    """
    Función principal que orquesta todo el pipeline de preprocesamiento.
    """
    # --- CONFIGURACIÓN DE RUTAS ---
    # Se define la ruta base del proyecto subiendo un nivel desde la ubicación
    # actual del script (que está en /src).
    RUTA_BASE = Path(__file__).resolve().parent.parent
    RUTA_DATOS_RAW = RUTA_BASE / "data" / "raw"
    RUTA_DATOS_PROCESADOS = RUTA_BASE / "data" / "processed"

    # --- NOMBRE DE LA IMAGEN A PROCESAR ---
    # Coloca aquí el nombre exacto de la carpeta de tu producto SPOT 6
    nombre_carpeta_imagen = "CO_2507090419315"
    ruta_completa_imagen = RUTA_DATOS_RAW / nombre_carpeta_imagen
    
    print(f"Buscando producto en: {ruta_completa_imagen}")

    # --- EJECUCIÓN DEL PIPELINE ---
    try:
        # 1. Se crea el objeto específico para el satélite.
        #    Le pasamos la ruta a los datos crudos y dónde debe guardar los resultados.
        producto_spot = ProductoSPOT6(
            ruta_producto=ruta_completa_imagen,
            ruta_salida=RUTA_DATOS_PROCESADOS
        )
        
        # 2. Se le da una única orden: "procesate".
        #    La clase se encargará de ejecutar todos los pasos internamente.
        producto_spot.ejecutar_preprocesamiento()

    except FileNotFoundError as e:
        print(f"Error: No se pudo cargar el producto. Revisa la ruta y el nombre de la carpeta.\n{e}")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado durante el procesamiento: {e}")

if __name__ == "__main__":
    main()