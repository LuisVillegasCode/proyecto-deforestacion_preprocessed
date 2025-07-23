# src/procesador.py

from .producto import Producto_Satelital_Base

class ProcesadorImagenes:
    """
    Contiene los algoritmos de preprocesamiento.
    Es agnóstico al tipo de satélite que procesa.
    """
    def __init__(self, ruta_salida):
        self.ruta_salida = ruta_salida
        # Asegurarse de que la carpeta de salida exista
        self.ruta_salida.mkdir(parents=True, exist_ok=True)

    def calibracion_radiometrica(self, producto: Producto_Satelital_Base):
        print(f"  -> Ejecutando: Calibración Radiométrica...")
        # TODO: Implementar la lógica de calibración aquí
        # Usarías los datos de producto.metadatos['ganancias_calibracion']
        pass
        
    def correccion_atmosferica(self, producto: Producto_Satelital_Base):
        print(f"  -> Ejecutando: Corrección Atmosférica...")
        # TODO: Implementar la lógica de corrección (ej. DOS1) aquí
        # Usarías producto.metadatos['elevacion_solar'], etc.
        pass

    def aplicar_mascara_nubes(self, producto: Producto_Satelital_Base):
        print(f"  -> Ejecutando: Aplicación de Máscara de Nubes...")
        ruta_mascara = producto.mascara_nubes_path
        if ruta_mascara and ruta_mascara.exists():
            print(f"     Máscara encontrada en: {ruta_mascara.name}")
            # TODO: Implementar la lógica para rasterizar y aplicar la máscara
        else:
            print("     Advertencia: No se encontró máscara de nubes predefinida.")
        pass