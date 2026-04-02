import os
import inspect

class InfoCodigo:
    """Clase de utilidad para introspección de código."""

    def obtener_nombre_fichero(self) -> str:
        """Devuelve el nombre del archivo actual con su extensión."""
        # __file__ es seguro en ficheros .py
        return os.path.basename(__file__)

    def obtener_nombre_clase(self) -> str:
        """Devuelve el nombre de la clase de la instancia actual."""
        return self.__class__.__name__

    def obtener_nombre_metodo(self) -> str:
        """Devuelve el nombre del método que está ejecutando esta línea."""
        # f_code.co_name extrae el nombre del objeto código en el frame actual
        return inspect.currentframe().f_code.co_name

# --- Ejemplo de uso ---
if __name__ == "__main__":
    info = InfoCodigo()
    
    print(f"Archivo: {info.obtener_nombre_fichero()}")
    print(f"Clase:   {info.obtener_nombre_clase()}")
    print(f"Método:  {info.obtener_nombre_metodo()}")