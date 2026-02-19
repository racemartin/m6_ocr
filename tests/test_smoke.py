import pytest
import mlflow
import os
from dotenv import load_dotenv

# Cargamos el archivo .env
load_dotenv()


def test_mlflow_connection():
    # 1. Configuración de la URI
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(uri)

    # 2. Verificación de formato (lo que ya tenías)
    current_uri = mlflow.get_tracking_uri()
    assert "http" in current_uri, (
        f"La URI {current_uri} no es HTTP. ¿Está bien el .env?"
    )

    # 3. Prueba de fuego: ¿El servidor responde y la DB funciona?
    try:
        # Intentamos obtener la lista de experimentos (operación de lectura en DB)
        experiments = mlflow.search_experiments()
        assert isinstance(experiments, list)
        print(f"\n✅ Conexión exitosa a MLflow en {current_uri}")
    except Exception as e:
        pytest.fail(f"❌ No se pudo conectar al servidor MLflow en Docker: {e}")


def test_imports():
    # Verifica que las librerías críticas carguen bien
    import catboost  # noqa: F401
    import lightgbm  # noqa: F401
    import xgboost  # noqa: F401

    assert True
