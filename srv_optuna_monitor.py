import os
import optuna
from dotenv import load_dotenv
from optuna_dashboard import run_server

# 1. Charger les variables du fichier .env
load_dotenv()

# 2. Récupérer l'URL depuis l'environnement (ou utiliser une valeur par défaut)
STORAGE_URL = os.getenv("OPTUNA_STORAGE_URL", "sqlite:///optuna_phase4.db")
HOST        = os.getenv("OPTUNA_DASHBOARD_HOST", "0.0.0.0")  
PORT        = os.getenv("OPTUNA_DASHBOARD_PORT", 8082)

if __name__ == "__main__":
    # 1. Cargamos el almacenamiento
    storage = optuna.storages.RDBStorage(STORAGE_URL)
    
    print(f"🚀 Lanzando Optuna Dashboard en http://{HOST}:{PORT}")
    print(f"📖 Leyendo base de datos: {STORAGE_URL}")
    
    # 2. Lanzamos el servidor (esto no usará gunicorn en Windows)
    run_server(storage, host=HOST, port=PORT)