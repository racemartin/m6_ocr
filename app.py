import subprocess, sys
from threading import Thread

def run_api():
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "src.api.main:application",
        "--host", "0.0.0.0", "--port", "8001"
    ])

def run_streamlit():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "monitoring/simulator.py",
        "--server.port", "7860",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ])

if __name__ == "__main__":
    Thread(target=run_api, daemon=True).start()
    run_streamlit()