FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/             ./src/
COPY model_artifact/  ./model_artifact/
COPY monitoring/      ./monitoring/
COPY app.py           ./app.py

EXPOSE 8501

ENV MODEL_BACKEND=onnx
ENV API_URL=http://localhost:8001

CMD ["python", "app.py"]
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
