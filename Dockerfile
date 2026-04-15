FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# 1. Actualizar pip e instalar setuptools primero
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 2. Luego instalar tus dependencias de producción
COPY . .
RUN pip install --no-cache-dir -r requirements-prod.txt



# ✅ Fix para "No module named 'config'"
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "app.py"]