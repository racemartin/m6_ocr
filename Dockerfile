# 1. Siempre empezar con la imagen base
FROM python:3.12-slim

# 2. Cambiar a root para instalar librerías del sistema
USER root

# 3. Instalar dependencias necesarias (libgomp1 para LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Configurar el directorio de trabajo
WORKDIR /app

# 5. Instalación de herramientas básicas
RUN pip install --no-cache-dir "setuptools<71.0.0"

# 6. Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copiar el resto del código
COPY . .

# 8. Permisos y variables de entorno
RUN chmod -R 777 /app/model_artifact
ENV PYTHONPATH=/app
ENV PORT=7860

# 9. Volver al usuario por defecto (Hugging Face usa el UID 1000)
# Es mejor no forzar "USER user" a menos que lo hayas creado, 
# pero dejarlo como root suele funcionar en Spaces si no hay restricciones.
# Si falla por permisos, borra la línea de abajo.
# USER 1000 

EXPOSE 7860

# 10. Comando de ejecución
CMD ["python", "app.py"]