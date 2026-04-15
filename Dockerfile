FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir setuptools
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# ✅ Fix para "No module named 'config'"
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "app.py"]