FROM python:3.12-slim
WORKDIR /app

COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

COPY . .

# ✅ Fix para "No module named 'config'"
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "app.py"]