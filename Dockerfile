FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY web/ ./web/
COPY data/ ./data/

# Hugging Face Spaces routes to port 7860; override PORT to run elsewhere.
ENV PORT=7860
EXPOSE 7860 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
