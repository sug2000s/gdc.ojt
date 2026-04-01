FROM python:3.11-slim

WORKDIR /app

# System deps required by pymupdf (fitz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY demo.py .

# Environment variables (override at runtime via docker run -e or --env-file)
ENV OPENAI_MODEL_NAME=gpt-4o
ENV PDF_OCR_MAX_PAGES=10

# Persist SQLite memory.db outside the container
VOLUME ["/app/memory.db"]

EXPOSE 8501

CMD ["streamlit", "run", "demo.py", "--server.port=8501", "--server.address=0.0.0.0"]
