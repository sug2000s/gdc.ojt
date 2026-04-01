FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY agent.py app.py constants.py config.py prompt_chain.py graph.py memory.py tools.py ./

EXPOSE 8501

# Streamlit needs a writable home for its config/cache
ENV HOME=/tmp

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
