FROM python:3.12-slim

WORKDIR /app

# Copy dependency specs
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy application code and model artifacts
COPY . .

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]