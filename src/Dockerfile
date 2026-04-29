FROM python:3.12-slim

WORKDIR /app

# Copy dependency specs
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Get Model files from Github Releases
ARG MODEL_VERSION=v0.1.0
ARG GITHUB_REPO=minimartzz/bg-reco

RUN apt-get update && apt-get isntall -y curl && rm -rf /var/lib/apt/lists/* && \
    mkdir -p models && \
    for file in emb_cfg.pkl game_index.json game_profiles.pkl meta.json reranker.pt \
    scaler.pkl tag_vocabs.pkl text_encoder.pkl two_tower.pt; do \
    echo "Downloading $file..." && \
    curl -L -o models/$file \
    "https://github.com/${GITHUB_REPO}/releases/download/${MODEL_VERSION}/$file"; \
    done

# Copy application code and model artifacts
COPY . .

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]