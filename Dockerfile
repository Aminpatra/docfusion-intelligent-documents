# ─────────────────────────────────────────────────────────────────────────────
# DocFusion – Production Container
# Includes: Python 3.13, Tesseract OCR, all Python deps, Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.13-slim

LABEL maintainer="DocFusion Team" \
      description="Intelligent Document Processing Pipeline"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .

# Install core packages (skip PaddleOCR in Docker for size; use Tesseract)
RUN pip install --no-cache-dir \
    numpy pandas scikit-learn scipy joblib \
    pytesseract Pillow \
    streamlit tqdm python-dotenv

# ── Copy project ─────────────────────────────────────────────────────────────
COPY . .

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ── Ports ─────────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Default: run the Streamlit UI ─────────────────────────────────────────────
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
