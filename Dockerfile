# Use a lightweight python base image (3.13 matches local dev + Streamlit Cloud;
# pinned numpy 2.3 / pandas 3.0 in requirements.txt require Python >= 3.11)
FROM python:3.13-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies, including Tesseract OCR for ocrmypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    icc-profiles-free \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies (ignoring system clashes and pinning pip limits)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Healthcheck to verify container health (python is guaranteed present; curl is not in slim images)
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Boot-time restore of the private knowledge bundle (public-repo deploys).
# Local Docker keeps knowledge mounted as a volume, so this is a fast no-op
# there; on fresh containers without the volume it downloads/extracts once.
# NOTE: shell form used so '&&' works (exec-form JSON would pass '&&' as argv).
CMD python core/bootstrap.py && streamlit run ui.py --server.port=8501 --server.address=0.0.0.0
