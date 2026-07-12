# Lightweight Python image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (for ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy full project
COPY . .

# Expose API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
