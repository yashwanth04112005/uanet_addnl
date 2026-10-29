# UANet Web Deployment Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and deployment dependencies
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy project files
COPY . .

# Expose ports: 8501 (Web Demo / Streamlit) and 8000 (API)
EXPOSE 8501 8000

# Default: Run the interactive web server
CMD ["python", "demo_server.py"]

