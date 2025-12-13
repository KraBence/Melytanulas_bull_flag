# 1. Base Image: Python 3.12
FROM python:3.12-slim

# Környezeti változók
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. Munkakönyvtár
WORKDIR /app

# Rendszer szintű függőségek telepítése
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Python csomagok
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY ./src ./src
COPY ./data ./data
COPY run.sh run.sh

# Kimeneti mappa
RUN mkdir -p /app/output
RUN mkdir -p /app/log


RUN chmod +x run.sh


ENV PYTHONPATH="/app/src"
# 6. Indítás
CMD ["bash", "run.sh"]