# 1. Base Image: Python 3.12
FROM python:3.12-slim

# Környezeti változók
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. Munkakönyvtár
WORKDIR /app

# Rendszer szintű függőségek telepítése (OpenCV-hez kötelező!)
# JAVÍTÁS: libgl1-mesa-glx -> libgl1
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Python csomagok
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Kód másolása (Megtartva a mappaszerkezetet!)
#COPY ./src ./src
COPY ./data ./data
COPY run.sh run.sh

# Kimeneti mappa
RUN mkdir -p /app/output

# 5. PYTHONPATH beállítása (Hogy a Python lássa a modulokat az src-ben)
# JAVÍTÁS: A $PYTHONPATH változót nem használjuk a definícióban, csak felülírjuk/beállítjuk
ENV PYTHONPATH="/app/src"

# 6. Indítás
CMD ["bash"]