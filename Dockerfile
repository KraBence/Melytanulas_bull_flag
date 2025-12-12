# 1. Base Image kiválasztása
FROM tensorflow/tensorflow:2.20.0-jupyter

# 2. Munkakönyvtár beállítása a konténeren belül
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./src .

# 6. A futtató script végrehajthatóvá tétele
RUN chmod +x run.sh

# 7. Alapértelmezett parancs a konténer indításakor
CMD ["bash", "run.sh"]