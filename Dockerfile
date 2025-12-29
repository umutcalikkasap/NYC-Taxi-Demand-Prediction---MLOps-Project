# Python 3.9 Slim imajını kullan (Hafif ve hızlı)
FROM python:3.9-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
# gcc ve python-dev gibi derleme araçlarını yükle (bazı kütüphaneler için gerekebilir)
RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Kaynak kodları ve modelleri kopyala
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed_inference_data_2025.parquet ./data/processed_inference_data_2025.parquet

# API portunu dışarı aç
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]