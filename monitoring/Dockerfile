# 1. Pakai Python versi ringan
FROM python:3.9-slim

# 2. Bikin folder kerja
WORKDIR /app

# 3. Copy daftar belanjaan & Install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy script exporter
COPY . .

# 5. Buka port 8000 (Pintu Exporter)
EXPOSE 8000

# 6. Perintah menyalakan script (tanpa buffering biar log realtime)
CMD ["python", "-u", "prometheus_exporter.py"]