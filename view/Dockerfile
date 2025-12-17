# 1. Pakai Python versi ringan
FROM python:3.9-slim

# 2. Bikin folder kerja di dalam container
WORKDIR /app

# 3. Copy daftar belanjaan & Install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy semua kode app.py
COPY . .

# 5. Buka port 8501 (Pintu Streamlit)
EXPOSE 8501

# 6. Perintah menyalakan aplikasi
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]