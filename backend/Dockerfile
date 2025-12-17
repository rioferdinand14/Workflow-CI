FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file (termasuk model.pkl dan main.py)
COPY . .

# Expose port (Render/Cloud biasanya meng-override ini, tapi standar 8000)
EXPOSE 8000

# Jalankan server uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]