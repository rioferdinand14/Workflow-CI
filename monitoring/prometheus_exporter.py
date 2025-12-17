from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# --- METRICS ---
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')

@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=None)) 
    RAM_USAGE.set(psutil.virtual_memory().percent) 
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    print("\n[INFO] 1. Menerima Request dari Streamlit...") # DEBUG
    start_time = time.time()
    REQUEST_COUNT.inc()

    # URL MLFLOW (Pastikan Port 5002)
    api_url = "http://127.0.0.1:5002/invocations"
    
    try:
        data = request.get_json()
        print(f"[INFO] 2. Data diterima: {str(data)[:100]}...") # DEBUG: Cek isi data
        
        print(f"[INFO] 3. Mengirim ke MLflow ({api_url})...") # DEBUG
        response = requests.post(api_url, json=data)
        
        print(f"[INFO] 4. Jawaban MLflow: Status {response.status_code}") # DEBUG
        
        if response.status_code != 200:
            print(f"[ERROR] MLflow Error: {response.text}") # DEBUG: Print error MLflow
            return jsonify({"error": "MLflow Error", "detail": response.text}), response.status_code

        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        
        return jsonify(response.json())

    except Exception as e:
        print(f"[ERROR] CRITICAL: {e}") # DEBUG
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("✅ Middleware Exporter jalan di port 8000...")
    app.run(host='127.0.0.1', port=8000)