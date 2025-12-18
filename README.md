# 🎗️ Breast Cancer Prediction MLOps Pipeline

[![CI Workflow](https://github.com/rioferdinand14/BreastCancer_Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/rioferdinand14/BreastCancer_Classification/actions)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)

> **Live Demo:** [Klik di sini untuk mencoba Aplikasi Streamlit](https://share.streamlit.io/rioferdinand14/Workflow-CI/main/frontend/app.py)

Proyek ini adalah implementasi **End-to-End MLOps** untuk diagnosa kanker payudara. Berbeda dengan proyek *machine learning* biasa, repositori ini berfokus pada **otomatisasi pipeline**, **reproducibility** (keterulangan), dan **skalabilitas infrastruktur**.

---

## 💡 Fitur Utama

### 1. Model Training
* Menggunakan algoritma **Random Forest Classifier**.
* Dilengkapi dengan **Recursive Feature Elimination (RFE)** yang otomatis menyeleksi **10 fitur medis terpenting** (seperti *mean perimeter, mean area, worst concave points*) untuk meningkatkan efisiensi model.

### 2. CI/CD Pipeline (GitHub Actions)
* Setiap kali ada perubahan kode (*push* ke `main`), GitHub Actions akan otomatis:
    * Menyiapkan environment Anaconda.
    * Menjalankan validasi kode (*Linting*).
    * Menjalankan ulang training model untuk memastikan tidak ada error pada script.

### 3. MLflow Tracking
* Script training dirancang dengan logika **Environment-Aware**:
    * **Mode CI:** Mendeteksi `MLFLOW_RUN_ID` dari GitHub Actions dan menggunakan ID tersebut (mencegah konflik/error "Run ID not found").
    * **Mode Lokal:** Secara otomatis membuat eksperimen dan Run ID baru jika dijalankan di laptop developer.

### 4. Monitoring
* Siap diintegrasikan dengan **Prometheus** untuk memantau metrik sistem (CPU, RAM, Request Latency) secara *real-time* 
---

## 📂 Struktur Proyek

```text
├── .github/workflows/    # Konfigurasi CI (GitHub Actions)
├── view/                 # Aplikasi User Interface berbasis Streamlit
│   ├── app.py            # Main application logic
│   └── model.pkl         # Model fisik untuk deployment Streamlit Cloud
├── modelling/            # Script Training & MLflow Project
│   ├── modelling.py      # Script training "Pintar" (Hybrid Logic)
│   ├── MLProject         # Definisi entry point MLflow
│   └── conda.yaml        # Environment dependencies
├── monitoring/           # Konfigurasi Prometheus (Observability)
│   ├── prometheus_exporter.py # Middleware untuk meminta prediksi
│   ├── prometheus.yml         # Konfigurasi untuk monitoring model
└── docker-compose.yml    # Orkestrasi Container untuk Local Development
└── requirements.txt      # Dependencies python


# 1. Clone Repository
   git clone [https://github.com/rioferdinand14/BreastCancer_Classification.git]
   cd BreastCancer_Classification

# 2. Buat virtual environment (Disarankan)
   python -m venv venv
   Aktifkan venv (Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate)

# 3. Install library
   pip install -r requirements.txt

# 4. Training Model
   cd modelling
   python modelling.py

# 5. Serve Model
   # Pastikan posisi di root folder project
   mlflow models serve -m "modelling/mlruns/<ID>/<RUN_ID>/artifacts/model" -p 5002 --env-manager=local

# 6. Middleware Exporter
   cd monitoring
   python prometheus_exporter.py

# 7. Prometheus (Monitoring Dashboard)
   Windows :
   docker run -p 9090:9090 -v "${PWD}/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus
   
   Linux/Mac:
   docker run -p 9090:9090 -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

# 8. Buka localhost:9090/targets dan pastikan statusnya tidak "DOWN"

# 9. Streamlit
   cd view
   streamlit run app.py

# 10. Buka localhost:9090/query untuk meliihat metrik yang dihasilkan oleh model prediksi.
