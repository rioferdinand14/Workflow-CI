# 🎗️ Breast Cancer Prediction MLOps Pipeline

[![CI Workflow](https://github.com/rioferdinand14/Workflow-CI/actions/workflows/ci.yml/badge.svg)](https://github.com/rioferdinand14/Workflow-CI/actions)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)

> **Live Demo:** [Klik di sini untuk mencoba Aplikasi Streamlit](https://share.streamlit.io/rioferdinand14/Workflow-CI/main/frontend/app.py)

Proyek ini adalah implementasi **End-to-End MLOps** untuk diagnosa kanker payudara. Berbeda dengan proyek *machine learning* biasa, repositori ini berfokus pada **otomatisasi pipeline**, **reproducibility** (keterulangan), dan **skalabilitas infrastruktur**.

---

## 💡 Fitur Utama

### 1. 🧠 Intelligent Model Training
* Menggunakan algoritma **Random Forest Classifier**.
* Dilengkapi dengan **Recursive Feature Elimination (RFE)** yang otomatis menyeleksi **10 fitur medis terpenting** (seperti *mean perimeter, mean area, worst concave points*) untuk meningkatkan efisiensi model.

### 2. 🤖 CI/CD Pipeline (GitHub Actions)
* Setiap kali ada perubahan kode (*push* ke `main`), GitHub Actions akan otomatis:
    * Menyiapkan environment Anaconda.
    * Menjalankan validasi kode (*Linting*).
    * Menjalankan ulang training model untuk memastikan tidak ada error pada script.

### 3. 🔄 Hybrid MLflow Tracking
* Script training dirancang dengan logika **Environment-Aware**:
    * **Mode CI:** Mendeteksi `MLFLOW_RUN_ID` dari GitHub Actions dan menggunakan ID tersebut (mencegah konflik/error "Run ID not found").
    * **Mode Lokal:** Secara otomatis membuat eksperimen dan Run ID baru jika dijalankan di laptop developer.

### 4. 🐳 Containerization & Monitoring
* Menyediakan **Dockerfile** dan **docker-compose** untuk isolasi environment.
* Siap diintegrasikan dengan **Prometheus** untuk memantau metrik sistem (CPU, RAM, Request Latency) secara *real-time* (tersedia pada mode *Local Docker Deployment*).

---

## 📂 Struktur Proyek

```text
├── .github/workflows/    # Konfigurasi CI (GitHub Actions)
├── backend/              # (Opsional) API Service untuk Microservices architecture
├── frontend/             # Aplikasi User Interface berbasis Streamlit
│   ├── app.py            # Main application logic
│   ├── Dockerfile        # Container recipe untuk frontend
│   └── model.pkl         # Model fisik untuk deployment Streamlit Cloud
├── modelling/            # Script Training & MLflow Project
│   ├── train_embedded.py # Script training "Pintar" (Hybrid Logic)
│   ├── MLProject         # Definisi entry point MLflow
│   └── conda.yaml        # Environment dependencies
├── monitoring/           # Konfigurasi Prometheus (Observability)
└── docker-compose.yml    # Orkestrasi Container untuk Local Development


# 1. Clone Repository
git clone [https://github.com/rioferdinand14/Workflow-CI.git](https://github.com/rioferdinand14/Workflow-CI.git)
cd Workflow-CI

# 2. Jalankan Environment (Frontend + Monitoring)
docker-compose up --build
