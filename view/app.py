import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import os
import pickle

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Breast Cancer AI System",
    page_icon="🎗️",
    layout="wide"
)

# --- CSS CUSTOM ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 3em;
        font-size: 20px;
        border-radius: 10px;
    }
    div[data-testid="stNumberInput"] {
        margin-top: -15px; 
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("frontend/model.pkl", "rb") as f: # Pastikan path benar
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("File model.pkl tidak ditemukan di folder frontend!")
    st.stop()

st.title("🎗️ Breast Cancer Prediction")
st.markdown("Sistem diagnosis berbasis AI.")

# --- FUNGSI INPUT PINTAR ---
def smart_input(label, key, min_val, max_val, default, step=0.01):
    k_slider = f"{key}_slider"
    k_input = f"{key}_input"

    if k_input not in st.session_state: st.session_state[k_input] = default
    if k_slider not in st.session_state: st.session_state[k_slider] = default

    def sync_slider_to_input(): st.session_state[k_input] = st.session_state[k_slider]
    def sync_input_to_slider(): st.session_state[k_slider] = st.session_state[k_input]

    col_s, col_n = st.sidebar.columns([3, 1]) 
    with col_s:
        st.slider(label, float(min_val), float(max_val), key=k_slider, step=step, on_change=sync_slider_to_input)
    with col_n:
        st.number_input("Nilai", float(min_val), float(max_val), key=k_input, step=step, on_change=sync_input_to_slider, label_visibility="collapsed")

    return st.session_state[k_input]

# --- SIDEBAR INPUT (10 FITUR SPESIFIK) ---
st.sidebar.header("📝 Input Data Klinis")
input_data = {}

# Kelompok 1: Mean Attributes (4 Fitur)
st.sidebar.subheader("1. Mean Attributes")
input_data['mean perimeter'] = smart_input("Mean Perimeter", "mp", 43.0, 190.0, 92.0)
input_data['mean area'] = smart_input("Mean Area", "ma", 143.0, 2500.0, 655.0, step=1.0)
input_data['mean concavity'] = smart_input("Mean Concavity", "mcc", 0.0, 0.43, 0.09, step=0.001)
input_data['mean concave points'] = smart_input("Mean Concave Points", "mcp", 0.0, 0.20, 0.05, step=0.001)

# Kelompok 2: Worst Attributes (6 Fitur)
st.sidebar.subheader("2. Worst Attributes")
input_data['worst radius'] = smart_input("Worst Radius", "wr", 7.0, 37.0, 16.0)
input_data['worst texture'] = smart_input("Worst Texture", "wt", 12.0, 50.0, 26.0) # <-- Fitur Baru
input_data['worst perimeter'] = smart_input("Worst Perimeter", "wp", 50.0, 260.0, 107.0)
input_data['worst area'] = smart_input("Worst Area", "wa", 185.0, 4300.0, 880.0, step=1.0)
input_data['worst concavity'] = smart_input("Worst Concavity", "wcc_worst", 0.0, 1.26, 0.27, step=0.01)
input_data['worst concave points'] = smart_input("Worst Concave Points", "wcp_worst", 0.0, 0.30, 0.11, step=0.001)

# --- VISUALISASI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Visualisasi Data")
    # Radar Chart
    categories = ['Perimeter', 'Area', 'Concavity', 'Concave Pts', 'Texture']
    
    # Normalisasi Visual (Agar grafik terbaca)
    # Mean
    mean_vals = [
        input_data['mean perimeter']/190,
        input_data['mean area']/2500,
        input_data['mean concavity']/0.43,
        input_data['mean concave points']/0.20,
        0 # Texture mean tidak ada di input
    ]
    # Worst
    worst_vals = [
        input_data['worst perimeter']/260,
        input_data['worst area']/4300,
        input_data['worst concavity']/1.26,
        input_data['worst concave points']/0.30,
        input_data['worst texture']/50
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=mean_vals, theta=categories, fill='toself', name='Mean Values'))
    fig.add_trace(go.Scatterpolar(r=worst_vals, theta=categories, fill='toself', name='Worst Values'))
    
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔍 Diagnosa AI")
    predict_btn = st.button("ANALISIS PASIEN", type="primary")

if predict_btn:
    try:
        with st.spinner('Memproses Diagnosa...'):
            # 1. Konversi input data (dict) menjadi Numpy Array 2D
            # input_data adalah dictionary yang Anda buat dari form slider
            input_values = list(input_data.values())
            input_array = np.array(input_values).reshape(1, -1)

            # 2. Prediksi Langsung (Tanpa Internet/API)
            start_time = time.time()
            prediction = model.predict(input_array)[0]
            
            # Coba ambil probabilitas jika model mendukung
            try:
                probability = model.predict_proba(input_array).max()
            except:
                probability = 0.0 # Fallback jika model tidak support probability
            
            end_time = time.time()
            latency = round((end_time - start_time), 4)

        # 3. Tampilkan Hasil
        st.success(f"Selesai dalam {latency} detik (Running Locally)")

        if prediction == 1: # Benign
            st.markdown("""
                <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb;">
                    <h2 style="color: #155724; text-align: center;">✅ BENIGN (JINAK)</h2>
                    <p style="text-align: center;">Tidak ditemukan indikasi keganasan.</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        elif prediction == 0: # Malignant
            st.markdown("""
                <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; border: 1px solid #f5c6cb;">
                    <h2 style="color: #721c24; text-align: center;">⚠️ MALIGNANT (GANAS)</h2>
                    <p style="text-align: center;">Terdeteksi karakteristik sel kanker ganas.</p>
                </div>
            """, unsafe_allow_html=True)
            
        # Tampilkan confidence score
        st.info(f"Tingkat Kepercayaan Model: {probability:.2%}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")