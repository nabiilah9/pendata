import streamlit as st
import pandas as pd
import pickle

# Load model
with open('breast.pkl', 'rb') as file:
    model = pickle.load(file)

# Judul aplikasi
st.title("Prediksi Diagnosis Kanker Payudara")
st.markdown("Model klasifikasi untuk memprediksi jenis kanker berdasarkan parameter medis.")

# Daftar fitur sesuai urutan model
fitur = [
    'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1',
    'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1',
    'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2',
    'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2',
    'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3',
    'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3'
]

# Form input user
st.sidebar.header("Input Data")

user_input = []
for f in fitur:
    value = st.sidebar.number_input(f"{f}", value=0.0)
    user_input.append(value)

# Konversi ke DataFrame
input_df = pd.DataFrame([user_input], columns=fitur)

# Prediksi
if st.button("Prediksi"):
    prediksi = model.predict(input_df)[0]
    label = "Ganas (Malignant)" if prediksi == 1 else "Jinak (Benign)"
    st.subheader("Hasil Prediksi:")
    st.success(f"Diagnosis: **{label}**")
