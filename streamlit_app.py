import streamlit as st
import pandas as pd
import pickle

# Fungsi untuk load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Fungsi untuk load scaler
@st.cache_resource
def load_scaler():
    with open("scaler2.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

# Load model dan scaler
model = load_model()
scaler = load_scaler()

# Judul aplikasi
st.title("Prediksi Machine Learning dengan Data Excel")

# Upload file Excel
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # Membaca file Excel
        data = pd.read_excel(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data)

        # Preprocessing data menggunakan scaler
        data_scaled = scaler.transform(data)
        
        # Prediksi menggunakan model
        if st.button("Prediksi"):
            predictions = model.predict(data_scaled)
            data["Prediction"] = predictions
            st.success("Prediksi berhasil!")
            st.write("Hasil Prediksi:")
            st.write(data)

            # Tombol untuk mengunduh hasil prediksi
            st.download_button(
                label="Unduh Hasil Prediksi",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error: {e}")
