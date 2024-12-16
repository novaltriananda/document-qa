import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Fungsi untuk load model
@st.cache_resource
def load_model_keras():
    model = load_model("model.keras")  # Memuat model Keras
    return model

# Fungsi untuk load scaler
@st.cache_resource
def load_scaler():
    with open("scaler2.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

# Load model dan scaler
model = load_model_keras()
scaler = load_scaler()

# Judul aplikasi
st.title("Prediksi Model Machine Learning dengan Data Excel")

# Upload file Excel
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # Membaca file Excel
        data = pd.read_excel(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data)

        # Preprocessing: Gunakan scaler untuk transformasi data
        try:
            data_scaled = scaler.transform(data)
        except Exception as e:
            st.error("Error saat transformasi data. Pastikan kolom data sesuai dengan scaler.")
            st.stop()

        # Prediksi menggunakan model
        if st.button("Prediksi"):
            predictions = model.predict(data_scaled)
            data["Prediction"] = predictions.argmax(axis=1)  # Jika output adalah probabilitas multi-kelas
            st.success("Prediksi berhasil!")
            st.write("Hasil Prediksi:")
            st.write(data)

            # Tombol unduh hasil prediksi
            st.download_button(
                label="Unduh Hasil Prediksi",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error: {e}")
