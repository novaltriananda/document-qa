import streamlit as st
import pandas as pd
import pickle

# Fungsi untuk load model
@st.cache_resource
def load_model():
    with open("model.keras", "rb") as file:
        model = pickle.load(file)
    return model

# Load model
model = load_model()

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

        # Prediksi menggunakan model
        if st.button("Prediksi"):
            predictions = model.predict(data)
            data["Prediction"] = predictions
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
