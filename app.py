import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load model
MODEL_PATH = "emotion_recognition_model.h5"
model = load_model(MODEL_PATH)

# Label klasifikasi (ubah sesuai dengan dataset Anda)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Fungsi untuk memproses gambar
def preprocess_image(image):
    image = image.resize((48, 48)).convert('L')  # Ubah ukuran ke 48x48 dan grayscale
    image_array = img_to_array(image) / 255.0  # Normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
    return image_array

# Judul aplikasi
st.title("Human Emotion Recognition")
st.write("Unggah gambar wajah untuk mengenali ekspresi emosional.")

# Input gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    
    # Prediksi
    st.write("Memproses gambar...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    
    # Tampilkan hasil prediksi
    st.write(f"Prediksi Ekspresi: **{predicted_class}**")
    
    # Buat DataFrame untuk visualisasi
    prediction_data = pd.DataFrame({
        "Emotion": class_labels,
        "Probability": prediction[0]
    })
    
    # Gunakan Altair untuk membuat grafik dengan label
    bars = alt.Chart(prediction_data).mark_bar().encode(
        x=alt.X("Emotion", sort=None),  # Label teks di sumbu x
        y=alt.Y("Probability"),
        color=alt.Color("Emotion", legend=None)  # Opsional: Warna berdasarkan emosi
    )
    
    # Tambahkan angka di atas batang
    text = bars.mark_text(
        align="center",
        baseline="bottom",
        dy=-5  # Jarak teks dari atas batang
    ).encode(
        text=alt.Text("Probability:Q", format=".2f")  # Format angka (2 desimal)
    )
    
    # Gabungkan batang dan teks
    chart = (bars + text).properties(
        width=600,
        height=400
    )
    
    st.altair_chart(chart, use_container_width=True)
