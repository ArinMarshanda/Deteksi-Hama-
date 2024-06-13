from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model yang telah disimpan
model = load_model('NASNetMobile.h5')

# Daftar kelas
class_labels = {
    "class1": "Ada hama",
    "class2": "Tanaman sehat"
}

def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Route untuk halaman beranda
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Route untuk prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "File tidak ditemukan", 400

    file = request.files['file']
    if file.filename == '':
        return "Nama file kosong", 400

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        img_array = prepare_image(file_path, target_size=(128, 128))

        predictions = model.predict(img_array)
        predicted_class = list(class_labels.keys())[np.argmax(predictions)]
        confidence = np.max(predictions)
        predicted_class_label = class_labels[predicted_class]

        result = {
            "predicted_class": predicted_class_label,
            "confidence": float(confidence)
        }

        return render_template('result.html', predicted_class=predicted_class_label, confidence=confidence)

# Menjalankan server Flask
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
