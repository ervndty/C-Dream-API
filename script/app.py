from flask import Flask, request, jsonify
from collections import OrderedDict
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow_hub as hub
from tensorflow.keras.utils import img_to_array
import os

# Menonaktifkan penggunaan GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Inisialisasi Flask
app = Flask(__name__)

# Load model dengan custom object
model = load_model(
    'script/model/Model_fix.h5', compile=False,
    custom_objects={'KerasLayer': hub.KerasLayer}
)

# Dictionary berisi deskripsi penyakit dan penanganannya
info_penyakit = {
    "Actinic keratosis": {
        "deskripsi": "Actinic keratosis adalah kondisi kulit praser kanker yang disebabkan oleh paparan sinar matahari berlebih.",
        "penanganan": "Gunakan pelindung kulit seperti sunscreen SPF tinggi, dan konsultasikan ke dokter untuk cryotherapy atau pengobatan topikal."
    },
    "Atopic Dermatitis": {
        "deskripsi": "Atopic Dermatitis adalah peradangan kulit kronis yang menyebabkan kulit kering, gatal, dan ruam.",
        "penanganan": "Hindari alergen dan gunakan pelembap. Jika parah, gunakan krim steroid atau konsultasikan ke dokter untuk terapi imunosupresif."
    },
    "Benign keratosis": {
        "deskripsi": "Benign keratosis adalah lesi kulit jinak yang biasanya muncul akibat penuaan atau paparan sinar matahari.",
        "penanganan": "Tidak memerlukan pengobatan khusus kecuali untuk alasan kosmetik. Dapat dilakukan pengangkatan dengan cryotherapy atau laser."
    },
    "Dermatofibroma": {
        "deskripsi": "Dermatofibroma adalah nodul kulit jinak yang biasanya muncul pada kaki atau lengan.",
        "penanganan": "Tidak memerlukan pengobatan kecuali mengganggu. Untuk alasan kosmetik, dapat diangkat melalui pembedahan."
    },
    "Melanocytic nevus": {
        "deskripsi": "Melanocytic nevus adalah tahi lalat atau lesi kulit jinak yang mengandung melanosit.",
        "penanganan": "Pantau perubahan bentuk, warna, atau ukuran. Jika mencurigakan, segera konsultasikan ke dokter untuk biopsi."
    },
    "Melanoma": {
        "deskripsi": "Melanoma adalah kanker kulit yang sangat serius, berkembang dari sel penghasil pigmen (melanosit).",
        "penanganan": "Segera konsultasikan ke dokter kulit. Penanganan melibatkan pembedahan, imunoterapi, atau kemoterapi, tergantung stadium."
    },
    "Squamous cell carcinoma": {
        "deskripsi": "Squamous cell carcinoma adalah jenis kanker kulit yang muncul pada lapisan luar kulit.",
        "penanganan": "Penanganan termasuk eksisi bedah, radioterapi, atau terapi topikal, tergantung pada tingkat keparahan."
    },
    "Tinea Ringworm Candidiasis": {
        "deskripsi": "Tinea atau Candidiasis adalah infeksi jamur yang menyebabkan ruam berbentuk cincin pada kulit.",
        "penanganan": "Gunakan obat antijamur topikal atau oral. Jaga area tetap kering dan bersih untuk mencegah infeksi berulang."
    },
    "Vascular lesion": {
        "deskripsi": "Vascular lesion adalah kelainan pembuluh darah pada kulit, seperti angioma atau malformasi vena.",
        "penanganan": "Penanganan melibatkan observasi, skleroterapi, atau pengangkatan laser, tergantung pada jenis lesi dan gejala."
    }
}

# Label kelas
label_kelas = [
    'Actinic keratosis',
    'Atopic Dermatitis',
    'Benign keratosis',
    'Dermatofibroma',
    'Melanocytic nevus',
    'Melanoma',
    'Squamous cell carcinoma',
    'Tinea Ringworm Candidiasis',
    'Vascular lesion',
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil file gambar dari request
        gambar = request.files['gambar']
        
        # Membuka gambar menggunakan PIL
        img = Image.open(gambar)
        
        # Mengubah ukuran gambar sesuai dengan input model (224x224)
        img = img.resize((224, 224))
        
        # Mengkonversi gambar ke array dan normalisasi
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalisasi ke rentang 0-1
        img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch

        # Melakukan prediksi
        prediksi = model.predict(img_array)
        
        # Mendapatkan kelas dengan probabilitas tertinggi
        kelas_prediksi = np.argmax(prediksi, axis=1)[0]
        nama_kelas = label_kelas[kelas_prediksi]
        probabilitas = float(np.max(prediksi))
        
        # Informasi penyakit
        info = info_penyakit.get(nama_kelas, {
            "deskripsi": "Informasi tidak tersedia.",
            "penanganan": "Konsultasikan ke dokter."
        })
        
        # Mengembalikan hasil prediksi sebagai JSON
        return jsonify(OrderedDict([
            ('penyakit', nama_kelas),
            ('deskripsi', info["deskripsi"]),
            ('penanganan', info["penanganan"]),
            ('probabilitas', probabilitas),
            ('status', 'success'),
        ]))
    except Exception as e:
        # Mengembalikan error jika ada masalah dengan proses prediksi
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    app.run(debug=True)