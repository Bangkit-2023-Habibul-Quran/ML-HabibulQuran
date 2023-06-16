import librosa
import os
import resampy
from flask import Flask, request, jsonify
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Fungsi untuk ekstraksi fitur menggunakan MFCC
def extract_mfcc(file_path, n_mfcc=13, max_len=60):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# Fungsi untuk ekstraksi fitur menggunakan Spectrogram
def extract_spectrogram(file_path, n_mels=128, n_fft=2048, hop_length=512, max_len=60):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.power_to_db(spectrogram)
    if (max_len > spectrogram.shape[1]):
        pad_width = max_len - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        spectrogram = spectrogram[:, :max_len]
    return spectrogram

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', '3gp', 'mp4', 'mpeg4', 'm4a', 'aac', 'mp2', 'mpeg' }

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load data dari file CSV
Xdata = pd.read_csv('./Dataload/filtered_10class.csv') #path to csv

# Load label encoder dan fitting
label_encoder = LabelEncoder()
label_encoder.fit(Xdata['label'])

# Memuat model yang telah disimpan
model = load_model('./model/model10huruf.h5') #path to model

# # Karakteristik audio yang diharapkan
# expected_sample_rate = 44100
# expected_duration = 0.8356235827664399

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if '' not in request.files:
        return jsonify({'messsage': 'No file part', 'error':True })

    file = request.files['']

    if file is None or file.filename == '':
        return jsonify({'message': 'No file selected', 'error':True })
    # Check if file has allowed extension
    if not allowed_file(file.filename):
        return jsonify({'message': 'Invalid file extension', 'error':True })

    filename = file.filename
    file.save(os.path.join(os.getcwd(), filename))
    
    # Operasi lainnya pada file baru
    # check_audio_characteristics(filename, expected_sample_rate, expected_duration)
    mfcc_features = extract_mfcc(filename)
    spectrogram_features = extract_spectrogram(filename)

    # Ubah dimensi data untuk model CNN
    mfcc_features = mfcc_features[np.newaxis, ..., np.newaxis]
    spectrogram_features = spectrogram_features[np.newaxis, ..., np.newaxis]

    # Prediksi label menggunakan model
    predictions = model.predict([mfcc_features, spectrogram_features])

    # Decode label menggunakan label_encoder
    predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])[0]
    os.remove(os.path.join(os.getcwd(), filename))

    print('Predicted label:', predicted_label)
    return jsonify({'Predicted label': predicted_label, 'error':False})

if __name__ == '__main__':
    app.run()
