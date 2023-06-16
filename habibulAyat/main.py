import os
import tensorflow as tf
from transformers import TFWav2Vec2ForCTC, Wav2Vec2Processor, AutoModel, AutoTokenizer
from flask import Flask, request, jsonify
import librosa
from werkzeug.utils import secure_filename
from tempfile import TemporaryFile

app = Flask(__name__)

custom_cache_dir = "./caching" #path untuk menyimpan cache model
my_model="./load" #path untuk menyimpan model.h5
processor = Wav2Vec2Processor.from_pretrained(my_model, cache_dir=custom_cache_dir, local_files_only=True)
model = TFWav2Vec2ForCTC.from_pretrained(my_model, cache_dir=custom_cache_dir, local_files_only=True)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', '3gp', 'mp4', 'mpeg4', 'm4a', 'aac', 'mp2', 'mpeg' }

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resample_audio(audio, original_sample_rate, target_sample_rate):
    resampled_audio = librosa.resample(audio, orig_sr=original_sample_rate, target_sr=target_sample_rate)
    return resampled_audio

def predict_from_file(file_path):
    speech, sample_rate = librosa.load(file_path)
    target_sample_rate = 16000                                                    # klu ga support hilangin aja yang ku comment ini
    speech = resample_audio(speech, sample_rate, target_sample_rate)              #  klu ga support hilangin aja yang ku comment ini
    input_values = processor(speech, return_tensors="tf").input_values
    logits = model(input_values).logits
    predicted_ids = tf.argmax(logits, axis=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

def calculate_cer(original_word, transcribed_word):
    original_len = len(original_word)
    transcribed_len = len(transcribed_word)
    distance = edit_distance(original_word, transcribed_word)
    cer = min((distance / original_len) * 100, 100)  # Limit the maximum CER to 100
    return cer

def get_rating(original_text, transcribed_text):
    original_words = original_text.split()
    transcribed_words = transcribed_text.split()
    wer = 0
    reduc1 = 0
    total_cer = 0
    average_cer=0
    reduc2 = 0
    rating_reduction = 0

    for original_word, transcribed_word in zip(original_words, transcribed_words):
        cer = calculate_cer(original_word, transcribed_word)
        if 60 <= cer <= 100:
            rating_reduction = 4
        elif cer == 0:
            rating_reduction = 0
        else:
            total_cer += cer

        print("cer :", cer)
        print("rate :",rating_reduction)
        reduc1 += rating_reduction

    if len(original_words) > 0:
        average_cer = total_cer / len(original_words)
        if average_cer < 15:
            reduc2 = 0
        elif average_cer < 25:
            reduc2 = 1
        elif average_cer < 45:
            reduc2 = 2
        else:
            reduc2 = 3
        print("aver",average_cer)
        rating = 5 - reduc1 - reduc2
        rating = max(rating, 1)

    return rating

def edit_distance(a, b):
    m = len(a)
    n = len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])

    return dp[m][n]

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'messsage': 'No file part', 'error':True })

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No file selected', 'error':True })
    # Check if file has allowed extension
    if not allowed_file(file.filename):
        return jsonify({'message': 'Invalid file extension', 'error':True })

    with TemporaryFile() as temp_file:
        file.save(temp_file)
        temp_file.seek(0)
        predicted_transcription = predict_from_file(temp_file)

    transcribed_text = predicted_transcription 

    # Mengekstrak teks asli
    original_text = request.form['original_text']
    rate = get_rating(original_text, transcribed_text)
    print("text :", transcribed_text )
    print("rating", rate)

    return jsonify({"Predicted transcription": predicted_transcription,
                    "Rating": rate,
                    'error':False})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
