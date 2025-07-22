import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

# Configurações do app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carrega modelo TFLite
interpreter = tflite.Interpreter(model_path="cats_vs_dogs_mobilenetv2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = (160, 160)

# Função de predição
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if prediction > 0.5:
        return f"🐶 Cão detectado com {prediction * 100:.2f}% de confiança"
    else:
        return f"🐱 Gato detectado com {(1 - prediction) * 100:.2f}% de confiança"

# Rota principal
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            image_path = filepath

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
