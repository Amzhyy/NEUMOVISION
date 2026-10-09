"""
Backend de PneumoVision AI (fase 1: login facial).

Uso (desde la carpeta PythonProject1):
    python main.py
y abre http://localhost:5000
"""
import base64
import os

import cv2
import imutils
import numpy as np
from flask import Flask, jsonify, redirect, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Data')
MODEL_PATH = os.path.join(DATA_PATH, 'modeloLBPH.yml')
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
FRONTEND_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', '..', 'Frontend'))

IMG_SIZE = (150, 150)  # igual que en entrenamiento.py
UMBRAL = 70            # LBPH: menor distancia = más parecido (igual que en Reconocimiento_Facial.py)

app = Flask(__name__, static_folder=FRONTEND_PATH, static_url_path='')

# ---------- Carga de modelo (una sola vez al arrancar) ----------
people = sorted(
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d)) and d != '__pycache__'
)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise SystemExit(f'No se pudo cargar el clasificador: {CASCADE_PATH}')

recognizer = None
if os.path.exists(MODEL_PATH):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    print('Modelo cargado. Personas:', people)
else:
    print('AVISO: no existe', MODEL_PATH, '- corre Data/entrenamiento.py')


def decode_image(data_url):
    """Convierte 'data:image/jpeg;base64,....' en una imagen de OpenCV."""
    encoded = data_url.split(',', 1)[-1]
    raw = base64.b64decode(encoded)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


@app.route('/')
def home():
    return redirect('/LoginFace.html')


@app.route('/api/login-face', methods=['POST'])
def login_face():
    if recognizer is None:
        return jsonify(ok=False, reason='model_missing'), 503

    body = request.get_json(silent=True) or {}
    if 'image' not in body:
        return jsonify(ok=False, reason='no_image'), 400

    try:
        frame = decode_image(body['image'])
    except Exception:
        frame = None
    if frame is None:
        return jsonify(ok=False, reason='bad_image'), 400

    # Mismo preprocesado que en Base_data.py / Reconocimiento_Facial.py
    frame = imutils.resize(frame, width=320)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return jsonify(ok=False, reason='no_face')

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # la cara más grande
    rostro = cv2.resize(gray[y:y + h, x:x + w], IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    label, distance = recognizer.predict(rostro)

    if distance < UMBRAL:
        return jsonify(ok=True, name=people[label], distance=round(float(distance), 1))
    return jsonify(ok=False, reason='unknown', distance=round(float(distance), 1))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)