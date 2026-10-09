import os

import cv2
import imutils

DATA_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DATA_PATH, 'modeloLBPH.yml')
UMBRAL = 70  # LBPH: menor = más parecido. Ajusta según lo que veas
IMG_SIZE = (150, 150)  # debe ser igual en entrenamiento.py

imagePaths = sorted(
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d)) and d != '__pycache__'
)
print('Personas:', imagePaths)

if not os.path.exists(MODEL_PATH):
    raise SystemExit('No existe el modelo. Corre entrenamiento.py primero.')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_PATH)

CASCADE_PATH = os.path.join(DATA_PATH, '..', 'haarcascade_frontalface_default.xml')
faceClassif = cv2.CascadeClassifier(CASCADE_PATH)
if faceClassif.empty():
    raise SystemExit(f'No se pudo cargar el clasificador: {CASCADE_PATH}')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=320)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = cv2.resize(gray[y:y + h, x:x + w], IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        # Distancia (menor = más parecido)
        cv2.putText(frame, '{:.0f}'.format(result[1]), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < UMBRAL:
            cv2.putText(frame, imagePaths[result[0]], (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break
    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()