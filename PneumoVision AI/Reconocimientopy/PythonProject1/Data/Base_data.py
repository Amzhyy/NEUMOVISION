import os
import sys

import cv2
import imutils

# Carpeta donde vive este script (Data/). Ahí se guardan las fotos de cada persona.
DATA_PATH = os.path.dirname(os.path.abspath(__file__))

# Uso: python Base_data.py NombreDeLaPersona
personName = sys.argv[1] if len(sys.argv) > 1 else input('Nombre de la persona: ').strip()
personPath = os.path.join(DATA_PATH, personName)

if not os.path.exists(personPath):
    print('Carpeta creada:', personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
CASCADE_PATH = os.path.join(DATA_PATH, '..', 'haarcascade_frontalface_default.xml')
faceClassif = cv2.CascadeClassifier(CASCADE_PATH)
if faceClassif.empty():
    raise SystemExit(f'No se pudo cargar el clasificador: {CASCADE_PATH}')
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=320)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(personPath, 'rostro_{}.jpg'.format(count)), rostro)
        count += 1
    cv2.imshow('frame', frame)

    # ESC para salir, o al llegar a 300 fotos
    if cv2.waitKey(1) == 27 or count >= 300:
        break

# Ahora sí se liberan la cámara y las ventanas (antes quedaban después de un break y nunca corrían)
cap.release()
cv2.destroyAllWindows()
print('Fotos guardadas:', count)