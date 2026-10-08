import os

import cv2
import numpy as np

DATA_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (150, 150)  # debe ser igual en Reconocimiento_Facial.py

peopleList = sorted(
    name for name in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, name)) and name != '__pycache__'
)
print('Lista de personas:', peopleList)

labels = []
facesData = []

for label, nameDir in enumerate(peopleList):
    personPath = os.path.join(DATA_PATH, nameDir)
    print('Leyendo imágenes de:', nameDir)

    for filename in os.listdir(personPath):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image = cv2.imread(os.path.join(personPath, filename), 0)
        if image is None:
            print('No se pudo leer:', filename)
            continue
        image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
        facesData.append(image)
        labels.append(label)

print('Número de imágenes:', len(facesData))
if not facesData:
    raise SystemExit('No hay imágenes. Corre Base_data.py primero.')

print('Entrenando...')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(facesData, np.array(labels))

modelPath = os.path.join(DATA_PATH, 'modeloLBPH.yml')
face_recognizer.write(modelPath)
print('Modelo guardado en:', modelPath)