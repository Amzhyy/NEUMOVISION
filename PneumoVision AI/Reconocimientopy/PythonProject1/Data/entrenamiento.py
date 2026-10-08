from cProfile import label
import cv2
import os
import numpy as np

dataPath = r'C:\Users\amzhy\Documents\PythonProject1\Data'

peopleList = [
    name for name in os.listdir(dataPath)
    if os.path.isdir(os.path.join(dataPath, name))
    and name != '__pycache__'
]

print('Lista de Personas:', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:

    personPath = os.path.join(dataPath, nameDir)

    print('Leyendo imágenes de:', nameDir)

    for filename in os.listdir(personPath):

        # Solo procesar imágenes
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        imagePath = os.path.join(personPath, filename)

        image = cv2.imread(imagePath, 0)

        # Verificar que la imagen se haya cargado correctamente
        if image is None:
            print('No se pudo leer:', imagePath)
            continue

        print('Rostro:', nameDir + '/' + filename)

        facesData.append(image)
        labels.append(label)

        cv2.imshow('image', image)
        cv2.waitKey(10)

    label += 1

cv2.destroyAllWindows()

print('Número de imágenes:', len(facesData))
print('Número de labels:', len(labels))

# Entrenamiento
print('Training...')

face_recognizer = cv2.face.EigenFaceRecognizer_create()

face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloEigenFace.xml')

print('Modelo entrenado y guardado correctamente.')