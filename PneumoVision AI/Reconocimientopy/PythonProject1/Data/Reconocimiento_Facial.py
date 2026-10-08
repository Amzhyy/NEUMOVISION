import cv2
import os

dataPath = r'C:\Users\amzhy\Documents\PythonProject1\Data'
imagePaths = [d for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, d)) and d != '__pycache__']
print('imagePath =',imagePaths)

# Tamaño real de las fotos con las que se entrenó
primera = None
for f in os.listdir(os.path.join(dataPath, imagePaths[0])):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        primera = cv2.imread(os.path.join(dataPath, imagePaths[0], f), 0)
        break
tam = (primera.shape[1], primera.shape[0])
print('Tamaño de entrenamiento =', tam)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

face_recognizer.read (r'C:\Users\amzhy\Documents\PythonProject1\Data\modeloEigenFace.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(r'C:\Users\amzhy\Documents\PythonProject1\haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,tam, interpolation = cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}' .format(result), (x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        if result [1] < 5700:
            cv2.putText(frame, '{}'. format(imagePaths[result[0]]), (x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame, 'Desconocido', (x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()