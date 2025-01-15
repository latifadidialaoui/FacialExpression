# -*- coding: utf-8 -*-
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

# Classe pour la prédiction des émotions
class FacialExpression(object):
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    
    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as file:
            loaded_model_json = file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
   
    def predict_emotion(self, img):
        self.pred = self.loaded_model.predict(img)
        return FacialExpression.EMOTIONS[np.argmax(self.pred)]

# Détection et prédiction en temps réel
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = FacialExpression("model.json", "model_weights.weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open the webcam")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(frame, pred, (x, y), font, 3, (255, 0, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Facial Expression', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

