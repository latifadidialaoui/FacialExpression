# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

# Importation de TensorFlow pour le framework de deep learning
import tensorflow as tf
print("Tensorflow version:", tf.__version__)

# Affichage du nombre d'images par expression
for expression in os.listdir(r"D:/Video_Analytics_Project/Video_Analytics_Project/train/"):
    print(str(len(os.listdir(r"D:/Video_Analytics_Project/Video_Analytics_Project/train/" + expression))) + " " + expression + " images")

image_size = 48
batch_size = 64

# Préparation des données d'entraînement
data_train = ImageDataGenerator(horizontal_flip=True)
train_generator = data_train.flow_from_directory(
    "D:/Video_Analytics_Project/Video_Analytics_Project/train/",
    target_size=(image_size, image_size), 
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Préparation des données de validation
data_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = data_validation.flow_from_directory(
    "D:/Video_Analytics_Project/Video_Analytics_Project/test/", 
    target_size=(image_size, image_size),  # Redimensionnement des images à 48x48.
    color_mode="grayscale",  # Conversion en niveaux de gris.
    batch_size=batch_size,  # Taille des lots.
    class_mode='categorical',  # Classification catégorielle pour les émotions.
    shuffle=False  # Pas de mélange des données pour la validation.
)

# Définition du modèle  CNN 
model = Sequential()

# Ajoutez la couche d'entrée
model.add(Input(shape=(48, 48, 1)))  # Utilisation de Input ici

# Couches Convolutionnelles et de Normalisation
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))  # Couche de sortie avec 7 neurones pour les 7 classes d'émotions.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entraînement du modèle
model.fit(x=train_generator, epochs=64, validation_data=validation_generator)

# Évaluation du modèle
scores = model.evaluate(validation_generator)
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])

# Sauvegarde du modèle
model_json = model.to_json()
model.save_weights('model_weights.weights.h5')  # Sauvegarde les poids du modèle dans un fichier.
with open("model.json", "w") as file: # Ouvre un fichier pour écrire le modèle JSON.
    file.write(model_json)

# Classe pour la prédiction des émotions
class FacialExpression(object):
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    
    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as file:  # Ouvre le fichier JSON pour lire le modèl
            loaded_model_json = file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
  
    def predict_emotion(self, img): # Méthode pour prédire l'émotion à partir d'une image.
        self.pred = self.loaded_model.predict(img)
        return FacialExpression.EMOTIONS[np.argmax(self.pred)]

# Détection et prédiction en temps réel
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = FacialExpression("model.json", "model_weights.weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open the webcam")

while True: # Boucle infinie pour traiter les images de la webcam
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        fc = gray[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(frame, pred, (x, y), font, 3, (255, 0, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Facial Expression', frame) # Affiche l'image traitée dans une fenêtre
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()