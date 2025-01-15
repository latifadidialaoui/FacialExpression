# Détection des Émotions avec un Modèle CNN

## Introduction
Ce projet utilise un réseau de neurones convolutifs (CNN) pour détecter les émotions humaines à partir d'images faciales.
Le modèle est entraîné sur un ensemble de données d'expressions faciales et prédit l'émotion en temps réel via une webcam.

## Fonctionnalités
- Détection des émotions à partir d'images faciales en temps réel.
- Prédictions pour 7 émotions différentes : Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
- Interface simple pour visualiser les résultats.

## Technologies Utilisées
- **Python** : Langage de programmation principal.
- **TensorFlow/Keras** : Framework de deep learning pour la construction et l'entraînement du modèle CNN.
- **OpenCV** : Bibliothèque pour le traitement d'images et d vidéos.
- **NumPy** : Pour la manipulation de tableaux numériques.

## Base de Données
Ce projet utilise une base de données d'images d'expressions faciales, qui est une collection d'images étiquetées pour différentes émotions. La base de données est structurée comme suit :

- **Dossier `train/`** : Contient les images d'entraînement, organisées par sous-dossiers. Chaque sous-dossier représente une classe d'émotion, par exemple :
train/
├── Angry/
├── Disgust/
├── Fear/
├── Happy/
├── Neutral/
├── Sad/
└── Surprise/

- **Dossier `test/`** : Contient les images de validation, également organisées par classes d'émotion, permettant d'évaluer la performance du modèle sur des données non vues pendant l'entraînement.
test/
├── Angry/
├── Disgust/
├── Fear/
├── Happy/
├── Neutral/
├── Sad/
└── Surprise/
Cette structure facilite le chargement des données et l'entraînement du modèle en utilisant des générateurs d'images de Keras.

## Installation
1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/latifadidialaoui/FacialExpression.git
   cd FacialExpression
2.Exécutez le script principal   
   python Expression faciale.py

## Qu'est-ce qu'un CNN et Pourquoi l'A-t-on Utilisé ?

### Qu'est-ce qu'un CNN ?

Un **réseau de neurones convolutifs (CNN)** est un type de modèle de deep learning particulièrement efficace pour traiter des données ayant une structure en grille, comme les images. 
Les CNN sont conçus pour reconnaître des motifs et des caractéristiques dans les données d'entrée, ce qui les rend idéaux pour des tâches telles que la classification d'images, la détection d'objets, et bien d'autres applications de vision par ordinateur.

### Fonctionnement d'un CNN

1. **Couches Convolutionnelles** :
   - Les CNN utilisent des couches de convolution pour extraire des caractéristiques des images. Chaque filtre (ou noyau) convolue sur l'image d'entrée pour détecter des motifs locaux, tels que des contours, des textures, ou des formes.
   - Ces couches permettent d'apprendre des représentations hiérarchiques, où les premières couches détectent des caractéristiques simples, et les couches plus profondes détectent des caractéristiques plus complexes.

2. **Couches de Normalisation et Activation** :
   - Après chaque couche de convolution, une couche de normalisation (comme Batch Normalization) est souvent ajoutée pour stabiliser l'apprentissage, suivie d'une fonction d'activation (comme ReLU) pour introduire de la non-linéarité.

3. **Couches de Max Pooling** :
   - Les couches de pooling réduisent la dimensionnalité des données tout en conservant les caractéristiques importantes, ce qui aide à réduire le temps de calcul et à éviter le surapprentissage.

4. **Couches Denses** :
   - Les caractéristiques extraites par les couches de convolution sont ensuite aplanies et passées à travers des couches denses, qui effectuent la classification finale. La dernière couche utilise généralement une fonction d'activation softmax pour fournir des probabilités pour chaque classe d'émotion.

### Pourquoi Utiliser un CNN pour la Détection des Émotions ?

- **Efficacité dans l'Extraction de Caractéristiques** : Les CNN sont capables d'extraire automatiquement des caractéristiques pertinentes des images sans nécessiter une ingénierie de caractéristiques manuelle, ce qui facilite le processus de développement.

- **Robustesse** : Les CNN sont robustes face à des variations dans les données d'entrée, comme les changements d'éclairage, d'angle, et d'expression faciale, ce qui est essentiel pour la détection des émotions dans des scénarios réels.

- **Performance** : Grâce à leur capacité à apprendre des représentations hiérarchiques, les CNN offrent généralement des performances supérieures par rapport à d'autres méthodes de classification d'images, ce qui les rend idéaux pour des applications de vision par ordinateur complexes, telles que la reconnaissance d'émotions.

## Exemple de Code
Voici un extrait de code montrant comment le modèle est défini et entraîné :

### Définition du modèle CNN
model = Sequential()
model.add(Input(shape=(48, 48, 1)))

### Ajout des couches : 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_generator, epochs=64, validation_data=validation_generator)

En résumé, l'utilisation d'un CNN pour ce projet permet de tirer parti de la puissance des réseaux de neurones profonds pour effectuer une tâche complexe de classification d'images de manière efficace et précise.

##  Détection d'Émotions Faciales en Temps Réel

### Exécution de l'Application

Le script principal utilise OpenCV et un modèle pré-entraîné pour détecter les émotions faciales en temps réel à partir d'une caméra.

## Fonctionnement du Script
   ### Chargement du Modèle :
Le modèle est chargé à partir d'un fichier JSON (model.json), qui contient l'architecture du modèle CNN.
Les poids du modèle, qui ont été sauvegardés dans un fichier H5 (model_weights.weights.h5), sont également chargés.
Ces poids représentent les paramètres appris par le modèle au cours de l'entraînement.

   ### Détection Faciale :
Le script utilise un classificateur en cascade de Haar pour détecter les visages dans le flux vidéo de la webcam.
Une fois un visage détecté, il est redimensionné à 48x48 pixels pour correspondre à l'entrée du modèle.

   ###  Prédiction :
L'image du visage est passée au modèle pour prédire l'émotion, qui est ensuite affichée en temps réel sur la vidéo de la webcam.
Les émotions sont affichées sur le cadre entourant le visage détecté, et un rectangle est dessiné autour de celui-ci pour indiquer la zone d'intérêt.

   ### Interface Utilisateur :
Une fenêtre s'ouvre pour afficher la vidéo de la webcam avec les émotions détectées.
Appuyez sur q pour quitter l'application.

### Fichiers du Modèle
Fichier JSON (model.json) :
Ce fichier contient l'architecture du modèle CNN sous forme JSON. Il décrit la disposition des couches, les types de couches, et les configurations nécessaires pour reconstruire le modèle.
Fichier H5 (model_weights.weights.h5) :
Ce fichier stocke les poids du modèle, c'est-à-dire les paramètres qui ont été appris lors de l'entraînement.
Les poids sont essentiels pour que le modèle fasse des prédictions précises, car ils déterminent comment chaque neurone contribue à l'output final basé sur l'entrée.
Ces fichiers sont cruciaux pour la reconstruction et l'utilisation du modèle une fois qu'il a été entraîné, permettant ainsi une utilisation facile et rapide pour les prédictions en temps réel.

## Conclusion
Ce script offre une solution simple et efficace pour la détection des émotions en temps réel, en utilisant un modèle de deep learning entraîné sur un ensemble de données d'expressions faciales.

## Démonstration Vidéo

Pour voir le projet en action, vous pouvez visionner la démonstration vidéo ci-dessous. Cette vidéo montre comment l'application détecte les émotions en temps réel à partir des expressions faciales capturées par la webcam.

Regarder la Démonstration Vidéo

https://github.com/user-attachments/assets/77ad2ed6-8511-4804-be31-3beca84f7a6f


## Ce que vous verrez dans la vidéo :
- L'interface utilisateur en temps réel avec la vidéo de la webcam.
- La détection des visages et l'affichage des émotions prédites sur l'écran.
- Un aperçu du fonctionnement de l'algorithme, y compris comment les émotions sont reconnues et affichées.

N'hésitez pas à poser des questions ou à laisser des commentaires sur la vidéo !


