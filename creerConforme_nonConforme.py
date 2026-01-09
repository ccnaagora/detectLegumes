import datetime

#from tensorflow.keras.applications import MobileNetV2
#from tensorflow.keras import layers, models
import numpy as np


#resize image en 242*224
import cv2
import numpy as np

import commun


def resize(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    target_h, target_w = target_size
    # Calcule le ratio pour conserver les proportions
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(image, target_size)
    return resized

def resize_with_pad(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calcule le ratio pour conserver les proportions
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)

    # Redimensionne
    resized = cv2.resize(image, (new_w, new_h))

    # Ajoute du padding pour atteindre target_size
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    padded = np.pad(resized, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    return padded
################################################################
'''
#creation d'un modele sequentiel à partir d'un modele pre-entrainé
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Gèle les couches pré-entraînées

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')  # Binaire : conforme/non-conforme
])

#compile le modele
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#charger et pre-traiter les images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.4)    #0.2 défini le nb d'images pour les tests

train_generator = train_datagen.flow_from_directory(
    'piece/',
    target_size=(224, 224),
    batch_size=32,          #taille bloc d'images pour l'entrainement
    class_mode='binary',
    subset='training'
)
print(train_generator.class_indices)
val_generator = train_datagen.flow_from_directory(
    'piece/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
#entrainer le modele
history = model.fit(
    train_generator,
    epochs=2,
    validation_data=val_generator
)
'''
import matplotlib.pyplot as plt

'''
#affiche quelques images conforme/non conforme à partir du modèle entrainé
# Prends un batch d'images et de labels depuis le générateur d'entraînement
images, labels = next(train_generator)
# Affiche 8 images avec leur label
plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i])
    # Récupère le nom de la classe à partir du label numérique
    label = "conforme" if labels[i] == 0 else "non_conforme"
    plt.title(f"Label: {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()
'''

#traitement
import cv2
import scipy
#pour dcs 935l
#cap = cv2.VideoCapture(commun.url935) # 0 pour la caméra par défaut
#pour dcs 5020l
cap = cv2.VideoCapture(commun.url5020)

cv2.namedWindow('Classification en temps réel', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #resize avec padding : fonctionne moins bien
    #frame = resize_with_pad(frame, target_size=(224, 224))
    # Prétraiter l'image (redimensionner, normaliser)
    #img = cv2.resize(frame, (224, 224))
    #img = img / 255.0


    # Afficher le résultat
    #cv2.resizeWindow('Classification en temps réel', 640, 480)
    cv2.imshow('Classification en temps réel', frame)
    key = cv2.waitKey(5) & 0xFF
    type_image = 'vert'
    if key == ord('q'):
        break
    if key == ord('t'):
        # Sauvegarde l'image dans le bon dossier
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img/train_{type_image}{timestamp}.jpg"
        enr = resize(frame, target_size=(224, 224))
        cv2.imwrite(filename, enr)
    if key == ord('v'):
        # Sauvegarde l'image dans le bon dossier
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img/val_{type_image}{timestamp}.jpg"
        enr = resize(frame, target_size=(224, 224))
        cv2.imwrite(filename, enr)

cap.release()
cv2.destroyAllWindows()


