from ultralytics import YOLO
import cv2
import numpy as np
import time
from threading import Thread

import commun


# Charger un modèle pré-entraîné
#penser à modifier le fichier yolo3.cfg:
#   classes = nb classes
#   filter = nbclasses*5 + 3
#   se mefier des num de classes dans les labels : doivent commencer à 0

#yolov8n doit être dans le reptertoire du projet.
#il existe une version plus précise de yolo qui nécessite une labelisation plus précise et plus longue des images train et val
model = YOLO("yolov8n.pt")

# Entraîner le modèle sur votre dataset
results = model.train(
    data="yolo.yaml",  # Fichier de configuration du dataset
    epochs=20,  # Nombre d'époques
    imgsz=640,   # Taille des images
    batch=16,    # Taille du batch
)

# Tester le modèle
results = model.val()
