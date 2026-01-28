from enum import Enum
from multiprocessing.dummy import Process

import torch
import numpy as np
import time
from threading import Thread
from ultralytics import YOLO
import cv2
from legume import *
import commun
###############################CLASS ENUM pour associer label et index
class label(Enum):
    patate=0
    carotte=1
###############fonction lancée dans un thread : traite en // chaque box detectée
def lancerTraitementBox(legume , frame):
    #start=time.perf_counter()
    contour = legume.getContour(frame , iter = 5)   #modifier iter si resultat incorrect
    if contour is not None:
        x1, y1, x2, y2 = legume.getVecteurDirection(contour)
        cv2.line(frame , (x1 , y1), (x2 , y2), (0,0,255), 2)
    old=time.perf_counter()
    #print(f'duree d\'un thread {old-start:.3f}')
################################################################################
#recherche de cuda pour activer le gpu0 de nvidia
# Vérifier si CUDA (NVIDIA) est bien disponible
if torch.cuda.is_available():
    print(f"GPU NVIDIA détecté : {torch.cuda.get_device_name(0)}")
    my_device = 'cuda'  # 0 est généralement l'index du GPU NVIDIA
else:
    print("GPU NVIDIA non détecté, utilisation du CPU")
    my_device = 'cpu'
# Charger le modèle entraîné
model = YOLO("runs-legumes/detect/train/weights/best.pt")  # Remplacez par le chemin vers votre modèle entraîné
model.to(my_device)

# Ouvrir la caméra (0 pour la webcam par défaut)
cap = cv2.VideoCapture(commun.url935_2  )           #, cv2.CAP_FFMPEG
#cap = cv2.VideoCapture(commun.url5020)
#desactive le buffer
#les cap.set ne fonctione pas avec certaines caméras ( dlink)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Désactive le buffer
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)
#cap.set(0 , cv2.CAP_DSHOW)
#old=0
occupe = False
while True:
    if occupe == False:
        # Lire une frame depuis la caméra
        ret, frame = cap.read()
        if not ret:
            break
        '''start= time.perf_counter()
        print(f'Intervale entre 2 captures {old-start:.3f}')
        old=start'''
        results = model(frame, imgsz=640 , conf=0.7 , verbose=False )  # Seuil de confiance à 0.5
        processus = []
        for result in results:
            print('nb box={}'.format(len(result.boxes)))
            for box in result.boxes:
                legume = Legume(box)
                '''print('label:{}'.format(box.cls[0]))
                c = int(box.cls[0])
                cc = label(c)
                print(cc.name)'''
                p = Process(target = lancerTraitementBox , args=[legume,frame])
                processus.append(p)
                p.start()
        for pro in processus:
            pro.join()
        cv2.imshow("test", frame)
    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('o'):
        print('occupe')
        occupe = True
    if cv2.waitKey(1) & 0xFF == ord('l'):
        print('libereoo')
        occupe = False
# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()