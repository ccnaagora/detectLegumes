from ultralytics import YOLO
import cv2
import numpy as np
import time
from threading import Thread

import commun


#atténue les ombres
def adjust_gamma(image, gamma=1.5):
    # Construction d'une table de correspondance pour la correction gamma
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Application de la table de correspondance
    return cv2.LUT(image, table)
def dessinerPointCentre(results , image):
    for result in results:
        for box in result.boxes:
            # Extraire les coordonnées xyxy
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Calculer le centre de la boîte
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            # Dessiner un gros point rouge (rayon = 10 pixels)
            cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)  # -1 pour remplir le cercle
            # Optionnel : Dessiner la boîte englobante et le label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Rectangle vert
            label = f"{model.names[int(box.cls)]} {box.conf[0]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#retourne les centres des boxes
def returnCentre(results )  :
    centres = []
    i=0
    for result in results:
        for box in result.boxes:
            print('Nb results: {}\tNb box: {}'.format(str(len(results)) , str(len(result.boxes))))
            # Extraire les coordonnées xyxy
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Calculer le centre de la boîte
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            centres.append( [center_x , center_y])
            print('x={}'.format(centres))
            i=i+1
    return centres
#dessine les contours des boundingbox de objetst trouvés
#delta1 >0 reduit le rectangle
#deltab >0 augmente le rectangle
def getcontour(box , image , iter=5 , delta=0):
    #print('confiance: {}'.format(box.conf))
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    rect = (max(0, x1 - delta), max(0, y1 - delta), x2 - x1 + delta, y2 - y1 + delta)
    # cv2.rectangle(image, rect, (0, 255, 0), 2)  # Rectangle vert

    # Initialiser le masque
    mask = np.zeros(image.shape[:2],np.uint8)  # les deux premiers : x et y et ignore le canal de couleur ( = 640,380,3 par exemple)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Appliquer GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iter, cv2.GC_INIT_WITH_RECT)
    # cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)  # Rectangle vert
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    #pas utilie
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Élimine les petits objets
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Ferme les petits trous
    # Trouver les contours
    #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # meilleur compromis
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        #retourne les coordonnées du vecteur 'direction de l'allongement'
        x,y,z,w = getVecteurDirection(image, commun.seuilsRougeH, largest_contour)
        cv2.drawContours(image, largest_contour, -1, (0, 255, 0), 2)
        cv2.line(image, (x, y), (z, w), (255, 0, 0), 2)


    #model = YOLO("yolov8n.pt")

# Détecter des objets sur une nouvelle image
from ultralytics import YOLO
import cv2

# Charger le modèle entraîné
model = YOLO("runs/detect/train/weights/best.pt")  # Remplacez par le chemin vers votre modèle entraîné
paire=0

# Ouvrir la caméra (0 pour la webcam par défaut)
cap = cv2.VideoCapture(commun.url935 , cv2.CAP_FFMPEG )
#cap = cv2.VideoCapture(commun.url5020)
#desactive le buffer
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Désactive le buffer
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)
old=0
while True:
    # Lire une frame depuis la caméra
    start = time.perf_counter()
    print(f"Thread grabcut: {start - old:.3f}s")
    ret, frame = cap.read()
    old=start
    if not ret:
        break

    #augmente le contraste : alpha>1 et beta =10
    #frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=15)
    #diminue les ombres
    #frame = adjust_gamma(frame, gamma=1.5)
    results = model(frame, imgsz=640 , conf=0.1)  # Seuil de confiance à 0.5

    # Effectuer la détection
    from fonction import *
    #getBoxes(results)                  #affiche les valeurs des boxs
    # Afficher les résultats
    #annotated_frame = results[0].plot()
    # Afficher la frame annotée
    #cv2.imshow("Détection de carottes et pommes de terre", annotated_frame)
    threads = []
    for result in results:
        for box in result.boxes:
            #1 thread par box
            t = Thread(target = getcontour , args=(box,frame,2,0))
            threads.append(t)
            t.start()
            #msk = getcontour(results , frame)
            #dessinerPointCentre(results , frame)                       #ok mais box rectangulaire
            #getcontour(box , frame , iter=2 , delta=0 )                #contour incorrect et prend trop de temps
            #retourne le tableau de centre de gravité
            #c = returnCentre(results)
            #print(c)
            #print(f"deltat : {(fin-debut):.4f} secondes")
    print('nb thread={}'.format(len(threads)))
    for t in threads:
        t.join()
    cv2.imshow("test", frame)
    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()


