import cv2
import numpy as np
import matplotlib.pyplot as plt

import commun

def seuil(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Seuillage (binarisation)
    _, thresholded = cv2.threshold(image_gray, 160, 255, cv2.THRESH_BINARY)

    # Détection des contours
    #contours, _ = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contour_image = cv2.drawContours(image_gray.copy(), contours, -1, (255, 255, 255), 1)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blanche = np.ones_like(image_gray) * 255
    contour_image = cv2.drawContours(blanche, contours, -1, (0 ,0, 0), 1)

    # Calcul de l'histogramme
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Affichage des résultats
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title("Image originale")
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title("Image seuillée")
    plt.imshow(thresholded, cmap='gray', vmin=0, vmax=255)

    plt.subplot(2, 2, 3)
    plt.title("Contours")
    plt.imshow(contour_image, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title("Histogramme")
    plt.plot(hist)

    plt.tight_layout()
    plt.show()

#prog
# Ouvrir la caméra (0 pour la webcam par défaut)
cap = cv2.VideoCapture(commun.url935 , cv2.CAP_FFMPEG )
'''cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Désactive le buffer
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 10)'''
while True:
    ret, frame = cap.read()
    seuil(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()