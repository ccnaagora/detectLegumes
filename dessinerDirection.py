import cv2
import numpy as np

from fonction import dessinerDirection

seuilsCarotte = [[5, 100, 100], [15, 255, 255]]
# patate
seuilsPatate = [[10, 30, 30], [30, 200, 200]]
seuilsVert = [[40 , 50 , 50] , [80 , 255 , 255]]
seuilsRougeH = [[160 , 50 , 50] , [180 , 255 , 255]]
seuilsRougeB = [[0,50,50]  , [10,255,255]]

# 1. Charger l'image et détecter le contour de l'objet
image = cv2.imread("0rouge1.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_orange = np.array(seuilsCarotte)  # Teinte basse, saturation et valeur minimales
upper_orange = np.array(seuilsCarotte[1])  # Teinte haute, saturation et valeur maximales
dessinerDirection(image , seuilsRougeH)
# Afficher l'image
cv2.imshow("Orientation de l'objet", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

