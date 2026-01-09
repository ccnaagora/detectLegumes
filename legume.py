import cv2
import numpy as np


class legume:
    def __init__(self , lab=None , centre = None , box=None , contour= None):
        self.label=lab
        self.centre = centre
        self.box=box       #[0,0,0,0]
        self.contour=contour

    def affiche(self):
        print('lab={}\tcentre={}\tBoxe={}'.format(self.label,self.centre, self.box))

    def setContour(self , image , seuils):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(seuils[0])  # Teinte basse, saturation et valeur minimales
        upper = np.array(seuils[1])  # Teinte haute, saturation et valeur maximales
        mask = cv2.inRange(hsv, lower, upper)
        # Appliquer le masque à l'image originale
        result = cv2.bitwise_and(image, image, mask=mask)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            self.contour = max(contours, key=cv2.contourArea)
        else :
            self.contour = None

    def afficheContour(self):
        if self.contour != None:
            print('nb points: {}'.format(len(self.contour)) )
            print('points : {}'.format(self.setContour()))