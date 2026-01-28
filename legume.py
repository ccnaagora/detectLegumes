import cv2
import numpy as np


class Legume:
    def __init__(self  , box ,lab=None ):
        self.label=lab
        self.box=box       #[0,0,0,0]

    def affiche(self):
        print('lab={}\tcentre={}\tBoxe={}'.format(self.label,self.centre, self.box))

    def setContour(self , contour):
        self.contour = contour

    def setBox(self , box):
        self.box = box



    def getVecteurDirection(self , contour):
        # 3. Calculer le centroïde
        M = cv2.moments(contour)
        ''' m00 : aire du contour
        m10 : moment selon l'axe X = somme des pixels*x
        m01 : moment selon l'axe Y
        '''
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # 4. Préparer les points du contour pour PCACompute
            points = contour[:, 0, :]  # Récupérer les points (format initial : int)
            points = points.astype(np.float32)  # Convertir en float32
            # 5. Calculer la PCA
            mean, eigenvectors = cv2.PCACompute(points, mean=None)
            # 6. Calculer l'angle d'orientation
            angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])  # Angle en radians
            angle_deg = np.degrees(angle)  # Convertir en degrés
            # 7. Dessiner l'axe principal
            axis_length = 50
            x2 = int(cX + axis_length * eigenvectors[0, 0])
            y2 = int(cY + axis_length * eigenvectors[0, 1])
            # Dessiner le contour, le centroïde et l'axe
            # cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            # cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)  # Centroïde en rouge
            # cv2.line(image, (cX, cY), (x2, y2), (255, 0, 0), 2)  # Axe principal en bleu
            return (cX, cY, x2, y2)

    def getContour(self , image , iter=5 , delta=0):
        x1, y1, x2, y2 = map(int, self.box.xyxy[0])
        rect = (max(0, x1 - delta), max(0, y1 - delta), x2 - x1 + delta, y2 - y1 + delta)
        # Initialiser le masque
        mask = np.zeros(image.shape[:2], np.uint8)  # les deux premiers : x et y et ignore le canal de couleur ( = 640,380,3 par exemple)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        # Appliquer GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iter, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # pas utilie
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Élimine les petits objets
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Ferme les petits trous
        # Trouver les contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # meilleur compromis
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour
        return None

    def setContour2(self , image , seuils):
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