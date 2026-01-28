import cv2
import numpy as np


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
def detect(results , image , seuils):

    # Convertir BGR en HSV (OpenCV utilise BGR au lieu de RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Définir une plage de couleurs pour l'orange (carottes)
    #
    lower_orange = np.array(seuils[0])   # Teinte basse, saturation et valeur minimales
    upper_orange = np.array(seuils[1])  # Teinte haute, saturation et valeur maximales

    # Créer un masque pour les pixels dans cette plage
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    # Appliquer le masque à l'image originale
    result = cv2.bitwise_and(image, image, mask=mask)
    #recherche des contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # Vérifie qu'au moins un contour a été trouvé
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest_contour], -1, (0, 0, 255), 3)  # En rouge, épaisseur = 3
        M = cv2.moments(largest_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # Dessiner le centroïde (cercle rouge)
        cv2.circle(result, (cX, cY), 3, (0, 0, 255), -1)  # -1 pour remplir le cercle
        '''for contour in contours:
            # Calculer les moments du contour
            M = cv2.moments(contour)
            # Vérifier que l'aire du contour est non nulle (éviter les erreurs)
            if M["m00"] != 0:
                # Calculer le centroïde (centre de gravité)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Dessiner le centroïde (cercle rouge)
                cv2.circle(result, (cX, cY), 5, (0, 0, 255), -1)  # -1 pour remplir le cercle
                # Optionnel : Dessiner le contour (vert)
            #    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                # Optionnel : Afficher les coordonnées du centroïde
                print(f"Centre de gravité : ({cX}, {cY})")
        # Dessiner tous les contours détectés (en vert, épaisseur = 2)
        #cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        # Optionnel : Dessiner uniquement le plus grand contour (supposé être la pomme de terre)
'''
    return result
# Afficher le résultat


#return les infos des box : confiance xyxy classe
def getBoxes(results):
    for result in results:
        for box in result.boxes:
            print('{}\t{}\t{}'.format(box.cls[0] , box.xyxy[0] , box.conf[0]))


if __name__ == "__main__":
    seuilsCarotte = [[5, 100, 100] , [15, 255, 255]]        #hsv: seuil bas et seuil haut
    seuilsPatate = [[10,30,30] , [30,200,200]]              #hsv seuil bas et seuil haut
    seuil = seuilsPatate
    image = cv2.imread("0rien.jpg")
    #image = cv2.imread("0carotte1.jpg")
    #image = cv2.imread("0carotte2.jpg")
    #image = cv2.imread("0melange.jpg")
    #image = cv2.imread("0patate1.jpg")
    im = detect(None , image , seuil)
    cv2.imshow("Masque orange", im)
    cv2.waitKey(0)