import datetime
import time
from threading import Thread

import numpy as np


#resize image en 242*224
import cv2
import numpy as np

import commun




import matplotlib.pyplot as plt

import cv2
import scipy
def perdreTemps(duree , a):
    time.sleep(duree)

#pour dcs 935l
#cap = cv2.VideoCapture(commun.url935) # 0 pour la caméra par défaut
#pour dcs 5020l
cap = cv2.VideoCapture(commun.url935 )
#desactive le buffer
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Désactive le buffer
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 1)



while True:
    ret, frame = cap.read()
    if not ret:
        print('pas image')
        break
    t = Thread(target=perdreTemps , args=(0.002 , 0))
    t.start()
    t.join()
    #time.sleep(0.050)
    cv2.imshow('Classification en temps réel', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


