from ultralytics import YOLO
import torch

# Vérifier si CUDA (NVIDIA) est bien disponible
if torch.cuda.is_available():
    print(f"GPU NVIDIA détecté : {torch.cuda.get_device_name(0)}")
    my_device = 1  # 0 est généralement l'index du GPU NVIDIA
else:
    print("GPU NVIDIA non détecté, utilisation du CPU")
    my_device = 'cpu'

# Charger le modèle sur le GPU spécifié
#model = YOLO("yolov8n.pt")
model = YOLO("runs/detect/train/weights/best.pt")
#model.to(my_device)
#print(model.info())

# Lancer la détection
