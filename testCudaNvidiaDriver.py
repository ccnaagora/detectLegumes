import torch
import sys

print(f"--- Diagnostic Système ---")
print(f"Version de Python : {sys.version}")
print(f"Version de PyTorch : {torch.__version__}")

print(f"\n--- Vérification GPU ---")
cuda_disponible = torch.cuda.is_available()
print(f"CUDA disponible (NVIDIA) : {cuda_disponible}")

if cuda_disponible:
    print(f"Nombre de GPU détectés : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} : {torch.cuda.get_device_name(i)}")
    print(f"GPU actuel par défaut : {torch.cuda.current_device()}")
else:
    print("ERREUR : PyTorch ne voit pas votre carte NVIDIA.")
    if "cpu" in torch.__version__:
        print("CAUSE PROBABLE : Vous avez installé la version 'CPU' de PyTorch au lieu de la version 'CUDA'.")
    else:
        print("CAUSE PROBABLE : Pilotes NVIDIA absents ou Toolkit CUDA non installé.")