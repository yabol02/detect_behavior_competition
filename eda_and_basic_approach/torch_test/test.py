import torch
print(f"¿CUDA disponible?: {torch.cuda.is_available()}")
print(f"Dispositivo: {torch.cuda.get_device_name(0)}")