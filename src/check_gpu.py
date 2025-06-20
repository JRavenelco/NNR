import torch
import torch.nn as nn
import torch.optim as optim
from vit_model import vit_small_patch4_32
import time

def check_gpu_memory():
    if not torch.cuda.is_available():
        print("CUDA no está disponible")
        return
    
    # Obtener información de la GPU
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # en GB
    reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # en GB
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # en GB
    free_memory = total_memory - allocated_memory
    
    print(f"\n=== Información de la GPU ===")
    print(f"Dispositivo: {gpu_name}")
    print(f"Memoria total: {total_memory:.2f} GB")
    print(f"Memoria reservada: {reserved_memory:.2f} GB")
    print(f"Memoria en uso: {allocated_memory:.2f} GB")
    print(f"Memoria disponible: {free_memory:.2f} GB")
    
    # Probar con un modelo pequeño
    print("\nProbando con un modelo pequeño...")
    try:
        model = vit_small_patch4_32(num_classes=100).cuda()
        x = torch.randn(64, 3, 32, 32).cuda()  # batch_size=64
        
        # Calcular memoria usada por el modelo
        model_memory = sum(p.numel() * 4 / (1024**2) for p in model.parameters())  # MB
        print(f"\nMemoria usada por el modelo: {model_memory:.2f} MB")
        
        # Realizar forward y backward
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Verificar memoria después de forward+backward
        allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"Memoria usada después de forward+backward: {allocated_after:.2f} GB")
        
        # Intentar con batch size más grande
        print("\nProbando con batch size más grande...")
        try:
            x_large = torch.randn(128, 3, 32, 32).cuda()  # batch_size=128
            output = model(x_large)
            print("¡Batch size de 128 funciona!")
        except RuntimeError as e:
            print(f"Error con batch_size=128: {str(e)}")
            
    except Exception as e:
        print(f"Error durante la prueba: {str(e)}")
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    check_gpu_memory()
