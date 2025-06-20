import torch

try:
    # Información básica
    print("=== Información de la GPU ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # Información del dispositivo
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
        # Memoria total
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"\nMemoria total de la GPU: {total_mem:.2f} GB")
        
        # Memoria usada/disponible
        print(f"Memoria reservada: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"Memoria en uso: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        # Probar asignación de memoria
        print("\nProbando asignación de memoria...")
        try:
            # Intentar asignar 1.5GB
            x = torch.zeros((400, 1000, 1000), device='cuda')
            print("¡Se asignaron 1.5GB correctamente!")
            print(f"Memoria después de asignar 1.5GB: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            
            # Liberar memoria
            del x
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error al asignar memoria: {str(e)}")
        
        # Capacidad de cómputo
        print(f"\nCapacidad de cómputo: {torch.cuda.get_device_capability(0)}")
        
        # Versión CUDA
        if hasattr(torch.version, 'cuda'):
            print(f"Versión CUDA: {torch.version.cuda}")
        
        # Recomendación para batch size
        if total_mem >= 10:  # 10GB o más
            print("\nRecomendación: Puedes usar batch_size=64 o superior")
        elif total_mem >= 6:  # 6-10GB
            print("\nRecomendación: Usa batch_size=32 para empezar")
        else:  # Menos de 6GB
            print("\nRecomendación: Usa batch_size=16 o menos")
            
except Exception as e:
    print(f"Error: {str(e)}")
