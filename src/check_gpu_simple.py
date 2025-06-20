import torch

def main():
    print("=== Información del Sistema ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("\n=== Información de la GPU ===")
        device = torch.device("cuda")
        print(f"Dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"Capacidad de cómputo: {torch.cuda.get_device_capability(0)}")
        
        # Memoria total y usada
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # en GB
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free = total_mem - allocated
        
        print(f"\nMemoria total de la GPU: {total_mem:.2f} GB")
        print(f"Memoria reservada: {reserved:.2f} GB")
        print(f"Memoria en uso: {allocated:.2f} GB")
        print(f"Memoria disponible: {free:.2f} GB")
        
        # Probar con un tensor grande
        try:
            print("\nProbando memoria con un tensor grande...")
            # Intentar asignar ~1GB de memoria
            x = torch.randn((250, 1000, 1000), device='cuda')  # ~1GB
            print("¡Se pudo asignar 1GB en la GPU!")
            
            # Intentar asignar más memoria
            try:
                y = torch.randn((500, 1000, 1000), device='cuda')  # ~2GB
                print("¡Se pudo asignar 2GB adicionales en la GPU!")
                del y
            except Exception as e:
                print(f"No se pudo asignar memoria adicional: {str(e)}")
            
            del x
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error al asignar memoria en la GPU: {str(e)}")
    else:
        print("No se detectó una GPU compatible con CUDA.")

if __name__ == "__main__":
    main()
