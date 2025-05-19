import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

from model import SimpleCNN # Asegúrate de que model.py esté en el mismo directorio o en PYTHONPATH
from data_loader import CIFAR100_CLASSES # Para obtener los nombres de las clases

# --- Parámetros ---
MODEL_PATH = './models_trained/cifar100_simplecnn.pth' # Ruta al modelo entrenado
NUM_CLASSES = 100 # CIFAR-100 tiene 100 clases
IMG_SIZE = 32 # Las imágenes de CIFAR-100 son 32x32

# Transformaciones para la imagen de la cámara (deben coincidir con las de entrenamiento/test)
# Usaremos las de test, ya que no hacemos data augmentation en la inferencia.
transform_cam = transforms.Compose([
    transforms.ToPILImage(), # Convertir frame de OpenCV (numpy array) a PIL Image
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

def predict_from_camera():
    print("Iniciando predicción desde la cámara...")

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar el modelo entrenado
    try:
        model = SimpleCNN(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() # Poner el modelo en modo evaluación
        print("Modelo cargado exitosamente.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en {MODEL_PATH}")
        print("Asegúrate de haber entrenado el modelo primero (ejecuta train.py).")
        return
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Obtener nombres de las clases (asegurarse de que data_loader.py tenga la lista)
    # Si CIFAR100_CLASSES está vacía en data_loader, esta lista será vacía aquí.
    # Deberíamos ejecutar get_cifar100_loaders una vez para poblarla o tenerla hardcodeada.
    class_names = CIFAR100_CLASSES
    if not class_names or len(class_names) != NUM_CLASSES:
        print(f"Advertencia: La lista de nombres de clases (CIFAR100_CLASSES) parece incorrecta o vacía.")
        print(f"Se esperaban {NUM_CLASSES} clases, pero se obtuvieron {len(class_names)}.")
        print("Usando índices numéricos para las clases.")
        class_names = [str(i) for i in range(NUM_CLASSES)]
    
    print(f"{len(class_names)} nombres de clases cargados.")

    # Iniciar captura de video
    cap = cv2.VideoCapture(0) # 0 es usualmente la cámara web por defecto

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Cámara abierta. Presiona 'q' para salir.")

    try:
        while True:
            # Capturar frame por frame
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame de la cámara.")
                break

            # Preprocesar el frame
            # OpenCV lee en BGR, PyTorch espera RGB. Convertir si es necesario.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform_cam(rgb_frame)
            input_batch = input_tensor.unsqueeze(0) # Crear un mini-lote de tamaño 1
            input_batch = input_batch.to(device)

            # Realizar la predicción
            with torch.no_grad():
                output = model(input_batch)
            
            # Obtener la clase predicha y la confianza (probabilidad)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            predicted_class = class_names[predicted_idx.item()]
            confidence_score = confidence.item()

            # Mostrar la predicción en el frame
            label_text = f"{predicted_class} ({confidence_score*100:.1f}%)"
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Mostrar el frame resultante
            cv2.imshow('Reconocimiento de Objetos CIFAR-100', frame)

            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Liberar la captura y cerrar ventanas
        cap.release()
        cv2.destroyAllWindows()
        print("Cámara cerrada y recursos liberados.")

if __name__ == '__main__':
    predict_from_camera()
