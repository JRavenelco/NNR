import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from picamera2 import Picamera2

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
    picam2 = None # Definir picam2 aquí para que esté en el scope del finally

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

    # Iniciar captura de video con Picamera2
    try:
        picam2 = Picamera2()
        # Configurar la cámara. Queremos un array RGB.
        # El modelo espera 32x32, pero la transformación ya se encarga de eso.
        # Capturamos a una resolución mayor para mejor calidad antes del reescalado.
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        print("Cámara Picamera2 iniciada. Presiona 'q' para salir.")
    except Exception as e:
        print(f"Error al iniciar Picamera2: {e}")
        if picam2 and hasattr(picam2, 'started') and picam2.started:
             picam2.stop()
        return

    try:
        while True:
            # Capturar frame por frame
            frame_rgb = picam2.capture_array() # Esto ya es un array NumPy RGB
            if frame_rgb is None:
                print("Error: No se pudo leer el frame de la cámara Picamera2.")
                break

            # Preprocesar el frame
            # 'frame_rgb' ya está en RGB. La transformación espera PIL.
            input_tensor = transform_cam(frame_rgb)
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
            # cv2.putText opera sobre el array (frame_rgb). Los colores son (B,G,R) para cv2.putText
            # pero como nuestro frame es RGB, (0,255,0) será verde.
            label_text = f"{predicted_class} ({confidence_score*100:.1f}%)"
            # Es buena práctica trabajar sobre una copia para visualización.
            output_display_frame_rgb = frame_rgb.copy()
            cv2.putText(output_display_frame_rgb, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA) # Verde en RGB
            
            # Convertir frame (RGB con texto) a BGR para cv2.imshow
            output_display_frame_bgr = cv2.cvtColor(output_display_frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Mostrar el frame resultante
            cv2.imshow('Reconocimiento de Objetos CIFAR-100', output_display_frame_bgr)

            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if picam2 and hasattr(picam2, 'started') and picam2.started:
            picam2.stop()
            print("Cámara Picamera2 detenida.")
        cv2.destroyAllWindows()
        print("Ventanas de OpenCV cerradas.")

if __name__ == '__main__':
    predict_from_camera()
