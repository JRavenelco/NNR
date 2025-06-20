import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import timm # Para cargar modelos ViT
import argparse
import os

from data_loader import CIFAR100_CLASSES # Para obtener los nombres de las clases

def predict_from_camera(args):
    print("Iniciando predicción desde la cámara con Vision Transformer...")

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Transformaciones para la imagen de la cámara (deben coincidir con las de entrenamiento/test del ViT)
    transform_cam = transforms.Compose([
        transforms.ToPILImage(), # Convertir frame de OpenCV (numpy array) a PIL Image
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Stats de ImageNet
    ])

    # Cargar el modelo entrenado
    try:
        # Crear el modelo ViT
        model = timm.create_model(
            args.model_name,
            pretrained=False, # No cargamos preentrenados de timm, cargaremos nuestro checkpoint
            num_classes=args.num_classes,
            img_size=args.img_size
        )
        
        # Cargar el checkpoint
        print(f"Cargando checkpoint desde: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # Cargar los pesos
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint: # A veces se guarda como 'state_dict'
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()  # Poner el modelo en modo evaluación
        print(f"Modelo ViT '{args.model_name}' cargado exitosamente.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en {args.checkpoint_path}")
        print("Asegúrate de que la ruta al checkpoint es correcta.")
        return
    except Exception as e:
        print(f"Error al cargar el modelo ViT: {e}")
        return

    # Obtener nombres de las clases
    class_names = CIFAR100_CLASSES
    if not class_names or len(class_names) != args.num_classes:
        print(f"Advertencia: La lista de nombres de clases (CIFAR100_CLASSES) parece incorrecta o vacía.")
        print(f"Se esperaban {args.num_classes} clases, pero se obtuvieron {len(class_names)}.")
        print("Usando índices numéricos para las clases.")
        class_names = [str(i) for i in range(args.num_classes)]
    
    print(f"{len(class_names)} nombres de clases cargados.")

    # Iniciar captura de video
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Usar DirectShow para mejor compatibilidad en Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Cámara abierta. Presiona 'q' para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame de la cámara.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform_cam(rgb_frame)
            input_batch = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_batch)
            
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            predicted_class_name = class_names[predicted_idx.item()]
            confidence_score = confidence.item()

            cv2.putText(frame, f"{predicted_class_name} ({confidence_score*100:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Conf: {confidence_score*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Presione 'q' para salir", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Reconocimiento de Objetos ViT CIFAR-100', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Cámara cerrada y recursos liberados.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicción con ViT desde la cámara.')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Ruta al checkpoint del modelo ViT entrenado.')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', 
                        help='Nombre del modelo ViT (ej: vit_base_patch16_224, vit_small_patch16_224).')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='Tamaño de la imagen de entrada para el ViT (ej: 224).')
    parser.add_argument('--num_classes', type=int, default=100, 
                        help='Número de clases (ej: 100 para CIFAR-100).')
    
    args = parser.parse_args()
    predict_from_camera(args)
