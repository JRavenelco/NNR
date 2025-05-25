import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont
from PyQt5.QtCore import QTimer, Qt
from picamera2 import Picamera2
import numpy as np
import torch
import torchvision.transforms as transforms
from model import SimpleCNN
from data_loader import CIFAR100_CLASSES
import os
import argparse

# Argumentos de línea de comandos para especificar el modelo
parser = argparse.ArgumentParser(description='Qt Camera Viewer con inferencia CIFAR-100')
# Default al modelo preentrenado original
default_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models_trained', 'cifar100_simplecnn.pth'))
parser.add_argument('-m', '--model', default=default_model, help='Ruta al .pth del modelo entrenado')
parser.add_argument('-a', '--arch', choices=['simplecnn','resnet18'], default='simplecnn', help='Arquitectura del modelo')
args = parser.parse_args()
MODEL_PATH = args.model
ARCH = args.arch

# --- Parámetros de inferencia ---
NUM_CLASSES = 100
IMG_SIZE = 32

# Transformación igual que entrenamiento
transform_cam = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

class CameraViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visor de Cámara Qt con Picamera2")
        self.setGeometry(100, 100, 640, 480) # x, y, width, height

        # Configurar Picamera2
        self.picam2 = Picamera2()
        # Trata de usar un formato que sea fácil de convertir a QImage
        # XRGB8888 es (ancho, alto, 4) y QImage lo maneja bien como QImage.Format_RGB32
        # RGB888 es (ancho, alto, 3) y QImage lo maneja como QImage.Format_RGB888
        self.preview_config = self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)})
        self.picam2.configure(self.preview_config)
        self.picam2.start()
        print("Cámara iniciada con Picamera2.")

        # --- Configurar modelo ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Instanciar modelo según arquitectura
        if ARCH == 'simplecnn':
            self.model = SimpleCNN(num_classes=NUM_CLASSES)
        else:
            from torchvision.models import resnet18
            self.model = resnet18(pretrained=False)
            # Ajustar la última capa fully-connected
            import torch.nn as nn
            self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
        # Cargar pesos (filtrando keys que no coincidan en shape)
        state = torch.load(MODEL_PATH, map_location=self.device)
        model_dict = self.model.state_dict()
        # Filtrar únicamente keys existentes y con mismas dimensiones
        filtered_state = {}
        for k, v in state.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_state[k] = v
                else:
                    print(f"Skipping '{k}' due to shape mismatch {v.shape} vs {model_dict[k].shape}")
            else:
                print(f"Skipping '{k}' as not found in model")
        # Cargar el dict filtrado
        self.model.load_state_dict(filtered_state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        # Cargar nombres de clase
        self.class_names = CIFAR100_CLASSES if len(CIFAR100_CLASSES)==NUM_CLASSES else [str(i) for i in range(NUM_CLASSES)]
        print(f"Modelo cargado en {self.device}, clases: {len(self.class_names)}")

        # Etiqueta para mostrar el video
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        # Temporizador para actualizar la imagen
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Actualizar cada ~33ms (aprox 30 FPS)

    def update_frame(self):
        # Capturar frame de Picamera2
        frame_array = self.picam2.capture_array()

        # El formato de picamera2.capture_array() con 'RGB888' es un array NumPy (alto, ancho, 3) en orden RGB.
        # QImage espera datos en formato RGB888 o RGB32 (con alfa).
        
        height, width, channel = frame_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # --- Inferencia ---
        # Preprocesar para PyTorch
        input_tensor = transform_cam(frame_array)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_batch)
        probs = torch.softmax(output, dim=1)[0]
        conf, idx = torch.max(probs, 0)
        label = f"{self.class_names[idx.item()]} ({conf.item()*100:.1f}%)"
        # --- Dibujar texto sobre el pixmap ---
        pixmap = QPixmap.fromImage(q_image)
        painter = QPainter(pixmap)
        painter.setPen(Qt.green)
        font = QFont('Arial', 16)
        painter.setFont(font)
        painter.drawText(10,30, label)
        painter.end()
        # Mostrar en label
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        print("Cerrando aplicación...")
        self.timer.stop()
        self.picam2.stop()
        print("Cámara detenida.")
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = CameraViewer()
    viewer.show()
    sys.exit(app.exec_())
