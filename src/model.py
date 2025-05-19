import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()
        # Capas convolucionales
        # Entrada: 3x32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # -> 32x32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> 32x16x16

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # -> 64x16x16
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> 64x8x8

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # -> 128x8x8
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> 128x4x4
        
        # Podríamos añadir más capas convolucionales aquí para mayor profundidad
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # -> 256x4x4
        self.bn4 = nn.BatchNorm2d(256)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # Si hiciéramos esto, quedaría 256x2x2

        # Capas totalmente conectadas
        # Aplanar la salida de la última capa de pooling/conv
        # Si pool3 es la última, la entrada aplanada es 128 * 4 * 4 = 2048
        # Si conv4 sin pool4 es la última, la entrada aplanada es 256 * 4 * 4 = 4096
        self.fc1_input_features = 256 * 4 * 4 
        self.fc1 = nn.Linear(self.fc1_input_features, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Bloque convolucional 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Bloque convolucional 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Bloque convolucional 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Bloque convolucional 4
        x = F.relu(self.bn4(self.conv4(x))) # Sin pooling después de esta

        # Aplanar para las capas FC
        x = x.view(-1, self.fc1_input_features)
        
        # Capas totalmente conectadas
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x) # Salida logits (sin softmax, CrossEntropyLoss lo incluye)
        return x

if __name__ == '__main__':
    # Prueba rápida del modelo
    print("Probando la definición del modelo CNN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = SimpleCNN(num_classes=100).to(device)
    print(model)

    # Crear un tensor de entrada de ejemplo (lote de 4 imágenes, 3 canales, 32x32)
    dummy_input = torch.randn(4, 3, 32, 32).to(device)
    print(f"Forma de la entrada: {dummy_input.shape}")

    # Pasar la entrada a través del modelo
    try:
        output = model(dummy_input)
        print(f"Forma de la salida: {output.shape}") # Debería ser [4, num_classes]
        assert output.shape == (4, 100)
        print("Prueba del modelo SimpleCNN completada con éxito.")
    except Exception as e:
        print(f"Error durante la prueba del modelo: {e}")
