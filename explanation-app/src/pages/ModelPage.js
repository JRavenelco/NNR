import React from 'react';
import './PageStyles.css';
import MermaidDiagram from '../components/MermaidDiagram'; // Importar el componente

// Definición del diagrama Mermaid para SimpleCNN
const simpleCnnMermaidChart = `
graph TD
    A[Input (3x32x32)] --> B(Conv1: 32x32x32\nBN1, ReLU, Pool1);
    B --> C(Conv2: 64x16x16\nBN2, ReLU, Pool2);
    C --> D(Conv3: 128x8x8\nBN3, ReLU, Pool3);
    D --> E(Conv4: 256x4x4\nBN4, ReLU);
    E --> F{Flatten (4096 features)};
    F --> G(FC1: 1024\nReLU, Dropout);
    G --> H(FC2: 512\nReLU, Dropout);
    H --> I[FC3: num_classes (Logits)];

    subgraph SimpleCNN_Architecture
        direction LR
        B
        C
        D
        E
        F
        G
        H
        I
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#ccf,stroke:#333,stroke-width:2px
`;

function ModelPage() {
  return (
    <div className="page-content">
      <h2>Arquitectura del Modelo CNN (<code>model.py</code>)</h2>
      <p>
        El archivo <code>model.py</code> define la arquitectura de nuestra Red Neuronal Convolucional (CNN), llamada <strong><code>SimpleCNN</code></strong>. 
        Esta clase hereda de <code>torch.nn.Module</code>, que es la clase base para todos los modelos de redes neuronales en PyTorch.
        La CNN es el cerebro del sistema, diseñada para aprender jerarquías de características directamente de los datos de imágenes del dataset CIFAR-100.
      </p>

      <h3>Diagrama Visual de la Arquitectura <code>SimpleCNN</code>:</h3>
      <MermaidDiagram chart={simpleCnnMermaidChart} />
      <p style={{textAlign: 'center', fontSize: '0.9em', marginTop: '5px'}}>
        <em>Diagrama simplificado mostrando el flujo principal y las dimensiones aproximadas de salida de cada bloque.</em>
      </p>

      <h3>Componentes Clave de <code>SimpleCNN</code>:</h3>

      <h4>1. Método <code>__init__(self, num_classes=100)</code>:</h4>
      <p>
        Este es el constructor de la clase. Aquí se definen todas las capas que componen la red. El parámetro <code>num_classes</code> 
        indica el número de categorías de salida (100 para CIFAR-100).
      </p>
      
      <h5>a. Bloques Convolucionales:</h5>
      <p>
        La red utiliza una secuencia de bloques convolucionales para la extracción de características. Cada bloque típicamente consiste en:
      </p>
      <ul>
        <li><strong><code>nn.Conv2d</code></strong>: Aplica filtros convolucionales a la entrada. Parámetros clave:
          <ul>
            <li><code>in_channels</code>: Número de canales en la imagen de entrada (3 para RGB al inicio).</li>
            <li><code>out_channels</code>: Número de filtros a aplicar (y por lo tanto, número de canales en la salida).</li>
            <li><code>kernel_size</code>: Tamaño del filtro convolucional (e.g., 3 para un filtro de 3x3).</li>
            <li><code>padding</code>: Añade píxeles alrededor de la imagen para controlar las dimensiones de salida. <code>padding=1</code> con <code>kernel_size=3</code> usualmente mantiene las dimensiones espaciales si <code>stride=1</code>.</li>
          </ul>
        </li>
        <li><strong><code>nn.BatchNorm2d</code></strong>: Normaliza las activaciones de la capa anterior. Ayuda a estabilizar y acelerar el entrenamiento.</li>
        <li><strong>Función de Activación (<code>F.relu</code> en <code>forward</code>)</strong>: Introduce no linealidad, permitiendo al modelo aprender funciones más complejas. ReLU (Rectified Linear Unit) es una elección común.</li>
        <li><strong><code>nn.MaxPool2d</code></strong>: Reduce las dimensiones espaciales (downsampling) de los mapas de características, ayudando a lograr invariancia a pequeñas traslaciones y a reducir la carga computacional. Parámetros:
            <ul>
                <li><code>kernel_size</code>: Tamaño de la ventana de pooling.</li>
                <li><code>stride</code>: Desplazamiento de la ventana. <code>stride=2</code> con <code>kernel_size=2</code> reduce a la mitad las dimensiones.</li>
            </ul>
        </li>
      </ul>
      <p>La arquitectura de <code>SimpleCNN</code> incluye cuatro bloques convolucionales principales:</p>
      <pre><code className="language-python">
# Ejemplo de la definición de las capas convolucionales en __init__
# Bloque 1 (Entrada: 3x32x32)
self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) # Salida: 32x32x32
self.bn1 = nn.BatchNorm2d(32)
self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Salida: 32x16x16

# Bloque 2 (Entrada: 32x16x16)
self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # Salida: 64x16x16
self.bn2 = nn.BatchNorm2d(64)
self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Salida: 64x8x8

# Bloque 3 (Entrada: 64x8x8)
self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # Salida: 128x8x8
self.bn3 = nn.BatchNorm2d(128)
self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Salida: 128x4x4

# Bloque 4 (Entrada: 128x4x4)
self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) # Salida: 256x4x4
self.bn4 = nn.BatchNorm2d(256)
# No hay pooling después de la última capa convolucional en este diseño particular.
      </code></pre>

      <h5>b. Aplanamiento (Flattening):</h5>
      <p>
        Después de los bloques convolucionales, los mapas de características 2D (o 3D con canales) se aplanan en un vector 1D. 
        Esto es necesario para pasar los datos a las capas totalmente conectadas. 
        En <code>SimpleCNN</code>, la salida del último bloque convolucional (<code>self.conv4</code>, <code>self.bn4</code>) es de <code>256x4x4</code>. 
        Por lo tanto, el número de características aplanadas (<code>self.fc1_input_features</code>) es <code>256 * 4 * 4 = 4096</code>.
      </p>

      <h5>c. Capas Totalmente Conectadas (Fully Connected / Dense Layers):</h5>
      <p>
        Estas capas realizan la clasificación final basada en las características extraídas por las capas convolucionales.
      </p>
      <ul>
        <li><strong><code>nn.Linear</code></strong>: Aplica una transformación lineal (<code>y = Wx + b</code>). Parámetros:
          <ul>
            <li><code>in_features</code>: Número de características de entrada (e.g., 4096 para la primera capa FC).</li>
            <li><code>out_features</code>: Número de neuronas en la capa (y características de salida).</li>
          </ul>
        </li>
        <li><strong><code>nn.Dropout</code></strong>: Técnica de regularización que desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento para prevenir el sobreajuste. El argumento es la probabilidad de desactivación (e.g., <code>0.5</code> para 50%).</li>
      </ul>
      <p>La <code>SimpleCNN</code> tiene tres capas totalmente conectadas:</p>
      <pre><code className="language-python">
# Definición de las capas totalmente conectadas en __init__
self.fc1_input_features = 256 * 4 * 4 
self.fc1 = nn.Linear(self.fc1_input_features, 1024)
self.dropout1 = nn.Dropout(0.5)
self.fc2 = nn.Linear(1024, 512)
self.dropout2 = nn.Dropout(0.5)
self.fc3 = nn.Linear(512, num_classes) # Capa de salida, produce logits
      </code></pre>

      <h4>2. Método <code>forward(self, x)</code>:</h4>
      <p>
        Este método define cómo los datos de entrada (<code>x</code>) fluyen a través de las capas definidas en <code>__init__</code>. 
        Es la implementación del "paso hacia adelante" de la red.
      </p>
      <pre><code className="language-python">
def forward(self, x):
    # Bloque convolucional 1
    x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    # Bloque convolucional 2
    x = self.pool2(F.relu(self.bn2(self.conv2(x))))
    # Bloque convolucional 3
    x = self.pool3(F.relu(self.bn3(self.conv3(x))))
    # Bloque convolucional 4
    x = F.relu(self.bn4(self.conv4(x))) # Sin pooling aquí

    # Aplanar para las capas FC
    x = x.view(-1, self.fc1_input_features)
    
    # Capas totalmente conectadas
    x = self.dropout1(F.relu(self.fc1(x)))
    x = self.dropout2(F.relu(self.fc2(x)))
    x = self.fc3(x) # Salida logits
    return x
      </code></pre>
      <p>
        Notar que la función de activación <code>F.relu</code> se aplica después de la convolución (y normalización de lote). 
        La operación <code>x.view(-1, self.fc1_input_features)</code> aplana el tensor <code>x</code>. El <code>-1</code> infiere automáticamente la dimensión del lote (batch size).
        La capa final (<code>self.fc3</code>) produce <strong>logits</strong> (valores brutos antes de aplicar una función Softmax). Esto es común porque la función de pérdida <code>nn.CrossEntropyLoss</code> (utilizada en el script de entrenamiento) combina internamente <code>LogSoftmax</code> y <code>NLLLoss</code>, por lo que espera logits como entrada.
      </p>

      <h4>3. Bloque <code>if __name__ == '__main__':</code> (Opcional pero útil):</h4>
      <p>
        El script <code>model.py</code> incluye un bloque que se ejecuta solo cuando el script se corre directamente (no cuando se importa). 
        Este bloque usualmente contiene código para probar rápidamente la definición del modelo: crea una instancia del modelo, 
        le pasa un tensor de entrada de ejemplo (dummy input) y verifica las dimensiones de la salida. 
        Es una buena práctica para depurar la arquitectura del modelo.
      </p>

      <h3>Conclusión:</h3>
      <p>
        La <code>SimpleCNN</code> es una arquitectura convolucional estándar que apila capas para aprender representaciones cada vez más abstractas de las imágenes.
        Las capas convolucionales actúan como extractores de características, mientras que las capas totalmente conectadas realizan la clasificación final.
        Esta arquitectura es un buen punto de partida para tareas de clasificación de imágenes como CIFAR-100.
      </p>
    </div>
  );
}

export default ModelPage;
