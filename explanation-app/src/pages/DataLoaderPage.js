import React from 'react';
import './PageStyles.css'; 

function DataLoaderPage() {
  return (
    <div className="page-content">
      <h2>Carga y Preparación de Datos (<code>data_loader.py</code>)</h2>
      <p>
        El script <code>data_loader.py</code> es fundamental para cualquier proyecto de aprendizaje automático, 
        ya que se encarga de la adquisición, transformación y organización de los datos que alimentarán al modelo.
        En este proyecto, su función principal es preparar el dataset <strong>CIFAR-100</strong>.
      </p>

      <h3>Funciones Clave y Proceso:</h3>
      
      <h4>1. Importaciones Necesarias:</h4>
      <p>El script comienza importando las librerías esenciales:</p>
      <ul>
        <li><strong><code>torch</code></strong>: La librería principal de PyTorch.</li>
        <li><strong><code>torchvision.datasets</code></strong>: Para acceder a datasets populares como CIFAR-100.</li>
        <li><strong><code>torchvision.transforms</code></strong>: Para aplicar transformaciones a las imágenes (normalización, data augmentation, etc.).</li>
        <li><strong><code>torch.utils.data.DataLoader</code></strong>: Para crear iteradores eficientes sobre el dataset, que manejan el batching, shuffling, etc.</li>
      </ul>
      <pre><code className="language-python">
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
      </code></pre>

      <h4>2. Definición de Transformaciones (<code>transforms</code>):</h4>
      <p>
        Las imágenes del dataset necesitan ser preprocesadas antes de ser introducidas en la red neuronal. Esto se hace definiendo una secuencia de transformaciones:
      </p>
      <ul>
        <li>
          <strong>Para el conjunto de entrenamiento (<code>train_transform</code>):</strong>
          <ul>
            <li><code>transforms.RandomCrop(32, padding=4)</code>: Recorta aleatoriamente la imagen a 32x32 píxeles después de añadir un padding de 4 píxeles. Esto ayuda a que el modelo sea más robusto a pequeñas traslaciones del objeto en la imagen (Data Augmentation).</li>
            <li><code>transforms.RandomHorizontalFlip()</code>: Invierte aleatoriamente la imagen horizontalmente (Data Augmentation).</li>
            <li><code>transforms.ToTensor()</code>: Convierte la imagen (que usualmente es un objeto PIL o NumPy) a un Tensor de PyTorch. También normaliza los valores de los píxeles del rango [0, 255] al rango [0.0, 1.0].</li>
            <li><code>transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))</code>: Normaliza el tensor de la imagen utilizando la media y la desviación estándar precalculadas para el dataset CIFAR-100. Cada canal (R, G, B) se normaliza independientemente. Esto ayuda a que el modelo converja más rápido.</li>
          </ul>
        </li>
        <li>
          <strong>Para el conjunto de prueba/validación (<code>test_transform</code>):</strong>
          <ul>
            <li><code>transforms.ToTensor()</code>: Misma función que en el entrenamiento.</li>
            <li><code>transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))</code>: Misma normalización. Es crucial usar las mismas medias y desviaciones estándar que en el entrenamiento. No se aplica data augmentation aquí para obtener una evaluación consistente del rendimiento del modelo.</li>
          </ul>
        </li>
      </ul>
      <pre><code className="language-python">
# Ejemplo de cómo se definen las transformaciones en el script original
# (los valores exactos de normalización pueden variar ligeramente si se recalcularon)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
      </code></pre>

      <h4>3. Carga del Dataset CIFAR-100:</h4>
      <p>
        Utilizando <code>torchvision.datasets.CIFAR100</code>, el script descarga (si no está ya presente) y carga los conjuntos de datos de entrenamiento y prueba. Se aplican las transformaciones definidas anteriormente a cada conjunto respectivo.
      </p>
      <ul>
        <li><code>root='./data'</code>: Especifica el directorio donde se guardarán/buscarán los datos.</li>
        <li><code>train=True</code>: Indica que se cargue el conjunto de entrenamiento.</li>
        <li><code>train=False</code>: Indica que se cargue el conjunto de prueba.</li>
        <li><code>download=True</code>: Permite la descarga del dataset si no se encuentra localmente.</li>
        <li><code>transform=train_transform</code> o <code>transform=test_transform</code>: Asigna las transformaciones correspondientes.</li>
      </ul>
      <pre><code className="language-python">
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
      </code></pre>

      <h4>4. Creación de DataLoaders:</h4>
      <p>
        Finalmente, se crean los <code>DataLoader</code> para los conjuntos de entrenamiento y prueba. Los DataLoaders son iterables que proporcionan lotes (batches) de datos al modelo durante el entrenamiento y la evaluación. Esto es eficiente y necesario para el proceso de aprendizaje.
      </p>
      <ul>
        <li><code>batch_size=64</code> (o el valor definido en constantes): Define cuántas imágenes se procesan en cada iteración.</li>
        <li><code>shuffle=True</code> (para el train_loader): Mezcla los datos de entrenamiento en cada época para evitar que el modelo aprenda el orden de los datos y mejorar la generalización.</li>
        <li><code>shuffle=False</code> (para el test_loader): No es necesario mezclar los datos de prueba.</li>
        <li><code>num_workers=2</code> (o un valor adecuado): Permite cargar los datos en paralelo utilizando múltiples subprocesos, lo que puede acelerar la carga de datos.</li>
      </ul>
      <pre><code className="language-python">
BATCH_SIZE = 64 # Ejemplo
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
      </code></pre>

      <h4>5. Obtención de Nombres de Clases:</h4>
      <p>
        El script también extrae los nombres de las 100 clases del dataset CIFAR-100. Esto es útil para interpretar las salidas del modelo y para la visualización.
      </p>
      <pre><code className="language-python">
# cifar100_classes = train_dataset.classes # Así se obtienen las clases
# num_classes = len(cifar100_classes)
      </code></pre>

      <h3>Importancia del <code>data_loader.py</code>:</h3>
      <p>
        Un buen pipeline de carga de datos es crucial. Si los datos no se preparan correctamente, 
        el modelo no podrá aprender eficazmente, independientemente de lo sofisticada que sea su arquitectura.
        El uso de Data Augmentation ayuda a prevenir el sobreajuste (overfitting) y a mejorar la capacidad 
        del modelo para generalizar a imágenes no vistas.
      </p>
      <p>
        En las siguientes secciones, veremos cómo se utiliza el <code>train_loader</code> y el <code>test_loader</code> en 
        el script de entrenamiento (<code>train.py</code>).
      </p>

    </div>
  );
}

export default DataLoaderPage;
