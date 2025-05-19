import React from 'react';
import './PageStyles.css';

function OverviewPage() {
  return (
    <div className="page-content">
      <h2>Visión General del Proyecto</h2>
      <p>
        Este proyecto demuestra la creación, entrenamiento y despliegue de una 
        Red Neuronal Convolucional (CNN) para el reconocimiento de objetos. 
        Utilizamos el conocido dataset <strong>CIFAR-100</strong>, que contiene 100 clases diferentes de objetos.
      </p>
      
      <h3>Objetivos Principales:</h3>
      <ul>
        <li>Cargar y preprocesar el dataset CIFAR-100.</li>
        <li>Definir una arquitectura de CNN adecuada para la clasificación de imágenes.</li>
        <li>Entrenar el modelo utilizando PyTorch y realizar un seguimiento del rendimiento.</li>
        <li>Utilizar el modelo entrenado para realizar predicciones en tiempo real a través de la cámara web del PC.</li>
        <li>Visualizar el proceso de entrenamiento y la arquitectura del modelo utilizando TensorBoard.</li>
      </ul>

      <h3>Componentes Clave del Proyecto Python:</h3>
      <p>
        El backend de este sistema de reconocimiento de objetos se ha desarrollado en Python y consta de varios módulos principales:
      </p>
      <ul>
        <li>
          <strong><code>data_loader.py</code>:</strong> Responsable de descargar el dataset CIFAR-100, aplicar las transformaciones necesarias (como normalización y data augmentation) y preparar los DataLoaders para el entrenamiento y la prueba.
        </li>
        <li>
          <strong><code>model.py</code>:</strong> Define la arquitectura de la Red Neuronal Convolucional (<code>SimpleCNN</code>). Describe las capas convolucionales, de pooling, de normalización por lotes (Batch Normalization), de dropout y las capas totalmente conectadas que componen la red.
        </li>
        <li>
          <strong><code>train.py</code>:</strong> Orquesta el proceso de entrenamiento. Inicializa el modelo, el optimizador y la función de pérdida. Itera sobre el dataset de entrenamiento por épocas, realiza la retropropagación (backpropagation) y actualiza los pesos del modelo. También evalúa el modelo en un conjunto de validación y guarda el mejor modelo obtenido. Integra TensorBoard para la visualización de métricas.
        </li>
        <li>
          <strong><code>predict_camera.py</code>:</strong> Utiliza el modelo entrenado para realizar predicciones en tiempo real. Captura vídeo de la cámara web, preprocesa cada fotograma de manera similar a como se prepararon las imágenes de entrenamiento/prueba, y pasa el fotograma a través del modelo para obtener una predicción de clase. La predicción se muestra superpuesta en la ventana de la cámara.
        </li>
        <li>
          <strong><code>utils.py</code>:</strong> Contiene funciones de utilidad, como por ejemplo, para guardar y cargar checkpoints del modelo o para graficar métricas (aunque la visualización principal se realiza con TensorBoard).
        </li>
      </ul>
      <p>
        En las siguientes secciones (que podríamos añadir), exploraremos cada uno de estos componentes con más detalle.
      </p>
    </div>
  );
}

export default OverviewPage;
