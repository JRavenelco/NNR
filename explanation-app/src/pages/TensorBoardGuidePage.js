import React from 'react';
import './PageStyles.css';
import MermaidDiagram from '../components/MermaidDiagram'; // Importar el componente

const tensorboardHighLevelGraph = `
graph TD
    Input[Datos de Entrada (Lotes)] --> Model(Modelo SimpleCNN);
    Model --> Output[Predicciones (Logits)];
    Model --> LossCalc{Cálculo de Pérdida};
    Output --> LossCalc;
    Labels[Etiquetas Reales] --> LossCalc;
    LossCalc --> Metrics[Métricas (Pérdida, Precisión)];
    Metrics --> TensorBoardLog[Logs para TensorBoard];

    subgraph TensorBoard_Visualization
        direction RL
        TensorBoardLog
    end

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Output fill:#ccf,stroke:#333,stroke-width:2px
    style Metrics fill:#lightgreen,stroke:#333,stroke-width:2px
`;

function TensorBoardGuidePage() {
  return (
    <div className="page-content">
      <h2>Guía de Visualización con TensorBoard</h2>
      <p>
        TensorBoard es una suite de herramientas de visualización proporcionada por TensorFlow, pero es ampliamente compatible con PyTorch 
        (a través de <code>torch.utils.tensorboard.SummaryWriter</code>). Es esencial para monitorizar y entender el entrenamiento 
        de los modelos de aprendizaje profundo, depurar problemas y optimizar el rendimiento.
      </p>

      <h3>1. ¿Cómo Iniciar TensorBoard?</h3>
      <p>
        En tu script <code>train.py</code>, los logs de TensorBoard se guardan en el directorio especificado por la variable <code>LOG_DIR</code> 
        (por defecto, <code>'./runs/cifar100_experiment'</code>).
      </p>
      <p>
        Para iniciar TensorBoard, abre una terminal en el directorio raíz de tu proyecto Python (<code>NNR</code>) y ejecuta el siguiente comando:
      </p>
      <pre><code className="language-bash">
tensorboard --logdir=./runs
      </code></pre>
      <p>
        Esto escaneará el directorio <code>./runs</code> (y sus subdirectorios) en busca de logs. Luego, podrás acceder a la interfaz de TensorBoard 
        en tu navegador, usualmente en la dirección <a href="http://localhost:6006" target="_blank" rel="noopener noreferrer">http://localhost:6006</a> (el puerto puede variar si el 6006 está ocupado).
      </p>

      <h3>2. Funcionalidades Clave y Qué Esperar de Tu Proyecto:</h3>
      
      <h4>a. Pestaña SCALARS (Escalares):</h4>
      <p>
        Esta es una de las pestañas más utilizadas. Muestra cómo cambian las métricas escalares (números individuales) a lo largo del tiempo 
        (épocas o pasos globales). En tu proyecto <code>train.py</code>, se registran los siguientes escalares:
      </p>
      <ul>
        <li><strong><code>Loss/train_batch</code></strong>: La pérdida calculada después de cada lote durante el entrenamiento. Útil para ver la estabilidad del entrenamiento a nivel granular.</li>
        <li><strong><code>Loss/train_epoch</code></strong>: La pérdida promedio de entrenamiento al final de cada época.</li>
        <li><strong><code>Accuracy/train_epoch</code></strong>: La precisión de entrenamiento al final de cada época.</li>
        <li><strong><code>LearningRate</code></strong>: La tasa de aprendizaje utilizada en cada época. Si usas un scheduler, verás cómo cambia.</li>
        <li><strong><code>Loss/validation_epoch</code></strong>: La pérdida promedio de validación al final de cada época. Crucial para detectar sobreajuste.</li>
        <li><strong><code>Accuracy/validation_epoch</code></strong>: La precisión de validación al final de cada época. Es la métrica principal para evaluar el rendimiento del modelo y decidir cuándo guardar el "mejor" modelo.</li>
      </ul>
      <p><strong>Interpretación:</strong></p>
      <ul>
        <li>Idealmente, las curvas de pérdida (<code>Loss/train_epoch</code>, <code>Loss/validation_epoch</code>) deberían descender con el tiempo.</li>
        <li>Las curvas de precisión (<code>Accuracy/train_epoch</code>, <code>Accuracy/validation_epoch</code>) deberían ascender.</li>
        <li>Si la pérdida de entrenamiento sigue bajando pero la de validación empieza a subir (o la precisión de validación se estanca/baja), es un signo de <strong>sobreajuste</strong> (overfitting).</li>
        <li>Puedes activar/desactivar la visualización de diferentes "runs" (experimentos) si tienes varios, y ajustar el suavizado (smoothing) de las curvas para una mejor visualización.</li>
      </ul>

      <h4>b. Pestaña GRAPHS (Grafos):</h4>
      <p>
        Esta pestaña te permite visualizar la arquitectura de tu modelo (el grafo computacional).
        En <code>train.py</code>, se intenta registrar el grafo con <code>writer.add_graph(model, dummy_images.to(device))</code>.
      </p>
      <MermaidDiagram chart={tensorboardHighLevelGraph} />
      <p style={{textAlign: 'center', fontSize: '0.9em', marginTop: '5px'}}>
        <em>Diagrama conceptual de alto nivel del flujo de datos en el entrenamiento y cómo se relaciona con TensorBoard.</em>
      </p>
      <p>
        En TensorBoard, verás nodos que representan las operaciones y capas de tu <code>SimpleCNN</code>. Puedes hacer doble clic en los nodos para expandirlos 
        y ver su estructura interna (por ejemplo, las capas Conv2d, BatchNorm, Linear dentro del módulo <code>SimpleCNN</code>).
        Esto es muy útil para:
      </p>
      <ul>
        <li>Verificar que la arquitectura del modelo se haya definido correctamente.</li>
        <li>Entender el flujo de datos a través del modelo.</li>
        <li>Depurar problemas relacionados con las dimensiones de los tensores.</li>
      </ul>
      <p>La imagen que proporcionaste anteriormente es un ejemplo de cómo se ve esta pestaña con tu modelo.</p>

      <h4>c. Pestaña TEXT (Texto):</h4>
      <p>
        Permite registrar información textual. En <code>train.py</code>, se utiliza para guardar:
      </p>
      <ul>
        <li><strong><code>Hyperparameters</code></strong>: Un resumen de los hiperparámetros clave del entrenamiento (e.g., tasa de aprendizaje, tamaño del lote, número de épocas).</li>
        <li><strong><code>Final Metrics</code></strong>: Las métricas finales del entrenamiento, incluyendo la mejor precisión de validación obtenida.</li>
      </ul>

      <h4>d. Otras Pestañas (Mención General):</h4>
      <ul>
        <li><strong>DISTRIBUTIONS (Distribuciones) e HISTOGRAMS (Histogramas):</strong> Estas pestañas pueden mostrar la evolución de la distribución de los pesos y biases de tus capas a lo largo del entrenamiento. Para esto, necesitarías añadir explícitamente <code>writer.add_histogram(...)</code> en tu código de entrenamiento. Son útiles para diagnosticar problemas como la desaparición/explosión de gradientes o neuronas "muertas".</li>
        <li><strong>IMAGES (Imágenes):</strong> Podrías usar <code>writer.add_image(...)</code> o <code>writer.add_images(...)</code> para visualizar imágenes (e.g., ejemplos del dataset, imágenes generadas, filtros convolucionales).</li>
      </ul>

      <h3>3. ¿Por Qué es Útil TensorBoard?</h3>
      <ul>
        <li><strong>Monitorización en Tiempo Real:</strong> Observa el rendimiento de tu modelo mientras entrena.</li>
        <li><strong>Comparación de Experimentos:</strong> Si cambias hiperparámetros o arquitecturas, puedes comparar fácilmente los resultados de diferentes "runs".</li>
        <li><strong>Depuración:</strong> Identifica problemas como el sobreajuste, el subajuste, o tasas de aprendizaje inadecuadas.</li>
        <li><strong>Comunicación:</strong> Es una excelente herramienta para compartir y explicar los resultados de tu modelo.</li>
      </ul>
      <p>
        Explorar TensorBoard a fondo te dará una comprensión mucho más profunda de cómo están aprendiendo tus modelos.
      </p>
    </div>
  );
}

export default TensorBoardGuidePage;
