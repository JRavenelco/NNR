# Documentación del Proyecto: Entrenamiento de ResNet-18 en CIFAR-100

## Resumen del Proyecto
Este proyecto implementa el entrenamiento de un modelo ResNet-18 en el conjunto de datos CIFAR-100, incluyendo la implementación de técnicas avanzadas de entrenamiento y un script para probar el modelo con una cámara web en tiempo real.

## Estructura del Proyecto
```
NNR/
├── data/                    # Conjunto de datos CIFAR-100 (se descarga automáticamente)
├── saved_models/            # Modelos entrenados y checkpoints
│   ├── best_model.pth       # Mejor modelo guardado
│   └── checkpoints/         # Checkpoints durante el entrenamiento
│       ├── checkpoint_epoch_010.pth
│       └── checkpoint_epoch_020.pth
├── runs/                    # Logs de TensorBoard
├── src/
│   ├── __init__.py
│   ├── train_resnet18_cifar100.py  # Script de entrenamiento principal
│   ├── predict_camera.py           # Script para probar con cámara web
│   ├── model.py
│   └── utils.py
├── requirements.txt          # Dependencias del proyecto
└── README.md                # Este archivo
```

## Configuración del Entorno

### Requisitos
- Python 3.8+
- PyTorch 1.9.0+
- CUDA 11.1+ (opcional, para entrenamiento en GPU)
- Otras dependencias en `requirements.txt`

### Instalación
1. Clonar el repositorio
2. Crear un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # En Windows
   source venv/bin/activate  # En Linux/Mac
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Entrenamiento del Modelo

### Configuración
El archivo `train_resnet18_cifar100.py` contiene la clase `Config` con los parámetros de entrenamiento:
- Tamaño de lote (batch size): 128 (GPU) / 64 (CPU)
- Número de épocas: 100
- Tasa de aprendizaje: 0.1
- Optimizador: SGD con momentum (0.9) y Nesterov
- Programación de tasa de aprendizaje: Cosine Annealing con calentamiento
- Aumento de datos: MixUp y CutMix
- Regularización: Dropout (0.25) y Stochastic Depth (0.1)

### Ejecutar Entrenamiento
```bash
python src/train_resnet18_cifar100.py
```

### Monitoreo
El entrenamiento guarda checkpoints periódicamente en `saved_models/checkpoints/` y el mejor modelo en `saved_models/best_model.pth`.

Para monitorear el entrenamiento con TensorBoard:
```bash
tensorboard --logdir=runs
```

## Probar el Modelo con Cámara Web

### Requisitos Adicionales
- OpenCV (`pip install opencv-python`)
- Cámara web conectada

### Ejecutar Prueba con Cámara
```bash
python src/predict_camera.py
```

### Instrucciones de Uso
1. Asegúrate de que el modelo `saved_models/best_model.pth` existe
2. Ejecuta el script `predict_camera.py`
3. La cámara web se encenderá y mostrará las predicciones en tiempo real
4. Presiona 'q' para salir

## Solución de Problemas

### Error: No se encuentra el modelo
Asegúrate de que el archivo `best_model.pth` existe en `saved_models/`. Si no existe, entrena el modelo primero.

### Bajo rendimiento en tiempo real
- Reduce el tamaño del frame de la cámara
- Usa una GPU para inferencia más rápida
- Ajusta el tamaño del lote de inferencia

## Resultados del Entrenamiento
El modelo alcanzó los siguientes resultados:
- Mejor precisión en validación: ~62%
- Épocas completadas: 20/100 (último checkpoint)

## Mejoras Futuras
- Entrenar por más épocas
- Ajustar hiperparámetros (tasa de aprendizaje, tamaño de lote, etc.)
- Probar diferentes arquitecturas o técnicas de aumento de datos
- Implementar early stopping basado en la pérdida de validación

## Licencia
[Incluir información de licencia aquí]
