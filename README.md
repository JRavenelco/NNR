# Proyecto de Reconocimiento de Objetos con CIFAR-100 y Cámara Web

Este proyecto implementa una red neuronal convolucional (CNN) para clasificar objetos basados en el dataset CIFAR-100. Una vez entrenado, el modelo puede utilizarse para el reconocimiento de objetos en tiempo real utilizando la cámara web del PC.

## Características

- Entrenamiento de una CNN con el dataset CIFAR-100.
- Preprocesamiento de datos y aumento de datos (data augmentation).
- Guardado y carga del modelo entrenado.
- Inferencia en tiempo real utilizando la cámara web.
- Visualización de la predicción sobre el fotograma de la cámara.

## Estructura del Proyecto

```
NNR/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── predict_camera.py
│   └── utils.py
└── models_trained/      # Directorio para guardar los modelos entrenados (se creará)
```

## Prerrequisitos

- Python 3.8+
- pip (Python package installer)
- Git
- (Opcional, para creación de repo en GitHub vía CLI) GitHub CLI (`gh`)

## Configuración

1.  **Clonar el repositorio (una vez que lo hayas creado en GitHub):**
    ```bash
    git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
    cd TU_REPOSITORIO
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    ```
    Activarlo:
    - Windows:
      ```bash
      .\venv\Scripts\activate
      ```
    - macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1.  **Entrenar el modelo:**
    ```bash
    python src/train.py
    ```
    Esto entrenará el modelo y guardará los pesos en el directorio `models_trained/`.

2.  **Ejecutar reconocimiento con la cámara:**
    ```bash
    python src/predict_camera.py
    ```
    Esto abrirá la cámara web y mostrará las predicciones en tiempo real.

## Dataset

Se utiliza el dataset CIFAR-100, que consta de 60000 imágenes en color de 32x32 en 100 clases, con 600 imágenes por clase. Hay 500 imágenes de entrenamiento y 100 imágenes de prueba por clase.

## Clases de CIFAR-100

El dataset CIFAR-100 tiene 100 clases agrupadas en 20 superclases. Este modelo predice las 100 clases individuales.

**Superclases (y algunas clases ejemplo):**
1.  aquatic mammals (beaver, dolphin, otter, seal, whale)
2.  fish (aquarium fish, flatfish, ray, shark, trout)
3.  flowers (orchid, poppy, rose, sunflower, tulip)
4.  food containers (bottle, bowl, can, cup, plate)
5.  fruit and vegetables (apple, mushroom, orange, pear, sweet pepper)
6.  household electrical devices (clock, computer keyboard, lamp, telephone, television)
7.  household furniture (bed, chair, couch, table, wardrobe)
8.  insects (bee, beetle, butterfly, caterpillar, cockroach)
9.  large carnivores (bear, leopard, lion, tiger, wolf)
10. large man-made outdoor things (bridge, castle, house, road, skyscraper)
11. large natural outdoor scenes (cloud, forest, mountain, plain, sea)
12. large omnivores and herbivores (camel, cattle, chimpanzee, elephant, kangaroo)
13. medium-sized mammals (fox, porcupine, possum, raccoon, skunk)
14. non-insect invertebrates (crab, lobster, snail, spider, worm)
15. people (baby, boy, girl, man, woman)
16. reptiles (crocodile, dinosaur, lizard, snake, turtle)
17. small mammals (hamster, mouse, rabbit, shrew, squirrel)
18. trees (maple_tree, oak_tree, palm_tree, pine_tree, willow_tree)
19. vehicles 1 (bicycle, bus, motorcycle, pickup truck, train)
20. vehicles 2 (lawn-mower, rocket, streetcar, tank, tractor)

## Ejecutar en Raspberry Pi 5 (Sistema Embebido)

Esta guía explica cómo configurar y ejecutar el proyecto NNR en una Raspberry Pi 5.

### 1. Prerrequisitos

*   Raspberry Pi 5 con Raspberry Pi OS (o una distribución de Linux compatible).
*   Conexión a internet.
*   Módulo de cámara conectado y configurado.

### 2. Instrucciones de Configuración

#### a. Actualizar Sistema e Instalar Git y Git LFS
Abre una terminal en tu Raspberry Pi y ejecuta:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install git -y

# Instalar Git LFS
# (Método recomendado usando el script oficial)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install --system # Instala LFS para todos los usuarios y repositorios
```
*Nota: Si el script de `packagecloud.io` falla, podrías necesitar descargar e instalar el paquete `.deb` de `git-lfs` manualmente para la arquitectura de tu RPi (ej. arm64) desde la [página de releases de Git LFS](https://github.com/git-lfs/git-lfs/releases).*

#### b. Clonar el Repositorio
Clona este proyecto desde GitHub:
```bash
git clone https://github.com/JRavenelco/NNR.git
cd NNR
```
Git LFS debería descargar automáticamente los archivos grandes del dataset. Si encuentras problemas con los archivos grandes, asegúrate de que LFS esté inicializado globalmente (como se hizo arriba) y prueba `git lfs pull` dentro del directorio `NNR`.

#### c. Configurar Entorno de Python y Dependencias
Se recomienda encarecidamente usar un entorno virtual de Python:
```bash
sudo apt install python3-pip python3-venv -y # Asegura que pip y venv estén instalados

python3 -m venv .venv   # Crea un entorno virtual llamado .venv
source .venv/bin/activate # Activa el entorno virtual
```
Ahora, instala los paquetes de Python requeridos:
```bash
pip install -r requirements.txt
```
*Nota sobre TensorFlow: El archivo `requirements.txt` podría especificar una versión de TensorFlow pensada para ordenadores de escritorio/servidores. Para Raspberry Pi, el rendimiento puede ser significativamente mejor con TensorFlow Lite, o una versión de TensorFlow específicamente compilada para ARM. Si encuentras problemas o lentitud con el modelo, considera instalar una versión compatible con Pi (ej. desde [piwheels.org](https://www.piwheels.org/) o usando el paquete `tensorflow-lite-runtime`). Tu script `predict_camera.py` podría necesitar ajustes si cambias a TensorFlow Lite.*

*Nota sobre OpenCV: Si `pip install opencv-python` da problemas, podrías necesitar bibliotecas de OpenCV a nivel de sistema:*
```bash
# Descomenta y ejecuta si es necesario:
# sudo apt install -y libopencv-dev python3-opencv
```

### 3. Ejecutar el Modelo de Predicción con Cámara

Asegúrate de que tu cámara esté conectada a la Raspberry Pi y habilitada (puedes hacerlo mediante `sudo raspi-config`, en la sección de Interfaces).

Navega al directorio `src` y ejecuta el script de predicción:
```bash
cd src
python3 predict_camera.py
```
Este script debería acceder a tu cámara y realizar predicciones en tiempo real usando el modelo entrenado.

### 4. Desactivar el Entorno Virtual
Cuando termines, puedes desactivar el entorno virtual:
```bash
deactivate
```

### 5. Solución de Problemas Comunes
*   **Archivos LFS no se descargan:** Asegúrate de que `git lfs install --system` se ejecutó correctamente. Dentro del repositorio, ejecuta `git lfs pull`.
*   **Dependencias de Python:** Algunos paquetes en `requirements.txt` podrían requerir bibliotecas del sistema. Revisa los mensajes de error durante `pip install` para obtener pistas (ej. `gcc` no encontrado, falta `libXYZ-dev`).
*   **Cámara no detectada:** Usa `ls /dev/video*` para verificar si la cámara es reconocida. Asegúrate de que esté habilitada en `raspi-config` y que los controladores estén cargados.
*   **Rendimiento del Modelo:** TensorFlow completo puede ser pesado para la Raspberry Pi. Para un mejor rendimiento, explora convertir tu modelo a TensorFlow Lite (`.tflite`) y usa el paquete de Python `tensorflow-lite-runtime`.
