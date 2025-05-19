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
