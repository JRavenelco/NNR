import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import shutil

# Define la ruta base donde se guardará el dataset
BASE_SAVE_PATH = os.path.join('data', 'cifar100_torchvision')

# CIFAR-100 class names (obtenidas de la documentación o metadata)
# No es estrictamente necesario para torchvision.datasets.CIFAR100 ya que los labels son numéricos,
# pero es útil para crear nombres de carpetas legibles si se desea.
# torchvision.datasets.CIFAR100().classes nos da los nombres de las clases

def download_and_save_cifar100():
    """
    Descarga el dataset CIFAR-100 usando torchvision y guarda las imágenes
    en una estructura de carpetas train/clase y test/clase.
    """
    print(f"Creando directorio base para el dataset en: {os.path.abspath(BASE_SAVE_PATH)}")
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)

    # Descargar y guardar el conjunto de entrenamiento
    print("Descargando conjunto de entrenamiento de CIFAR-100...")
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
    print("Organizando imágenes de entrenamiento...")
    save_images(train_dataset, 'train')

    # Descargar y guardar el conjunto de prueba
    print("\nDescargando conjunto de prueba de CIFAR-100...")
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    print("Organizando imágenes de prueba...")
    save_images(test_dataset, 'test')

    print(f"\nDataset CIFAR-100 descargado y organizado en: {os.path.abspath(BASE_SAVE_PATH)}")
    print("Puedes eliminar la carpeta './data/cifar-100-python' si deseas, ya que las imágenes han sido reorganizadas.")

def save_images(dataset, split):
    """
    Guarda las imágenes del dataset en la estructura de carpetas especificada.
    split: 'train' o 'test'
    """
    save_dir_split = os.path.join(BASE_SAVE_PATH, split)
    os.makedirs(save_dir_split, exist_ok=True)

    # Obtener los nombres de las clases del dataset
    class_names = dataset.classes

    # Crear carpetas para cada clase
    for class_name in class_names:
        class_path = os.path.join(save_dir_split, class_name.replace(' ', '_').replace('-', '_')) # Sanitize class names for paths
        os.makedirs(class_path, exist_ok=True)

    num_images = len(dataset)
    for i in range(num_images):
        image, label_idx = dataset[i]
        class_name = class_names[label_idx].replace(' ', '_').replace('-', '_') # Sanitize class names
        
        # El nombre del archivo será img_split_indice.png
        # Usamos el índice original del dataset para asegurar unicidad y trazabilidad si es necesario
        # El nombre original de las imágenes de CIFAR-100 no se preserva directamente por torchvision.datasets
        filename = f"img_{split}_{i:05d}.png"
        image_path = os.path.join(save_dir_split, class_name, filename)
        
        try:
            image.save(image_path)
        except Exception as e:
            print(f"Error guardando imagen {filename} para la clase {class_name}: {e}")
            print(f"Tipo de imagen: {type(image)}, Modo: {image.mode if hasattr(image, 'mode') else 'N/A'}")

        if (i + 1) % 5000 == 0:
            print(f"  Procesadas {i + 1}/{num_images} imágenes para el conjunto {split}...")
    print(f"Se guardaron {num_images} imágenes en {save_dir_split}")

if __name__ == '__main__':
    # Limpiar directorio de destino si ya existe para evitar duplicados o errores
    if os.path.exists(BASE_SAVE_PATH):
        print(f"Eliminando directorio existente: {os.path.abspath(BASE_SAVE_PATH)}")
        shutil.rmtree(BASE_SAVE_PATH)
    
    download_and_save_cifar100()
