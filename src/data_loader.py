import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image, ImageEnhance

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al. (https://arxiv.org/abs/1708.04896)
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(self, img):
        if random.random() < self.p:
            return F.erase(img, 0, 0, img.size(1), img.size(2), self.value, self.inplace)
        return img

# --- Lista de Clases CIFAR-100 (Fine Labels) ---
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

def get_cifar100_loaders(batch_size=64, data_dir='./data', use_mixup=False, use_cutmix=False, cutmix_alpha=1.0, cutmix_prob=0.5):
    """Carga y preprocesa el dataset CIFAR-100 con aumentos de datos mejorados.
    
    Args:
        batch_size: Tamaño del lote
        data_dir: Directorio donde se almacenan/descargan los datos
        use_mixup: Si es True, aplica MixUp a los datos de entrenamiento
        use_cutmix: Si es True, aplica CutMix a los datos de entrenamiento
        cutmix_alpha: Parámetro alpha para la distribución Beta en CutMix
        cutmix_prob: Probabilidad de aplicar CutMix a un lote
        
    Returns:
        train_loader, test_loader, num_classes, train_dataset, test_dataset
    """
    print("Preparando los cargadores de datos de CIFAR-100 con aumentos mejorados...")
    print(f"Directorio de datos: {os.path.abspath(data_dir)}")
    print(f"Tamaño de lote: {batch_size}")
    print(f"Usar MixUp: {use_mixup}")
    print(f"Usar CutMix: {use_cutmix} (alpha: {cutmix_alpha}, prob: {cutmix_prob})")

    # Media y desviación estándar para CIFAR-100
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    
    # Transformaciones de aumento de datos mejoradas
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        Cutout(n_holes=1, length=16),
        RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False)
    ])

    # Transformaciones para validación (sin aumentos)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    # Descargar y cargar el conjunto de entrenamiento
    print("\nDescargando datos de entrenamiento...")
    try:
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        print(f"Datos de entrenamiento cargados. Tamaño: {len(train_dataset)} muestras")
    except Exception as e:
        print(f"Error al cargar datos de entrenamiento: {str(e)}")
        raise

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Descargar y cargar el conjunto de prueba
    print("\nDescargando datos de prueba...")
    try:
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        print(f"Datos de prueba cargados. Tamaño: {len(test_dataset)} muestras")
    except Exception as e:
        print(f"Error al cargar datos de prueba: {str(e)}")
        raise

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Número de clases CIFAR-100 (fine): {len(CIFAR100_CLASSES)}")
    print("Cargadores de datos listos.")
    return train_loader, test_loader, CIFAR100_CLASSES

if __name__ == '__main__':
    # Ejemplo de uso:
    print("Probando el cargador de datos...")
    train_loader, test_loader, classes = get_cifar100_loaders(batch_size=4)
    
    print(f"Número de lotes en el cargador de entrenamiento: {len(train_loader)}")
    print(f"Número de lotes en el cargador de prueba: {len(test_loader)}")
    print(f"Total de clases: {len(classes)}")
    print(f"Primeras 10 clases: {classes[:10]}")

    # Visualizar algunas imágenes del primer lote
    import matplotlib.pyplot as plt
    import numpy as np

    def imshow(img):
        img = img / 2 + 0.5     # Desnormalizar (si la normalización fue (-0.5, 0.5))
                                # Para la normalización de CIFAR-100, esto es más complejo.
                                # La normalización usada es: mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                                # Para desnormalizar: img * std + mean
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        img = img * std + mean
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Obtener algunas imágenes de entrenamiento
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Mostrar imágenes
    #imshow(torchvision.utils.make_grid(images))
    # Imprimir etiquetas
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    print("Ejemplo de lote cargado.")
    print(f"Forma de las imágenes: {images.shape}") # Debería ser [batch_size, 3, 32, 32]
    print(f"Forma de las etiquetas: {labels.shape}") # Debería ser [batch_size]
