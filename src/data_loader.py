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

# --- Lista de Clases CIFAR-100 (Fine Labels, Español) ---
CIFAR100_CLASSES = [
    'manzana', 'pez de acuario', 'bebé', 'oso', 'castor', 'cama', 'abeja', 'escarabajo',
    'bicicleta', 'botella', 'tazón', 'niño', 'puente', 'autobús', 'mariposa', 'camello',
    'lata', 'castillo', 'oruga', 'ganado', 'silla', 'chimpancé', 'reloj',
    'nube', 'cucaracha', 'sofá', 'cangrejo', 'cocodrilo', 'taza', 'dinosaurio',
    'delfín', 'elefante', 'pez plano', 'bosque', 'zorro', 'niña', 'hámster',
    'casa', 'canguro', 'teclado', 'lámpara', 'cortacésped', 'leopardo', 'león',
    'lagarto', 'langosta', 'hombre', 'arce', 'motocicleta', 'montaña', 'ratón',
    'hongo', 'roble', 'naranja', 'orquídea', 'nutria', 'palma', 'pera',
    'camioneta', 'pino', 'llanura', 'plato', 'amapola', 'puercoespín',
    'zarigüeya', 'conejo', 'mapache', 'raya', 'carretera', 'cohete', 'rosa', 'mar',
    'foca', 'tiburón', 'musaraña', 'zorrillo', 'rascacielos', 'caracol', 'serpiente', 'araña',
    'ardilla', 'tranvía', 'girasol', 'pimiento', 'mesa', 'tanque',
    'teléfono', 'televisión', 'tigre', 'tractor', 'tren', 'trucha', 'tulipán',
    'tortuga', 'armario', 'ballena', 'sauce', 'lobo', 'mujer', 'gusano'
]

def get_cifar100_loaders(batch_size=64, num_workers=2, data_dir='./data', img_size=32):
    """Carga y preprocesa el dataset CIFAR-100 con aumentos de datos mejorados.
    
    Args:
        batch_size: Tamaño del lote.
        num_workers: Número de workers para DataLoader.
        data_dir: Directorio donde se almacenan/descargan los datos.
        img_size: Tamaño de la imagen (e.g., 32 para CIFAR-100).
        
    Returns:
        train_loader: DataLoader para el conjunto de entrenamiento con aumentos.
        val_loader: DataLoader para el conjunto de validación (test set de CIFAR-100).
        test_loader: DataLoader para el conjunto de prueba (mismo que val_loader en este caso).
    """
    print("Preparando los cargadores de datos de CIFAR-100...")
    print(f"Directorio de datos: {os.path.abspath(data_dir)}")
    print(f"Tamaño de lote: {batch_size}, Workers: {num_workers}, Tamaño de imagen: {img_size}")

    # CIFAR-100 specific stats
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    
    # Transformaciones para el conjunto de entrenamiento con aumentos avanzados
    train_transform = transforms.Compose([
        # Aumentos geométricos
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        
        # Aumentos de color
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomInvert(p=0.1),
        transforms.RandomPosterize(bits=4, p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        
        # Aumentos avanzados
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # Normalización y conversión final
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        
        # Cutout para regularización adicional
        Cutout(n_holes=2, length=16),
    ])

    # Transformaciones para validación/prueba (sin aumentos, solo normalización y ToTensor)
    transform_val_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    # Descargar y cargar el conjunto de entrenamiento
    print("\nCargando datos de entrenamiento CIFAR-100...")
    try:
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        print(f"Datos de entrenamiento cargados. Tamaño: {len(train_dataset)} muestras")
    except Exception as e:
        print(f"Error al cargar datos de entrenamiento: {str(e)}")
        raise

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    # Descargar y cargar el conjunto de prueba (usado como validación y prueba)
    print("\nCargando datos de validación/prueba (test set de CIFAR-100)...")
    try:
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_val_test
        )
        print(f"Datos de validación/prueba cargados. Tamaño: {len(test_dataset)} muestras")
    except Exception as e:
        print(f"Error al cargar datos de validación/prueba: {str(e)}")
        raise

    # Usamos el mismo conjunto para validación y prueba
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Número de clases CIFAR-100 (fine): {len(CIFAR100_CLASSES)}")
    print("Cargadores de datos CIFAR-100 listos.")
    
    return train_loader, val_loader, test_loader

def get_ood_loaders(ood_dataset_name, batch_size, num_workers, data_dir, img_size):
    """Carga y preprocesa un dataset OOD especificado.

    Args:
        ood_dataset_name (str): Nombre del dataset OOD ('svhn', 'cifar10', 'texture', 'places365_small').
        batch_size (int): Tamaño del lote.
        num_workers (int): Número de workers para DataLoader.
        data_dir (str): Directorio base para los datos.
        img_size (int): Tamaño al que se redimensionarán las imágenes OOD.

    Returns:
        torch.utils.data.DataLoader: DataLoader para el dataset OOD, o None si no se especifica/encuentra.
    """
    if not ood_dataset_name:
        return None

    print(f"\nPreparando cargador de datos OOD: {ood_dataset_name}...")
    ood_data_path = os.path.join(data_dir, ood_dataset_name.lower())
    os.makedirs(ood_data_path, exist_ok=True)

    # Usar la misma normalización que CIFAR-100 para consistencia, aunque podría no ser óptima para todos los OOD.
    # Idealmente, se usaría la normalización con la que el modelo fue pre-entrenado si es de ImageNet.
    # Para este caso, asumimos que el modelo se entrena en CIFAR-100 y evaluamos OOD con la misma escala.
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    ood_transform = transforms.Compose([
        transforms.Resize((img_size + 4, img_size + 4)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    ood_dataset = None
    try:
        if ood_dataset_name.lower() == 'svhn':
            ood_dataset = torchvision.datasets.SVHN(
                root=ood_data_path, split='test', download=True, transform=ood_transform
            )
            print(f"SVHN (test set) cargado. Tamaño: {len(ood_dataset)} muestras.")
        elif ood_dataset_name.lower() == 'cifar10':
            # Usar el test set de CIFAR-10 como OOD
            ood_dataset = torchvision.datasets.CIFAR10(
                root=ood_data_path, train=False, download=True, transform=ood_transform
            )
            print(f"CIFAR-10 (test set) cargado como OOD. Tamaño: {len(ood_dataset)} muestras.")
        # TODO: Añadir más datasets OOD como Texture, Places365 (requerirían descarga manual o scripts)
        # elif ood_dataset_name.lower() == 'texture':
        #     # DTD (Describable Textures Dataset)
        #     # Necesitaría una clase Dataset personalizada y descarga manual.
        #     print("Dataset de Texturas (DTD) no implementado automáticamente. Requiere descarga manual.")
        # elif ood_dataset_name.lower() == 'places365_small':
        #     # Places365 subset
        #     # Necesitaría una clase Dataset personalizada y descarga manual.
        #     print("Dataset Places365 (small) no implementado automáticamente. Requiere descarga manual.")
        else:
            print(f"Dataset OOD '{ood_dataset_name}' no soportado o no implementado.")
            return None
    except Exception as e:
        print(f"Error al cargar el dataset OOD '{ood_dataset_name}': {str(e)}")
        return None

    if ood_dataset:
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        print(f"Cargador de datos OOD para '{ood_dataset_name}' listo.")
        return ood_loader
    return None

if __name__ == '__main__':
    # Ejemplo de uso:
    print("Probando el cargador de datos CIFAR-100...")
    train_loader_cifar, val_loader_cifar = get_cifar100_loaders(batch_size=4, num_workers=1, data_dir='../data', img_size=32)
    
    print(f"Número de lotes en el cargador de entrenamiento CIFAR-100: {len(train_loader_cifar)}")
    print(f"Número de lotes en el cargador de validación CIFAR-100: {len(val_loader_cifar)}")
    print(f"Total de clases CIFAR-100: {len(CIFAR100_CLASSES)}")

    # Probar cargador OOD
    print("\nProbando el cargador de datos OOD (SVHN)...")
    ood_loader_svhn = get_ood_loaders('svhn', batch_size=4, num_workers=1, data_dir='../data', img_size=32)
    if ood_loader_svhn:
        print(f"Número de lotes en el cargador OOD SVHN: {len(ood_loader_svhn)}")
        ood_images, _ = next(iter(ood_loader_svhn))
        print(f"Forma de las imágenes OOD SVHN: {ood_images.shape}")

    print("\nProbando el cargador de datos OOD (CIFAR-10)...")
    ood_loader_cifar10 = get_ood_loaders('cifar10', batch_size=4, num_workers=1, data_dir='../data', img_size=32)
    if ood_loader_cifar10:
        print(f"Número de lotes en el cargador OOD CIFAR-10: {len(ood_loader_cifar10)}")
        ood_images_c10, _ = next(iter(ood_loader_cifar10))
        print(f"Forma de las imágenes OOD CIFAR-10: {ood_images_c10.shape}")

    # Visualizar algunas imágenes del primer lote de CIFAR-100
    import matplotlib.pyplot as plt
    import numpy as np

    def imshow(img, labels=None):
        # Desnormalizar para visualización
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1) # Asegurar que esté en [0,1] después de desnormalizar
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        if labels is not None:
            plt.title(' | '.join([CIFAR100_CLASSES[int(lbl)] for lbl in labels]))
        plt.show()

    # Obtener algunas imágenes de entrenamiento CIFAR-100
    dataiter = iter(train_loader_cifar)
    images, labels = next(dataiter)

    # Mostrar imágenes y etiquetas en español
    # imshow(torchvision.utils.make_grid(images), labels=labels)
    print("\nEjemplo de lote CIFAR-100 cargado.")
    print(f"Forma de las imágenes: {images.shape}") # Debería ser [batch_size, 3, img_size, img_size]
    print(f"Forma de las etiquetas: {labels.shape}") # Debería ser [batch_size]
    print('Etiquetas en español:', ', '.join([CIFAR100_CLASSES[int(lbl)] for lbl in labels]))
