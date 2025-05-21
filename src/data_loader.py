import torch
import torchvision
import torchvision.transforms as transforms

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

def get_cifar100_loaders(batch_size=64, data_dir='./data'):
    """Carga y preprocesa el dataset CIFAR-100."""
    print("Preparando los cargadores de datos de CIFAR-100...")

    # Transformaciones para el conjunto de entrenamiento (con data augmentation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # Medias y std de CIFAR-100
    ])

    # Transformaciones para el conjunto de prueba/validación (sin augmentation, solo normalización)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Descargar y cargar el conjunto de entrenamiento
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Descargar y cargar el conjunto de prueba
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Número de clases CIFAR-100 (fine): {len(CIFAR100_CLASSES)}")
    print("Cargadores de datos listos.")
    return trainloader, testloader, CIFAR100_CLASSES

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
