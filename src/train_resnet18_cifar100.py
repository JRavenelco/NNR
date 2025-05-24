import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import timm
from timm import create_model

# Configuración
class Config:
    # Configuración de entrenamiento
    BATCH_SIZE = 16  # Reducido a 16 para usar menos memoria
    NUM_EPOCHS = 100  # Reducido a 100 épocas
    WARMUP_EPOCHS = 3  # Reducido de 5 a 3
    GRAD_ACCUM_STEPS = 8  # Aumentar la acumulación de gradientes
    
    # Reducir tamaño de imagen para usar menos memoria
    IMG_SIZE = 32  # Tamaño original de CIFAR-100
    CROP_SIZE = 32  # Sin recorte adicional
    
    # Regularización
    DROPOUT_RATE = 0.1  # Tasa de dropout para regularización
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 5  # Reducido de 10 a 5
    MIN_DELTA = 0.01  # Aumentado para ser menos sensible
    
    # Hiperparámetros optimizados para memoria
    LEARNING_RATE = 0.05  # Reducido de 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4  # Reducido de 5e-4
    NUM_WORKERS = 0  # Desactivar workers para ahorrar memoria
    
    # Configuración de AMP (Automatic Mixed Precision)
    USE_AMP = True  # Mantener activado para ahorrar memoria
    
    # Configuración de OneCycleLR
    MAX_LR = 0.05  # Reducido de 0.1
    MIN_LR = 1e-5
    
    # Configuración de la red
    NUM_CLASSES = 100
    
    # Directorios
    SAVE_DIR = './saved_models'
    LOG_DIR = f'./runs/resnet18_cifar100_{int(time.time())}'
    
    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith('_')])

# Cargar datos
def get_data_loaders(config):
    # Tamaño de imagen esperado por el modelo pre-entrenado
    img_size = 224
    
    # Aumentos de datos más ligeros para usar menos memoria
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomCrop(config.CROP_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Conjuntos de datos
    train_set = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform)
    
    val_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=val_transform)
    
    # DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE * 2, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader

# Función para crear el modelo con timm
def create_resnet18_model(num_classes=100, pretrained=False, dropout_rate=0.1):
    # Cargar modelo básico sin parámetros adicionales
    model = timm.create_model('resnet18', pretrained=pretrained)
    
    # Reemplazar la capa final para el número correcto de clases
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    # Ajustar la primera capa para CIFAR-100 (32x32)
    if hasattr(model, 'conv1'):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Reducir el tamaño del modelo
    def _reduce_model(m):
        if isinstance(m, nn.Conv2d):
            # No modificar las capas ya que puede causar problemas
            pass
    
    # Aplicar reducción de tamaño al modelo
    model.apply(_reduce_model)
    
    return model

# Funciones de aumento de datos
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    if alpha <= 0:
        return x, y, None, 1.0
    
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(device)
    
    # Generar región de corte
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    # Ajustar lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y, rand_index, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Función de entrenamiento
def train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, epoch, config, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Configuración para usar menos memoria
    torch.backends.cudnn.benchmark = True  # Mejor rendimiento para tamaños fijos
    torch.backends.cudnn.deterministic = False
    
    # Limpiar caché de CUDA
    torch.cuda.empty_cache()
    
    # Forzar recolección de basura
    import gc
    gc.collect()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass con autocast para mixed precision
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            outputs = model(inputs)
            loss = criterion(outputs, targets) / config.GRAD_ACCUM_STEPS  # Escalar la pérdida
        
        # Backward pass y optimización
        scaler.scale(loss).backward()
        
        # Acumular gradientes y actualizar pesos cada GRAD_ACCUM_STEPS pasos
        if batch_idx % config.GRAD_ACCUM_STEPS == 0 or batch_idx == len(train_loader):
            # Escalar gradientes y actualizar pesos
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # Más eficiente que zero_grad()
        
        # Estadísticas
        with torch.no_grad():
            running_loss += loss.item() * config.GRAD_ACCUM_STEPS * inputs.size(0)  # Deshacer escalado y promediar
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Imprimir progreso
        if batch_idx % 10 == 0 or batch_idx == len(train_loader):
            avg_loss = running_loss / total
            acc = 100. * correct / total
            print(f'\rÉpoca: {epoch + 1} | Lote: {batch_idx}/{len(train_loader)} | '
                  f'Pérdida: {avg_loss:.4f} | Precisión: {acc:.2f}%', end='')
    
    # Calcular métricas finales
    avg_loss = running_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

# Función de validación
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total

def main():
    print("Iniciando función main...")
    # Configuración
    try:
        print("Creando configuración...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')
        config = Config()
        print("Configuración creada exitosamente")
        print("Configuración:")
        print('-' * 50)
        print(config)
        print('-' * 50)
    except Exception as e:
        print(f"Error al crear configuración: {e}")
        raise
    
    # Directorios
    try:
        print(f"Creando directorios: {config.SAVE_DIR} y {config.LOG_DIR}")
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.join(config.SAVE_DIR, 'checkpoints'), exist_ok=True)
        print("Directorios creados exitosamente")
    except Exception as e:
        print(f"Error al crear directorios: {e}")
        raise

    # Cargar datos
    train_loader, val_loader = get_data_loaders(config)

    # Inicializar modelo con timm
    model = create_resnet18_model(
        num_classes=config.NUM_CLASSES, 
        pretrained=True,  # Usar pesos pre-entrenados
        dropout_rate=config.DROPOUT_RATE
    ).to(device)
    
    # Contador para early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Función de pérdida
    criterion = nn.CrossEntropyLoss()

    # Optimizador SGD estándar
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,  # Usar LEARNING_RATE en lugar de BASE_LR
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=True  # Habilitar Nesterov momentum
    )
    
    # Scheduler de calentamiento y decaimiento coseno
    warmup_epochs = config.WARMUP_EPOCHS
    
    # Scheduler de calentamiento lineal
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Scheduler de decaimiento coseno
    cosine_sched = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS - warmup_epochs,
        eta_min=config.MIN_LR
    )
    
    # Combinar ambos schedulers
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_sched],
        milestones=[warmup_epochs]
    )

    # El warmup ya está manejado en la creación del lr_scheduler

    # Inicializar GradScaler para mixed precision training
    scaler = GradScaler(enabled=config.USE_AMP)

    # TensorBoard
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # Entrenamiento
    print("\nIniciando entrenamiento...")
    best_acc = 0.0
    best_epoch = 0
    training_start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Entrenar por una época
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, epoch, config, device
        )
        
        # Validar
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Actualizar scheduler
        if not isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        # Calcular tiempo por época
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        remaining_epochs = config.NUM_EPOCHS - (epoch + 1)
        remaining_time = remaining_epochs * epoch_time
        
        # Guardar mejor modelo
        if val_acc > best_acc:
            print(f'Mejor precisión de validación: {val_acc:.2f}%')
            best_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Guardar el mejor modelo
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(config.SAVE_DIR, 'best_model.pth'))
        else:
            epochs_without_improvement += 1
        
        # Guardar checkpoint cada 10 épocas
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(config.SAVE_DIR, 'checkpoints', f'checkpoint_epoch_{epoch+1:03d}.pth'))
            
            # Eliminar checkpoints antiguos (mantener solo los últimos 3)
            checkpoints = sorted([f for f in os.listdir(os.path.join(config.SAVE_DIR, 'checkpoints')) 
                               if f.startswith('checkpoint_epoch_')])
            for old_checkpoint in checkpoints[:-3]:
                os.remove(os.path.join(config.SAVE_DIR, 'checkpoints', old_checkpoint))
        
        # Early stopping
        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f'\nEarly stopping después de {epoch + 1} épocas sin mejora en la pérdida de validación')
            break
            
        # Logging en TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Mostrar progreso
        print(f'Época {epoch + 1}/{config.NUM_EPOCHS} - '
              f'Loss: {train_loss:.4f} (train) / {val_loss:.4f} (val) - '
              f'Acc: {train_acc:.2f}% (train) / {val_acc:.2f}% (val) - '
              f'Tiempo: {epoch_time//60:.0f}m {epoch_time%60:.0f}s - '
              f'ETA: {remaining_time//3600:.0f}h {(remaining_time%3600)//60:.0f}m - '
              f'Paciencia: {epochs_without_improvement}/{config.EARLY_STOPPING_PATIENCE}')
    
    # Guardar el modelo final
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, os.path.join(config.SAVE_DIR, 'final_model.pth'))
    
    total_time = time.time() - training_start_time
    print(f'\nEntrenamiento completado en {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m')
    print(f'Mejor precisión de validación: {best_acc:.2f}% en la época {best_epoch + 1}')
    
    # Cerrar el escritor de TensorBoard
    writer.close()

def print_system_info():
    import platform
    import subprocess
    
    print("\n=== Información del sistema ===")
    print(f"Sistema operativo: {platform.system()} {platform.release()}")
    print(f"Procesador: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
        print(f"Versión CUDA: {torch.version.cuda}")
        print(f"CuDNN versión: {torch.backends.cudnn.version()}")
    print("==============================\n")

if __name__ == '__main__':
    import os
    import signal
    import sys
    
    # Configurar variable de entorno para manejo de memoria de CUDA
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    def signal_handler(sig, frame):
        print('\nDeteniendo el entrenamiento...')
        # Limpiar memoria CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("1. Iniciando script...")
    print_system_info()
    
    # Verificar si hay GPU disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"2. Usando dispositivo: {device}")
    
    # Configurar multiprocesamiento
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError as e:
        print(f"Advertencia al configurar multiprocesamiento: {e}")
    
    mp.freeze_support()
    print("3. Soporte multiprocesamiento configurado")
    
    try:
        print("4. Iniciando función main...")
        main()
    except KeyboardInterrupt:
        print('\nEntrenamiento detenido por el usuario')
        sys.exit(0)
    except Exception as e:
        print(f"\nError en main: {e}")
        import traceback
        traceback.print_exc()
        # Limpiar memoria CUDA en caso de error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(1)
