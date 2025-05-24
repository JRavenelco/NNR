import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class EarlyStopping:
    """Detiene el entrenamiento cuando la pérdida de validación no mejora."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): Número de épocas a esperar después de la última mejora.
            verbose (bool): Si es True, imprime un mensaje por cada validación.
            delta (float): Cambio mínimo para considerar una mejora.
            path (str): Ruta para guardar el modelo.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')  # Usando float('inf') en lugar de np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping contador: {self.counter} de {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Guarda el modelo cuando la pérdida de validación disminuye."""
        if self.verbose:
            print(f'Validación pérdida disminuyó ({self.val_loss_min:.6f} --> {val_loss:.6f}). Guardando modelo...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

from data_loader import get_cifar100_loaders, CIFAR100_CLASSES

# --- Configuración de Entrenamiento ---
class Config:
    # Tamaño de lote optimizado para la memoria GPU disponible (3GB)
    BATCH_SIZE = 128 if torch.cuda.is_available() else 64
    
    # Configuración de épocas
    NUM_EPOCHS = 300  # Aumentado a 300 épocas como en el modelo de referencia
    WARMUP_EPOCHS = 5  # Reducido ya que usaremos un warmup más corto
    
    # Hiperparámetros de optimización
    LEARNING_RATE = 0.1  # Aumentado a 0.1 como en el modelo de referencia
    MIN_LR = 1e-4  # Mínimo learning rate (ajustado para el rango más amplio)
    WEIGHT_DECAY = 5e-4  # Mantenido igual que en el modelo de referencia
    MOMENTUM = 0.9  # Momentum para SGD
    NESTEROV = True  # Usar momentum Nesterov para mejor convergencia
    
    # Configuración del optimizador
    OPTIMIZER = 'sgd'  # 'sgd' funciona mejor que 'adamw' para ResNets en CIFAR
    LABEL_SMOOTHING = 0.1  # Suavizado de etiquetas para mejor generalización
    GRAD_CLIP = 1.0  # Clip de gradiente
    
    # Configuración del scheduler - Usando CosineAnnealing con warmup
    LR_SCHEDULER = 'cosine'  # Cambiado a 'cosine' para seguir el modelo de referencia
    WARMUP_LR = 1e-4  # Learning rate inicial más bajo para warmup suave
    
    # Aumento de datos
    USE_MIXUP = True  # Habilitar MixUp
    MIXUP_ALPHA = 0.1  # Reducido para un efecto más sutil
    USE_CUTMIX = True  # Habilitar CutMix
    CUTMIX_ALPHA = 1.0  # Mantenido igual
    CUTMIX_PROB = 0.3  # Reducida la probabilidad para menos distorsión
    
    # Configuración del modelo
    NUM_CLASSES = 100  # Número de clases en CIFAR-100
    DROPOUT_RATE = 0.25  # Ligeramente reducido para permitir más capacidad de aprendizaje
    STOCHASTIC_DEPTH = 0.1  # Reducido para el entrenamiento más largo
    
    # Configuración de entrenamiento
    EARLY_STOPPING_PATIENCE = 15  # Reducido para ahorrar tiempo de entrenamiento
    GRAD_ACCUMULATION_STEPS = 2  # Acumular gradientes para simular batch size más grande
    
    # Rutas
    MODEL_SAVE_PATH = './models_trained/'
    CHECKPOINT_PATH = './checkpoints/'
    DATA_DIR = './data'
    LOG_DIR = f'./runs/cifar100_resnet18_advanced_{time.strftime("%Y%m%d_%H%M%S")}'
    
    # Configuración de rendimiento
    USE_AMP = True  # Mixed Precision Training
    NUM_WORKERS = 4 if torch.cuda.is_available() else 2
    PIN_MEMORY = torch.cuda.is_available()
    
    # Frecuencia de guardado y logging
    SAVE_EVERY = 5  # Guardar checkpoint cada N épocas
    LOG_INTERVAL = 50  # Frecuencia de logging
    
    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith('_')])

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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

def train_model():
    print("Iniciando el proceso de entrenamiento...")
    
    # Verificar disponibilidad de CUDA
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Dispositivo CUDA actual: {torch.cuda.current_device()}")
        print(f"Nombre del dispositivo: {torch.cuda.get_device_name(0)}")
    
    config = Config()
    print("1. Configuración cargada")
    
    # Configuración de directorios
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(config.LOG_DIR), exist_ok=True)
    
    # Configuración de TensorBoard
    writer = SummaryWriter(log_dir=config.LOG_DIR, flush_secs=10)
    print(f"Los logs de TensorBoard se guardarán en: {os.path.abspath(config.LOG_DIR)}")
    print("\nConfiguración:")
    print(str(config))

    # Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Cargar datos con aumentos mejorados
    print("\nCargando datos...")
    print("2. Iniciando carga de datos...")
    try:
        train_loader, val_loader, _ = get_cifar100_loaders(
            batch_size=config.BATCH_SIZE, 
            data_dir=config.DATA_DIR,
            use_mixup=config.USE_MIXUP,
            use_cutmix=config.USE_CUTMIX,
            cutmix_alpha=config.CUTMIX_ALPHA,
            cutmix_prob=config.CUTMIX_PROB
        )
        print("3. Datos cargados exitosamente")
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        raise
    
    # Inicializar el modelo ResNet-18 con pesos pre-entrenados
    print("\nInicializando modelo...")
    try:
        print("4. Cargando modelo ResNet-18...")
        model = models.resnet18(weights=None, num_classes=config.NUM_CLASSES)
        print("5. Modelo cargado exitosamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        raise
    
    # Modificar la primera capa para CIFAR-100 (imágenes de 32x32)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Eliminar maxpool inicial para mantener la resolución
    
    # Inicialización mejorada para las capas convolucionales
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Aplicar inicialización
    model.apply(init_weights)
    
    # Añadir dropout antes de la capa final
    model.fc = nn.Sequential(
        nn.Dropout(config.DROPOUT_RATE),
        nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    )
    
    # Implementar Stochastic Depth si está habilitado
    if hasattr(config, 'STOCHASTIC_DEPTH') and config.STOCHASTIC_DEPTH > 0:
        from torchvision.ops import StochasticDepth
        for block in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for basic_block in block:
                basic_block.conv1 = nn.Sequential(
                    StochasticDepth(p=config.STOCHASTIC_DEPTH, mode='batch'),
                    basic_block.conv1
                )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    # Guardar el número de características de entrada
    num_ftrs = model.fc.in_features
    
    # Crear la nueva capa fully connected
    new_fc = nn.Sequential(
        nn.Dropout(0.3),  # Reducido dropout para permitir más aprendizaje
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),  # Reducido dropout
        nn.Linear(512, config.NUM_CLASSES)
    )
    
    # Reemplazar la capa fully connected
    model.fc = new_fc
    
    # Inicializar las capas lineales
    for m in model.fc.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Descongelar las últimas capas para fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model = model.to(device)
    
    # Función de pérdida estándar (sin label smoothing inicialmente)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizador mejorado
    if config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.BASE_LR,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:  # SGD con Nesterov
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.BASE_LR,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            nesterov=config.NESTEROV  # Habilitar Nesterov para una convergencia más rápida
        )
    
    # Planificador de tasa de aprendizaje mejorado
    def get_lr_scheduler(optimizer, config, train_loader=None):
        """Crea un planificador de tasa de aprendizaje con warmup y coseno annealing"""
        if config.LR_SCHEDULER == 'cosine':
            # Crear un annealing coseno simple sin reinicios
            main_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS,  # Duración del decay
                eta_min=config.MIN_LR
            )
            
            # Crear un warmup lineal
            warmup_scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-4,  # Comenzar desde un LR muy bajo
                end_factor=1.0,
                total_iters=config.WARMUP_EPOCHS
            )
            
            # Combinar warmup con el scheduler principal
            return lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config.WARMUP_EPOCHS]
            )
            
        elif config.LR_SCHEDULER == 'onecycle' and train_loader is not None:
            # OneCycleLR con warmup integrado para comparación
            return lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.LEARNING_RATE,
                epochs=config.NUM_EPOCHS,
                steps_per_epoch=len(train_loader) // config.GRAD_ACCUMULATION_STEPS,
                pct_start=config.WARMUP_EPOCHS / config.NUM_EPOCHS,
                anneal_strategy='cos',
                final_div_factor=1e4,
                div_factor=25.0,
                three_phase=False
            )
            
        elif config.LR_SCHEDULER == 'plateau':
            # Reducción en meseta
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=10,
                verbose=True,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=0,
                min_lr=config.MIN_LR,
                eps=1e-8
            )
            
        # Planificador por defecto: paso de reducción
        return lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    
    # Obtener el scheduler configurado
    scheduler = get_lr_scheduler(optimizer, config, train_loader)
    # Inicializar GradScaler para mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP and device.type == 'cuda')
    
    # Inicializar early stopping
    early_stopping = EarlyStopping(
        patience=10, 
        verbose=True, 
        delta=0.001,
        path=os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
    )

    print("Modelo, función de pérdida y optimizador inicializados.")
    
    # Inicializar el mejor puntaje
    best_val_accuracy = 0.0
    
    # Añadir grafo del modelo a TensorBoard
    try:
        dataiter = iter(train_loader)
        dummy_images, _ = next(dataiter)
        
        # Bucle principal de entrenamiento
        print("\nIniciando entrenamiento...")
        start_time = time.time()
        best_epoch = 0
        
        for epoch in range(config.NUM_EPOCHS):
            # Entrenamiento
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            print('-' * 10)
            
            # Inicializar métricas
            epoch_loss = 0.0
            correct = 0
            total = 0
            val_loss = float('inf')
            val_accuracy = 0.0
            
            # Bucle de entrenamiento
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # Barra de progreso
            train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}', ncols=100)
            
            for batch_idx, (inputs, targets) in enumerate(train_loader_tqdm):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Verificar que los datos sean correctos (solo en el primer lote)
                if batch_idx == 0 and epoch == 0:  # Solo en la primera época
                    print(f"\nPrimer lote de entrenamiento:")
                    print(f"  - Forma de las entradas: {inputs.shape}")
                    print(f"  - Rango de valores de entrada: [{inputs.min():.3f}, {inputs.max():.3f}]")
                    print(f"  - Media de las entradas: {inputs.mean():.3f}")
                    print(f"  - Desviación estándar de las entradas: {inputs.std():.3f}")
                    print(f"  - Etiquetas: {targets[:10].cpu().numpy()}")
                
                # Gestionar la acumulación de gradientes
                if batch_idx % config.GRAD_ACCUMULATION_STEPS == 0:
                    optimizer.zero_grad()
                
                # Aplicar técnicas de aumento de datos avanzadas si están habilitadas
                apply_mixup = config.USE_MIXUP and np.random.rand() < 0.5
                apply_cutmix = config.USE_CUTMIX and np.random.rand() < config.CUTMIX_PROB
                
                if apply_mixup and not apply_cutmix:  # Aplicar MixUp
                    # Crear versión mezclada de las entradas y objetivos
                    lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
                    rand_index = torch.randperm(inputs.size()[0]).to(device)
                    mixed_inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]
                    targets_a, targets_b = targets, targets[rand_index]
                    
                    # Forward pass con mixed precision
                    with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                        outputs = model(mixed_inputs)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                
                elif apply_cutmix:  # Aplicar CutMix
                    # Obtener parámetros para el recorte
                    lam = np.random.beta(config.CUTMIX_ALPHA, config.CUTMIX_ALPHA)
                    rand_index = torch.randperm(inputs.size()[0]).to(device)
                    targets_a, targets_b = targets, targets[rand_index]
                    
                    # Generar el recuadro para el recorte
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    
                    # Aplicar el recorte a las imágenes
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    
                    # Ajustar lambda para que coincida con la proporción de píxeles
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    
                    # Forward pass con mixed precision
                    with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                        outputs = model(inputs)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                
                else:  # Entrenamiento normal sin MixUp ni CutMix
                    # Forward pass con mixed precision
                    with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                
                # Normalizar la pérdida por el número de pasos de acumulación
                loss = loss / config.GRAD_ACCUMULATION_STEPS
                
                # Backward pass con escalado de gradiente
                scaler.scale(loss).backward()
                
                # Actualizar los pesos solo cuando se haya acumulado el número de pasos configurado
                if (batch_idx + 1) % config.GRAD_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    # Aplicar recorte de gradiente
                    if config.GRAD_CLIP > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                    
                    # Actualizar pesos
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Actualizar el scheduler si es de tipo OneCycle que se actualiza por paso
                    if config.LR_SCHEDULER == 'onecycle':
                        scheduler.step()
                
                # Calcular métricas
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                epoch_loss += loss.item() * inputs.size(0)
                
                # Actualizar barra de progreso
                train_loader_tqdm.set_postfix({
                    'loss': epoch_loss / total,
                    'acc': 100. * correct / total
                })
            
            # Calcular métricas de entrenamiento
            train_loss = epoch_loss / len(train_loader.dataset)
            train_accuracy = 100. * correct / total
            
            # Validación
            val_loss, val_accuracy = validate(model, val_loader, criterion, device, config)
            
            # Imprimir métricas
            print(f'\nÉpoca {epoch+1}/{config.NUM_EPOCHS}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            # Actualizar el scheduler
            if config.LR_SCHEDULER == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Guardar el mejor modelo basado en la precisión de validación
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model_accuracy.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                }, best_model_path)
                print(f'\nMejor modelo guardado en {best_model_path} con precisión: {val_accuracy:.2f}%')
            
            # Early stopping basado en la pérdida de validación
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f'\nEarly stopping en la época {epoch+1} (sin mejora en {early_stopping.patience} épocas)')
                break
        
        # Guardar el modelo final
        final_model_path = os.path.join(config.MODEL_SAVE_PATH, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f'\nEntrenamiento completado en {(time.time() - start_time)/3600:.2f} horas')
        print(f'Mejor precisión de validación: {best_val_accuracy:.2f}% en la época {best_epoch+1}')
        print(f'Modelo final guardado en: {os.path.abspath(final_model_path)}')
        
        # Cerrar el writer de TensorBoard
        writer.close()
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Asegurarse de cerrar el writer de TensorBoard en caso de error
        writer.close()

def validate(model, val_loader, criterion, device, config):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with amp.autocast(enabled=config.USE_AMP and device.type == 'cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def add_model_graph(writer, model, dummy_input):
    try:
        writer.add_graph(model, dummy_input.to(next(model.parameters()).device))
        print("Grafo del modelo añadido a TensorBoard.")
    except Exception as e:
        print(f"No se pudo añadir el grafo del modelo a TensorBoard: {e}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler_cosine, scheduler_plateau, device, config, scaler=None):
    """
    Carga un checkpoint del modelo.
    
    Args:
        checkpoint_path: Ruta al archivo de checkpoint
        model: Modelo a cargar
        optimizer: Optimizador a cargar
        scheduler_cosine: Scheduler de coseno a cargar
        scheduler_plateau: Scheduler de plateau a cargar
        device: Dispositivo a utilizar
        config: Configuración del entrenamiento
        scaler: Objeto GradScaler para mixed precision (opcional)
        
    Returns:
        epoch: Época en la que se guardó el checkpoint
    """
    if os.path.isfile(checkpoint_path):
        print(f"Cargando checkpoint '{checkpoint_path}'")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Cargar los estados del modelo y optimizador
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Cargar los schedulers si están disponibles
            if 'scheduler_cosine' in checkpoint and scheduler_cosine is not None:
                scheduler_cosine.load_state_dict(checkpoint['scheduler_cosine'])
            if 'scheduler_plateau' in checkpoint and scheduler_plateau is not None:
                scheduler_plateau.load_state_dict(checkpoint['scheduler_plateau'])
            
            # Cargar el scaler si se está usando mixed precision
            if config.USE_AMP and 'scaler' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            
            print(f"Checkpoint cargado en la época {checkpoint['epoch']}")
            return checkpoint['epoch']
            
        except Exception as e:
            print(f"Error al cargar el checkpoint: {str(e)}")
            return 0
    else:
        print(f"No se encontró el checkpoint en {checkpoint_path}")
        return 0

if __name__ == '__main__':
    train_model()
