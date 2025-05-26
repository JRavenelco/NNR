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
from torch.cuda.amp import autocast, GradScaler
import timm # For timm.utils.accuracy if available, or implement locally
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import json
from datetime import datetime
import signal
import sys
import platform
from tqdm import tqdm

from vit_model import vit_small_patch4_32, vit_base_patch4_32 # Assuming these are in vit_model.py
from data_loader import get_cifar100_loaders, CIFAR100_CLASSES, get_ood_loaders # Assuming get_ood_loaders exists
from utils import save_checkpoint as util_save_checkpoint, load_checkpoint as util_load_checkpoint, plot_metrics

# --- Helper Classes ---
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # Cambiado de np.Inf a np.inf para compatibilidad con NumPy 2.0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, epoch, optimizer, scheduler, scaler):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, scaler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, scaler)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer, scheduler, scaler):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'val_loss': val_loss
        }
        util_save_checkpoint(state, filename=self.path)
        self.val_loss_min = val_loss

# --- Configuración ---
class Config:
    # Hyperparámetros
    MODEL_TYPE = 'vit_small'  # 'vit_small' o 'vit_base'
    NUM_CLASSES = 100
    IMG_SIZE = 32
    PATCH_SIZE = 4
    BATCH_SIZE = 64  # Aumentado para mejor estabilidad
    NUM_EPOCHS = 300  # Más épocas para convergencia
    WARMUP_EPOCHS = 20  # Warmup más largo
    GRAD_ACCUM_STEPS = 1
    DROPOUT_RATE = 0.2
    DROP_PATH_RATE = 0.2
    EARLY_STOPPING_PATIENCE = 30  # Más paciencia para early stopping
    MIN_DELTA = 0.0001  # Más sensible a mejoras pequeñas
    
    # Optimizador
    OPTIMIZER = 'adamw'  # 'adamw' o 'sgd'
    LEARNING_RATE = 6e-4  # Tasa de aprendizaje más baja
    WEIGHT_DECAY = 0.05
    MOMENTUM = 0.9  # Para SGD
    
    # Scheduler
    LR_SCHEDULER = 'cosine_with_warmup'  # 'cosine', 'step', 'cosine_with_warmup'
    MIN_LR = 1e-6
    WARMUP_LR = 1e-7  # Tasa de aprendizaje inicial para warmup
    
    # Mezcla de datos mejorada
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4  # Más agresivo
    USE_CUTMIX = True
    CUTMIX_ALPHA = 1.2  # Más agresivo
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING = 0.1  # Suavizado de etiquetas
    
    # Detección OOD mejorada
    OOD_DATASETS = ['svhn', 'cifar10', 'texture']  # Múltiples conjuntos OOD
    OOD_METHODS = ['energy', 'max_prob', 'odin', 'mahalanobis', 'head']  # Múltiples métodos
    
    # Monitoreo de gradientes
    GRAD_CLIP = 1.0  # Clip de gradiente
    TRACK_GRADIENTS = True  # Monitorear estadísticas de gradientes
    
    # Otros
    NUM_WORKERS = 4  # Más workers para carga de datos más rápida
    USE_AMP = True
    LOG_INTERVAL = 50
    SAVE_EVERY = 10
    SAVE_DIR = '../models_trained/vit/'
    CHECKPOINT_DIR = '../checkpoints/vit/'
    LOG_DIR = f'../runs/vit_{MODEL_TYPE}_cifar100_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    DATA_DIR = '../data'

    def __init__(self):
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith('_')])

# --- Data Augmentation Functions (from train_resnet18_cifar100.py) ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# --- Model Creation ---
def create_vit_model(config):
    """
    Crea un modelo ViT según la configuración especificada.
    Los parámetros específicos del modelo ya están definidos en las funciones de fábrica.
    """
    if config.MODEL_TYPE == 'vit_small':
        model = vit_small_patch4_32(
            num_classes=config.NUM_CLASSES
        )
    elif config.MODEL_TYPE == 'vit_base':
        model = vit_base_patch4_32(
            num_classes=config.NUM_CLASSES
        )
    else:
        raise ValueError(f"Unsupported ViT model type: {config.MODEL_TYPE}")
    return model

# --- Training and Validation ---
def train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, epoch, config, device, writer):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    grad_norms = AverageMeter('Grad Norm', ':.4f')
    
    # Para seguimiento de gradientes
    if config.TRACK_GRADIENTS:
        grad_stats = {'mean': [], 'std': [], 'max': []}
    
    end = time.time()
    
    for i, (images, target) in enumerate(train_loader):
        # Medir tiempo de carga de datos
        data_time.update(time.time() - end)
        
        # Mover datos al dispositivo
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Mezcla de datos mejorada (MixUp o CutMix)
        if config.USE_MIXUP and np.random.rand() < 0.5:
            images, target_a, target_b, lam = mixup_data(images, target, config.MIXUP_ALPHA, device)
        elif config.USE_CUTMIX and np.random.rand() < 0.5:
            images, target_a, target_b, lam = cutmix_data(images, target, config.CUTMIX_ALPHA, device)
        else:
            target_a, target_b = target, target
            lam = 1.0
        
        # Entrenamiento con autocast para mixed precision
        with autocast(enabled=config.USE_AMP):
            # Forward pass
            output = model(images)
            
            # Calcular pérdida con label smoothing si está habilitado
            if config.USE_LABEL_SMOOTHING:
                smooth_loss = F.cross_entropy(output, target_a, label_smoothing=config.LABEL_SMOOTHING)
                if lam < 1.0:  # Si se usó MixUp/CutMix
                    smooth_loss = lam * smooth_loss + (1 - lam) * F.cross_entropy(
                        output, target_b, label_smoothing=config.LABEL_SMOOTHING)
                loss = smooth_loss
            else:
                if lam < 1.0:  # Si se usó MixUp/CutMix
                    loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
                else:
                    loss = criterion(output, target)
            
            # Escalar la pérdida para el gradiente acumulado
            loss = loss / config.GRAD_ACCUM_STEPS
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Actualizar pesos cada GRAD_ACCUM_STEPS pasos
        if (i + 1) % config.GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
            # Escalar gradientes y aplicar clip
            scaler.unscale_(optimizer)
            
            # Monitorear estadísticas de gradientes
            if config.TRACK_GRADIENTS:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                        
                        # Guardar estadísticas de gradientes
                        if i % config.LOG_INTERVAL == 0:
                            grad_stats['mean'].append(p.grad.detach().mean().item())
                            grad_stats['std'].append(p.grad.detach().std().item())
                            grad_stats['max'].append(p.grad.detach().abs().max().item())
                
                total_norm = total_norm ** 0.5
                grad_norms.update(total_norm)
            
            # Aplicar clip de gradiente
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
            
            # Actualizar parámetros
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Actualizar scheduler
            if scheduler is not None and config.LR_SCHEDULER == 'cosine_with_warmup':
                scheduler.step_update(epoch * len(train_loader) + i)
        
        # Calcular precisión
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item() * config.GRAD_ACCUM_STEPS, images.size(0))
        top1.update(acc1[0], images.size(0))
        
        # Medir tiempo de procesamiento
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Registrar en consola
        if i % config.LOG_INTERVAL == 0:
            log_msg = (f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t')
            
            if config.TRACK_GRADIENTS:
                log_msg += (f'Grad Norm {grad_norms.val:.4f} ({grad_norms.avg:.4f})\t')
                
                # Registrar estadísticas de gradientes
                if grad_stats['mean']:
                    writer.add_scalar('grad/mean', np.mean(grad_stats['mean']), epoch * len(train_loader) + i)
                    writer.add_scalar('grad/std', np.mean(grad_stats['std']), epoch * len(train_loader) + i)
                    writer.add_scalar('grad/max', np.max(grad_stats['max']), epoch * len(train_loader) + i)
                    grad_stats = {'mean': [], 'std': [], 'max': []}  # Reset stats
            
            log_msg += f'LR {optimizer.param_groups[0]["lr"]:.6f}'
            print(log_msg)
    
    # Registrar en TensorBoard
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/acc1', top1.avg, epoch)
    writer.add_scalar('train/grad_norm', grad_norms.avg, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    
    return losses.avg, top1.avg

def validate(model, val_loader, ood_loaders, criterion, config, device, writer, epoch):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # Para detección OOD con múltiples métodos y conjuntos
    ood_metrics = {}
    
    with torch.no_grad():
        # Validación en el conjunto de validación (in-distribution)
        all_logits, all_targets, all_features = [], [], []
        
        # Primera pasada: recolectar logits y características
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            
            with autocast(enabled=config.USE_AMP):
                features = model.forward_features(images)
                logits = model.head(features)
                loss = criterion(logits, targets)
            
            acc1 = accuracy(logits, targets, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_features.append(features.cpu())
    
    # Calculate statistics for Mahalanobis
    all_features = torch.cat(all_features).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Calculate class means and covariances for Mahalanobis
    class_mean = []
    for c in range(config.NUM_CLASSES):
        idx = (all_targets == c)
        if idx.sum() > 0:
            class_mean.append(all_features[idx].mean(0, keepdims=True))
    
    if class_mean:
        class_mean = np.concatenate(class_mean, axis=0)
        model.class_mean = torch.from_numpy(class_mean).to(device)
        
        # Calculate pooled covariance
        centered = all_features - class_mean[all_targets]
        cov = np.cov(centered, rowvar=False) + 1e-6 * np.eye(centered.shape[1])
        model.class_cov = torch.from_numpy(cov).to(device)
    
    # Evaluate on each OOD dataset
    if ood_loaders:
        for ood_name, ood_loader in ood_loaders.items():
            ood_metrics[ood_name] = {}
            
            # Collect OOD scores for each method
            id_scores = {method: [] for method in config.OOD_METHODS}
            ood_scores = {method: [] for method in config.OOD_METHODS}
            
            # Calculate scores for ID data
            for images, _ in val_loader:
                images = images.to(device)
                with autocast(enabled=config.USE_AMP):
                    features, logits, scores = model(images, return_features=True, ood_score=True)
                
                for method in config.OOD_METHODS:
                    id_scores[method].append(scores[method].cpu().numpy())
            
            # Calculate scores for OOD data
            for images, _ in ood_loader:
                images = images.to(device)
                with autocast(enabled=config.USE_AMP):
                    _, _, scores = model(images, return_features=True, ood_score=True)
                
                for method in config.OOD_METHODS:
                    ood_scores[method].append(scores[method].cpu().numpy())
            
            # Calculate metrics for each method
            for method in config.OOD_METHODS:
                try:
                    id_scores_concat = np.concatenate(id_scores[method])
                    ood_scores_concat = np.concatenate(ood_scores[method])
                    
                    # Create labels (0 for ID, 1 for OOD)
                    labels = np.concatenate([
                        np.zeros_like(id_scores_concat),
                        np.ones_like(ood_scores_concat)
                    ])
                    
                    # Concatenate scores
                    all_scores = np.concatenate([id_scores_concat, ood_scores_concat])
                    
                    # Calculate metrics
                    if len(np.unique(labels)) > 1:  # Need both classes for AUROC/AUPR
                        auroc = roc_auc_score(labels, all_scores)
                        aupr = average_precision_score(labels, all_scores)
                        fpr95 = fpr_at_95_tpr(labels, all_scores)
                        
                        ood_metrics[ood_name][f'{method}_auroc'] = auroc
                        ood_metrics[ood_name][f'{method}_aupr'] = aupr
                        ood_metrics[ood_name][f'{method}_fpr95'] = fpr95
                        
                        # Log to TensorBoard
                        writer.add_scalar(f'OOD/{ood_name}/{method}_auroc', auroc, epoch)
                        writer.add_scalar(f'OOD/{ood_name}/{method}_aupr', aupr, epoch)
                        writer.add_scalar(f'OOD/{ood_name}/{method}_fpr95', fpr95, epoch)
                    else:
                        print(f"Skipping OOD metrics for {method}: not enough diversity in OOD labels.")
                        ood_metrics[ood_name][f'{method}_auroc'] = 0.0
                        ood_metrics[ood_name][f'{method}_aupr'] = 0.0
                        ood_metrics[ood_name][f'{method}_fpr95'] = 1.0
                        
                except ValueError as e:
                    print(f"Error calculating OOD metrics for {method}: {e}")
    else:
        print("No OOD loaders provided, skipping OOD metrics.")
    
    # Calculate classification metrics
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    
    # Calculate per-class accuracy
    class_acc = {}
    preds = all_logits.argmax(dim=1)
    for c in range(config.NUM_CLASSES):
        idx = (all_targets == c)
        if idx.sum() > 0:
            class_acc[f'class_{c}_acc'] = (preds[idx] == all_targets[idx]).float().mean().item()
    
    # Log metrics
    print(f' * Val Loss {losses.avg:.4f} Acc@1 {top1.avg:.3f}')
    
    # Log classification metrics
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc1', top1.avg, epoch)
    
    # Log per-class accuracy
    for name, acc in class_acc.items():
        writer.add_scalar(f'val/class_acc/{name}', acc, epoch)
    
    # Log summary OOD metrics
    if ood_loaders:
        for ood_name, metrics in ood_metrics.items():
            for metric_name, value in metrics.items():
                if 'auroc' in metric_name:
                    writer.add_scalar(f'summary/ood_{ood_name}_auroc', value, epoch)
    
    # Return main metrics
    main_auroc = np.mean([m.get('energy_auroc', 0) for m in ood_metrics.values()] or [0])
    return losses.avg, top1.avg, main_auroc, 0.0  # AUPR not currently used

# --- Main Training Function ---
def train_with_config(config_dict=None, log_dir=None):
    # Permite pasar un diccionario de hiperparámetros para tuning
    class DynamicConfig(Config):
        def __init__(self, config_dict=None):
            super().__init__()
            if config_dict:
                for k, v in config_dict.items():
                    setattr(self, k, v)
    config = DynamicConfig(config_dict)
    if log_dir is not None:
        config.LOG_DIR = log_dir
    print("--- Configuration ---")
    print(config)
    print("---------------------")

    writer = SummaryWriter(log_dir=config.LOG_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Loaders
    print("Loading data...")
    train_loader, val_loader, _ = get_cifar100_loaders(
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS,
        data_dir=config.DATA_DIR,
        img_size=config.IMG_SIZE # Pass img_size for transforms
    )
    # Cargar múltiples conjuntos de datos OOD
    ood_loaders = {}
    if config.OOD_DATASETS:
        for ood_name in config.OOD_DATASETS:
            print(f"Loading OOD dataset: {ood_name}...")
            try:
                ood_loader = get_ood_loaders(ood_name, config.BATCH_SIZE, config.NUM_WORKERS, config.DATA_DIR, config.IMG_SIZE)
                ood_loaders[ood_name] = ood_loader
                print(f"  - {ood_name} loaded successfully")
            except Exception as e:
                print(f"  - Error loading {ood_name}: {str(e)}")

    # Model, Criterion, Optimizer, Scheduler
    print("Creating model...")
    model = create_vit_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else: # Default to SGD if not AdamW, or add more options
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY)

    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Scheduler
    if config.LR_SCHEDULER == 'cosine_with_warmup':
        warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=config.MIN_LR/config.LEARNING_RATE, end_factor=1.0, total_iters=config.WARMUP_EPOCHS)
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS, eta_min=config.MIN_LR)
        main_scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config.WARMUP_EPOCHS])
    else: # Fallback or other schedulers
        main_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    start_epoch = 0
    best_val_acc = 0.0
    best_val_auroc = 0.0

    # Early Stopping
    early_stopping_path = os.path.join(config.CHECKPOINT_DIR, f'{config.MODEL_TYPE}_early_stop_best_val_loss.pth.tar')
    early_stopper = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True, path=early_stopping_path)

    # Optionally load checkpoint
    latest_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{config.MODEL_TYPE}_latest.pth.tar')
    if os.path.exists(latest_checkpoint_path):
        print(f"Attempting to load latest checkpoint from: {latest_checkpoint_path}")
        start_epoch, loaded_val_loss, loaded_val_acc, loaded_val_auroc = util_load_checkpoint(
            latest_checkpoint_path, model, optimizer, main_scheduler, scaler, device
        )
        best_val_acc = loaded_val_acc
        best_val_auroc = loaded_val_auroc
        # Initialize early stopper's state if resuming
        if loaded_val_loss != float('inf'):
            early_stopper.val_loss_min = loaded_val_loss
            early_stopper.best_score = -loaded_val_loss # best_score is negative val_loss
        print(f"Resumed from epoch {start_epoch}. Best Val Acc: {best_val_acc:.2f}%, Best Val AUROC: {best_val_auroc:.4f}, Early Stopper Val Loss Min: {early_stopper.val_loss_min:.4f}")
    else:
        print(f"No latest checkpoint found at {latest_checkpoint_path}. Starting from scratch.")

    print("Starting training...")
    training_start_time = time.time()

    train_losses_hist, val_losses_hist = [], []
    train_accs_hist, val_accs_hist = [], []

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, main_scheduler, epoch, config, device, writer)
        val_loss, val_acc, val_auroc = validate(model, val_loader, ood_loaders, criterion, config, device, writer, epoch)

        train_losses_hist.append(train_loss)
        val_losses_hist.append(val_loss)
        train_accs_hist.append(train_acc)
        val_accs_hist.append(val_acc)

        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Train_Acc@1', train_acc, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Val_Acc@1', val_acc, epoch)
        if ood_loader: writer.add_scalar('Epoch/Val_AUROC', val_auroc, epoch)
        writer.add_scalar('Epoch/LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} Summary | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% AUROC: {val_auroc:.4f}")
        epoch_time_taken = time.time() - epoch_start_time
        print(f"Time for epoch {epoch+1}: {epoch_time_taken:.2f}s")

        # Save best model based on validation accuracy or AUROC
        current_metric_for_best = val_auroc if ood_loader else val_acc
        best_metric_so_far = best_val_auroc if ood_loader else best_val_acc

        if current_metric_for_best > best_metric_so_far:
            print(f"New best model found! Metric: {current_metric_for_best:.4f} (Epoch {epoch+1})")
            if ood_loader: best_val_auroc = current_metric_for_best
            else: best_val_acc = current_metric_for_best
            
            best_model_path = os.path.join(config.SAVE_DIR, f'{config.MODEL_TYPE}_best_metric.pth.tar')
            util_save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': main_scheduler.state_dict() if main_scheduler else None,
                'scaler_state_dict': scaler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'val_auroc': val_auroc,
                'config': vars(config)
            }, filename=best_model_path)

        # Early stopping check (based on validation loss)
        early_stopper(val_loss, model, epoch, optimizer, main_scheduler, scaler)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
        
        # Save latest checkpoint periodically or at the end of every epoch
        if (epoch + 1) % config.SAVE_EVERY == 0 or (epoch + 1) == config.NUM_EPOCHS:
            current_latest_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'{config.MODEL_TYPE}_latest.pth.tar')
            util_save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': main_scheduler.state_dict() if main_scheduler else None,
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss, # Save current val_loss for early stopper re-initialization
                'val_acc': val_acc,
                'val_auroc': val_auroc,
                'config': vars(config) # Save config for reference
            }, filename=current_latest_checkpoint_path)
            print(f"Latest checkpoint saved to {current_latest_checkpoint_path} at epoch {epoch+1}")

    total_training_time = time.time() - training_start_time
    print(f"Total training time: {total_training_time/3600:.2f} hours")
    writer.close()

    # Plot metrics
    if train_losses_hist and val_losses_hist and train_accs_hist and val_accs_hist:
        plot_metrics(train_losses_hist, val_losses_hist, train_accs_hist, val_accs_hist, len(train_losses_hist))

    print("Training finished.")
    # Devuelve los resultados clave para tuning
    return {
        'best_val_acc': best_val_acc,
        'best_val_auroc': best_val_auroc,
        'train_losses_hist': train_losses_hist,
        'val_losses_hist': val_losses_hist,
        'train_accs_hist': train_accs_hist,
        'val_accs_hist': val_accs_hist,
        'total_training_time': total_training_time
    }

# --- System Info and Entry Point ---
def print_system_info():
    print("\n=== System Information ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CuDNN Version: {torch.backends.cudnn.version()}")
    print("========================\n")

def signal_handler(sig, frame):
    print('\nTraining interrupted. Cleaning up...')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    print_system_info()
    
    # Set seed for reproducibility (optional)
    # torch.manual_seed(42)
    # np.random.seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    try:
        main()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(1)

# Permite importar train_with_config desde otros scripts
