import os
import time
import argparse
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
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy

# Configuración
class Config:
    # Entrenamiento
    batch_size = 64
    num_epochs = 200
    warmup_epochs = 5
    num_workers = 4
    # Modelo
    model_name = 'vit_base_patch16_224'  # Usaremos timm para cargar el modelo
    img_size = 224  # Tamaño de imagen para ViT
    num_classes = 100  # CIFAR-100
    
    # Optimizador
    learning_rate = 3e-4
    weight_decay = 0.05
    
    # Scheduler
    min_lr = 1e-6
    warmup_lr = 1e-6
    
    # Regularización
    drop_path = 0.1
    drop_rate = 0.1
    
    # Augmentations
    hflip = 0.5
    color_jitter = 0.4
    auto_augment = 'rand-m9-mstd0.5-inc1'
    
    # Mixup/Cutmix
    mixup = 0.8
    cutmix = 1.0
    mixup_prob = 1.0
    mixup_switch_prob = 0.5
    smoothing = 0.1
    
    # Checkpoints
    save_dir = 'checkpoints/vit'
    log_dir = 'runs/vit'


def get_transforms(config, is_training=False):
    if is_training:
        return transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.RandomHorizontalFlip(p=config.hflip),
            transforms.RandomCrop(config.img_size, padding=4, padding_mode='reflect'),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(config):
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    train_set = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=val_transform)
    
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, 
        shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size * 2, 
        shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    return train_loader, val_loader

def create_model(config):
    # Crear modelo ViT preentrenado
    model = timm.create_model(
        config.model_name,
        pretrained=True,
        num_classes=config.num_classes,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path,
        img_size=config.img_size
    )
    
    # Ajustar el tamaño del patch para CIFAR-100
    if hasattr(model, 'patch_embed'):
        model.patch_embed = timm.models.vision_transformer.PatchEmbed(
            img_size=config.img_size,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
        )
    
    return model

def train_epoch(model, loader, optimizer, criterion, scaler, scheduler, epoch, config, writer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step(epoch + batch_idx / len(loader))
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0:
            acc = 100. * correct / total
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%')
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100. * correct / total
    
    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/acc', avg_acc, epoch)
    
    return avg_loss, avg_acc

def validate(model, loader, criterion, epoch, writer=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100. * correct / total
    
    if writer:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/acc', avg_acc, epoch)
    
    return avg_loss, avg_acc

def main():
    # Configuración
    config = Config()
    
    # Directorios
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Datos
    train_loader, val_loader = get_dataloaders(config)
    
    # Modelo
    model = create_model(config).to(device)
    
    # Criterio
    if config.mixup > 0.0 or config.cutmix > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.smoothing)
    
    # Optimizador
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    num_steps = len(train_loader) * config.num_epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        step -= warmup_steps
        decay_steps = num_steps - warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = GradScaler()
    
    # TensorBoard
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Entrenamiento
    best_acc = 0.0
    start_epoch = 0

    # Cargar checkpoint si existe
    checkpoint_path = os.path.join(config.save_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        print(f"Reanudando entrenamiento desde {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('val_acc', 0.0) # Usar get para compatibilidad con checkpoints antiguos
        print(f"Checkpoint cargado. Época de inicio: {start_epoch}, Mejor Acc Val: {best_acc:.2f}%")
    else:
        print("No se encontró checkpoint. Iniciando entrenamiento desde cero.")
    
    for epoch in range(start_epoch, config.num_epochs):
        print(f'\nEpoch: {epoch+1}/{config.num_epochs}')
        
        # Entrenar
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, 
            scaler, scheduler, epoch, config, writer
        )
        
        # Validar
        val_loss, val_acc = validate(
            model, val_loader, criterion, epoch, writer
        )
        
        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(config.save_dir, 'best_model.pth'))
        
        print(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    
    writer.close()
    print(f'Mejor precisión de validación: {best_acc:.2f}%')

if __name__ == '__main__':
    import math
    main()
