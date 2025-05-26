import torch
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Guarda el estado del modelo y del optimizador."""
    print("=> Guardando checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device='cpu'):
    """Carga el estado del modelo, optimizador, scheduler y scaler desde un checkpoint."""
    print(f"=> Cargando checkpoint desde '{checkpoint_path}'")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"Checkpoint no encontrado en {checkpoint_path}. No se carga nada.")
        return 0 # Devuelve epoch 0 si no se encuentra el checkpoint

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint: # Compatibilidad con formato antiguo
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("Advertencia: No se encontró 'model_state_dict' o 'state_dict' en el checkpoint.")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif optimizer and 'optimizer' in checkpoint: # Compatibilidad con formato antiguo
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif optimizer:
        print("Advertencia: No se encontró 'optimizer_state_dict' o 'optimizer' en el checkpoint para el optimizador.")

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    elif scheduler:
        print("Advertencia: No se encontró 'scheduler_state_dict' en el checkpoint para el scheduler.")

    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    elif scaler:
        print("Advertencia: No se encontró 'scaler_state_dict' en el checkpoint para el scaler.")

    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    best_val_acc = checkpoint.get('val_acc', 0.0)
    best_val_auroc = checkpoint.get('val_auroc', 0.0)
    
    print(f"Checkpoint cargado. Resumiendo desde la época {start_epoch}.")
    print(f"Valores del checkpoint: Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%, Val AUROC: {best_val_auroc:.4f}")

    # Devolver la época de inicio y las métricas relevantes para que el script de entrenamiento pueda usarlas
    return start_epoch, best_val_loss, best_val_acc, best_val_auroc

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    """Grafica las métricas de entrenamiento y validación."""
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Pérdida de Entrenamiento')
    plt.plot(epochs_range, val_losses, label='Pérdida de Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida de Entrenamiento y Validación')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Precisión de Entrenamiento')
    plt.plot(epochs_range, val_accuracies, label='Precisión de Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión (%)')
    plt.title('Precisión de Entrenamiento y Validación')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Ejemplo de cómo se podría usar plot_metrics después del entrenamiento en train.py:
# (Esto es solo un ejemplo, la lógica real estaría en train.py)
if __name__ == '__main__':
    print("Ejecutando ejemplo de utils.py - plot_metrics")
    # Datos de ejemplo
    num_epochs_example = 10
    example_train_losses = np.random.rand(num_epochs_example) * 2 + 0.5 # Pérdidas entre 0.5 y 2.5
    example_val_losses = np.random.rand(num_epochs_example) * 1.5 + 0.8   # Pérdidas entre 0.8 y 2.3
    example_train_accuracies = np.sort(np.random.rand(num_epochs_example) * 50 + 30) # Acc entre 30 y 80, creciente
    example_val_accuracies = np.sort(np.random.rand(num_epochs_example) * 40 + 25)   # Acc entre 25 y 65, creciente

    plot_metrics(example_train_losses, example_val_losses, 
                 example_train_accuracies, example_val_accuracies, 
                 num_epochs_example)
    print("Gráfica de ejemplo mostrada.")
