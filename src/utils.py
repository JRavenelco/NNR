import torch
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Guarda el estado del modelo y del optimizador."""
    print("=> Guardando checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer=None):
    """Carga el estado del modelo y opcionalmente del optimizador."""
    print("=> Cargando checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    # También se podría devolver la época, etc.
    # return checkpoint.get('epoch', 0), checkpoint.get('best_accuracy', 0.0)

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
