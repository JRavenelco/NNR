import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_cifar100_loaders
from model import SimpleCNN

# --- Parámetros de Entrenamiento ---
LEARNING_RATE = 0.001
BATCH_SIZE = 128 # Aumentado para un entrenamiento potencialmente más rápido si la GPU lo permite
NUM_EPOCHS = 50 # Incrementar para un mejor rendimiento, pero tarda más
MODEL_SAVE_PATH = './models_trained/'
MODEL_NAME = 'cifar100_simplecnn.pth'
DATA_DIR = './data'
LOG_DIR = './runs/cifar100_experiment' # Directorio para logs de TensorBoard

def train_model():
    print("Iniciando el proceso de entrenamiento...")

    # Inicializar TensorBoard SummaryWriter
    writer = SummaryWriter(LOG_DIR) 
    print(f"Los logs de TensorBoard se guardarán en: {os.path.abspath(LOG_DIR)}")

    # Verificar si CUDA está disponible y configurar el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    # Crear directorio para guardar modelos si no existe
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
        print(f"Directorio creado: {MODEL_SAVE_PATH}")

    # Cargar datos
    train_loader, test_loader, cifar100_classes = get_cifar100_loaders(batch_size=BATCH_SIZE, data_dir=DATA_DIR)
    num_classes = len(cifar100_classes)
    print(f"Número de clases detectadas: {num_classes}")

    # Inicializar modelo, función de pérdida y optimizador
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler para ajustar la tasa de aprendizaje (opcional, pero bueno para el rendimiento)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    print("Modelo, función de pérdida y optimizador inicializados.")
    # print(f"Modelo: {model}") # Comentado para no llenar la consola, se puede ver en TensorBoard

    # Añadir grafo del modelo a TensorBoard (opcional)
    # Se necesita un lote de entrada de ejemplo
    try:
        dataiter = iter(train_loader)
        dummy_images, _ = next(dataiter)
        writer.add_graph(model, dummy_images.to(device))
        print("Grafo del modelo añadido a TensorBoard.")
    except Exception as e:
        print(f"No se pudo añadir el grafo del modelo a TensorBoard: {e}")

    # Bucle de entrenamiento
    best_val_accuracy = 0.0
    start_time_total = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train() # Poner el modelo en modo entrenamiento
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Global step para TensorBoard (número total de lotes procesados)
        global_step_train = epoch * len(train_loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Poner a cero los gradientes del optimizador
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass y optimización
            loss.backward()
            optimizer.step()

            current_batch_loss = loss.item()
            running_loss += current_batch_loss * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Registrar pérdida de entrenamiento por lote en TensorBoard
            writer.add_scalar('Loss/train_batch', current_batch_loss, global_step_train + i)

            if (i + 1) % 100 == 0: # Imprimir progreso cada 100 lotes
                 train_acc_batch = 100 * correct_train / total_train
                 print(f'  Lote {i+1}/{len(train_loader)}, Pérdida: {loss.item():.4f}, Precisión ent. (lote): {train_acc_batch:.2f}%')

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_train = 100 * correct_train / total_train
        epoch_duration = time.time() - epoch_start_time

        print(f"Fin Epoch {epoch+1}/{NUM_EPOCHS} - Duración: {epoch_duration:.2f}s")
        print(f"  Pérdida de Entrenamiento: {epoch_loss:.4f}, Precisión de Entrenamiento: {epoch_acc_train:.2f}%")
        # Registrar métricas de entrenamiento por época en TensorBoard
        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch + 1)
        writer.add_scalar('Accuracy/train_epoch', epoch_acc_train, epoch + 1)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1) # Registrar LR actual

        # Validación
        model.eval() # Poner el modelo en modo evaluación
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # No necesitamos calcular gradientes durante la validación
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss_val_batch = criterion(outputs, labels)
                val_loss += loss_val_batch.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_accuracy = 100 * correct_val / total_val
        print(f"  Pérdida de Validación: {epoch_val_loss:.4f}, Precisión de Validación: {epoch_val_accuracy:.2f}%")
        # Registrar métricas de validación por época en TensorBoard
        writer.add_scalar('Loss/validation_epoch', epoch_val_loss, epoch + 1)
        writer.add_scalar('Accuracy/validation_epoch', epoch_val_accuracy, epoch + 1)

        # Ajustar tasa de aprendizaje con el scheduler (basado en la pérdida de validación)
        scheduler.step(epoch_val_loss)

        # Guardar el modelo si la precisión de validación es la mejor hasta ahora
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            save_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"  Mejor modelo guardado en {save_path} con precisión: {best_val_accuracy:.2f}%")
        print("-"*30)

    total_training_time = time.time() - start_time_total
    print("Entrenamiento completado.")
    print(f"Tiempo total de entrenamiento: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"Mejor precisión de validación: {best_val_accuracy:.2f}%")

    # Añadir hiperparámetros y métricas finales a TensorBoard (opcional)
    hparams = {'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'epochs': NUM_EPOCHS}
    final_metrics = {'hparam/best_validation_accuracy': best_val_accuracy,
                     'hparam/final_train_accuracy': epoch_acc_train, # última época, no necesariamente la mejor
                     'hparam/final_validation_loss': epoch_val_loss}
    # writer.add_hparams(hparams, final_metrics) # add_hparams puede ser un poco quisquilloso con las versiones
    # Alternativa más simple para guardar los hiperparámetros como texto:
    writer.add_text('Hyperparameters', str(hparams), 0)
    writer.add_text('Final Metrics', str(final_metrics), 0)

    # Cerrar el writer de TensorBoard
    writer.close()
    print("Logs de TensorBoard guardados y writer cerrado.")

if __name__ == '__main__':
    train_model()
