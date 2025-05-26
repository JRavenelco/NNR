import itertools
import csv
import os
from datetime import datetime
from train_vit import train_with_config

grid = {
    'LEARNING_RATE': [3e-4, 6e-4, 1e-3],
    'BATCH_SIZE': [64, 128],
    'DROPOUT_RATE': [0.1, 0.2, 0.3],
    'DROP_PATH_RATE': [0.1, 0.2, 0.3],
    'MIXUP_ALPHA': [0.2, 0.4, 0.8],
    'CUTMIX_ALPHA': [0.8, 1.0, 1.2],
    'LABEL_SMOOTHING': [0.0, 0.05, 0.1],
}

if __name__ == "__main__":
    # Genera todas las combinaciones posibles de hiperparámetros
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Archivo donde se guardarán los resultados
    results_file = 'results_vit_tuning.csv'

    # Si el archivo no existe, escribe el encabezado
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(grid.keys()) + ['best_val_acc', 'best_val_auroc', 'total_training_time'])

    for i, config in enumerate(combinations):
        print(f"\n===== Experimento {i+1}/{len(combinations)} =====")
        # Crea un log_dir único para cada experimento
        log_dir = f"../runs/tune_vit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_exp{i+1}"
        # Entrena el modelo con estos hiperparámetros
        result = train_with_config(config, log_dir=log_dir)
        # Guarda los resultados
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([config[k] for k in grid.keys()] + [result['best_val_acc'], result['best_val_auroc'], result['total_training_time']])
        print(f"Resultados guardados en {results_file}")

    print("\nTuning finalizado. Puedes revisar los resultados en el archivo CSV.")
