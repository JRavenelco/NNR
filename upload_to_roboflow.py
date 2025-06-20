import os
import roboflow
import argparse
import time

# --- CONFIGURACIÓN ---
# Reemplaza con tu clave API de Roboflow
ROBOFLOW_API_KEY = "1QaioZ6lSVket4h0AjJP"
# Nombre de tu workspace en Roboflow (si es diferente al predeterminado, ajústalo)
# Puedes encontrarlo en la URL de tu dashboard de Roboflow, ej: app.roboflow.com/NOMBRE_WORKSPACE
ROBOFLOW_WORKSPACE_NAME = "roboteam-x6iuc" # Si es None, intentará usar el workspace predeterminado
# Nombre de tu proyecto en Roboflow
ROBOFLOW_PROJECT_NAME = "cifar_santana"
# Ruta base donde están las imágenes organizadas por torchvision
LOCAL_DATASET_PATH = os.path.join('data', 'cifar100_torchvision')
# --------------------

def upload_dataset_to_roboflow(api_key, workspace_name, project_name, local_path):
    """
    Sube un dataset de clasificación de imágenes a Roboflow.
    El dataset local debe estar organizado en carpetas:
    local_path/
        train/
            clase1/
                img1.jpg
                ...
            clase2/
                img1.jpg
                ...
        test/
            clase1/
                img1.jpg
                ...
            clase2/
                img1.jpg
                ...
    """
    print(f"Iniciando la subida del dataset a Roboflow...")
    print(f"  Workspace: {workspace_name if workspace_name else 'Predeterminado'}")
    print(f"  Proyecto: {project_name}")
    print(f"  Ruta local del dataset: {os.path.abspath(local_path)}")

    if not os.path.exists(local_path):
        print(f"Error: La ruta del dataset local no existe: {local_path}")
        return

    train_path = os.path.join(local_path, "train")
    test_path = os.path.join(local_path, "test")

    if not os.path.exists(train_path):
        print(f"Error: No se encontró la carpeta de entrenamiento en: {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"Error: No se encontró la carpeta de prueba en: {test_path}")
        return

    try:
        rf = roboflow.Roboflow(api_key=api_key)
        
        if workspace_name:
            # Si se proporciona un nombre/ID de workspace, se intenta acceder a él directamente.
            # El usuario debe asegurarse de que este 'workspace_name' sea el ID correcto o el slug de la URL.
            try:
                workspace = rf.workspace(workspace_name)
                print(f"Intentando usar el workspace especificado: {workspace_name}")
                
            except Exception as e:
                print(f"Error al acceder al workspace '{workspace_name}': {e}")
                print("Verifica que el nombre/ID del workspace sea correcto o déjalo vacío para usar el predeterminado.")
                return
        else:
            # Si no se especifica un nombre de workspace, se usa el predeterminado.
            try:
                workspace = rf.workspace() # Obtiene el workspace predeterminado
                print(f"Intentando usar el workspace predeterminado.")
                
            except Exception as e:
                print(f"Error al obtener el workspace predeterminado: {e}")
                print("Asegúrate de tener un workspace predeterminado configurado en Roboflow o especifica uno.")
                return

        project = None
        try:
            project = workspace.project(project_name)
            print(f"Proyecto '{project_name}' encontrado en el workspace.")
        except Exception:
            print(f"Proyecto '{project_name}' no encontrado en workspace '{workspace.name}'. Creándolo...")
            try:
                project = workspace.create_project(
                    project_name=project_name,
                    project_license="MIT",
                    project_type="classification",
                    annotation_group="CIFAR-100"
                )
                print(f"Proyecto '{project_name}' creado exitosamente.")
            except Exception as create_e:
                print(f"Error al crear el proyecto '{project_name}': {create_e}")
                return

        # Subir imágenes de entrenamiento
        print(f"\nSubiendo imágenes de ENTRENAMIENTO desde: {train_path}")
        workspace.upload_dataset(
            project_name=project.name,
            dataset_path=train_path,
            project_type="classification",
            batch_name=f"cifar100_train_upload_{int(time.time())}",
            num_workers=10
        )
        print("Imágenes de entrenamiento subidas.")

        # Subir imágenes de prueba
        print(f"\nSubiendo imágenes de PRUEBA desde: {test_path}")
        workspace.upload_dataset(
            project_name=project.name,
            dataset_path=test_path,
            project_type="classification",
            batch_name=f"cifar100_test_upload_{int(time.time())}",
            num_workers=10
        )
        print("Imágenes de prueba subidas.")

        print(f"\n¡Subida del dataset '{project_name}' completada!")
        print("Ahora puedes ir a la app de Roboflow para generar una versión de tu dataset y obtener el snippet de descarga.")

    except Exception as e:
        print(f"Ocurrió un error durante la subida a Roboflow: {e}")
        print("Asegúrate de que tu clave API es correcta y tiene permisos para el workspace y proyecto.")
        print("Verifica también que el nombre del workspace y proyecto sean correctos.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sube el dataset CIFAR-100 (organizado localmente) a Roboflow.")
    parser.add_argument("--api_key", type=str, default=ROBOFLOW_API_KEY, help="Tu clave API de Roboflow.")
    parser.add_argument("--workspace", type=str, default=ROBOFLOW_WORKSPACE_NAME, help="Nombre de tu workspace en Roboflow. Si es None, se usará el predeterminado o el primero encontrado.")
    parser.add_argument("--project", type=str, default=ROBOFLOW_PROJECT_NAME, help="Nombre de tu proyecto en Roboflow.")
    parser.add_argument("--path", type=str, default=LOCAL_DATASET_PATH, help="Ruta al directorio local del dataset (ej: data/cifar100_torchvision).")
    
    args = parser.parse_args()

    if args.api_key == "TU_API_KEY_AQUI" or not args.api_key:
        print("Error: Debes proporcionar tu clave API de Roboflow.")
        print("Puedes pasarla como argumento --api_key, o editar la variable ROBOFLOW_API_KEY en el script.")
    else:
        upload_dataset_to_roboflow(args.api_key, args.workspace, args.project, args.path)
