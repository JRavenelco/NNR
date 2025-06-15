from ultralytics import YOLO
import os

def main():
    """
    Main function to train the YOLOv8 model.
    """
    # Define the absolute path to the data.yaml file.
    # This makes the script runnable from any directory.
    try:
        # This assumes the script is in the 'src' folder and the dataset is in the root.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_yaml_path = os.path.join(project_root, 'WRO Round 2.v3-red_green_xpark_320.yolov8', 'data.yaml')

        # Check if the YAML file exists
        if not os.path.exists(data_yaml_path):
            print(f"Error: data.yaml not found at {data_yaml_path}")
            print("Please ensure the dataset is downloaded and in the project root directory.")
            return

        # Load a pre-trained YOLOv8 model. 'yolov8n.pt' is the smallest and fastest.
        # Other options: 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
        model = YOLO('yolov8n.pt')

        # Train the model using the dataset specified in data.yaml.
        # - epochs: Number of training epochs. 100 is a good starting point.
        # - imgsz: Image size. The dataset name suggests 320px.
        # - project: Directory to save results.
        # - name: Name for the specific training run folder.
        print(f"Starting training with dataset: {data_yaml_path}")
        results = model.train(
            data=data_yaml_path,
            epochs=100,
            imgsz=320,
            project='runs',
            name='wro_yolov8n_run1',
            exist_ok=True # Allows re-running the script and overwriting previous results
        )
        
        print("Training finished successfully.")
        # The results object is not a string, so we access the save_dir attribute for the path
        # In recent ultralytics versions, the final save directory is an attribute of the trainer.
        # However, the trainer object itself is a good thing to log.
        print(f"Model and results saved to the 'runs/wro_yolov8n_run1' directory.")

    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    main()
