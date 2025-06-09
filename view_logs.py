import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_logs(log_dir):
    # Find all event files in the log directory
    event_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print("No TensorBoard event files found in", log_dir)
        return
    
    # Sort by modification time (newest first)
    event_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Found {len(event_files)} event files. Loading the most recent one...")
    print("File:", event_files[0])
    
    # Load the most recent event file
    event_acc = EventAccumulator(os.path.dirname(event_files[0]))
    event_acc.Reload()
    
    # Print available tags
    print("\nAvailable tags:", event_acc.Tags())
    
    # Print scalar data
    for tag in event_acc.Tags()['scalars']:
        print(f"\nTag: {tag}")
        events = event_acc.Scalars(tag)
        for event in events:
            print(f"  Step {event.step}: {event.value}")

if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(__file__), "runs")
    load_tensorboard_logs(log_dir)
