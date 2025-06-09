import os
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator

def read_tfevents_file(file_path):
    """Read a TensorBoard event file and print its contents."""
    try:
        for event in summary_iterator(file_path):
            for value in event.summary.value:
                print(f"Step: {event.step}, Tag: {value.tag}, Value: {value.simple_value}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def find_tfevents_files(directory):
    """Find all TensorBoard event files in the given directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('events.out.tfevents'):
                yield os.path.join(root, file)

def main():
    log_dir = os.path.join(os.path.dirname(__file__), "runs")
    if not os.path.exists(log_dir):
        print(f"Directory not found: {log_dir}")
        return
    
    print(f"Looking for TensorBoard event files in: {log_dir}")
    event_files = list(find_tfevents_files(log_dir))
    
    if not event_files:
        print("No TensorBoard event files found.")
        return
    
    print(f"Found {len(event_files)} event files. Reading...\n")
    for file_path in sorted(event_files, key=os.path.getmtime, reverse=True):
        print(f"\nReading file: {file_path}")
        print("-" * 80)
        read_tfevents_file(file_path)

if __name__ == "__main__":
    main()
