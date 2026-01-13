# File: src/data/cifar_loader.py
import pickle
import numpy as np
import os

def unpickle(file):
    """Helper provided by CIFAR to load the raw pickle files."""
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def load_cifar_batch(file_path):
    """Loads a single CIFAR batch and reshapes it for plotting/training."""
    batch_content = unpickle(file_path)
    
    # Extract data and labels
    raw_images = batch_content[b'data']
    raw_labels = batch_content[b'labels']
    
    # Reshape to (Batch_Size, 3, 32, 32) then Transpose to (Batch_Size, 32, 32, 3)
    images = raw_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return images, raw_labels

def get_label_names(root_path):
    """Loads the metadata to get string class names."""
    meta_file = os.path.join(root_path, 'batches.meta')
    meta_data = unpickle(meta_file)
    return [label.decode('utf-8') for label in meta_data[b'label_names']]
