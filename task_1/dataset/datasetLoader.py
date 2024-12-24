import os
import pickle
import numpy as np

def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        return data.reshape(-1, 3, 32, 32).astype(np.float32), np.array(labels)

def load_cifar10(data_dir):
    train_data, train_labels = [], []
    for i in range(1,6):
        #file_path = os.path.join(data_dir, 'cifar-10-batches-py')
        file_path = os.path.join(data_dir, f"data_batch_{i}")
        data, labels = load_cifar10_batch(file_path)
        train_data.append(data)
        train_labels.append(labels)

    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, "test_batch"))

    return train_data, train_labels, test_data, test_labels



