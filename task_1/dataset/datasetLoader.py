import os
import pickle
import numpy as np

def test_IDE_issue():
    print("This new function is recognised and working")

def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']
    labels = np.array(batch['labels'])
    return data, labels

def load_cifar10_data(data_dir):
    X_train, y_train = [], []

    for i in range (1, 6):
        data, labels = load_cifar10_batch(f"{data_dir}/data_batch_{i}")
        X_train.append(data)
        y_train.append(labels)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = load_cifar10_batch(f"{data_dir}/test_batch")

    return X_test, X_train, y_test, y_train

def one_hot_encode(labels, num_classes=10):
    labels = np.array(labels)
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

def load_preprocess_cifar10(data_dir):
    X_train, X_test, y_train, y_test = load_cifar10_data(data_dir)

    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    mean = np.mean(X_train, axis=(0, 1, 2))
    std = np.std(X_train, axis=(0, 1, 2))
    X_train = (X_train / 255.0 - mean) / std
    X_test = (X_test / 255.0 - mean) / std

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return X_train, y_train, X_test, y_test



