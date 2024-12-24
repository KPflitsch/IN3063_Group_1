import os
from datasetLoader import load_cifar10

# Define the dataset path
DATASET_DIR = "task_1/dataset/cifar-10-batches-py"

# Load CIFAR-10
train_data, train_labels, test_data, test_labels = load_cifar10(DATASET_DIR)

# Print dataset info
print(f"Training data shape: {train_data.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Test labels shape: {test_labels.shape}")
