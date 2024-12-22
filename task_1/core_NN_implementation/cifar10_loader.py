import numpy as np

# CIFAR-10 data loading and preprocessing
def preprocess_cifar10(train_data, train_labels, test_data, test_labels, num_classes=10):
    # Normalize pixel values to [0, 1]
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # Flatten image data
    train_data = train_data.reshape(train_data.shape[0], -1)  # (n_samples, 3072)
    test_data = test_data.reshape(test_data.shape[0], -1)

    # One-hot encode labels
    train_labels = one_hot_encode(train_labels, num_classes)
    test_labels = one_hot_encode(test_labels, num_classes)

    return train_data, train_labels, test_data, test_labels

def one_hot_encode(labels, num_classes):
    """Manual one-hot encoding."""
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot