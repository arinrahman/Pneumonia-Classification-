import numpy as np

# Load the files
train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')
test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')

# Print the shape of the arrays
print("Train features shape:", train_features.shape)
print("Train labels shape:", train_labels.shape)
print("Test features shape:", test_features.shape)
print("Test labels shape:", test_labels.shape)

# Accessing elements
print("\nFirst feature vector in training set:\n", train_features[0])
print("\nFirst label in training set:", train_labels[0])
