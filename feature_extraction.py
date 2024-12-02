import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Path to the chest_xray dataset
base_dir = "/Users/lutfarrahman/Desktop/chest_xray"

# Image parameters
IMG_SIZE = (224, 224)  # Input size for VGG16
BATCH_SIZE = 32

# Load the VGG16 model without the top layer (for feature extraction)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))


# Function to extract features
def extract_features(data_dir, batch_size=BATCH_SIZE):
    """
    Extracts features from images using a pre-trained VGG16 model.
    :param data_dir: Path to the dataset directory (e.g., train/test)
    :param batch_size: Batch size for processing images
    :return: Features and labels as numpy arrays
    """
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Create a generator to load images from subdirectories
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='binary',  # Assuming binary classification (NORMAL, PNEUMONIA)
        shuffle=False
    )

    # Use the pre-trained model to predict features
    features = base_model.predict(generator, verbose=1)
    labels = generator.classes  # Get the labels

    return features, labels


# Paths to train and test directories
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Extract features for training and testing datasets
print("Extracting features from training set...")
train_features, train_labels = extract_features(train_dir)

print("Extracting features from testing set...")
test_features, test_labels = extract_features(test_dir)

# Save the features and labels for later use
np.save('train_features.npy', train_features)
np.save('train_labels.npy', train_labels)
np.save('test_features.npy', test_features)
np.save('test_labels.npy', test_labels)

print("Feature extraction complete. Features saved to disk.")
