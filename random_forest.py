import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Load the saved features and labels
train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')
test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')

# Flatten the features
train_features_flat = train_features.reshape(train_features.shape[0], -1)  # Flatten to (num_samples, 25088)
test_features_flat = test_features.reshape(test_features.shape[0], -1)    # Flatten to (num_samples, 25088)

print("Features and labels loaded and flattened successfully!")
print("Train Features Shape:", train_features_flat.shape)
print("Train Labels Shape:", train_labels.shape)
print("Test Features Shape:", test_features_flat.shape)
print("Test Labels Shape:", test_labels.shape)

# Parameters for Random Forest
estimators = [30, 100, 500]

for n_estimators in estimators:
    start_train = time.time()
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(train_features_flat, train_labels)  # Training
    elapsed_train = time.time() - start_train

    start_test = time.time()
    y_pred = rf.predict(test_features_flat)  # Testing
    elapsed_test = time.time() - start_test

    # Evaluate the model
    acc = accuracy_score(test_labels, y_pred)
    cm = confusion_matrix(test_labels, y_pred)

    print(f"Number of Estimators={n_estimators}")
    print(f"Elapsed time training={elapsed_train:.4f} secs")
    print(f"Elapsed time testing={elapsed_test:.4f} secs")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion matrix:\n{cm}\n")
