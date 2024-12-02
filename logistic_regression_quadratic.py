import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
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

# Create quadratic features
poly = PolynomialFeatures(degree=2, include_bias=False)  # Generate quadratic features
train_features_quad = poly.fit_transform(train_features_flat)  # Transform training features
test_features_quad = poly.transform(test_features_flat)  # Transform test features

# Logistic Regression for Quadratic Features
start_train = time.time()
logistic_quad = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
logistic_quad.fit(train_features_quad, train_labels)  # Training
elapsed_train = time.time() - start_train

start_test = time.time()
y_pred_quad = logistic_quad.predict(test_features_quad)  # Testing
elapsed_test = time.time() - start_test

# Compute metrics
acc_quad = accuracy_score(test_labels, y_pred_quad)
cm_quad = confusion_matrix(test_labels, y_pred_quad)

# Display results
print("Logistic Regression with Quadratic Features")
print(f"Elapsed time training={elapsed_train:.4f} secs")
print(f"Elapsed time testing={elapsed_test:.4f} secs")
print(f"Accuracy: {acc_quad:.4f}")
print(f"Confusion matrix:\n{cm_quad}\n")
