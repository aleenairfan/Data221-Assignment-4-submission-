from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()

X = data.data
y = data.target

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# The dataset has 569 samples and 30 features.
# Class 0 = malignant, Class 1 = benign.

# The dataset is slightly imbalanced (more benign than malignant cases).
# Class balance is important because models may become biased toward the majority class and fail to correctly predict minority cases.