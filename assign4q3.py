from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

dt = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
dt.fit(X_train, y_train)

print("Training Accuracy:", accuracy_score(y_train, dt.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, dt.predict(X_test)))

importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 5 Features:")
for i in range(5):
    print(data.feature_names[indices[i]], importances[indices[i]])

# Limiting tree depth reduces overfitting.
# Feature importance helps interpret which variables influence decisions.