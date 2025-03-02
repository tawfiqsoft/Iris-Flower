# Iris-Flower
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load and explore the dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target
data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("First 5 rows of the dataset:")
print(data.head())

# Step 2: Visualize the data
sns.pairplot(data, hue='species', palette='Dark2')
plt.suptitle("Pair Plot of Iris Dataset", y=1.02)
plt.show()

# Step 3: Preprocess the data
X = iris.data  # Features
y = iris.target  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for k-NN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train a k-Nearest Neighbors (k-NN) classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = knn.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 7: Test the model with a new sample (example)
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input (sepal length, sepal width, petal length, petal width)
new_sample_scaled = scaler.transform(new_sample)
predicted_species = knn.predict(new_sample_scaled)
print(f"\nPredicted Species for New Sample: {iris.target_names[predicted_species][0]}")
