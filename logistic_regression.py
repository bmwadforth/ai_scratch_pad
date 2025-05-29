# 1. Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch

# 2. Load the dataset
# The Iris dataset is a classic dataset in machine learning and statistics.
# It contains 150 samples of iris flowers, each with 4 features (sepal length, sepal width, petal length, petal width)
# and a target variable indicating the species of iris (Setosa, Versicolour, Virginica).
iris = load_iris()
X, y = iris.data, iris.target

# 3. Split the data into training and testing sets
# We'll use 80% of the data for training the model and 20% for testing its performance.
# random_state ensures that the split is the same every time you run the code, for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the model
# We'll use Logistic Regression, which is a simple and effective algorithm for classification tasks.
# We set max_iter to a higher value to ensure convergence for this dataset.
model = LogisticRegression(max_iter=200)

# 5. Train the model
# The model learns the relationship between the features (X_train) and the target variable (y_train).
model.fit(X_train, y_train)

# 6. Make predictions on the test set
# The trained model predicts the species for the unseen test data.
y_pred = model.predict(X_test)

# 7. Evaluate the model
# We compare the predicted species (y_pred) with the actual species (y_test) to calculate the accuracy.
accuracy = accuracy_score(y_test, y_pred)

# Print the results
print(f"Features (first 5 rows):\n{X_train[:5]}")
print(f"\nTarget (first 5 training labels):\n{y_train[:5]}")
print(f"\nPredicted labels for the test set:\n{y_pred}")
print(f"Actual labels for the test set:\n{y_test}")
print(f"\nAccuracy of the Logistic Regression model: {accuracy:.2f}")

# You can also see the target names (species of Iris)
print(f"\nIris species: {iris.target_names}")



x = torch.rand(10000,3)
print(x)