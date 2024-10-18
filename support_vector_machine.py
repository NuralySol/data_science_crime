import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
data = datasets.load_iris()

# Convert to a DataFrame for easier handling
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns="target"), df["target"], test_size=0.2, random_state=42
)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM model (using the RBF kernel by default)
svm_model = SVC(
    kernel="rbf", gamma="scale"
)  # 'rbf' is the Radial Basis Function kernel
svm_model.fit(X_train_scaled, y_train)

# Predict the classes of the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names,
)
plt.title("Confusion Matrix for SVM Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 1.	True Setosa (First Row):
# 	•	10 samples of Setosa were correctly predicted as Setosa.
# 	•	0 samples of Setosa were incorrectly predicted as Versicolor or Virginica.
# 	•	Conclusion: The model perfectly classified all Setosa samples.
# 	2.	True Versicolor (Second Row):
# 	•	9 samples of Versicolor were correctly classified as Versicolor.
# 	•	1 sample of Versicolor was misclassified as Virginica.
# 	•	Conclusion: Most Versicolor samples were classified correctly, but there was 1 misclassification.
# 	3.	True Virginica (Third Row):
# 	•	10 samples of Virginica were correctly classified as Virginica.
# 	•	0 samples of Virginica were misclassified.
# 	•	Conclusion: The model perfectly classified all Virginica samples.

# Key Metrics Derived from the Confusion Matrix:

# 	1.	True Positives (TP):
# 	•	These are cases where the model correctly predicted the class.
# 	•	For example, the model correctly predicted 10 Setosa, 9 Versicolor, and 10 Virginica samples.
# 	2.	False Positives (FP):
# 	•	These are cases where the model incorrectly predicted a class.
# 	•	For example, 1 Versicolor sample was incorrectly predicted as Virginica.
# 	3.	False Negatives (FN):
# 	•	These are cases where the model failed to predict the correct class.
# 	•	For example, 1 Versicolor sample was classified as Virginica instead of Versicolor.
# 	4.	Accuracy:
# 	•	Accuracy is the ratio of correctly predicted observations to the total observations:
