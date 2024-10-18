import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Load the iris dataset
data = load_iris()

# Convert to a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Map target values to flower names and add a new column 'flower' for better variable key calls (readability)
df["flower"] = df["target"].apply(lambda x: data.target_names[x])

# Print the first few rows of the DataFrame
print(df.head())

# Split the data into training and test sets (exclude the 'flower' column)
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["flower", "target"]), df["target"], test_size=0.2, random_state=42
)

# Initialize the KNN model with K=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict the classes of the test set
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Scatter plot of sepal length vs sepal width, colored by flower type
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="flower",
    data=df,
    palette="Set1",
    s=100,
)

# Add title and labels
plt.title("Sepal Length vs Sepal Width by Flower Type")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")

# Show the plot of
plt.show()

# Pairplot to show relationships between all features, colored by flower type
sns.pairplot(df, hue="flower", palette="Set1", markers=["o", "s", "D"])
plt.suptitle("Pairplot of All Features by Flower Type", y=1.02)  # Adjust title position
plt.show()

# FacetGrid for petal length and sepal length, with hue as flower type and distinct colors for each
palette = {
    "setosa": "red",
    "versicolor": "green",
    "virginica": "blue",
}  # Define custom palette for each flower type

g = sns.FacetGrid(df, col="flower", hue="flower", palette=palette, height=4, aspect=1)
g.map(sns.scatterplot, "petal length (cm)", "sepal length (cm)").add_legend()

g.set_axis_labels("Petal Length (cm)", "Sepal Length (cm)")
plt.suptitle("FacetGrid of Petal Length vs Sepal Length by Flower Type", y=1.02)

# Show the plot
plt.show()

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

#! Generate the confusion matrix (confusion matrix to see how the model)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

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
plt.title("Confusion Matrix for KNN Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

#! Summary of the confusion matrix of the above!
# 1. True Setosa (Row 1):
# •	10 samples of Setosa were correctly classified as Setosa (True Positive).
# •	0 samples of Setosa were misclassified as Versicolor (False Positive for Versicolor).
# •	0 samples of Setosa were misclassified as Virginica (False Positive for Virginica).
# 2.	True Versicolor (Row 2):
# •	9 samples of Versicolor were correctly classified as Versicolor (True Positive).
# •	0 samples of Versicolor were misclassified as Setosa (False Positive for Setosa).
# •	1 sample of Versicolor was misclassified as Virginica (False Positive for Virginica).
# 3.	True Virginica (Row 3):
# •	8 samples of Virginica were correctly classified as Virginica (True Positive).
# •	2 samples of Virginica were misclassified as Versicolor (False Positive for Versicolor).
# •	0 samples of Virginica were misclassified as Setosa (False Positive for Setosa).
