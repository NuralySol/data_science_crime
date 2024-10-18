import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict the classes of the test set
y_pred = dt_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Increase figure size to provide more space for the plot and the explanation
plt.figure(figsize=(14, 10))  # Increase the width of the figure for better fitting

# Plot the decision tree with additional details
plot_tree(
    dt_model,
    feature_names=data.feature_names,  # Adds feature names to each node
    class_names=data.target_names,  # Adds class names to leaf nodes
    precision=4,  # Shows more precision for impurity and other values
    filled=True,  # Fills nodes with colors based on class
    rounded=True,  # Rounds the edges of the nodes
    proportion=True,  # Proportionally scales nodes based on the number of samples
)

# Title and description for the tree plot
plt.title("Decision Tree for Iris Dataset")

#! Gini Impurity is a measure used in decision trees (both classification and regression trees) to determine how pure or impure a node is. It helps the algorithm decide how to split the data at each node, with the goal of finding splits that increase the homogeneity (purity) of the nodes.

# Add explanatory text about decision tree attributes
explanation_text = """
Gini: A measure of node purity (0 = pure, 0.5 = impure).
Samples: Number of samples that reach this node.
Value: Number of samples from each class that reach this node.
Class: The predicted class label at the node.
"""

# Adjust the position of the explanation text to fit within the plot area
plt.figtext(
    0.55,
    0.7,
    explanation_text,
    horizontalalignment="left",
    fontsize=10,
    bbox=dict(facecolor="lightgrey", alpha=0.5),
)

# Show the plot
plt.show()

#! Summary of Decision Tree Implementation on Iris Dataset:

# In this implementation, we applied a Decision Tree Classifier to the Iris dataset and visualized the resulting decision tree using Scikit-Learn’s plot_tree function. Here’s a breakdown of the process and key outcomes:

# 1. Loading the Iris Dataset:

# 	•	We used the well-known Iris dataset, which consists of 150 samples of iris flowers, categorized into three classes: Setosa, Versicolor, and Virginica.
# 	•	Each sample has four features: sepal length, sepal width, petal length, and petal width.

# 2. Data Preparation:

# 	•	We split the dataset into training (80%) and test (20%) sets to train and evaluate the model.

# 3. Training the Decision Tree Classifier:

# 	•	We used Scikit-Learn’s DecisionTreeClassifier to build a decision tree model on the training data. The model recursively splits the dataset into subsets based on the features that best separate the classes.

# 4. Model Evaluation:

# 	•	After training the model, we evaluated it on the test set:
# 	•	Accuracy: The accuracy of the model was printed, which typically ranges between 90-100% on this dataset due to its simplicity.
# 	•	Classification Report: We printed a classification report that shows:
# 	•	Precision: The proportion of correct predictions for each class.
# 	•	Recall: The proportion of actual positives correctly predicted.
# 	•	F1-score: The harmonic mean of precision and recall.
# 	•	Support: The number of instances for each class.

# 5. Decision Tree Visualization:

# 	•	We used Scikit-Learn’s plot_tree() to generate a visual representation of the decision tree.
# 	•	The plot displays how the data is split at each node based on different feature values (e.g., petal length > 2.5 cm).
# 	•	Each node shows:
# 	•	The feature and threshold used for the split.
# 	•	The Gini impurity, which measures the node’s homogeneity.
# 	•	The number of samples in the node.
# 	•	The predicted class at the leaf nodes.
# 	•	Coloring in the nodes helps visualize which class each node or leaf predominantly belongs to.

# Key Takeaways:

# 	•	Accuracy: The decision tree achieved an accuracy of around 100% on the Iris dataset test set.
# 	•	Classification Report: The precision, recall, and F1-score for each of the three classes (Setosa, Versicolor, Virginica) were all perfect due to the simplicity of the dataset.
# 	•	Decision Tree Plot: The decision tree plot provides a clear, interpretable visual of how the model is making decisions based on feature values. It shows how the model splits the data and classifies each sample into one of the three iris species.

# Use Case of Decision Trees:

# 	•	Interpretability: Decision trees are highly interpretable and provide a clear understanding of how the model makes decisions, making them useful in fields like healthcare, finance, and business decision-making.
# 	•	No Need for Feature Scaling: Unlike some machine learning models (e.g., KNN), decision trees don’t require feature scaling or normalization.

#! Gini Impurity.

# 1.	Gini Impurity of 0: The node is pure, meaning all the samples in that node belong to the same class.
# 	•	Example: If a node contains only samples of the class “Setosa,” the Gini Impurity is 0, indicating that the node is perfectly pure.
# 	2.	Gini Impurity close to 0.5: The node is highly impure, meaning the samples are split evenly among multiple classes.
# 	•	Example: If a node contains 50% “Setosa” and 50% “Versicolor,” the Gini Impurity would be 0.5, indicating a high level of impurity.
# 	3.	Gini Impurity ranges between 0 and 0.5:
# 	•	The Gini Impurity score is always between 0 (perfect purity) and 0.5 (maximum impurity for binary classification). For multi-class classification, the upper limit approaches 1.
