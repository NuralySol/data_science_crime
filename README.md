# Crime Data and Social Media Spending (Data Science)

## Confustion Matrix Summary

A confusion matrix is a tool used to evaluate the performance of a classification model. It breaks down the results of classification into four categories: true positives, false positives, true negatives, and false negatives.

Key Terminology:

* **True Negatives** (TN): Correctly predicted that the actual class was No (0).
* **False Positives** (FP): Incorrectly predicted Yes (1) when the actual class was No (0).
* **False Negatives** (FN): Incorrectly predicted No (0) when the actual class was Yes (1).
* **True Positives** (TP): Correctly predicted that the actual class was Yes (1).

|                   | Predicted: No (0)    | Predicted: Yes (1)   |
|-------------------|----------------------|----------------------|
| **Actual: No (0)**| True Negatives (TN)   | False Positives (FP)  |
| **Actual: Yes (1)**| False Negatives (FN) | True Positives (TP)   |

## The Standard Scaler

> The Standard Scaler is a data preprocessing technique used to normalize or standardize features by removing the mean and scaling to unit variance. It is commonly used in machine learning to ensure that features are on a similar scale, which helps many algorithms (especially distance-based ones like K-Nearest Neighbors or Support Vector Machines) perform better.

## K-Nearest Neighbors (KNN)

> K-Nearest Neighbors is a popular machine learning algorithm used for both classification and regression tasks. The core idea behind KNN is to classify or predict the output for a given data point based on the outputs of its K nearest neighbors in the feature space.

## Support Vector Machine (SVM)

> Support Vector Machine (SVM) is a powerful supervised machine learning algorithm used for both classification and regression tasks, though it is mostly used for classification. The goal of an SVM is to find the best decision boundary (also called a hyperplane) that separates the classes in the dataset.

## A Decision Tree

> A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. It works by recursively splitting the dataset into subsets based on the most significant feature at each step, forming a tree-like structure where each node represents a decision based on the value of a feature.

## Cheat Sheets for Data Science

[Cheat Sheets for Data Science](https://www.theinsaneapp.com/2020/12/machine-learning-and-data-science-cheat-sheets-pdf.html#Data-Science-Cheat-Sheet)

The above **link** provides a comprehensive collection of cheat sheets for Data Science and Machine Learning topics. You can find cheat sheets on topics like machine learning algorithms, Python libraries, data wrangling, statistics, deep learning, and more. These resources are great for quick references and revising key concepts in data science.

### Gini Coefficient in Data Science

The **Gini Coefficient** is a measure of statistical dispersion that represents the inequality in a distribution, typically used in economics to measure income inequality, but also relevant in **data science** to understand the distribution of certain features.

#### Key Concepts

* **Range**: The Gini coefficient ranges from **0** (perfect equality) to **1** (maximum inequality).
  * A **Gini coefficient of 0** indicates a perfectly equal distribution (e.g., all individuals have the same income).
  * A **Gini coefficient of 1** indicates total inequality (e.g., one individual has all the income).

#### Applications in Data Science

1. **Imbalance in Target Classes**: The Gini Coefficient can help measure the imbalance in target classes, such as binary or multi-class classification problems. A higher Gini value may indicate an imbalance in class distribution.
2. **Decision Trees**: In **decision tree algorithms** like CART (Classification and Regression Trees), the **Gini Impurity** is used to measure the "purity" of nodes. It helps the tree decide where to split the data to achieve more homogeneous nodes.
   * **Gini Impurity Formula**:
     \[
     Gini = 1 - \sum_{i=1}^{n} p_i^2
     \]
     Where \(p_i\) is the probability of a sample belonging to class \(i\).

#### Example Usage

* In a **binary classification problem**, if a node has a Gini Impurity of 0, it means that all samples in the node belong to the same class, making it a "pure" node. A higher Gini Impurity indicates that the node contains a mix of classes.

#### Advantages

* **Simple Interpretation**: The Gini Coefficient is easy to interpret and helps visualize how evenly or unevenly a feature is distributed.

* **Widely Used**: Popular in both economics and machine learning, particularly in **decision trees**.

#### Gini vs. Entropy

* The **Gini Coefficient** is faster to compute than entropy and is more commonly used in decision tree algorithms like CART.

* **Entropy** measures uncertainty, while Gini focuses on class distribution inequality.
