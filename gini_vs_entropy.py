import numpy as np
import matplotlib.pyplot as plt


def gini_index(p):
    return 1 - np.sum(p**2)


def entropy(p):
    p_log2_p = np.where(p > 0, p * np.log2(p), 0)
    return -np.sum(p_log2_p)


# Generate sample data with varying impurity
probabilities = np.linspace(0.01, 0.99, 50)  # Array from 0.01 to 0.99 for one class
data = np.array([probabilities, 1 - probabilities])  # Two classes

# Calculate Gini Index and Entropy
gini_values = [gini_index(p) for p in data.T]
entropy_values = [entropy(p) for p in data.T]

# Plotting
plt.figure(figsize=(8, 5))

plt.plot(probabilities, gini_values, label="Gini Index")
plt.plot(probabilities, entropy_values, label="Entropy")

plt.xlabel("Probability of Class 1")
plt.ylabel("Impurity Measure")
plt.title("Gini Index and Entropy vs. Class Probability")

plt.legend()
plt.grid(True)
plt.show()

#! Gini vs Entropy in class probabilities.

# In the above example, we calculate and visualize how the Gini Index and Entropy change with varying class probabilities. Both metrics are commonly used in decision tree algorithms to measure the impurity of a node and help determine the best splits in the data.

# 	•	Gini Index: Measures the impurity of a node based on the probability of each class. A Gini Index of 0 represents a pure node, while higher values indicate more class mixing.
#  	•	Entropy: Measures the uncertainty or disorder in the data. Entropy is 0 when a node is pure, and it increases as the classes become more mixed.

#   Visualization:

# 	•	The plot shows how Gini Index and Entropy change as the probability of one class (Class 1) increases from 0.01 to 0.99.
# 	•	As the probability of one class increases, both Gini Index and Entropy decrease, indicating that the node becomes purer.

# Key Observations:

# 	1.	Gini Index and Entropy are both highest when the class probabilities are evenly split (e.g., 0.5 for both classes), meaning the node is maximally impure.
# 	2.	Gini Index is generally lower than Entropy but follows a similar curve, with both reaching 0 when one class dominates completely (i.e., probability is 0 or 1).
# 	3.	The Gini Index tends to be more computationally efficient and is commonly used in algorithms like CART.
