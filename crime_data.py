import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


try:
    df = pd.read_csv("./Social_Network_Ads.csv")
    print("CSV file loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print(
        "Error: File not found. Please check the path to the 'canada_per_capita_income.csv' file."
    )
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except pd.errors.ParserError:
    print("Error: The file is corrupted or improperly formatted.")

#! Do the EDA for the file. (Cleaning up the data for the further analysis)

#! Drop the 'User ID' column
if "User ID" in df.columns:
    df.drop(columns=["User ID"], inplace=True)
    print("Dropped 'User ID' column")

# Plot a pie chart showing the number of people who purchased
purchased_counts = df["Purchased"].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    purchased_counts,
    labels=["Not Purchased", "Purchased"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["lightcoral", "lightgreen"],
)
plt.title("Distribution of Purchases")
plt.show()

# Plot Purchased vs Age
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="Age",
    y="Purchased",
    data=df,
    hue="Purchased",
    palette="coolwarm",
    s=100,
    alpha=0.7,
)
plt.title("Purchased vs Age")
plt.xlabel("Age")
plt.ylabel("Purchased (0 = Not Purchased, 1 = Purchased)")
plt.legend(title="Purchased")
plt.show()

# Create age bins (grouping age into intervals)
df["AgeGroup"] = pd.cut(df["Age"], bins=5)  # You can adjust the number of bins

# Plot a bar plot to show the average purchase rate for each age group
plt.figure(figsize=(8, 8))
sns.barplot(x="AgeGroup", y="Purchased", data=df, palette="Blues")
plt.title("Purchase Rate by Age Grouping!")
plt.xlabel("Age Group")
plt.ylabel("Average Purchased (0 = Not Purchased, 1 = Purchased) True and Fale value!")
plt.xticks(rotation=90)
plt.show()


# Create age bins (grouping age into intervals)
df["AgeGroup"] = pd.cut(df["Age"], bins=5)

# Create a crosstab of AgeGroup and Purchased
crosstab = pd.crosstab(df["AgeGroup"], df["Purchased"], margins=True, normalize="index")

# Check if 'All' exists before trying to drop it
if "All" in crosstab.columns:
    crosstab = crosstab.drop(columns="All")

print("Crosstab between AgeGroup and Purchased:")
print(crosstab)

# Plot the crosstab result
crosstab.plot(kind="bar", stacked=True, figsize=(8, 8), colormap="coolwarm")
plt.title("Purchase Distribution Across Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Proportion of Purchases")
plt.xticks(rotation=45)
plt.legend(title="Purchased", loc="upper left")
plt.show()

#! Drop rows where age is less than 21
dropped_27 = df[df["Age"] >= 27]

# Create age bins (grouping age into intervals)
dropped_27["AgeGroup"] = pd.cut(dropped_27["Age"], bins=15)

# Create a crosstab of AgeGroup and Purchased
crosstab = pd.crosstab(
    dropped_27["AgeGroup"], dropped_27["Purchased"], margins=True, normalize="index"
)
print("Crosstab between AgeGroup and Purchased:")
print(crosstab)
crosstab.plot(kind="bar", stacked=True, figsize=(8, 8), colormap="coolwarm")
plt.title("Purchase Distribution Across Age Groups (Age >= 27)")
plt.xlabel("Age Group")
plt.ylabel("Proportion of Purchases")
plt.xticks(rotation=45)
plt.legend(title="Purchased", loc="upper left")
plt.show()

#! Crosstab: Gender vs Purchased
gender_purchase_crosstab = pd.crosstab(
    df["Gender"], df["Purchased"], margins=True, normalize="index"
)

# Check if 'All' exists before dropping it
if "All" in gender_purchase_crosstab.columns:
    gender_purchase_crosstab = gender_purchase_crosstab.drop(columns="All")

# Display the crosstab
print("Crosstab between Gender and Purchased:")
print(gender_purchase_crosstab)

# Plot the crosstab result
gender_purchase_crosstab.plot(
    kind="bar", stacked=True, figsize=(8, 6), colormap="coolwarm"
)
plt.title("Purchase Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Proportion of Purchases")
plt.legend(title="Purchased", loc="upper left")
plt.show()

# Boxplot of EstimatedSalary (Income) vs Gender and Purchased
plt.figure(figsize=(10, 8))
sns.boxplot(x="Purchased", y="EstimatedSalary", hue="Gender", data=df, palette="Set2")
plt.title("Income Distribution by Gender and Purchase Status")
plt.xlabel("Purchased (0 = Not Purchased, 1 = Purchased)")
plt.ylabel("Estimated Salary (Income)")
plt.legend(title="Gender")
plt.show()

# Catplot (Categorical Plot) of EstimatedSalary (Income) vs Gender and Purchased
g = sns.catplot(
    x="Purchased",
    y="EstimatedSalary",
    hue="Gender",
    kind="box",
    data=df,
    height=6,
    aspect=1.2,
    palette="Set2",
)

# Set plot title and labels
plt.title("Income Distribution by Gender and Purchase Status (Female and Male)")
plt.xlabel("Purchased (0 = Not Purchased, 1 = Purchased)")
plt.ylabel("Estimated Salary (Income)")

# Add Gender labels to the plot
ax = g.ax  # Accessing the underlying Axes
for patch, label in zip(ax.artists, df["Gender"].unique()):
    # Get the center of the box
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_y() + patch.get_height() / 2

    # Add text annotation for gender
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=10,
        color="black",
        fontweight="bold",
    )

plt.show()

#! Train test split and use a Logistic Regression Algorithm to train the model of the train_test_split 80% and 20%.
# Features (EstimatedSalary) and Target (Purchased)
X = df["EstimatedSalary"].values.reshape(-1, 1)  # Reshape for a single feature
y = df["Purchased"].values  # Target values

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print the shapes of the split datasets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Features (EstimatedSalary) and Target (Purchased)
X = df["EstimatedSalary"].values.reshape(-1, 1)  # Reshape for a single feature
y = df["Purchased"].values  # Target values

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualizing the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Not Purchased", "Purchased"],
    yticklabels=["Not Purchased", "Purchased"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#! Correction for the gender column:
#! Correct the encoding of Gender column
df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "Male" else 0)

# Features (Gender and EstimatedSalary) and Target (Purchased)
X = df[
    ["Gender", "EstimatedSalary"]
]  # Using both Gender and EstimatedSalary as features
y = df["Purchased"].values  # Target values

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualizing the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Not Purchased", "Purchased"],
    yticklabels=["Not Purchased", "Purchased"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

