import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

try:
    df = pd.read_csv("./NYPD.csv")
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
    
smf_by_boro = df.groupby(["STATISTICAL_MURDER_FLAG","BORO"])["INCIDENT_KEY"].count()
smf_by_boro.loc["Y"]

# Plot the murders by borough
plt.figure(figsize=(10, 8))
smf_by_boro.plot(kind='bar', color='orange')
plt.title('Number of Murders by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Murders')
plt.xticks(rotation=45)
plt.show()

#! Plot a count of incidents by borough
plt.figure(figsize=(8, 8))
sns.countplot(x='BORO', data=df)
plt.title('Distribution of Incidents by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.show()

#! Fixture of population data to ajust the incident for per capita basis.
population_data = {
    'BRONX': 1418207,
    'BROOKLYN': 2559903,
    'MANHATTAN': 1628706,
    'QUEENS': 2253858,
    'STATEN ISLAND': 476143
}

df['BORO_POPULATION'] = df['BORO'].map(population_data)
borough_incidents = df['BORO'].value_counts()

borough_incidents_per_capita = pd.DataFrame({
    'Borough': borough_incidents.index,
    'Incidents': borough_incidents.values,
    'Population': [population_data[boro] for boro in borough_incidents.index]
})

#! calculation based on incidents per capita
borough_incidents_per_capita['Incidents_per_100k'] = (borough_incidents_per_capita['Incidents'] / borough_incidents_per_capita['Population']) * 100000

# Plot incidents per capita by borough for better vis and representation
plt.figure(figsize=(10, 8))
sns.barplot(x='Borough', y='Incidents_per_100k', data=borough_incidents_per_capita, palette='viridis')
plt.title('Incidents per 100,000 People by Borough')
plt.xlabel('Borough')
plt.ylabel('Incidents per 100,000 People')
plt.xticks(rotation=45)
plt.show()

# Plot a count of incidents by perpetrator age group
plt.figure(figsize=(8, 6))
sns.countplot(x='PERP_AGE_GROUP', data=df, order=df['PERP_AGE_GROUP'].value_counts().index)
plt.title('Distribution of Perpetrator Age Groups')
plt.xlabel('Perpetrator Age Group')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.show()

# Compare the distribution of victim and perpetrator age groups
plt.figure(figsize=(8, 6))
sns.countplot(x='VIC_AGE_GROUP', data=df, order=df['VIC_AGE_GROUP'].value_counts().index, color='lightblue', label='Victims')
sns.countplot(x='PERP_AGE_GROUP', data=df, order=df['VIC_AGE_GROUP'].value_counts().index, color='salmon', label='Perpetrators')
plt.title('Comparison of Victim and Perpetrator Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Plot a scatter plot of incident locations based on Latitude and Longitude
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Longitude', y='Latitude', data=df)
plt.title('Scatter Plot of Incident Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Convert the OCCUR_TIME column to datetime and extract the hour
df['OCCUR_TIME'] = pd.to_datetime(df['OCCUR_TIME'], format='%H:%M:%S').dt.hour

# Plot the distribution of incidents by time of day
plt.figure(figsize=(8, 6))
sns.histplot(df['OCCUR_TIME'], bins=24, kde=False)
plt.title('Distribution of Incidents by Time of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Incidents')
plt.xticks(range(0, 24))
plt.show()

# Select numeric columns and calculate the correlation matrix
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Replace 'NUMERIC_COLUMN1', 'NUMERIC_COLUMN2', 'NUMERIC_COLUMN3' with actual numeric column names
plt.figure(figsize=(10, 10))
sns.pairplot(df[numeric_columns])
plt.title('Pairplot of Numerical Features')
plt.show()

# Create a FacetGrid to show the distribution of incidents by time of day, faceted by borough
g = sns.FacetGrid(df, col="BORO", height=4, aspect=1)
g.map(sns.histplot, "OCCUR_TIME", bins=24, color="blue", kde=False)
g.set_axis_labels("Hour of Day", "Number of Incidents")
g.set_titles("{col_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Incidents by Borough and Time of Day")
plt.show()

# Plot the distribution of time of incidents by borough using a violin plot
plt.figure(figsize=(10, 8))
sns.violinplot(x='BORO', y='OCCUR_TIME', data=df, palette='Set2', inner='quartile')
plt.title('Time of Incidents by Borough')
plt.xlabel('Borough')
plt.ylabel('Time of Day (Hours)')
plt.show()

