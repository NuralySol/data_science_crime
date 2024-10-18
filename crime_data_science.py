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

df.columns
#! plot incidents as a scatter x = "Longitude" y = "Latitude"
#! GOOGLE Long Lat for the Empire State Building and plot it as a star

plt.figure(figsize=(10, 10))
sns.scatterplot(x="Longitude", y="Latitude", data=df, alpha=0.25, s=50, color="blue")

# Coordinates for the Empire State Building (GOOGLE maps) and save them as variables.
empire_state_longitude = -73.9857
empire_state_latitude = 40.748817

# Scatter plot for the empire state building as a star
plt.scatter(
    empire_state_longitude,
    empire_state_latitude,
    color="orange",
    marker="*",
    s=300,
    label="Empire State Building",
)
plt.title("Incident Locations in NYC")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.annotate(
    "Empire State Building",
    xy=(empire_state_longitude, empire_state_latitude),
    xytext=(-73.98, 40.75),
    fontsize=10,
    color="black",
    fontweight="bold",
)
plt.show()

#! Plot incidents using relplot and hue for the 'STATISTICAL_MURDER_FLAG'
# Plot incidents using relplot and hue for the 'STATISTICAL_MURDER_FLAG'
plt.figure(figsize=(10, 10))
sns.relplot(
    x="Longitude",
    y="Latitude",
    data=df,
    hue="STATISTICAL_MURDER_FLAG",  # Color based on 'STATISTICAL_MURDER_FLAG'
    alpha=0.5,
    size="INCIDENT_KEY",  # Optional: size based on number of incidents
    palette="Set1",       # Use Set1 palette for more contrast
    height=10,            # Adjust height of the plot
    aspect=1,             # Aspect ratio
)

# Coordinates for the Empire State Building (GOOGLE maps) and save them as variables.
empire_state_longitude = -73.9857
empire_state_latitude = 40.748817

# Scatter plot for the empire state building as a star
plt.scatter(
    empire_state_longitude,
    empire_state_latitude,
    color="orange",
    marker="*",
    s=300,
    label="Empire State Building",
)

# Annotate the Empire State Building
plt.annotate(
    "Empire State Building",
    xy=(empire_state_longitude, empire_state_latitude),
    xytext=(-73.98, 40.75),
    fontsize=10,
    color="black",
    fontweight="bold",
)


plt.title("Incident Locations in NYC with Statistical Murder Flag")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

#! Plotting as a bar chart to see what is the most dangerous time to be outside
df["OCCUR_TIME"] = pd.to_datetime(df["OCCUR_TIME"], format='%H:%M:%S', errors='coerce')  
df["HOUR"] = df["OCCUR_TIME"].dt.hour

plt.figure(figsize=(10, 10))
sns.histplot(df['HOUR'], bins=24, color='lightblue') # 24 bins for 24 hours of the day. 
plt.title('Most Dangerous hours to be outside')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Incidents')
plt.xticks(range(0, 24))
plt.grid(True)
plt.show()


#! chance to be murdered based two independent values of time of day and location in the ragards to total population of nyc
#! np.array to hold two independent values for this calculation and create a heatmap for chance to murdered.
#! use the fixture for the population to see the chance of being murdered for per-capita basis. 
population_data = {
    'BRONX': 1418207,
    'BROOKLYN': 2559903,
    'MANHATTAN': 1628706,
    'QUEENS': 2253858,
    'STATEN ISLAND': 476143
}

df["OCCUR_TIME"] = pd.to_datetime(df["OCCUR_TIME"], format='%H:%M:%S', errors='coerce')
df["HOUR_OF_INCIDENT"] = df["OCCUR_TIME"].dt.hour

murder_cases = df[df['STATISTICAL_MURDER_FLAG'] == 'Y']

murder_by_hour_location = murder_cases.groupby(['BORO', 'HOUR_OF_INCIDENT'])['INCIDENT_KEY'].count().unstack(fill_value=0)

#! The below code converts the entire murder_by_hour_location DataFrame (which now contains the murder rates per 100,000 people for each borough and hour) into a NumPy array.
#! This NumPy array is useful for plotting, as some plotting functions (such as the heatmap in Seaborn) require or work better with array-like structures.

for boro in murder_by_hour_location.index:
    murder_by_hour_location.loc[boro] = (murder_by_hour_location.loc[boro] / population_data[boro]) * 100000

murder_heatmap_data = murder_by_hour_location.to_numpy()

#! Heat map is good and closer to 1 more change to be killed
plt.figure(figsize=(12, 10))
sns.heatmap(murder_heatmap_data, cmap='coolwarm', annot=True, fmt='.2f', xticklabels=range(24), yticklabels=murder_by_hour_location.index)
plt.title('Chance to be Murdered by Hour and Borough (per 100,000 people)')
plt.xlabel('Hour of Day')
plt.ylabel('Borough')
plt.show()