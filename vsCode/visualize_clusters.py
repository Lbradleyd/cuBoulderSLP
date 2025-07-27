import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from clean_data import add_location_clusters

# Load data
df = pd.read_csv("data.csv")  # Update with your path
df = add_location_clusters(df, n_clusters=10)

# Plot clusters on map
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="longitude", y="latitude", hue="location_cluster", palette="tab10", alpha=0.7)
plt.title("Airbnb Listings Colored by Location Cluster")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()
