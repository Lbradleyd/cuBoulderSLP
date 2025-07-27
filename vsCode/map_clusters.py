import pandas as pd
import folium
from clean_data import add_location_clusters

# Load data
df = pd.read_csv("data.csv")
df = add_location_clusters(df, n_clusters=10)

# Initialize map centered on NYC
nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# Define color palette for clusters
cluster_colors = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "lightblue", "lightgreen", "gray", "black"
]

# Plot each listing
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=2,
        color=cluster_colors[row["location_cluster"] % len(cluster_colors)],
        fill=True,
        fill_opacity=0.5
    ).add_to(nyc_map)

# Save and/or display map
nyc_map.save("airbnb_cluster_map.html")
print("Map saved to airbnb_cluster_map.html")
