import pandas as pd
from sklearn.cluster import KMeans

def clean_airbnb_data(df):
    print("Cleaning data...")

    # Drop irrelevant columns
    df = df.drop(columns=["id", "name", "host_name", "last_review"])
    print("Dropped columns: id, name, host_name, last_review.")

    # Fill missing values in reviews_per_month
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    print("Filled missing values in 'reviews_per_month' with 0.")

    # Drop remaining missing values
    before_drop = df.shape[0]
    df = df.dropna()
    print(f"Dropped {before_drop - df.shape[0]} rows with remaining null values.")

    # Remove listings with extreme or zero prices
    before_filter = df.shape[0]
    df = df[(df["price"] > 0) & (df["price"] < 1000)]
    print(f"Removed {before_filter - df.shape[0]} listings with price outliers.")

    # Add location clusters
    df = add_location_clusters(df, n_clusters=10)

    # Drop raw lat/lon if desired (optional)
    df = df.drop(columns=["latitude", "longitude"])
    print("Dropped raw 'latitude' and 'longitude' columns after clustering.")

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=["neighbourhood_group", "room_type", "location_cluster"], drop_first=True)
    print("Encoded 'neighbourhood_group', 'room_type', and 'location_cluster' using one-hot encoding.")

    print(f"Final cleaned dataset shape: {df.shape}")
    return df


def add_location_clusters(df, n_clusters=10):
    print(f"Creating {n_clusters} location clusters using latitude and longitude...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["location_cluster"] = kmeans.fit_predict(df[["latitude", "longitude"]])
    return df
