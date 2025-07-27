import pandas as pd

def load_airbnb_data(path="data.csv"):
    print("Loading Airbnb dataset...")
    df = pd.read_csv(path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df
