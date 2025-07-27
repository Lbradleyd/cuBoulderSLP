from load_data import load_airbnb_data
from clean_data import clean_airbnb_data
from train_model_pipeline import run_model_pipeline
import matplotlib.pyplot as plt


print("Starting Airbnb price prediction pipeline...\n")

df_raw = load_airbnb_data()
df_cleaned = clean_airbnb_data(df_raw)
model = run_model_pipeline(df_cleaned)

print("\nPipeline execution complete.")
