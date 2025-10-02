# src/load_data.py
import pandas as pd
from pathlib import Path

def load_data(path="data/CloudWatch_Traffic_Web_Attack.csv"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found. Please put the CSV in the data/ folder.")
    df = pd.read_csv(p, low_memory=False)
    return df

if __name__ == "__main__":
    df = load_data()
    print("✅ Dataset loaded successfully")
    print("Shape:", df.shape)
    print("First 5 rows:")
    print(df.head())
