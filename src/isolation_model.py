# src/isolation_model.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from load_data import load_data
from preprocess import preprocess


def train_isolation(df: pd.DataFrame, features=None, contamination=0.1):
    """
    Train Isolation Forest on selected features.
    contamination: fraction of anomalies to expect (0.05 = 5%)
    """
    if features is None:
        features = ["bytes_in", "bytes_out", "session_duration", "bytes_total", "avg_packet_size"]

    X = df[features].fillna(0)

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)

    preds = model.predict(X)  # -1 = anomaly, 1 = normal
    df["anomaly"] = preds
    df["anomaly_label"] = df["anomaly"].map({1: "Normal", -1: "Suspicious"})

    return model, df


def save_model(model, path="models/isolation_model.joblib"):
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")


if __name__ == "__main__":
    # 1. Load and preprocess data
    df = load_data()
    df = preprocess(df)

    # 2. Train model
    model, df_out = train_isolation(df)

    # 3. Save model
    save_model(model)

    # 4. Show results
    print("✅ Isolation Forest training complete")
    print("Total suspicious:", (df_out["anomaly_label"] == "Suspicious").sum())
    print(df_out[["bytes_in", "bytes_out", "bytes_total", "avg_packet_size", "anomaly_label"]].head())
