# src/eda_plot.py
import matplotlib.pyplot as plt
from load_data import load_data
from preprocess import preprocess
from isolation_model import train_isolation

def plot_anomalies(df):
    # Scatter plot: bytes_in vs bytes_out
    normal = df[df["anomaly_label"] == "Normal"]
    suspicious = df[df["anomaly_label"] == "Suspicious"]

    plt.figure(figsize=(8,6))
    plt.scatter(normal["bytes_in"], normal["bytes_out"], alpha=0.6, label="Normal")
    plt.scatter(suspicious["bytes_in"], suspicious["bytes_out"], alpha=0.9, marker="x", label="Suspicious")

    plt.xlabel("Bytes In")
    plt.ylabel("Bytes Out")
    plt.title("Traffic Anomalies: Bytes In vs Bytes Out")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    model, df_out = train_isolation(df)
    print("✅ Data prepared, plotting anomalies...")
    plot_anomalies(df_out)
