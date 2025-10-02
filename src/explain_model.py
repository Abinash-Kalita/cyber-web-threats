# explain_model.py
import shap
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def explain_model(df, model, features=None):
    """
    Generate SHAP explanations for tree-based models (RandomForest, etc.).
    IsolationForest is skipped (not supported).

    Returns:
        shap_values: raw SHAP values (or None)
        explainer: SHAP explainer (or None)
        X: feature DataFrame (or None)
        mean_shap_df: DataFrame with mean absolute SHAP per feature (or None)
    """
    if features is None:
        features = ["bytes_in", "bytes_out", "session_duration", "bytes_total", "avg_packet_size"]

    # Prepare features
    X = df[[f for f in features if f in df.columns]].fillna(0)
    if X.empty:
        return None, None, None, None

    # Skip IsolationForest (not supported by SHAP)
    model_name = type(model).__name__.lower()
    if "isolationforest" in model_name:
        return None, None, None, None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_matrix = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_matrix = np.abs(shap_values)

        mean_shap_df = pd.DataFrame({
            "Feature": X.columns,
            "MeanAbsSHAP": shap_matrix.mean(axis=0)
        }).sort_values(by="MeanAbsSHAP", ascending=False)

        # Save static SHAP summary plot
        if not os.path.exists("models"):
            os.makedirs("models")
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig("models/shap_summary.png")
        plt.close()
        print("✅ SHAP summary plot saved to models/shap_summary.png")

        return shap_values, explainer, X, mean_shap_df

    except Exception as e:
        print(f"⚠ SHAP computation failed: {e}")
        return None, None, X, None


def format_classification_report(report_dict):
    """
    Convert sklearn classification_report (dict) into a tidy DataFrame.
    """
    if not isinstance(report_dict, dict):
        return None

    # Convert nested dict to DataFrame
    df_report = pd.DataFrame(report_dict).T.reset_index()
    df_report = df_report.rename(columns={"index": "Class"})

    # Round numeric columns
    for col in ["precision", "recall", "f1-score", "support"]:
        if col in df_report.columns:
            df_report[col] = pd.to_numeric(df_report[col], errors="coerce").round(3)

    return df_report
