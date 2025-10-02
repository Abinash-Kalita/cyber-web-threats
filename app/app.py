import seaborn as sns
import plotly.express as px
import sys, os
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# Add project root and src folder to Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Import user modules (these should exist in src/)
from load_data import load_data
from preprocess import preprocess
from isolation_model import train_isolation
from explain_model import explain_model
from logger import get_logger
from random_forest_model import train_random_forest, save_model as save_rf_model
from explain_model import explain_model, format_classification_report
# ------------------------------
# Setup logger
# ------------------------------
logger = get_logger()
logger.info("App started")

# ------------------------------
# App config
# ------------------------------
st.set_page_config(page_title="Cyber Threat Explorer", layout="wide")
st.title("🔎 Cybersecurity Web Traffic Anomaly Detection")

# ------------------------------
# Sidebar Filters + Hyperparameters
# ------------------------------
st.sidebar.header("Filters & Hyperparameters")

# Load + preprocess
df = load_data()
df = preprocess(df)
logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} cols")

# Ensure required derived features exist (defensive)
def ensure_features(df):
    if "session_duration" not in df.columns:
        if "creation_time" in df.columns and "end_time" in df.columns:
            df["creation_time"] = pd.to_datetime(df["creation_time"], errors="coerce")
            df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
            df["session_duration"] = (df["end_time"] - df["creation_time"]).dt.total_seconds().fillna(0)
        else:
            df["session_duration"] = 0

    if "bytes_total" not in df.columns:
        if "bytes_in" in df.columns and "bytes_out" in df.columns:
            df["bytes_total"] = (df["bytes_in"].fillna(0) + df["bytes_out"].fillna(0))
        else:
            df["bytes_total"] = 0

    if "avg_packet_size" not in df.columns:
        df["avg_packet_size"] = df["bytes_total"] / df["session_duration"].replace(0, np.nan)
        df["avg_packet_size"] = df["avg_packet_size"].fillna(0)

    if "dest_ip" not in df.columns and "dst_ip" in df.columns:
        df["dest_ip"] = df["dst_ip"]

    return df

df = ensure_features(df)

# Anomaly detection parameter
contamination = st.sidebar.slider("Expected anomaly fraction", 0.01, 0.30, 0.10)

# RandomForest hyperparams
st.sidebar.header("RandomForest Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 50, 500, 150, step=10)
max_depth = st.sidebar.slider("Max Depth", 1, 50, 10)

# ------------------------------
# Train Isolation Forest
# ------------------------------
if st.sidebar.button("Retrain Isolation Forest"):
    model_iso, df_out = train_isolation(df, contamination=contamination)
    suspicious_df = df_out[df_out.get("anomaly_label", "") == "Suspicious"]
    st.success("Isolation Forest retrained!")
else:
    model_iso, df_out = train_isolation(df, contamination=contamination)
    suspicious_df = df_out[df_out.get("anomaly_label", "") == "Suspicious"]

df_out = ensure_features(df_out)
suspicious_df = df_out[df_out.get("anomaly_label", "") == "Suspicious"]

# ------------------------------
# Cache RandomForest Training (robust)
# ------------------------------
@st.cache_resource
def cached_train_rf(df_local, n_estimators, max_depth):
    label_col = None
    info_msg = None

    if "detection_types" in df_local.columns:
        vals = df_local["detection_types"].dropna().unique()
        if len(vals) >= 2:
            label_col = "detection_types"

    if label_col is None and "anomaly_label" in df_local.columns:
        vals = df_local["anomaly_label"].dropna().unique()
        if len(vals) >= 2:
            label_col = "anomaly_label"

    if label_col is None:
        info_msg = "Not enough label diversity for supervised RandomForest training. Need >=2 classes."
        logger.warning(info_msg)
        return None, None, info_msg, None, (None, None), []

    try:
        model_rf, cm, report, feature_importances, cv_mean_std, all_labels = train_random_forest(
            df_local, label_col=label_col, n_estimators=n_estimators, max_depth=max_depth
        )
        return model_rf, cm, report, feature_importances, cv_mean_std, all_labels
    except Exception as e:
        logger.exception("RandomForest training failed")
        return None, None, f"RandomForest training failed: {e}", None, (None, None), []

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Summary",
    "Scatter",
    "Network",
    "GeoIP Map",
    "Explainability",
    "Download",
    "RandomForest Classification",
    "Anomalies + Classification"
])

# ------------------------------
# Tab1: Summary
# ------------------------------
with tab1:
    st.write("### Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df_out.shape[0])
    col2.metric("Suspicious Rows", int((df_out.get('anomaly_label') == 'Suspicious').sum()))
    col3.metric("Normal Rows", int((df_out.get('anomaly_label') == 'Normal').sum()))
    st.write("### Data Sample")
    st.dataframe(df_out.head())
    st.write("### Anomaly Counts")
    if "anomaly_label" in df_out.columns:
        st.bar_chart(df_out["anomaly_label"].value_counts())
    else:
        st.info("No anomaly_label column found in dataset.")

# ------------------------------
# Tab2: Scatter Plot
# ------------------------------
with tab2:
    st.write("### Bytes In vs Bytes Out (Anomalies Highlighted)")
    normal = df_out[df_out.get("anomaly_label") == "Normal"]
    fig, ax = plt.subplots(figsize=(8, 6))
    if not normal.empty:
        ax.scatter(normal["bytes_in"], normal["bytes_out"], alpha=0.6, label="Normal")
    if not suspicious_df.empty:
        ax.scatter(suspicious_df["bytes_in"], suspicious_df["bytes_out"], alpha=0.9,
                   marker="x", label="Suspicious", color="red")
    ax.set_xlabel("Bytes In")
    ax.set_ylabel("Bytes Out")
    ax.legend()
    st.pyplot(fig)

# ------------------------------
# Tab3: Suspicious Traffic Network
# ------------------------------
with tab3:
    st.write("### Suspicious Traffic Network")
    if not suspicious_df.empty and "src_ip" in suspicious_df.columns and ("dest_ip" in suspicious_df.columns or "dst_ip" in suspicious_df.columns):
        target_col = "dest_ip" if "dest_ip" in suspicious_df.columns else "dst_ip"
        try:
            G = nx.from_pandas_edgelist(suspicious_df, source="src_ip", target=target_col, create_using=nx.DiGraph())
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
            net.from_nx(G)
            net.save_graph("app/network.html")
            with open("app/network.html", "r", encoding="utf-8") as HtmlFile:
                components.html(HtmlFile.read(), height=600)
        except Exception as e:
            st.error(f"Could not render network graph: {e}")
    else:
        st.info("No suspicious IP connections available.")

# ------------------------------
# Tab4: GeoIP Map
# ------------------------------
with tab4:
    st.write("### Suspicious Traffic by Country")
    if "src_ip_country_code" in df_out.columns and not suspicious_df.empty:
        country_counts = suspicious_df["src_ip_country_code"].value_counts().reset_index()
        country_counts.columns = ["country", "count"]
        try:
            fig = px.choropleth(
                country_counts,
                locations="country",
                locationmode="ISO-3",
                color="count",
                hover_name="country",
                color_continuous_scale="Reds",
                title="Suspicious Traffic Sources"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render choropleth: {e}")
    else:
        st.warning("No country code data available.")

# ------------------------------
# Tab5: SHAP Explainability (IsolationForest)
# ------------------------------
with tab5:
    st.write("### SHAP Explainability (Isolation Forest)")
    shap_values_iso, explainer_iso, X_iso, mean_shap_iso = explain_model(df_out, model_iso)
    
    if shap_values_iso is None:
        st.info("⚠ SHAP explainability is not supported for IsolationForest models.")
    else:
        st.write("### Top SHAP Features (IsolationForest)")
        st.dataframe(mean_shap_iso.head(10))



# ------------------------------
# Tab6: Download Suspicious Records
# ------------------------------
with tab6:
    st.write("### Download Suspicious Records")
    if not suspicious_df.empty:
        csv = suspicious_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Suspicious Records", csv, "suspicious_records.csv", "text/csv")
    else:
        st.info("No suspicious rows to download.")

# ------------------------------
# Tab7: RandomForest Classification & Explainability
# ------------------------------
with tab7:
    st.write("### RandomForest Classification & Explainability")
    model_rf, cm, report, feature_importances, cv_mean_std, all_labels = cached_train_rf(
        df_out, n_estimators, max_depth
    )

    if model_rf is None:
        st.warning(report if isinstance(report, str) else "RandomForest not trained.")
    else:
        # Confusion Matrix
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=all_labels, yticklabels=all_labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Confusion matrix could not be plotted: {e}")

        # Classification Report (tidy DataFrame)
        st.write("### Classification Report")
        try:
            report_df = format_classification_report(report)
            if report_df is not None:
                st.dataframe(report_df, use_container_width=True)
            else:
                st.json(report)  # fallback
        except Exception as e:
            st.warning(f"Could not format classification report: {e}")

        # Feature Importances
        if feature_importances is not None:
            fi_df = pd.DataFrame(feature_importances, columns=["feature", "importance"])\
                        .sort_values("importance", ascending=False)
            st.write("### Top RandomForest Feature Importances")
            st.dataframe(fi_df.head(15), use_container_width=True)

            # Optional bar plot
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="importance", y="feature", data=fi_df.head(10), ax=ax, palette="viridis")
            ax.set_title("Top 10 RandomForest Features")
            st.pyplot(fig)

        # SHAP Explainability
        try:
            shap_values_rf, explainer_rf, X_rf, mean_shap_rf = explain_model(df_out, model_rf)
            if mean_shap_rf is not None:
                st.write("### Top SHAP Features (RandomForest)")
                st.dataframe(mean_shap_rf.head(10))

                # SHAP bar plot
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x="MeanAbsSHAP", y="Feature", data=mean_shap_rf.head(10), ax=ax, palette="magma")
                ax.set_title("Top 10 SHAP Features (RandomForest)")
                st.pyplot(fig)
            else:
                st.info("No SHAP values available for RandomForest.")
        except Exception as e:
            st.warning(f"SHAP postprocessing failed: {e}")

        # Save trained RF model
        try:
            save_rf_model(model_rf)
            st.success("✅ RandomForest model saved to disk.")
        except Exception as e:
            st.warning(f"Could not save RandomForest model: {e}")


# ------------------------------
# Tab8: Combined Anomalies + Classification
# ------------------------------
with tab8:
    st.write("### Combined Isolation Forest + RandomForest Results")
    try:
        if model_rf is not None:
            candidate_feats = ["bytes_in", "bytes_out", "session_duration", "bytes_total", "avg_packet_size"]
            feats = [c for c in candidate_feats if c in df_out.columns]
            if feats:
                preds = model_rf.predict(df_out[feats].fillna(0))
                df_out["RF_Prediction"] = preds
            else:
                df_out["RF_Prediction"] = "N/A"
        else:
            df_out["RF_Prediction"] = "N/A"

        src = "src_ip" if "src_ip" in df_out.columns else None
        dst = "dest_ip" if "dest_ip" in df_out.columns else ("dst_ip" if "dst_ip" in df_out.columns else None)
        merged_cols = [c for c in [src, dst, "bytes_in", "bytes_out", "anomaly_label", "RF_Prediction"] if c is not None]
        merged_df = df_out[merged_cols]
        st.dataframe(merged_df.head(50))

        fig, ax = plt.subplots(figsize=(8, 6))
        if "anomaly_label" in merged_df.columns:
            for label, subset in merged_df.groupby("anomaly_label"):
                ax.scatter(subset["bytes_in"], subset["bytes_out"], alpha=0.6, label=f"{label} (Anomaly)")
        if "RF_Prediction" in merged_df.columns:
            for cls, subset in merged_df.groupby("RF_Prediction"):
                ax.scatter(subset["bytes_in"], subset["bytes_out"], alpha=0.5, marker="x", label=f"RF: {cls}")
        ax.set_xlabel("Bytes In")
        ax.set_ylabel("Bytes Out")
        ax.legend()
        st.pyplot(fig)

        csv = merged_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Combined Results", csv, "combined_results.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠ Could not generate combined view: {e}")
