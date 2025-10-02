# src/combined_rf_if_model.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
 


from load_data import load_data
from preprocess import preprocess

# Suppress undefined metric warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def train_random_forest(df, label_col="detection_types", n_estimators=200, max_depth=None, cv_folds=5):
    """Train Random Forest with class balancing and stratified split."""

    # Ensure required features exist
    features = ["bytes_in", "bytes_out", "session_duration", "bytes_total", "avg_packet_size"]
    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("No numeric features available for training RandomForest.")

    # Prepare data
    X = df[features].fillna(0)
    y = df[label_col].fillna("Unknown")
    all_labels = sorted(y.unique())

    # If dataset too small for stratified split, fallback to random split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Train Random Forest
    model_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight=class_weights
    )
    model_rf.fit(X_train, y_train)

    # Predictions
    y_pred = model_rf.predict(X_test)

    # Confusion matrix & classification report
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    report_dict = classification_report(y_test, y_pred, labels=all_labels, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, labels=all_labels, zero_division=0)

    # Feature importances (return as list of tuples for app.py)
    feature_importances = list(zip(features, model_rf.feature_importances_))

    # Cross-validation (weighted F1)
    try:
        cv_scores = cross_val_score(model_rf, X, y, cv=cv_folds, scoring="f1_weighted")
        cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)
    except Exception:
        cv_mean, cv_std = 0.0, 0.0

    return model_rf, cm, report_dict, feature_importances, (cv_mean, cv_std), all_labels


def save_model(model, path="models/random_forest.joblib"):
    """Save trained model to disk."""
    if not path.endswith(".joblib"):
        path += ".joblib"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")


def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    """Plot confusion matrix with Seaborn heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importances(feature_importances, title="Feature Importances"):
    """Plot feature importances as horizontal bar chart."""
    plt.figure(figsize=(7, 5))
    fi_df = pd.DataFrame(feature_importances, columns=["feature", "importance"]).sort_values("importance")
    fi_df.plot(kind="barh", x="feature", y="importance", legend=False, ax=plt.gca())
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load and preprocess
    df = load_data()
    df = preprocess(df)

    # Train Random Forest
    model_rf, cm, report, feature_importances, (cv_mean, cv_std), all_labels = train_random_forest(
        df, label_col="detection_types", n_estimators=150, max_depth=10
    )
    save_model(model_rf)

    print("\n✅ RandomForest training complete")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", pd.DataFrame(report).transpose())
    print("\nFeature Importances:\n", feature_importances)
    print(f"\nCross-Validation F1 (weighted): {cv_mean:.3f} ± {cv_std:.3f}")

    # Train Isolation Forest (optional)
    features = ["bytes_in", "bytes_out", "session_duration", "bytes_total", "avg_packet_size"]
    features = [f for f in features if f in df.columns]
    if features:
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_forest.fit(df[features].fillna(0))

        # Generate combined results
        df["rf_pred"] = model_rf.predict(df[features].fillna(0))
        df["iso_pred"] = iso_forest.predict(df[features].fillna(0))
        combined_results = df[["rf_pred", "iso_pred", "detection_types"]]
        print("\nCombined RandomForest + Isolation Forest results:\n", combined_results.head())

    # Optional plots
    plot_confusion_matrix(cm, all_labels)
    plot_feature_importances(feature_importances)
