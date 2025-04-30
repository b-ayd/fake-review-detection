import csv
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(y_true, y_pred, save_path="outputs/cm.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, save_path="outputs/rc.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def save_metrics(accuracy, precision, recall, f1, auc_roc, filename="outputs/metrics_baseline.csv", experiment="Baseline"):
    """Save evaluation metrics to a CSV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Experiment", experiment])
        writer.writerow(["Accuracy", f"{accuracy:.4f}"])
        writer.writerow(["Precision", f"{precision:.4f}"])
        writer.writerow(["Recall", f"{recall:.4f}"])
        writer.writerow(["F1-score", f"{f1:.4f}"])
        writer.writerow(["AUC-ROC", f"{auc_roc:.4f}"])

def plot_feature_importance(feature_names, weights, title="Custom Feature Importance"):
    """Bar plot of feature importance for custom (non-TFIDF) features."""
    plt.figure(figsize=(10, 4))
    bars = plt.barh(feature_names, weights, color='steelblue')
    plt.xlabel("Weight (Model Coefficient)")
    plt.title(title)
    plt.axvline(0, color='gray', linestyle='--')
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/custom_feature_importance.png")
    plt.close()

def plot_tfidf_feature_importance(feature_names, weights, top_n=20, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Get top N positive and negative features
    sorted_indices = np.argsort(weights)
    top_pos = [(feature_names[i], weights[i]) for i in sorted_indices[-top_n:]]
    top_neg = [(feature_names[i], weights[i]) for i in sorted_indices[:top_n]]

    # Combine and reverse for horizontal bar chart
    top_features = top_neg + top_pos
    features, coefs = zip(*top_features)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, coefs, color=['red' if c < 0 else 'green' for c in coefs])
    plt.xlabel("Weight (Model Coefficient)")
    plt.title("Top TF-IDF Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tfidf_feature_importance.png"))
    plt.close()