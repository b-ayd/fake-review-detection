def evaluate_model(y_true, y_pred, y_pred_proba):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    return accuracy, precision, recall, f1, auc_roc