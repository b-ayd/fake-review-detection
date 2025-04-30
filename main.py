import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

from features.preprocessing import clean_text
from models.baseline_model import train_logistic_regression, train_random_forest
from models.dimensionality import reduce_dimensionality
from models.feature_selection import select_top_features
from utils.evaluation import evaluate_model
from utils.visualization import plot_confusion_matrix, plot_roc_curve, save_metrics, plot_feature_importance, plot_tfidf_feature_importance
from features.readability import extract_readability_features
from features.sentiment import extract_sentiment_scores, compute_sentiment_rating_inconsistency
from features.lexical_diversity import compute_lexical_diversity

# Load dataset
df = pd.read_csv("fake_reviews_dataset.csv", encoding="utf-8")
df['text_'] = df['text_'].apply(clean_text)
df['label'] = df['label'].apply(lambda x: 1 if x == "CG" else 0)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text_'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Apply SVD on TF-IDF matrices
#X_train_tfidf, X_test_tfidf, svd_model = reduce_dimensionality(X_train_tfidf, X_test_tfidf, n_components=100)

# Apply feature selection
X_train_tfidf, X_test_tfidf, selector = select_top_features(X_train_tfidf, y_train, X_test_tfidf, k=3000)

# Readability Feature Extraction
flesch, grade, avg_sent_len = extract_readability_features(X_train.tolist())
flesch_test, grade_test, avg_sent_len_test = extract_readability_features(X_test.tolist())

readability_train = np.vstack([flesch, grade, avg_sent_len]).T
readability_test = np.vstack([flesch_test, grade_test, avg_sent_len_test]).T

scaler = StandardScaler()
readability_train_scaled = scaler.fit_transform(readability_train)
readability_test_scaled = scaler.transform(readability_test)

# Sentiment Feature Extraction
sentiment_train = extract_sentiment_scores(X_train.tolist())
sentiment_test = extract_sentiment_scores(X_test.tolist())

sentiment_train = np.array(sentiment_train).reshape(-1, 1)
sentiment_test = np.array(sentiment_test).reshape(-1, 1)

sentiment_scaler = StandardScaler()
sentiment_train_scaled = sentiment_scaler.fit_transform(sentiment_train)
sentiment_test_scaled = sentiment_scaler.transform(sentiment_test)

rating_train = df.loc[X_train.index, 'rating'].tolist()
rating_test = df.loc[X_test.index, 'rating'].tolist()

inconsistency_train = compute_sentiment_rating_inconsistency(X_train.tolist(), rating_train)
inconsistency_test = compute_sentiment_rating_inconsistency(X_test.tolist(), rating_test)

# Standardize Inconsistency Scores
inconsistency_scaler = StandardScaler()
inconsistency_train_scaled = inconsistency_scaler.fit_transform(inconsistency_train)
inconsistency_test_scaled = inconsistency_scaler.transform(inconsistency_test)

# --- Lexical Diversity Feature Extraction ---
lexical_div_train = compute_lexical_diversity(X_train.tolist())
lexical_div_test = compute_lexical_diversity(X_test.tolist())

# Reshape to column vectors
import numpy as np
lexical_div_train = np.array(lexical_div_train).reshape(-1, 1)
lexical_div_test = np.array(lexical_div_test).reshape(-1, 1)

# Standardize
from sklearn.preprocessing import StandardScaler
lexical_scaler = StandardScaler()
lexical_div_train_scaled = lexical_scaler.fit_transform(lexical_div_train)
lexical_div_test_scaled = lexical_scaler.transform(lexical_div_test)

# --- Lexical Diversity Feature Extraction ---
lexical_div_train = compute_lexical_diversity(X_train.tolist())
lexical_div_test = compute_lexical_diversity(X_test.tolist())

lexical_div_train = np.array(lexical_div_train).reshape(-1, 1)
lexical_div_test = np.array(lexical_div_test).reshape(-1, 1)

lexical_scaler = StandardScaler()
lexical_div_train_scaled = lexical_scaler.fit_transform(lexical_div_train)
lexical_div_test_scaled = lexical_scaler.transform(lexical_div_test)

# Combine TF-IDF + Features
X_train_combined = hstack([
        X_train_tfidf, 
        readability_train_scaled, 
        sentiment_train_scaled,
        inconsistency_train_scaled,
        lexical_div_train_scaled])

X_test_combined = hstack([
    X_test_tfidf, 
    readability_test_scaled, 
    sentiment_test_scaled,
    inconsistency_test_scaled,
    lexical_div_test_scaled])

# Train baseline
model = train_logistic_regression(X_train_combined, y_train)

# Predict
y_pred = model.predict(X_test_combined)
y_pred_proba = model.predict_proba(X_test_combined)[:, 1]

# Evaluate
accuracy, precision, recall, f1, auc_roc = evaluate_model(y_test, y_pred, y_pred_proba)

# --- TF-IDF Feature Importance Visualization ---
# Extract only the coefficients for TF-IDF part
if selector is not None:
    # Apply feature selection to the names too
    tfidf_feature_names = vectorizer.get_feature_names_out()
    selected_indices = selector.get_support(indices=True)
    tfidf_feature_names = tfidf_feature_names[selected_indices]
else:
    tfidf_feature_names = vectorizer.get_feature_names_out()

num_tfidf = len(tfidf_feature_names)

# Check model type for compatibility
if hasattr(model, "coef_"):
    # Logistic Regression
    weights = model.coef_[0]
    tfidf_weights = weights[:num_tfidf]
    custom_weights = weights[num_tfidf:]

    # Plot both
    plot_tfidf_feature_importance(tfidf_feature_names, tfidf_weights, top_n=20)

elif hasattr(model, "feature_importances_"):
    # Random Forest
    importances = model.feature_importances_
    tfidf_weights = importances[:num_tfidf]
    custom_weights = importances[num_tfidf:]

    # Plot both
    plot_tfidf_feature_importance(tfidf_feature_names, tfidf_weights, top_n=20)

else:
    print("⚠️ Model type does not support feature importance visualization.")
    tfidf_weights, custom_weights = None, None

custom_feature_names = [
    "Flesch Reading Ease",
    "Flesch-Kincaid Grade",
    "Avg Sentence Length",
    "Sentiment Polarity",
    "Sentiment Inconsistency",
    "Lexical Diversity"
]

plot_feature_importance(custom_feature_names, custom_weights)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

# Save metrics
#save_metrics(accuracy, precision, recall, f1, auc_roc, experiment="final")
print("\n✅ Metrics saved to outputs/metrics_baseline.csv")

# Visualize results
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred_proba)