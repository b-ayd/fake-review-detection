from sklearn.feature_selection import SelectKBest, chi2

def select_top_features(X_train_tfidf, y_train, X_test_tfidf, k=3000):
    """Select top-k TF-IDF features using chi-squared test."""
    selector = SelectKBest(chi2, k=k)
    X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
    X_test_selected = selector.transform(X_test_tfidf)
    return X_train_selected, X_test_selected, selector
