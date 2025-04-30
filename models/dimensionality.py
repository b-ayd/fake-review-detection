from sklearn.decomposition import TruncatedSVD

def reduce_dimensionality(X_train_tfidf, X_test_tfidf, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_reduced = svd.fit_transform(X_train_tfidf)
    X_test_reduced = svd.transform(X_test_tfidf)
    return X_train_reduced, X_test_reduced, svd