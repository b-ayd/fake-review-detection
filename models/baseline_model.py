def train_logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model