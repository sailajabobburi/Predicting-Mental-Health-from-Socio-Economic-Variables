from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_random_forest(model,X_train, y_train):
    """
    Train a Random Forest Classifier.

    Parameters:
    - X_train (DataFrame): The training input samples.
    - y_train (Series): The target values.
    - n_estimators (int): The number of trees in the forest. Defaults to 100.
    - random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
    - RandomForestClassifier: The trained random forest model.
    """

    model.fit(X_train, y_train)
    return model


def evaluate_random_forest(clf, X_test, y_test):
    """
    Evaluate a trained Random Forest Classifier and print top 20 features.

    Parameters:
    - clf (RandomForestClassifier): The trained random forest model.
    - X_test (DataFrame): The test input samples.
    - y_test (Series): The true target values for the test set.

    Returns:
    - dict: A dictionary containing the model, accuracy, confusion matrix, classification report, and top features.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    feature_importances = clf.feature_importances_

    # Get the top 20 features
    feature_names = X_test.columns
    top_features = sorted(zip(feature_importances, feature_names), reverse=True)[:20]

    performance = {
        'model': 'RandomForestClassifier',
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'top_features': top_features
    }

    return performance


def evaluate_random_forest(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    feature_importances = model.feature_importances_

    # Get the top 20 features
    feature_names = X_test.columns
    top_features = sorted(zip(feature_importances, feature_names), reverse=True)[:20]

    performance = {
        'model': 'RandomForestClassifier',
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'top_features': top_features
    }

    return performance