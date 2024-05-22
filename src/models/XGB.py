from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42,scale_pos_weight=1):
    """
    Train an XGBoost Classifier.

    Parameters:
    - X_train (DataFrame): The training input samples.
    - y_train (Series): The target values.
    - n_estimators (int): Number of boosting rounds.
    - learning_rate (float): Step size shrinkage used in update to prevent overfitting.
    - max_depth (int): Maximum depth of a tree.
    - random_state (int): Random seed for reproducibility.
    - scale_pos_weight (float): Control the balance of positive and negative weights.

    Returns:
    - XGBClassifier: The trained XGBoost model.
    """
    clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                        scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False,
                        eval_metric='logloss')
    clf.fit(X_train, y_train)
    return clf


def evaluate_xgboost(model, X_test, y_test):
    """
    Evaluate a trained XGBoost Classifier.

    Parameters:
    - model (XGBClassifier): The trained XGBoost model.
    - X_test (DataFrame): The test input samples.
    - y_test (Series): The true target values for the test set.

    Returns:
    - dict: A dictionary containing the model, accuracy, confusion matrix, classification report, and top features.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    feature_importances = model.feature_importances_

    # Get the top 20 features
    feature_names = X_test.columns
    top_features = sorted(zip(feature_importances, feature_names), reverse=True)[:20]

    performance = {
        'model': 'XGBClassifier',
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'top_features': top_features
    }

    return performance