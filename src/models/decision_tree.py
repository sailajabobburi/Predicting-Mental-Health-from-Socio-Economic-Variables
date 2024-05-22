from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def train_decision_tree(model,X_train, y_train, random_state=42):
    """
    Train a Decision Tree Classifier.

    Parameters:
    - X_train (DataFrame): The training input samples.
    - y_train (Series): The target values.

    Returns:
    - DecisionTreeClassifier: The trained decision tree model.
    """

    model.fit(X_train, y_train)
    return model


def evaluate_decision_tree(clf, X_test, y_test):
    """
    Evaluate a trained Decision Tree Classifier.

    Parameters:
    - clf (DecisionTreeClassifier): The trained decision tree model.
    - X_test (DataFrame): The test input samples.
    - y_test (Series): The true target values for the test set.

    Returns:
    - dict: A dictionary containing the model, accuracy, confusion matrix, and classification report.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Get the top 20 features
    feature_importances = clf.feature_importances_
    feature_names = X_test.columns
    top_features = sorted(zip(feature_importances, feature_names), reverse=True)[:20]


    performance = {
        'model': "Decision_tree_Classifier",
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'top_features': top_features
    }

    return performance
