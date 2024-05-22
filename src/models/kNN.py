from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
def train_knn(X_train, y_train, n_neighbors=5, metric='euclidean'):
    """
    Train a k-Nearest Neighbors (kNN) classifier.

    Parameters:
    - X_train (DataFrame): The training input samples.
    - y_train (Series): The target values.
    - n_neighbors (int): Number of neighbors to use. Defaults to 5.
    - metric (str): The distance metric to use. Defaults to 'euclidean'.

    Returns:
    - KNeighborsClassifier: The trained kNN model.
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    clf.fit(X_train, y_train)
    return clf


def evaluate_knn(model, X_test, y_test):
    """
    Evaluate a trained k-Nearest Neighbors (kNN) classifier.

    Parameters:
    - model (KNeighborsClassifier): The trained kNN model.
    - X_test (DataFrame): The test input samples.
    - y_test (Series): The true target values for the test set.

    Returns:
    - dict: A dictionary containing the model, accuracy, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    performance = {
        'model': 'KNeighborsClassifier',
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report
    }

    return performance