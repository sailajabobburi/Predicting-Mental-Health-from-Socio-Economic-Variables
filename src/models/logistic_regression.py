from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model


def evaluate_logistic_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    performance = {
        'model': model,
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report
    }
    return performance
