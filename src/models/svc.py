from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_svc(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model


def evaluate_svc(model, X_test, y_test):
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
