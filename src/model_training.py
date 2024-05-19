from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def split_data(df, target_column, test_size=0.2, random_state=42):
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Data split successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Failed to split data: {e}")
        return None, None, None, None


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    try:
        model.fit(X_train, y_train)
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
        print(f"Model: {model.__class__.__name__}")
        print(f'Accuracy: {accuracy}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
        return performance
    except Exception as e:
        print(f"Failed to train and evaluate model: {e}")
        return None


def compare_models(X_train, y_train, X_test, y_test):
    models = [
        LogisticRegression(),
        RandomForestClassifier(),
        SVC()
    ]

    performance_list = []
    for model in models:
        performance = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        if performance:
            performance_list.append(performance)

    return performance_list
