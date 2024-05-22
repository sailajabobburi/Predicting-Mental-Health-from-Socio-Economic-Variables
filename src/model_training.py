from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import cross_val_score


def split_data(df, target_columns, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets for each target variable.

    Parameters:
    - df (DataFrame): The DataFrame to split.
    - target_columns (list of str): List of target columns.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - dict: A dictionary containing the training and testing sets for each target variable.
    """
    splits = {}
    X = df.drop(columns=target_columns)

    for target in target_columns:
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        splits[target] = (X_train, X_test, y_train, y_test)

    return splits

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
        SVC(),
        HistGradientBoostingClassifier()
    ]

    performance_list = []
    for model in models:
        performance = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        if performance:
            performance_list.append(performance)

    return performance_list

def compute_class_weights(y):
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(zip(np.unique(y), class_weights))


def cross_validate_model(model, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Perform cross-validation on the given model.

    Parameters:
    - model: The machine learning model to cross-validate.
    - X_train (DataFrame): Features of the training set.
    - y_train (Series): Target variable of the training set.
    - cv (int): Number of cross-validation folds. Default is 5.
    - scoring (str): Scoring method to use. Default is 'accuracy'.

    Returns:
    - mean_score (float): Mean cross-validation score.
    - std_score (float): Standard deviation of cross-validation scores.
    """
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return mean_score, std_score
# Compute class weights
# class_weights = compute_class_weights(y_train)
# scale_pos_weight = class_weights[1] / class_weights[0]  # For binary classification
#
# clf = train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=6,
#                     scale_pos_weight=scale_pos_weight)
# result = evaluate_xgboost(clf, X_test, y_test)
