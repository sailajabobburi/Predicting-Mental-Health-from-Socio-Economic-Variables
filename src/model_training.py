from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


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

def train_model(model, X_train, y_train):
    """
    Train a generic model.

    Parameters:
    - model (estimator): Any scikit-learn-style estimator.
    - X_train (DataFrame): The training input samples.
    - y_train (Series): The target values.

    Returns:
    - model: The trained model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a generic trained model.

    Parameters:
    - model (estimator): Any scikit-learn-style estimator.
    - X_test (DataFrame): The test input samples.
    - y_test (Series): The true target values for the test set.

    Returns:
    - dict: A dictionary containing the model's name, accuracy, confusion matrix, classification report, and top features if available.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    results = {
        'model': model.__class__.__name__,
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
    }

    # Handling feature importances if the model has them
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_names = X_test.columns
        top_features = sorted(zip(feature_importances, feature_names), reverse=True)[:20]
        results['top_features'] = top_features

    return results

