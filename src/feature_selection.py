from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def select_features_lasso(X, y, alpha=0.01):
    """
    Select features using Lasso regression.

    Parameters:
    - X (DataFrame): Normalized feature DataFrame.
    - y (Series): Target variable.
    - alpha (float): Regularization strength. Default is 0.01.

    Returns:
        - list: List of selected feature names.
    """
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    coef = lasso.coef_
    selected_features = X.columns[(coef != 0)]

    return selected_features


def select_features_rfe(X, y, n_features_to_select=50):
    """
    Select features using Recursive Feature Elimination (RFE).

    Parameters:
    - X (DataFrame): Feature DataFrame.
    - y (Series): Target variable.
    - n_features_to_select (int): Number of features to select. Default is 10.

    Returns:
    - list: List of selected feature names.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(model, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]

    if len(selected_features) == 0:
        raise ValueError("No features were selected. Try increasing the number of features to select.")

    return selected_features