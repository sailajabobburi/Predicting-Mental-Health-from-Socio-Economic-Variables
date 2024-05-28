import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def plot_feature_importance(model, feature_names):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 6))
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Model {model.__class__.__name__} does not support feature importances.")
    except Exception as e:
        print(f"Failed to plot feature importance: {e}")

def get_shap_explainer(model,X_train):
    """
    Initialize the appropriate SHAP explainer based on the model type.

    Parameters:
    model: The trained model.

    Returns:
    explainer: The SHAP explainer.
    """
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_train)
    else:
        raise ValueError("Model type not supported for SHAP explainer")
    return explainer


def get_feature_importance_shap(model,X_train,X_test, top_n=20, model_folder='.', target_name='Target'):
    """
    Calculate and rank features by their SHAP values for a random forest model.

    Parameters:
    model: The trained random forest model.
    X_test: DataFrame containing the test features.
    top_n: The number of top features to return (default is 20).
    model_folder: The folder where the output CSV file will be saved (default is current directory).
    target_name: The name of the target variable for the output CSV (default is 'Target').

    Returns:
    DataFrame containing the top N features ranked by their importance, along with their ranks and target variable.
    """
    # Create the appropriate SHAP explainer
    explainer = get_shap_explainer(model,X_train)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    # Print the type and shape of shap_values for debugging
    print(f"Type of shap_values: {type(shap_values)}")
    print(f"Shape of shap_values: {shap_values.shape}")

    # Initialize shap_values_class to ensure it is always defined
    shap_values_class = shap_values  # Default assignment

    if isinstance(shap_values, list):
        # For binary classification, use the SHAP values for the positive class (class 1)
        shap_values_class = shap_values[1]
    elif shap_values.ndim == 3:
        # For 3D arrays, use the last dimension to determine class
        shap_values_class = shap_values[:, :, 1]  # Assuming binary classification and interested in class 1



    # SHAP values for the class of interest
    #shap_values_class = shap_values_class_1  # Assuming binary classification and class of interest is 1

    # Compute mean absolute SHAP values for each feature
    shap_values_mean = np.mean(np.abs(shap_values_class), axis=0)

    # Debug: Print lengths and shapes to ensure they match and are 1-dimensional
    print(f"Length of X_test.columns: {len(X_test.columns)}")
    print(f"Length of shap_values_mean: {len(shap_values_mean)}")
    print(f"Shape of shap_values_mean: {shap_values_mean.shape}")
    print(f"Shape of X_test.columns: {X_test.columns.shape}")

    # Ensure shap_values_mean is 1-dimensional
    if shap_values_mean.ndim != 1:
        raise ValueError("shap_values_mean is not 1-dimensional")

    # Create a DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': shap_values_mean
    })

    # Sort features by importance and get the top N
    top_features = feature_importance.sort_values(by='Importance', ascending=False).head(top_n).reset_index(drop=True)

    # Add rank and target columns
    top_features['Feature_Rank'] = top_features.index + 1
    top_features['Target'] = target_name

    # # Save to CSV
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # output_file = os.path.join(model_folder, f'feature_importances_{timestamp}.csv')
    # top_features.to_csv(output_file, index=False)
    # print(f"Feature importances saved to {output_file}")

    # Return the DataFrame with selected columns
    return top_features[['Target', 'Feature_Rank', 'Feature']]


def get_feature_importance_logistic_regression(model, X_train, model_folder='Results', target_name='Target', top_n=20):
    """
    Extract the top N feature importance scores from a trained logistic regression model.

    Parameters:
    model (LogisticRegression): Trained logistic regression model.
    X_train (DataFrame): Training feature set.
    model_folder (str): Folder where the output CSV file will be saved.
    target_name (str): Name of the target variable for the output CSV.
    top_n (int): Number of top features to return.

    Returns:
    DataFrame: DataFrame containing the top N features and their importance scores.
    """
    # Extract the coefficients
    coefficients = model.coef_[0]  # Assuming binary classification

    # Calculate feature importance scores (absolute values of the coefficients)
    feature_importance = np.abs(coefficients)

    # Create a DataFrame to store feature names and their importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    })

    # Rank the features based on their importance scores
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Select the top N features
    top_features = feature_importance_df.head(top_n)

    # Add rank and target columns
    # top_features['Feature_Rank'] = top_features.index + 1
    # top_features['Target'] = target_name
    # Add rank and target columns using .loc to avoid SettingWithCopyWarning
    top_features.loc[:, 'Feature_Rank'] = top_features.index + 1
    top_features.loc[:, 'Target'] = target_name

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(model_folder, f'feature_importances_logistic_regression_{timestamp}.csv')
    os.makedirs(model_folder, exist_ok=True)
    top_features.to_csv(output_file, index=False)
    print(f"Feature importances saved to {output_file}")

    return top_features[['Target', 'Feature_Rank', 'Feature']]

def get_feature_importance_scores(model, X_train, model_folder='Results', target_name='Target', top_n=20):
    """
    Extract the top N feature importance scores from a trained tree-based model.

    Parameters:
    model: Trained tree-based model (e.g., RandomForestClassifier, GradientBoostingClassifier, XGBClassifier).
    X_train (DataFrame): Training feature set.
    model_folder (str): Folder where the output CSV file will be saved.
    target_name (str): Name of the target variable for the output CSV.
    top_n (int): Number of top features to return.

    Returns:
    DataFrame: DataFrame containing the top N features and their importance scores.
    """
    # Extract the feature importances
    feature_importance = model.feature_importances_

    # Create a DataFrame to store feature names and their importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    })

    # Rank the features based on their importance scores
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # Select the top N features
    top_features = feature_importance_df.head(top_n).copy()  # Create a copy to avoid SettingWithCopyWarning

    # Add rank and target columns using .loc to avoid SettingWithCopyWarning
    top_features.loc[:, 'Feature_Rank'] = top_features.index + 1
    top_features.loc[:, 'Target'] = target_name

    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(model_folder, f'feature_importances_{model.__class__.__name__}_{timestamp}.csv')
    os.makedirs(model_folder, exist_ok=True)
    top_features.to_csv(output_file, index=False)
    print(f"Feature importances saved to {output_file}")

    return top_features[['Target', 'Feature_Rank', 'Feature']]