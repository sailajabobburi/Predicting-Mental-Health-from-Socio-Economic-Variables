# utils.py
import shap
import matplotlib.pyplot as plt

def get_shap_values(model, X):
    """
    Get SHAP values from a trained model.

    Parameters:
    - model: Trained model.
    - X (DataFrame): Feature DataFrame.

    Returns:
    - shap_values: SHAP values for the input features.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer

def plot_shap_importance(shap_values, X, max_display=20):
    """
    Plot SHAP feature importance bar plot.

    Parameters:
    - shap_values: SHAP values for the input features.
    - X (DataFrame): Feature DataFrame.
    - max_display (int): Maximum number of features to display.

    Returns:
    - None
    """
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display)
    plt.show()
