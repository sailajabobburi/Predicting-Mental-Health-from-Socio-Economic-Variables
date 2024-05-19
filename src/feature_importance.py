import numpy as np
import matplotlib.pyplot as plt

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
