import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def generate_report(performances):
    report = []
    for perf in performances:
        model_name = perf['model'].__class__.__name__
        accuracy = perf['accuracy']
        report.append(f"Model: {model_name}")
        report.append(f"Accuracy: {accuracy:.4f}")
        report.append("Confusion Matrix:")
        report.append(str(perf['conf_matrix']))
        report.append("Classification Report:")
        report.append(str(perf['class_report']))
        report.append("\n" + "=" * 50 + "\n")

    return "\n".join(report)


def save_evaluation_metrics(results, model_name, output_folder):
    """
    Save model evaluation metrics to a CSV file.

    Parameters:
    - results (dict): A dictionary containing evaluation results for each target variable.
    - model_name (str): The name of the model.
    - output_folder (str): The folder to save the CSV file.

    Returns:
    - None
    """
    data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = os.path.join(output_folder, model_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    for target, result in results.items():
        try:
            accuracy = result['accuracy']
            for class_label, scores in result['class_report'].items():
                if class_label in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                row = {
                    'Target': target,
                    'Model': model_name,
                    'Class': class_label,
                    'Accuracy': accuracy,
                    'Precision': scores['precision'],
                    'Recall': scores['recall'],
                    'F1-Score': scores['f1-score'],
                    'Support': scores['support']
                }
                data.append(row)
        except KeyError as e:
            print(f"KeyError: Missing key {e} in result for target {target}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        df = pd.DataFrame(data)
        output_file = os.path.join(model_folder, f'evaluation_metrics_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        print(f"Evaluation metrics saved to {output_file}")


def save_feature_importances(results, model_name, output_folder):
    """
    Save model feature importances to a CSV file and return a DataFrame of some details.

    Parameters:
    - results (dict): A dictionary containing evaluation results for each target variable.
    - model_name (str): The name of the model.
    - output_folder (str): The folder to save the CSV file.

    Returns:
    - DataFrame: A DataFrame containing the target, feature rank, and feature name.
    """
    data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = os.path.join(output_folder, model_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    for target, result in results.items():
        if 'top_features' in result:
            for rank, (importance, feature) in enumerate(result['top_features'], start=1):
                row = {
                    'Target': target,
                    'Model': model_name,
                    'Feature_Rank': rank,
                    'Feature': feature,
                    'Importance': importance
                }
                data.append(row)

    if data:
        df = pd.DataFrame(data)
        output_file = os.path.join(model_folder, f'feature_importances_{timestamp}.csv')
        df.to_csv(output_file, index=False)
        print(f"Feature importances saved to {output_file}")
        return df[['Target', 'Feature_Rank', 'Feature']]
    else:
        print("No data to save.")
        return pd.DataFrame()


def save_confusion_matrix(results, model_name, output_folder):
    """
    Save all confusion matrices as JPEG images.

    Parameters:
    - results (dict): A dictionary containing evaluation results for each target variable.
    - model_name (str): The name of the model.
    - output_folder (str): The folder to save the JPEG files.

    Returns:
    - None
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = os.path.join(output_folder, model_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    for target, result in results.items():
        try:
            conf_matrix = result['conf_matrix']
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
            plt.title(f'Confusion Matrix for {model_name} - {target}')
            plt.xlabel('Predicted')
            plt.ylabel('True')

            output_file = os.path.join(model_folder, f'conf_matrix_{target}_{timestamp}.jpeg')
            plt.savefig(output_file)
            plt.close()
            print(f"Confusion matrix for {target} saved to {output_file}")
        except KeyError as e:
            print(f"KeyError: Missing key {e} in result for target {target}")
        except Exception as e:
            print(f"Unexpected error: {e}")
