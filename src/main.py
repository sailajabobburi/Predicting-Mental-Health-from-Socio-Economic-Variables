
from src.model_training import *
from src.report_generation import *
from src.data_preprocessing import *
from src.data_loading import split_train_test
import src.config as config
from feature_selection import *
from src.model_interpretation import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def main(data_file_path, model_type, target_variables):
    """
    Main function to process data, train and evaluate models, and save feature importances.

    Parameters:
    - data_file_path (str): The path to the CSV data file.
    - model_type (str): The type of model to train.
    - target_variables (list): A list of target variables to process.

    Returns:
    - DataFrame: A DataFrame containing feature importances for all target variables.
    """

    # Model dictionary
    models = {
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42)
    }

    # Load data
    df = read_csv_data(data_file_path)
    df = drop_columns(df, config.columns_to_drop)
    # Split train and test data
    df_train, df_test = split_train_test(df, test_size=0.2, output_folder='../Data/raw')

    # Handle missing data in gender specific columns
    df_train = handle_gender_specific_columns(df_train, "female1", config.gender_specific_columns_file_path)
    df_test = handle_gender_specific_columns(df_test, "female1", config.gender_specific_columns_file_path)

    # Handle missing data
    df_train = handle_missing_data(df_train, config.numerical_columns, config.categorical_columns)
    df_test = handle_missing_data(df_test, config.numerical_columns, config.categorical_columns)

    # Normalize numerical features
    df_train = normalize_features(df_train, config.numerical_columns)
    df_test = normalize_features(df_test, config.numerical_columns)

    # Save the processed data
    df_train.to_csv("../Data/processed/processed_data_train.csv", index=False)
    df_test.to_csv("../Data/processed/processed_data_test.csv", index=False)

    if model_type not in models:
        raise ValueError("Model type not supported")

    model = models[model_type]

    # Train and evaluate models for each target variable
    all_results = {}
    test_accuracies = []

    for target in target_variables:
        print(f"Training and evaluating models for target: {target}")

        # Separate features and target for training and testing sets
        X_train = drop_columns(df_train, config.target_columns)
        y_train = df_train[target]
        X_test = drop_columns(df_test, config.target_columns)
        y_test = df_test[target]

        # Perform feature selection using Lasso
        selected_features = select_features_rfe(X_train, y_train, n_features_to_select=50)
        print(f"Selected Features for {target}: {selected_features}")
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # SMOTE sampling
        X_train_res, y_train_res = apply_smote(X_train_selected, y_train)
        counts = y_train_res.value_counts()
        print(counts)

        #
        # # Cross Validatiom  Random Forest
        # model = RandomForestClassifier(n_estimators=100, random_state=42)
        # # Perform cross-validation
        # mean_cv_score, std_cv_score = cross_validate_model(model, X_train_res, y_train_res)
        # print(f"Cross-Validation Mean Score for {target}: {mean_cv_score}")
        # print(f"Cross-Validation Std Dev for {target}: {std_cv_score}")

        #Train the model on the entire training set

        trained_model = train_model(model, X_train_res, y_train_res)
        # Evaluate model
        result = evaluate_model(trained_model, X_test_selected, y_test)
        all_results[target] = result
        test_accuracies.append({'Target': target, 'Accuracy': result['accuracy']})

        # # Get SHAP values
        # shap_values, explainer = get_shap_values(trained_model, X_test_selected)
        # shap.summary_plot(shap_values, X_test_selected, feature_names=selected_features)
        # plot_shap_importance(shap_values, X_test_selected, max_display=20)


        # model_dt = DecisionTreeClassifier(random_state=42)
        # trained_model=train_decision_tree(model_dt,X_train_res, y_train_res)
        # result= evaluate_decision_tree(trained_model, X_test_selected, y_test)
        # all_results[target] = result


    save_evaluation_metrics(all_results, model_type, '../Results')
    imp_features_df=save_feature_importances(all_results, model_type, '../Results/')
    save_confusion_matrix(all_results, model_type, '../Results/')
    test_accuracies_df = pd.DataFrame(test_accuracies)

    return imp_features_df,test_accuracies_df

#
# if __name__ == "__main__":
#     df=main(config.raw_data_file_path,"decision_tree",["kessler_dummy2","kessler_dummy3"])
#     print(df)
