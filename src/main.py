
from src.model_training import *
from src.report_generation import *
from src.data_preprocessing import *
from src.data_loading import *
import src.config as config
from src.feature_selection import *
from src.feature_importance import *


def main(data_file_path, model_type, target_variables, feature_selection_criteria='SHAP'):
    """
    Main function to process data, train and evaluate models, and save feature importances.

    Parameters:
    - data_file_path (str): The path to the CSV data file.
    - model_type (str): The type of model to train.
    - target_variables (list): A list of target variables to process.
    - feature_selection_criteria (str): The criteria for selecting feature importance ('SHAP' or 'feature_importance_scores').

    Returns:
    - DataFrame: A DataFrame containing feature importances for all target variables.
    """

    # Model dictionary
    models = {
        'decision_tree': DecisionTreeClassifier(
            criterion='gini',  # Measure of split quality ('gini' or 'entropy')
            max_depth=3,  # Maximum depth of the tree
            min_samples_split=2,  # Minimum number of samples required to split an internal node
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,  # Number of trees in the forest
            random_state=42
        ),
        'logistic_regression': LogisticRegression(
            penalty='l2',  # Norm used in the penalization ('l1', 'l2', or 'elasticnet')
            C=1.0,  # Inverse of regularization strength
            solver='lbfgs',  # Algorithm to use in the optimization problem
            max_iter=1000,  # Maximum number of iterations taken for the solvers to converge
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,  # Number of boosting rounds
            learning_rate=0.1,  # Boosting learning rate
            max_depth=5,  # Maximum depth of a tree
            subsample=1,  # Subsample ratio of the training instances
            colsample_bytree=1,  # Subsample ratio of columns when constructing each tree
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
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

    save_to_csv(df_train,config.processed_data_dir,'processed_data_train.csv', index=False)
    save_to_csv(df_test,config.processed_data_dir,'processed_data_test.csv', index=False)

    if model_type not in models:
        raise ValueError("Model type not supported")

    model = models[model_type]

    # Train and evaluate models for each target variable
    all_results = {}
    test_accuracies = []
    all_feature_importances = []

    for target in target_variables:
        print(f"Training and evaluating models for target: {target}")

        # Separate features and target for training and testing sets
        X_train = drop_columns(df_train, config.target_columns)
        y_train = df_train[target]
        X_test = drop_columns(df_test, config.target_columns)
        y_test = df_test[target]

        # Perform feature selection using RFE
        selected_features = select_features_rfe(X_train, y_train, n_features_to_select=50)
        print(f"Selected Features for {target}: {selected_features}")
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # SMOTE sampling
        X_train_res, y_train_res = apply_smote(X_train_selected, y_train)
        counts = y_train_res.value_counts()
        print(counts)

        trained_model = train_model(model, X_train_res, y_train_res)
        # Evaluate model
        result = evaluate_model(trained_model, X_test_selected, y_test)
        test_accuracies.append({'Target': target, 'Accuracy': result['accuracy']})

        if feature_selection_criteria == 'SHAP':
            top_features = get_feature_importance_shap(trained_model, X_train_res, X_test_selected, top_n=20,
                                                       model_folder='../Results', target_name=target)
            all_feature_importances.append(top_features)

        elif feature_selection_criteria=='FIS':
            if model_type == 'logistic_regression':
                top_features = get_feature_importance_logistic_regression(trained_model, X_train_res,
                                                                          model_folder='../Results/Logistic_regression',
                                                                          target_name=target, top_n=20)
            else:
                top_features = get_feature_importance_scores(trained_model, X_train_res, model_folder='../Results',
                                                             target_name=target, top_n=20)
            all_feature_importances.append(top_features)

        #result['top_features'] = top_features
        all_results[target] = result

    save_evaluation_metrics(all_results, model_type, '../Results')

    # Concatenate all feature importances
    all_feature_importances_df = pd.concat(all_feature_importances)
    output_file = os.path.join(f'../Results/{model_type}', f'feature_importances_{feature_selection_criteria}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    all_feature_importances_df.to_csv(output_file, index=False)
    print(f"All feature importances saved to {output_file}")

    save_confusion_matrix(all_results, model_type, '../Results/')
    test_accuracies_df = pd.DataFrame(test_accuracies)

    return all_feature_importances_df, test_accuracies_df


# if __name__ == "__main__":
#     #df_st=main(config.raw_data_file_path,"decision_tree",["kessler_dummy2","kessler_dummy3","kessler_dummy4"],'FIS')
#     # df_rf = main(config.raw_data_file_path, "random_forest", ["kessler_dummy2","kessler_dummy3","kessler_dummy4"],'FIS')
#     # # df_gb = main(config.raw_data_file_path, "XGBoost", ["kessler_dummy2","kessler_dummy3","kessler_dummy4"],'FIS')
#     # #df_ls= main(config.raw_data_file_path, "logistic_regression", ["kessler_dummy2","kessler_dummy3","kessler_dummy4"],'FIS')
