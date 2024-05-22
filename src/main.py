import os
from data_preprocessing import drop_columns
from data_loading import read_csv_data
from src.model_training import *
from src.report_generation import *
from feature_importance import plot_feature_importance
from src.models.logistic_regression import train_logistic_regression, evaluate_logistic_regression
from src.models.decision_tree import train_decision_tree, evaluate_decision_tree
from src.models.random_forest import train_random_forest, evaluate_random_forest
from src.models.kNN import train_knn, evaluate_knn
from src.models.XGB import train_xgboost, evaluate_xgboost
from src.model_training import compute_class_weights
from src.data_preprocessing import *
from src.data_loading import split_train_test
import src.config as config
from feature_selection import *
from sklearn.tree import DecisionTreeClassifier


def main():

    # Load data
    df = read_csv_data(config.raw_data_file_path)
    df= drop_columns(df, config.columns_to_drop)
    # Split train and test data
    df_train,df_test = split_train_test(df,test_size=0.2,output_folder='../Data/raw')

    # Handle missing data in gender specific columns
    df_train = handle_gender_specific_columns(df_train, "female1", config.gender_specific_columns_file_path)
    df_test= handle_gender_specific_columns(df_test, "female1", config.gender_specific_columns_file_path)

    # Handle missing data
    df_train = handle_missing_data(df_train, config.numerical_columns, config.categorical_columns)
    df_test = handle_missing_data(df_test, config.numerical_columns, config.categorical_columns)

    # Normalize numerical features
    df_train = normalize_features(df_train, config.numerical_columns)
    df_test = normalize_features(df_test, config.numerical_columns)

    # Save the processed data
    df_train.to_csv("../Data/processed/processed_data_train.csv", index=False)
    df_test.to_csv("../Data/processed/processed_data_test.csv", index=False)

    analyze_dataframe(df_train)
    analyze_dataframe(df_test)

    # Train and evaluate models for each target variable
    all_results = {}
    model_name = None
    for target in config.target_columns:
        print(f"Training and evaluating models for target: {target}")

        # Separate features and target for training and testing sets
        X_train = drop_columns(df_train, config.target_columns)
        y_train = df_train[target]
        X_test = drop_columns(df_test, config.target_columns)
        y_test = df_test[target]

        # Perform feature selection using Lasso
        selected_features = select_features_rfe(X_train, y_train, n_features_to_select=50)
        print(f"Selected Features for {target}: {selected_features}")
        X_train_selected= X_train[selected_features]
        X_test_selected = X_test[selected_features]

        #SMOTE sampling
        X_train_res, y_train_res = apply_smote(X_train_selected, y_train)

        #Cross Validatiom  Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Perform cross-validation
        mean_cv_score, std_cv_score = cross_validate_model(model, X_train_res, y_train_res)
        print(f"Cross-Validation Mean Score for {target}: {mean_cv_score}")
        print(f"Cross-Validation Std Dev for {target}: {std_cv_score}")

        #Train the model on the entire training set
        trained_model = train_random_forest(model,X_train_res, y_train_res)
        # Evaluate model
        result = evaluate_random_forest(trained_model, X_test_selected, y_test)
        all_results[target] = result
        # model_dt = DecisionTreeClassifier(random_state=42)
        # trained_model=train_decision_tree(model_dt,X_train_res, y_train_res)
        # result= evaluate_decision_tree(trained_model, X_test_selected, y_test)
        # all_results[target] = result

    model_name = "random_forest_with_rfe"
    save_evaluation_metrics(all_results, model_name,'../Results')
    save_feature_importances(all_results, model_name,'../Results/')
    save_confusion_matrix(all_results, model_name,'../Results/')

if __name__ == "__main__":
    main()
