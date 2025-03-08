# Ghana Panel Survey: Benchmarking and Model Interpretability for Mental Health Analysis

This project utilizes the **Ghana Panel Survey** dataset to benchmark multiple **machine learning models** and apply **SHAP (SHapley Additive exPlanations)** for model interpretability. The primary goal is to **identify the most influential socio-economic variables impacting mental health** by analyzing feature importance rather than solely focusing on classification accuracy.

## Project Overview

- **Benchmarking Machine Learning Models**: Evaluate **Logistic Regression, Decision Trees, Random Forest, and XGBoost** to determine the most effective model for predicting mental health distress.
- **Model Interpretability Using SHAP**: Utilize SHAP values to **explain feature importance**, enabling policymakers to identify key socio-economic factors affecting mental health.
- **Strategic Feature Importance Analysis**: Rank features based on SHAP values to ensure the focus is on the most significant predictors.
- **Data-Driven Insights**: Leverage extensive data from **18,000 participants across 5,000 households** over multiple survey waves to assess trends in mental health distress.

## Features

- **Comprehensive Model Benchmarking**: Compare performance metrics across multiple machine learning models.
- **SHAP-Based Interpretability**: Identify and visualize key socio-economic variables influencing mental health.
- **Class Imbalance Handling**: Use **SMOTE (Synthetic Minority Oversampling Technique)** and oversampling techniques to ensure balanced model training.
- **Feature Engineering**: Address gender-based missing values and create generalized features to improve model robustness.

## Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/ghana-panel-survey-analysis.git
cd ghana-panel-survey-analysis
```

### **2. Set Up a Virtual Environment** (Optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

*Ensure that the `requirements.txt` file includes all necessary libraries, such as `pandas`, `scikit-learn`, `shap`, `matplotlib`, and `imbalanced-learn`.*

## Usage

### **Run the Model Benchmarking and SHAP Analysis**

```bash
python main.py
```

### **View Results**

- Model performance metrics and SHAP visualizations will be saved in the `results/` directory.
- SHAP plots will be generated to highlight the most influential features.


## Model Evaluation and Benchmarking

- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score.
- **Best Model**: Random Forest achieves the highest test accuracy across all Kessler variables.
- **Feature Importance Ranking**: SHAP values are used to rank key socio-economic factors.


## SHAP-Based Feature Importance Analysis

- **SHAP Value Calculation**:
  - Feature A: **+0.9** (Significantly increases distress probability)
  - Feature B: **-0.8** (Significantly decreases distress probability)
  - Feature C: **+0.01** (Marginal impact)

- **Top Features Identified Using SHAP**:
  - **Age (age_yrs1)**
  - **Educational Attainment (educ1)**
  - **Employment Status of Mother (work_mother1)**
  - **Wealth (wealth1), Borrowing (borrow_yn1), Savings (savings_yn1)**
  - **Personality Traits (big5consc_score1, big5agreeable_score1)**
  - **Community Alertness (alert_community1)**
  - **Living in Shared Housing (shared_dwelling1)**

## Future Scope

- **Impact Analysis**: Quantify the direction and magnitude of each socio-economic variable’s impact on mental health.
- **Longitudinal Studies**: Track individual mental health changes over time using socio-economic indicators.
- **Policy Recommendations**: Use insights to inform mental health interventions in Ghana.




