import joblib
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, f1_score
from credit_fraud_utils_data import load_data, split_data, resample_data
from models import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
PATH_TRAIN = r'C:\Users\lap shop\OneDrive\Documents\machin lear\project 2\split\train.csv'
PATH_VAL = r'C:\Users\lap shop\OneDrive\Documents\machin lear\project 2\split\val.csv'
THRESHOLDS = {
    'RandomForest': 0.8,
    'MLPClassifier': 0.9,
    'LogisticRegression': 0.5,
    'VotingClassifier': 0.65
}
SELECTED_COLUMNS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V11', 'V12',
                    'V14', 'V16', 'V17', 'V18', 'V19', 'V21', 'Time_log', 'Amount_log']


def evaluate_model(model, X: np.ndarray, t: np.ndarray, threshold: float, data: str) -> dict:
    """
    Evaluates the given model using the provided dataset.

    Parameters:
        model: The model to evaluate.
        X (np.ndarray): Features for evaluation.
        t (np.ndarray): True labels.
        threshold (float): Threshold for binary classification.
        data (str): Name of the dataset (e.g., 'Training', 'Validation').

    Returns:
        dict: F1 score and PR AUC of the model.
    """
    model_name = type(model.named_steps['Model']).__name__ if hasattr(model, 'named_steps') else type(model).__name__

    # Predict probabilities for the positive class
    t_pred_prob = model.predict_proba(X)[:, 1]

    # Convert probabilities to binary predictions using the specified threshold
    t_pred = (t_pred_prob >= threshold).astype(int)

    # Calculate metrics
    f1 = f1_score(t, t_pred)
    precision, recall, _ = precision_recall_curve(t, t_pred)
    pr_auc = auc(recall, precision)

    # Separate logging results for each metric
    print(f"\n{data} Evaluation for {model_name}")
    print("-" * 30)
    print(f"F1 Score: {f1:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print("-" * 30)

    # Log results if needed
    logging.info(f"{data} F1 Score for {model_name}: {f1:.4f}")
    logging.info(f"{data} PR AUC for {model_name}: {pr_auc:.4f}")

    return {
        'f1': f1,
        'pr_auc': pr_auc,
    }


def apply_log_transform(X: pd.DataFrame) -> pd.DataFrame:
    """
    Applies log transformation to 'Time' and 'Amount' columns.

    Parameters:
        X (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    X['Time_log'] = np.log1p(X['Time'])
    X['Amount_log'] = np.log1p(X['Amount'])
    return X


if __name__=='__main__':
    # Load and process the data
    train = load_data(PATH_TRAIN)
    val = load_data(PATH_VAL)

    X_train, t_train = split_data(train)
    X_val, t_val = split_data(val)

    # Apply log transformation
    X_train = apply_log_transform(X_train)
    X_val = apply_log_transform(X_val)

    # Select relevant features
    X_train = X_train[SELECTED_COLUMNS]
    X_val = X_val[SELECTED_COLUMNS]

    # Resample training data
    X_train_over, t_train_over = resample_data(X_train, t_train, 'under_over')

    # Load models
    random_forest = joblib.load('random_forest_model.pkl')
    mlp_classifier = joblib.load('mlp_classifier_model.pkl')
    logistic_regression = joblib.load('logistic_regression_model.pkl')
    voting_classifier = joblib.load('voting_classifier_model.pkl')
    bosting=joblib.load('xgboost_model.pkl')

    # Evaluate models on training and validation sets
    evaluate_model(random_forest, X_train_over, t_train_over, THRESHOLDS['RandomForest'], 'Training')
    evaluate_model(random_forest, X_val, t_val, THRESHOLDS['RandomForest'], 'Validation')

    evaluate_model(mlp_classifier, X_train_over, t_train_over, THRESHOLDS['MLPClassifier'], 'Training')
    evaluate_model(mlp_classifier, X_val, t_val, THRESHOLDS['MLPClassifier'], 'Validation')

    evaluate_model(logistic_regression, X_train_over, t_train_over, THRESHOLDS['LogisticRegression'], 'Training')
    evaluate_model(logistic_regression, X_val, t_val, THRESHOLDS['LogisticRegression'], 'Validation')

    evaluate_model(voting_classifier, X_train_over, t_train_over, THRESHOLDS['VotingClassifier'], 'Training')
    evaluate_model(voting_classifier, X_val, t_val, THRESHOLDS['VotingClassifier'], 'Validation')

    evaluate_model(bosting,X_val,t_val,0.9,'Training')
    evaluate_model(bosting,X_val,t_val,0.9,'Valdtion')



"""

Training F1 Score for RandomForestClassifier is : 0.9637
Training PR AUC for RandomForestClassifier is : 0.9665
valing F1 Score for RandomForestClassifier is : 0.8743
valing PR AUC for RandomForestClassifier is : 0.8797

Training F1 Score for MLPClassifier is : 0.9885
Training PR AUC for MLPClassifier is : 0.9890
valing F1 Score for MLPClassifier is : 0.8276
valing PR AUC for MLPClassifier is : 0.8287

Training F1 Score for LogisticRegression is : 0.8723
Training PR AUC for LogisticRegression is : 0.8901
valing F1 Score for LogisticRegression is : 0.7929
valing PR AUC for LogisticRegression is : 0.7965

Training F1 Score for VotingClassifier is : 0.9578
Training PR AUC for VotingClassifier is : 0.9613
valing F1 Score for VotingClassifier is : 0.8757
valing PR AUC for VotingClassifier is : 0.8796


XGBF1 Score: 0.8554
XGBPR AUC: 0.8617
-------------------

"""
