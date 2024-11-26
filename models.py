import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from credit_fraud_utils_data import load_data, split_data, resample_data

# Set random state for reproducibility
random_state = 43

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def random_search_random_forest(X_train, t_train, X_val, t_val):
    """Perform random search for hyperparameters of Random Forest model."""
    param_distributions_rf = {
        'n_estimators': [10, 50, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [{0: 1, 1: 0.1}, {0: 1, 1: 0.2}, {0: 100, 1: 10}]
    }

    rf = RandomForestClassifier(random_state=random_state)
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    random_search_rf = RandomizedSearchCV(
        rf, param_distributions_rf, n_iter=20, scoring='f1', cv=stratified_kfold,
        n_jobs=-1, verbose=2, random_state=random_state
    )

    random_search_rf.fit(X_train, t_train)
    best_params_rf = random_search_rf.best_params_

    model_rf = random_search_rf.best_estimator_
    t_predict_rf_val = model_rf.predict(X_val)
    score_val_f1 = f1_score(t_val, t_predict_rf_val)

    logging.info(f"Best Random Forest Params: {best_params_rf} | Validation F1 Score: {score_val_f1}")

    return model_rf, best_params_rf, score_val_f1


def random_search_mlp(X_train, t_train, X_val, t_val):
    """Perform random search for hyperparameters of MLP model."""
    param_distributions_mlp = {
        'hidden_layer_sizes': [(100,), (150,), (200,)],
        'activation': ['tanh', 'relu'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 300, 400],
        'solver': ['adam', 'sagd']
    }

    mlp = MLPClassifier(random_state=random_state)
    random_search_mlp = RandomizedSearchCV(
        mlp, param_distributions_mlp, n_iter=20, scoring='f1',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
        n_jobs=-1, verbose=2, random_state=random_state
    )

    random_search_mlp.fit(X_train, t_train)
    best_params_mlp = random_search_mlp.best_params_

    model_mlp = random_search_mlp.best_estimator_
    t_predict_mlp_val = model_mlp.predict(X_val)
    score_val_f1 = f1_score(t_val, t_predict_mlp_val)

    logging.info(f"Best MLP Params: {best_params_mlp} | Validation F1 Score: {score_val_f1}")

    return model_mlp, best_params_mlp, score_val_f1


def random_search_logistic_regression(X_train, t_train, X_val, t_val):
    """Perform random search for hyperparameters of Logistic Regression model."""
    param_distributions_lr = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['elasticnet'],
        'l1_ratio': [0.1, 0.5, 0.9],
        'solver': ['saga'],
        'class_weight': [{0: 1, 1: 0.1}, {0: 1, 1: 0.2}]
    }

    lr = linear_model.LogisticRegression(random_state=random_state)
    random_search_lr = RandomizedSearchCV(
        lr, param_distributions_lr, n_iter=20, scoring='f1',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
        n_jobs=-1, verbose=2, random_state=random_state
    )

    random_search_lr.fit(X_train, t_train)
    best_params_lr = random_search_lr.best_params_

    model_lr = random_search_lr.best_estimator_
    t_predict_lr_val = model_lr.predict(X_val)
    score_val_f1 = f1_score(t_val, t_predict_lr_val)

    logging.info(f"Best Logistic Regression Params: {best_params_lr} | Validation F1 Score: {score_val_f1}")

    return model_lr, best_params_lr, score_val_f1


def random_search_classifier_with_best_models(X_train, t_train, X_val, t_val, model_rf, model_mlp, model_lr):
    """Perform random search for weights in the Voting Classifier."""
    voting_clf = VotingClassifier(
        estimators=[('RandomForest', model_rf), ('MLPClassifier', model_mlp), ('LogisticRegression', model_lr)
        ],
        voting='soft'
    )

    param_distributions_voting = {
        'weights': [
            [1, 1, 1], [1, 2, 1], [1, 1, 2], [2, 1, 1],
            [2, 2, 1], [1, 2, 2], [2, 1, 2], [2, 2, 2],
            [3, 1, 1], [1, 3, 1], [1, 1, 3]
        ]
    }

    random_search_voting = RandomizedSearchCV(
        voting_clf, param_distributions_voting, n_iter=10, scoring='f1',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
        n_jobs=-1, verbose=2, random_state=random_state
    )

    random_search_voting.fit(X_train, t_train)
    best_params_voting = random_search_voting.best_params_

    model_voting = random_search_voting.best_estimator_
    t_predict_voting_val = model_voting.predict(X_val)
    score_val_f1 = f1_score(t_val, t_predict_voting_val)

    logging.info(f"Best Voting Classifier Params: {best_params_voting} | Validation F1 Score: {score_val_f1}")

    return model_voting, best_params_voting, score_val_f1

def random_search_xgboost(X_train, t_train, X_val, t_val):
    """Perform random search for hyperparameters of XGBoost model."""
    param_distributions_xgb = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.8, 0.9, 1]
    }

    xgb = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    random_search_xgb = RandomizedSearchCV(
        xgb, param_distributions_xgb, n_iter=20, scoring='f1', cv=stratified_kfold,
        n_jobs=-1, verbose=2, random_state=random_state
    )

    random_search_xgb.fit(X_train, t_train)
    best_params_xgb = random_search_xgb.best_params_

    model_xgb = random_search_xgb.best_estimator_
    t_predict_xgb_val = model_xgb.predict(X_val)
    score_val_f1 = f1_score(t_val, t_predict_xgb_val)

    logging.info(f"Best XGBoost Params: {best_params_xgb} | Validation F1 Score: {score_val_f1}")

    return model_xgb, best_params_xgb, score_val_f1
def create_pipeline(model):
    """Creates a pipeline with standard scaling followed by the specified model."""
    return Pipeline(steps=[
        ("Scaling", StandardScaler()),
        ("Model", model)
    ])


def main():
    """Main function to execute the model training and evaluation."""
    path_train = r'C:\Users\lap shop\OneDrive\Documents\machin lear\project 2\split\train.csv'
    path_val = r'C:\Users\lap shop\OneDrive\Documents\machin lear\project 2\split\val.csv'

    # Load and process the data
    try:
        train = load_data(path_train)
        val = load_data(path_val)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    X_train, t_train = split_data(train)
    X_val, t_val = split_data(val)

    # Log transformation
    X_train['Time_log'] = np.log1p(X_train['Time'])
    X_val['Time_log'] = np.log1p(X_val['Time'])
    X_train['Amount_log'] = np.log1p(X_train['Amount'])
    X_val['Amount_log'] = np.log1p(X_val['Amount'])

    selected_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V11', 'V12',
                        'V14', 'V16', 'V17', 'V18', 'V19', 'V21', 'Time_log', 'Amount_log']
    X_train = X_train[selected_columns]
    X_val = X_val[selected_columns]

    # Resample data
    X_train_over, t_train_over = resample_data(X_train, t_train, 'under_over')

    # Define and train models
    model_rf = create_pipeline(RandomForestClassifier(random_state=random_state))
    model_mlp = create_pipeline(MLPClassifier(random_state=random_state))
    model_lr = create_pipeline(linear_model.LogisticRegression(random_state=random_state))
    model_xgb=create_pipeline(XGBClassifier(random_state=random_state))


    model_rf, best_rf_params, rf_score_val = random_search_random_forest(X_train_over, t_train_over, X_val, t_val)
    model_mlp, best_mlp_params, mlp_score_val = random_search_mlp(X_train_over, t_train_over, X_val, t_val)
    model_lr, best_lr_params, lr_score_val = random_search_logistic_regression(X_train_over, t_train_over, X_val, t_val)
    model_xgb, best_xgb_params, xgb_score_val = random_search_xgboost(X_train_over, t_train_over, X_val, t_val)

    # Train Voting Classifier
    model_voting, best_voting_params, voting_score_val = random_search_classifier_with_best_models(X_train_over, t_train_over, X_val, t_val, model_rf, model_mlp, model_lr)

    #Save models
    try:
        joblib.dump(model_rf, 'random_forest_model.pkl')
        joblib.dump(model_mlp, 'mlp_classifier_model.pkl')
        joblib.dump(model_lr, 'logistic_regression_model.pkl')
        joblib.dump(model_voting, 'voting_classifier_model.pkl')
        joblib.dump(model_xgb, 'xgboost_model.pkl')

    except Exception as e:
        logging.error(f"Error saving models: {e}")


""" 
in grid searvh
Note : here threshold is not best thereshold here equall default =0.5
:    and i change some hyperramters in found it better 

Best Hyperparameters Summary:
Random Forest: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight': {0: 1, 1: 0.1}} 

MLP Classifier: {'solver': 'adam', 'max_iter': 150, 'hidden_layer_sizes': (150,), 'alpha': 0.0001, 'activation': 'relu'} 

Logistic Regression: {'solver': 'saga', 'penalty': 'elasticnet', 'l1_ratio': 0.9, 'class_weight': {0: 1, 1: 0.2}, 'C': 10} 
Best XGBoost Params: {'subsample': 0.9, 'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.2, 'colsample_bytree': 0.9}
Voting Classifier: {'weights': [1, 3, 1]}
"""

if __name__=='__main__':
    main()
