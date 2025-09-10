import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.pipeline import Pipeline as imbpipeline

# Set a global random state for reproducibility
RANDOM_STATE = 42

# Function to load dataset
def load_data(path_dataset):
    """
    Load dataset from a CSV file.

    Parameters:
    - path_dataset (str): Path to the dataset.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(path_dataset)

# Function to split data into features and target
def split_data(data, target_column='Class'):
    """
    Split the dataset into features (X) and target (y).

    Parameters:
    - data (pd.DataFrame): The dataset.
    - target_column (str): The name of the target column. Default is 'Class'.

    Returns:
    - X_data (pd.DataFrame): Feature data.
    - y_data (pd.Series): Target labels.
    """
    X_data = data.drop(columns=[target_column])
    y_data = data[target_column]
    return X_data, y_data

# Function for scaling data
def scale_data(X_train, X_val):
    """
    Scale training and validation data using StandardScaler.

    Parameters:
    - X_train (pd.DataFrame): Training feature data.
    - X_val (pd.DataFrame): Validation feature data.

    Returns:
    - X_train_scaled (np.ndarray): Scaled training data.
    - X_val_scaled (np.ndarray): Scaled validation data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled

# Function for undersampling and oversampling
def resample_data(X, y, strategy="under_over", random_state=RANDOM_STATE):
    """
    Apply a specified resampling strategy to balance the dataset.

    Parameters:
    - X (pd.DataFrame): Input features.
    - y (pd.Series): Target labels.
    - strategy (str): Resampling strategy to use. Options:
        - "under": Perform undersampling.
        - "over": Perform SMOTE oversampling.
        - "under_over": Perform undersampling followed by SMOTE oversampling.
    - random_state (int): Seed for reproducibility. Default is RANDOM_STATE.

    Returns:
    - X_resampled (pd.DataFrame): Resampled features.
    - y_resampled (pd.Series): Resampled target labels.
    """
    print(f"Original class distribution: {Counter(y)}")

    if strategy == "under":
        # Perform Random Undersampling
        minority_size = Counter(y)[1]
        undersample = RandomUnderSampler(random_state=random_state, sampling_strategy={0: 10 * minority_size})
        X_resampled, y_resampled = undersample.fit_resample(X, y)

    elif strategy == "over":
        # Perform SMOTE Oversampling
        smote = SMOTE(sampling_strategy=0.05, random_state=random_state, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)

    elif strategy == "under_over":
        # Perform undersampling followed by SMOTE oversampling
        majority_size = Counter(y)[0]
        new_maj_size = int(majority_size * 0.4)
        new_min_size = int(new_maj_size * 0.10)

        new_maj_size = max(new_maj_size, 1)
        new_min_size = max(new_min_size, 1)

        undersample_strategy = {0: new_maj_size}
        oversample_strategy = {1: new_min_size}

        # Define pipeline for resampling
        resample_pipeline = imbpipeline(steps=[
            ('under', RandomUnderSampler(random_state=random_state, sampling_strategy=undersample_strategy)),
            ('over', SMOTE(random_state=random_state, sampling_strategy=oversample_strategy, k_neighbors=5))
        ])
        X_resampled, y_resampled = resample_pipeline.fit_resample(X, y)

    else:
        raise ValueError(f"Invalid strategy: {strategy}. Use 'under', 'over', or 'under_over'.")

    print(f"Class distribution after '{strategy}' sampling: {Counter(y_resampled)}")
    return X_resampled, y_resampled

# Function for cost-sensitive analysis (for further development)
def cost_sensitive(X, y):
    """
    Calculate the imbalance ratio (IR) of the dataset.

    Parameters:
    - X (pd.DataFrame): Input features.
    - y (pd.Series): Target labels.

    Returns:
    - None: Prints the imbalance ratio for class 0 to class 1.
    """
    class_counts = Counter(y)
    print(f"Class distribution: {class_counts}")
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"Imbalance Ratio (IR): {imbalance_ratio}")
