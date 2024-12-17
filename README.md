

# Credit Card Fraud Detector

The **Credit Card Fraud Detector** project aims to develop a robust machine learning model to identify fraudulent transactions in credit card datasets. By utilizing advanced sampling techniques, preprocessing methods, and various classification algorithms, the project seeks to enhance the accuracy and reliability of fraud detection.

## Table of Contents
1. [Key Insights](#key-insights)  
2. [Sampling Techniques](#sampling-techniques)  
3. [Modeling](#modeling)  
4. [Results](#results)  

## Key Insights

Key findings from the Exploratory Data Analysis (EDA) include:

- **Class Imbalance**: The dataset is heavily imbalanced, requiring oversampling or undersampling techniques to improve model performance.
- **Feature Distributions**: Some features, such as `Amount`, exhibit skewness and outliers that could affect predictive modeling.
- **Correlations**: Strong correlations between specific features and the target variable highlight important predictors of fraud.
- **Fraud Patterns**: Certain transaction patterns, like specific amounts, indicate identifiable markers of fraudulent activity.

## Sampling Techniques

The project employs the following resampling methods to address class imbalance:

- **Oversampling**: Replicates minority class samples.  
- **Undersampling**: Reduces majority class samples.  
- **SMOTE**: Generates synthetic samples for the minority class.  
- **Combination**: Combines undersampling with oversampling or SMOTE.

## Modeling

Several classification algorithms were utilized:

1. **Logistic Regression**: A simple yet effective linear classification model.  
2. **Random Forest Classifier**: An ensemble method using decision trees.  
3. **MLP Classifier**: A neural network-based classifier.  
4. **Voting Classifier**: Combines predictions from multiple models for enhanced performance.

## Results

Performance metrics for training and testing sets:

| **Model**                 | **Training F1** | **Testing F1** | **Training AUC-PR** | **Testing AUC-PR** |
|---------------------------|-----------------|----------------|---------------------|--------------------|
| Logistic Regression       | 0.809           | 0.802          | 0.655               | 0.646             |
| Random Forest Classifier  | 0.998           | 0.835          | 0.997               | 0.698             |
| MLP Classifier            | 0.902           | 0.794          | 0.814               | 0.634             |
| Voting Classifier         | 0.963           | 0.800          | 0.929               | 0.644             |

---

