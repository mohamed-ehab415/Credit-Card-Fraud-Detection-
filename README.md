# Credit Card Fraud Detection

<p align="center">
  <img src="assets/4.png" alt="Project Banner" width="600"/>
</p>

Detect fraudulent credit card transactions using machine learning.  
Designed for developers and data scientists interested in advanced fraud detection workflows.

---

## Table of Contents

- [Problem Definition](#problem-definition)
- [Dataset](#dataset)
- [Data Analysis](#data-analysis)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Best Model](#best-model)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Problem Definition

Financial institutions face significant risks from credit card fraud.  
The goal: **build a machine learning model to accurately detect fraudulent transactions**—even with highly imbalanced data.

---

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features:**
  - `Time`: Seconds since first transaction in the dataset
  - `V1`–`V28`: PCA-transformed anonymized features
  - `Amount`: Transaction amount
  - `Class`: Target (0 = legitimate, 1 = fraudulent)

---

## Data Analysis

- **Imbalanced Target:**  
  <p align="center">
    <img src="assets/1.png" alt="Imbalanced Dataset" width="400"/>
  </p>
- **Feature Correlations:**  
  <p align="center">
    <img src="assets/2.png" alt="Feature Correlations" width="400"/>
  </p>

---

## Preprocessing

- **Scaling:**  
  - All features standardized (`StandardScaler`)
- **Resampling:**  
  - **Random Undersampling**: Balances by reducing majority class
  - **SMOTE Oversampling**: Synthesizes minority class samples
  - **Combined**: Undersample majority, then SMOTE minority

---

## Modeling

The following models were implemented, tuned, and evaluated:

### Models

- **Logistic Regression** (baseline, efficient, handles imbalance)
- **Random Forest** (ensemble, robust, captures feature interactions)
- **MLP Classifier** (neural network for complex patterns)
- **Voting Classifier** (ensemble for improved performance)
- **XGBoost** (gradient boosting for structured data)

### Hyperparameter Tuning

Used `RandomizedSearchCV` for all models.

#### Best Hyperparameters

<details>
<summary>Expand for details</summary>

- **Random Forest**
  - `n_estimators`: 100
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1
  - `class_weight`: {0: 1, 1: 0.1}
- **MLP Classifier**
  - `solver`: 'adam'
  - `max_iter`: 150
  - `hidden_layer_sizes`: (150,)
  - `alpha`: 0.0001
  - `activation`: 'relu'
- **Logistic Regression**
  - `solver`: 'saga'
  - `penalty`: 'elasticnet'
  - `l1_ratio`: 0.9
  - `class_weight`: {0: 1, 1: 0.2}
  - `C`: 10
- **XGBoost**
  - `subsample`: 0.9
  - `n_estimators`: 150
  - `max_depth`: 6
  - `learning_rate`: 0.2
  - `colsample_bytree`: 0.9
- **Voting Classifier**
  - `weights`: [1, 3, 1]
</details>

---

## Evaluation

| Model                | Training F1 | Training PR AUC | Validation F1 | Validation PR AUC |
|----------------------|-------------|-----------------|---------------|-------------------|
| Random Forest        | 0.9637      | 0.9665          | 0.8743        | 0.8797            |
| MLP Classifier       | 0.9885      | 0.9890          | 0.8276        | 0.8287            |
| Logistic Regression  | 0.8723      | 0.8901          | 0.7929        | 0.7965            |
| Voting Classifier    | 0.9578      | 0.9613          | 0.8757        | 0.8796            |
| XGBoost              |     —       |      —           | 0.8554        | 0.8617            |

- **Voting Classifier** is the best performer (see [Best Model](#best-model) below).

---

## Best Model

- **Voting Classifier**
  - **Test Results:**
    - F1 Score: `0.8528`
    - PR AUC: `0.8531`

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/mohamed-ehab415/Credit-Card-Fraud-Detection-.git
cd Credit-Card-Fraud-Detection-

# Install dependencies
pip install -r requirements.txt

# Run main analysis (update with your main script name)
python main.py
```

---

## Installation

> **Python 3.8+ required**

```bash
pip install -r requirements.txt
```

---

## Usage

1. **Download the dataset** from Kaggle and place it in the `data/` directory.
2. **Set up your parameters** in `config.py` (if available).
3. **Run analysis or training:**
   ```bash
   python main.py
   ```

---

## Contributing

Pull requests, issues, and suggestions are welcome!  
- Fork the repository
- Create a feature branch (`git checkout -b feature/my-feature`)
- Commit your changes
- Open a pull request

---

## License

This project is for educational and research use.  
See `LICENSE` for details.

---

## Author

**Mohamed Ehab**  
[GitHub](https://github.com/mohamed-ehab415)

---

<p align="center">
  <img src="assets/4.png" alt="Logo" width="120"/>
</p>
