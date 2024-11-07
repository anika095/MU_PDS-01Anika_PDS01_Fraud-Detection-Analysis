# Credit Card Fraud Detection

This repository contains a comprehensive analysis and modeling workflow for detecting credit card fraud using a dataset of transactions. The objective is to identify fraudulent transactions effectively using machine learning techniques.

## Introduction

Credit card fraud is a significant issue affecting financial institutions and consumers worldwide. This project utilizes machine learning algorithms to detect fraudulent transactions. The primary focus is on building a logistic regression model to predict fraudulent activities based on transaction features.

## Dataset

The dataset used in this project is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/dalpozz/creditcard-fraud). It contains various attributes related to credit card transactions, with a significant class imbalance between fraudulent and non-fraudulent transactions.

### Features
- **Time**: Time elapsed since the first transaction in the dataset.
- **Amount**: Transaction amount.
- **Class**: Label indicating whether the transaction is fraudulent (1) or legitimate (0).

## Installation

To run this project, you need to have the following packages installed. You can install them using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/anika095/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Open the Jupyter Notebook or Google Colab to execute the code.

3. Load the dataset:
   ```python
   data = pd.read_csv('path/to/credit_card_fraud_dataset.csv')
   ```

4. Follow the steps outlined in the notebook to preprocess the data, build the model, and evaluate its performance.

## Modeling

The modeling process includes:

1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
2. **Balancing the Dataset**: Using SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
3. **Splitting the Data**: Dividing the dataset into training and testing sets.
4. **Training the Model**: Implementing a Logistic Regression model.
5. **Cross-Validation**: Evaluating the model's performance using cross-validation techniques.

## Evaluation

Model performance is evaluated using various metrics:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Ratio of true positive predictions to the total positive predictions.
- **Recall**: Ratio of true positive predictions to the actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

Example code for evaluation:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
```

## Results

The results of the model are discussed in detail in the notebook, including visualizations that highlight the performance metrics. Key insights and suggestions for improvement are provided based on the model's performance.

