# XGBoost ML Model - InsuranceProposalEvaluator

This repository contains code for generating synthetic life insurance proposal data and training a machine learning (ML) model to predict proposal acceptance.

## Overview

This project simulates the process of evaluating life insurance proposals using machine learning. It generates a dataset of synthetic applicant data (age, income, health history, etc.) and trains an XGBoost model to predict whether a proposal will be accepted or rejected. The model is then exposed via a Flask API, allowing for real-time evaluation of new proposals.

## Features

* **Synthetic Data Generation:** Generates realistic synthetic insurance proposal data using `pandas` and `numpy`.
* **XGBoost ML Model Training:** Trains an XGBoost classification model to predict proposal acceptance.
* **Rejection Rules:** The system is able to reject proposals based on explicit rules, such as age outside the training range.
* **ML Model Saving and Loading:** The trained ML model is saved and loaded for easy reuse.

## Getting Started

### Prerequisites

* Python 3.6+
* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn xgboost
```

### Usage
 **Generate Data and Train ML Model**:
Run the data_generation.py to create the dataset and train the ML model.
```bash
python "Dataset generation.py"
````
And

```bash
python "ML Model generation.py"
````

### Data Generation Details
The Dataset generation.py script creates a dataset with the following columns:
- age
- gender
- health_history
- income
- occupation
- marital_status
- family_history
- smoker
- alcohol_consumption
- bmi
- accepted (target variable)
The accepted column is generated based on a scoring system that simulates insurance underwriting criteria. Explicit rejection rules are applied for ages outside the 20-60 range.

### ML Model Details
The XGBoost ML model is trained using the generated dataset. It is configured for binary classification with the following parameters:
- objective='binary:logistic'
- n_estimators=100
- learning_rate=0.1
- max_depth=3
- random_state=42

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request.

## Contact Information
For any questions or issues, feel free to reach out at saksalstha@gmail.com.

