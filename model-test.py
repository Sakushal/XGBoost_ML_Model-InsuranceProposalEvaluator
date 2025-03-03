import pandas as pd
import xgboost as xgb
import numpy as np

# 1. Load the Trained Model
loaded_model = xgb.XGBClassifier()
loaded_model.load_model('Insurance Model.json')  # Load the saved model

# 2. Prepare New Data for Prediction (Example: Single Data Point)
new_data = {
    'age': 21,
    'gender': 'Male',
    'health_history': 'Excellent',
    'income': 80000,
    'occupation': 'Engineer',
    'marital_status': 'Single',
    'family_history': 'No',
    'smoker': 'No',
    'alcohol_consumption': 'Light',
    'bmi': 24.5
}

new_df = pd.DataFrame([new_data])

# 3. Preprocess the New Data (One-Hot Encoding)
# Important: You need to know the columns used during training.

#First, load the training data again, just to get the columns.
training_data = pd.read_csv('realisticinsurancedata_10000.csv')
X_train_data = training_data.drop('accepted', axis=1)
X_train_encoded_data = pd.get_dummies(X_train_data, drop_first=True)
training_columns = X_train_encoded_data.columns

new_df_encoded = pd.get_dummies(new_df, drop_first=True).reindex(columns=training_columns, fill_value=0)

# 4. Make a Prediction
prediction = loaded_model.predict(new_df_encoded)

if prediction[0] == 1:
    print("\nPrediction: Policy Accepted")
else:
    print("\nPrediction: Policy Rejected")

