import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. Load the Data from CSV
data = pd.read_csv('realisticinsurancedata_10000.csv')  # Replace with your CSV file name

# 2. Preprocess the Data
# Separate features (X) and target (y)
X = data.drop('accepted', axis=1)
y = data['accepted']

# Handle categorical features (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the XGBoost Model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Evaluate the Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Feature Importance
feature_importance = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)

# 6. Save the Trained ModelS
model.save_model('Insurance Model.json')
print("\nModel saved as Insurance Model.json")

