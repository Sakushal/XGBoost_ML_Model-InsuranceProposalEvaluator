import pandas as pd
import numpy as np
import random

num_rows = 10000

data = {
    'age': np.random.randint(0, 100, num_rows),
    'gender': np.random.choice(['Male', 'Female'], num_rows),
    'health_history': np.random.choice(['Good', 'Moderate', 'Excellent', 'Diabetes', 'Heart Disease', 'Arthritis'], num_rows),
    'income': np.random.randint(30000, 120000, num_rows),
    'occupation': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Manager', 'Salesperson'], num_rows),
    'marital_status': np.random.choice(['Married', 'Single', 'Divorced', 'Widowed'], num_rows),
    'family_history': np.random.choice(['Yes', 'No'], num_rows),
    'smoker': np.random.choice(['Yes', 'No'], num_rows),
    'alcohol_consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], num_rows),
    'bmi': np.random.uniform(18, 35, num_rows)
}

df = pd.DataFrame(data)

# Create 'accepted' column with logic, including rejection rules
def determine_acceptance(row):
    score = 0

    # Explicit rejection rules
    if row['age'] < 20 or row['age'] > 60:
        return 0  # Reject immediately

    # Increase score for younger age, good health, higher income, etc.
    if row['age'] < 45:
        score += 2
    if row['health_history'] in ['Excellent', 'Good']:
        score += 3
    if row['income'] > 70000:
        score += 2
    if row['smoker'] == 'No':
        score += 1
    if row['alcohol_consumption'] in ['None', 'Light']:
        score += 1
    if row['bmi'] < 30:
        score += 1
    if row['family_history'] == 'No':
        score += 1

    # Decrease score for risk factors
    if row['health_history'] in ['Diabetes', 'Heart Disease', 'Arthritis']:
        score -= 3
    if row['smoker'] == 'Yes':
        score -= 2
    if row['alcohol_consumption'] == 'Heavy':
        score -= 2
    if row['bmi'] > 30:
        score -= 1
    if row['family_history'] == 'Yes':
        score -= 1

    # Determine acceptance based on score threshold
    if score >= 3:
        return 1  # Accepted
    else:
        return 0  # Rejected

df['accepted'] = df.apply(determine_acceptance, axis=1)

# Save to CSV (optional)
df.to_csv('realisticinsurancedata_10000.csv', index=False)

print(df.head())