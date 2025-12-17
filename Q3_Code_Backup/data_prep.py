import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Define Columns
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# 2. Load Data
# The raw data does not have a header, and the separator is ', '
try:
    df = pd.read_csv(
        'adult.csv',
        names=COLUMNS,
        sep=r',\s*',
        engine='python',
        na_values='?'
    )
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 3. Data Cleaning and Preprocessing (Addressing Q1 requirements)

# Standardise categorical labels (strip whitespace)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# Handle missing values (imputation strategy: mode for categorical)
# Missing values are now represented as NaN after using na_values='?'
for col in ['workclass', 'occupation', 'native-country']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Target variable transformation (<=50K -> 0, >50K -> 1)
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# 4. Feature Engineering and Selection (Addressing Q1/Q2 requirements)

# Drop redundant/irrelevant variables
# 'fnlwgt' is a sampling weight, not a predictive feature for an individual
# 'education' is redundant with 'education-num'
df.drop(columns=['fnlwgt', 'education'], inplace=True)

# Identify categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('income') # Target variable

# One-Hot Encoding for categorical features
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Final check for any remaining NaNs (should be none)
if df_encoded.isnull().sum().sum() > 0:
    print("Warning: NaNs still present after cleaning.")

# 5. Prepare for Modeling (Q3)
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Save the processed data and split sets for use in the next phase
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=True)
y_test.to_csv('y_test.csv', index=False, header=True)

print("Data preparation complete. X_train.csv, X_test.csv, y_train.csv, y_test.csv saved.")
