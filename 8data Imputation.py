import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r'C:\Users\anike\Desktop\TCS\ADS\Automobile_data.csv')

# Identify missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Initialize SimpleImputer with strategy='mean' for numerical columns and 'most_frequent' for categorical columns
imputer = SimpleImputer(strategy='mean')  # You can change the strategy as needed

# Select numeric columns for imputation
numeric_columns = df.select_dtypes(include=['int', 'float']).columns

# Impute missing values in numeric columns
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Verify imputation
missing_values_after_imputation = df.isnull().sum()
print("\nMissing values after imputation:\n", missing_values_after_imputation)
