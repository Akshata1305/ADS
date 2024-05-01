import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# Load the dataset
df = pd.read_csv(r'C:\Users\anike\Desktop\TCS\ADS\olympics.csv', index_col=0)

# Drop the 'Combined total' row as it is not needed for outlier detection
df.drop('Totals', inplace=True)

# Select the numerical columns for outlier detection
numeric_cols = df.columns[1:-1].tolist()  # Exclude the last column which indicates outliers
X = df[numeric_cols]

# Convert non-numeric values to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
X.dropna(inplace=True)

# Fit the Local Outlier Factor model
lof = LocalOutlierFactor()
outliers = lof.fit_predict(X)

# Create a Series with outliers aligned with the DataFrame index
outliers_series = pd.Series(outliers, index=X.index)

# Assign the Series to a new column in the DataFrame
df['Outlier'] = outliers_series

# Display the outliers
print("Outliers:")
print(df[df['Outlier'] == -1])

# Display the non-outliers
print("\nNon-Outliers:")
print(df[df['Outlier'] == 1])