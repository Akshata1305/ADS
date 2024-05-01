# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset into a DataFrame
auto_df = pd.read_csv(r'C:\Users\anike\Desktop\TCS\ADS\Automobile_data.csv')


# Replace '?' with NaN
auto_df.replace('?', pd.NA, inplace=True)

# Convert numeric columns to float
numeric_cols = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm', 'price']
auto_df[numeric_cols] = auto_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Exclude non-numeric columns from correlation calculation
numeric_df = auto_df.select_dtypes(include=['float64', 'int64'])

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Visualize correlation matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Automobile Dataset')
plt.show()