#Perform univariate analysis
#like Mean, median, variance, Standard deviation, skewness, and kurtosis on Diabetes
#dataset.
import pandas as pd
import numpy as np

# Load the Diabetes dataset
diabetes_data = pd.read_csv(r'C:\Users\anike\Desktop\TCS\ADS\diabetes.csv')

# Display the first few rows of the dataset
print(diabetes_data.head())

# Calculate mean
mean_values = diabetes_data.mean()
print("\nMean:")
print(mean_values)

# Calculate median
median_values = diabetes_data.median()
print("\nMedian:")
print(median_values)

# Calculate variance
variance_values = diabetes_data.var()
print("\nVariance:")
print(variance_values)

# Calculate standard deviation
std_dev_values = diabetes_data.std()
print("\nStandard Deviation:")
print(std_dev_values)

# Calculate skewness
skewness_values = diabetes_data.skew()
print("\nSkewness:")
print(skewness_values)

# Calculate kurtosis
kurtosis_values = diabetes_data.kurtosis()
print("\nKurtosis:")
print(kurtosis_values)
