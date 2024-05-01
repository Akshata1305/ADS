import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
diabetes_df = pd.read_csv(r'C:\Users\anike\Desktop\TCS\ADS\diabetes.csv')

# Specify the column for which you want to create the box plot
column_name = "Glucose"

# Create a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(diabetes_df[column_name], vert=False)
plt.title("Box Plot of " + column_name)
plt.xlabel(column_name)
plt.grid(True)
plt.show()
