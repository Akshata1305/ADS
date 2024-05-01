import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\anike\Desktop\TCS\ADS\placement.csv')

# Plot histograms for numerical variables
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Corrected scatter plot code
plt.figure(figsize=(10, 6))
plt.scatter(df['cgpa'], df['placement_exam_marks'], c=df['placed'], cmap='coolwarm', alpha=0.7)
plt.title('CGPA vs Placement Exam Marks')
plt.xlabel('CGPA')
plt.ylabel('Placement Exam Marks')
plt.colorbar(label='Placed')
plt.grid(True)
plt.show()

# boxplot
plt.figure(figsize=(10, 6))
df.boxplot()
plt.title('Box Plot of Numerical Variables')
plt.show()

#bar chart
plt.figure(figsize=(8, 6))
df['placed'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Count of Placed vs Not Placed')
plt.xlabel('Placed')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

#pairplot
import seaborn as sns
sns.pairplot(df, hue='placed', diag_kind='kde')
plt.show()


