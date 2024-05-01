import pandas as pd
df=pd.read_csv(r'C:\Users\anike\Desktop\TCS\ADS\olympics.csv')
df.columns
# Calculate descriptive statistics for the 'Combined_total' column
combined_total_column = df['Combined total']
mean = combined_total_column.mean()
median = combined_total_column.median()
mode = combined_total_column.mode()[0]  # Mode returns a Series, so we get the first value
minimum = combined_total_column.min()
maximum = combined_total_column.max()
std_dev = combined_total_column.std()

# Print the results
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Minimum:", minimum)
print("Maximum:", maximum)
print("Standard Deviation:", std_dev)


from scipy import stats
df=pd.read_csv(r'C:\Users\anike\Desktop\TCS\ADS\olympics2.csv')


# Calculate descriptive statistics for the 'Combined_total' column
combined_total_column = df['Combined total']
mean = combined_total_column.mean()



# Hypothesis Testing (One-Sample t-test)
# Example: Testing if the mean of 'Combined total' is significantly different from a specific value (e.g., 100)
null_hypothesis = "The mean of 'Combined total' is equal to 100"
alternative_hypothesis = "The mean of 'Combined total' is not equal to 100"
test_value = 100
t_statistic, p_value = stats.ttest_1samp(df['Combined total'], test_value)


print("\nOne-Sample t-test Results:")
print("Mean:", mean)
print("Null Hypothesis:", null_hypothesis)
print("Alternative Hypothesis:", alternative_hypothesis)
print("t-statistic:", t_statistic)
print("p-value:", p_value)

# Step 4: Confidence Interval
confidence_level = 0.95
confidence_interval = stats.norm.interval(confidence_level, loc=mean, scale=std_dev/len(df)**0.5)

print("\nConfidence Interval (95%):", confidence_interval)