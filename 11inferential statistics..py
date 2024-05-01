import numpy as np
from scipy import stats

# Generate some sample data (replace this with your data)
data = np.random.normal(loc=0, scale=1, size=100)
print(data)

# Define the null hypothesis (replace this with your null hypothesis)
null_mean = 0

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(data, null_mean)

# Print results
print("One-Sample T-Test:")
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
