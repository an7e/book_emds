import math
import random
import plotly.express as px
from scipy.stats import norm, t

# Population Variance and Standard Deviation
data = [0, 1, 5, 7, 9, 10, 14]


def variance(values, is_sample: bool = False):
    mean = sum(values) / len(values)
    _variance = sum((v - mean) ** 2 for v in values) / (len(values) - (1 if is_sample else 0))
    return _variance


def std_dev(values, is_sample: bool = False):
    return math.sqrt(variance(values, is_sample))


print("VARIANCE = {}".format(variance(data, is_sample=True)))
print("STD DEV = {}".format(std_dev(data, is_sample=True)))


# Normal Distribution

# Probability Density Function (PDF)
def normal_pdf(x: float, mean: float, std_dev: float) -> float:
    return (1.0 / (2.0 * math.pi * std_dev ** 2) ** 0.5) * math.exp(-1.0 * ((x - mean) ** 2 / (2.0 * std_dev ** 2)))


# Cumulative Distribution Function (CDF)
mean = 64.43
std_dev = 2.99
x = norm.cdf(64.43, mean, std_dev)
print(x)
mean = 64.43
std_dev = 2.99
x = norm.cdf(66, mean, std_dev) - norm.cdf(62, mean, std_dev)
print(x)

# Inverse CDF
x = norm.ppf(.95, loc=64.43, scale=2.99)
print(x)

# Central Limit Theorem
sample_size = 31
sample_count = 1000
x_values = [(sum([random.uniform(0.0, 1.0) for i in range(sample_size)]) / sample_size) for _ in range(sample_count)]
y_values = [1 for _ in range(sample_count)]


# px.histogram(x=x_values, y=y_values, nbins=20).show()

# Confidence Intervals
def critical_z_value(p):
    norm_dist = norm(loc=0.0, scale=1.0)
    left_tail_area = (1.0 - p) / 2.0
    upper_area = 1.0 - ((1.0 - p) / 2.0)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)


print(critical_z_value(p=.95))


def confidence_interval(p, sample_mean, sample_std, n):
    # Sample size must be greater than 30
    lower, upper = critical_z_value(p)
    lower_ci = lower * (sample_std / math.sqrt(n))
    upper_ci = upper * (sample_std / math.sqrt(n))
    return sample_mean + lower_ci, sample_mean + upper_ci


print(confidence_interval(p=.95, sample_mean=64.408, sample_std=2.05, n=31))

# Hypothesis Testing
# Cold has 18 day mean recovery, 1.5 std dev
mean = 18
std_dev = 1.5
# 95% probability recovery time takes between 15 and 21 days.
x = norm.cdf(21, mean, std_dev) - norm.cdf(15, mean, std_dev)
print(x)
# What x-value has 5% of area behind it?
x = norm.ppf(.05, mean, std_dev)
print(x)
# Probability of 16 or less days
p_value = norm.cdf(16, mean, std_dev)
print(p_value)
# What x-value has 2.5% of area behind it?
x1 = norm.ppf(.025, mean, std_dev)
# What x-value has 97.5% of area behind it
x2 = norm.ppf(.975, mean, std_dev)
print(x1)
print(x2)
# Probability of 16 or less days
p1 = norm.cdf(16, mean, std_dev)
# Probability of 20 or more days
p2 = 1.0 - norm.cdf(20, mean, std_dev)
# P-value of both tails
p_value = p1 + p2
print(p_value)

# T-Distribution
# get critical value range for 95% confidence
# with a sample size of 25
n = 25
lower = t.ppf(.025, df=n - 1)
upper = t.ppf(.975, df=n - 1)
print(lower, upper)

