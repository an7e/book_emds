from scipy.stats import binom
from scipy.stats import beta

# Binomial Distribution
n = 10
p = 0.9
q = 0
for k in range(n + 1):
    probability = binom.pmf(k, n, p)
    print("{0} - {1}".format(k, probability))

# Beta Distribution
a = 8
b = 2
p = beta.cdf(.90, a, b)
print(p)
a = 30
b = 6
p = 1.0 - beta.cdf(.90, a, b)
print(p)
a = 8
b = 2
p = beta.cdf(.90, a, b) - beta.cdf(.80, a, b)
print(p)