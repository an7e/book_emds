import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit
from numpy.linalg import qr, inv
import numpy as np
import random
import sympy as sympy
from sympy.plotting import plot3d
from scipy.stats import t
from math import sqrt

# Basic Linear Regression
# Import points
df = pd.read_csv('https://bit.ly/3goOAnt', delimiter=",")
# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]
# Extract output column (all rows, last column)
Y = df.values[:, -1]
# Fit a line to the points
fit = LinearRegression().fit(X, Y)
# m = 1.7867224, b = -16.51923513
m = fit.coef_.flatten()
b = fit.intercept_.flatten()
print("m = {0}".format(m))
print("b = {0}".format(b))
# show in chart
plt.plot(X, Y, 'o')  # scatterplot
plt.plot(X, m * X + b)  # line
plt.show()

# Residuals and Squared Errors
# Import points
points = pd.read_csv('https://bit.ly/3goOAnt', delimiter=",").itertuples()
# Test with a given line
m = 1.93939
b = 4.73333
# Calculate the residuals
for p in points:
    y_actual = p.y
    y_predict = m * p.x + b
    residual = y_actual - y_predict
    print(residual)

# Import points
points = pd.read_csv("https://bit.ly/2KF29Bd").itertuples()
# Test with a given line
m = 1.93939
b = 4.73333
sum_of_squares = 0.0
# calculate sum of squares
for p in points:
    y_actual = p.y
    y_predict = m * p.x + b
    residual_squared = (y_predict - y_actual) ** 2
    sum_of_squares += residual_squared
print("sum of squares = {}".format(sum_of_squares))
# sum of squares = 28.096969704500005

# Closed Form Equation
# Load the data
points = list(pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",").itertuples())
n = len(points)
m = (n * sum(p.x * p.y for p in points) - sum(p.x for p in points) *
     sum(p.y for p in points)) / (n * sum(p.x ** 2 for p in points) -
                                  sum(p.x for p in points) ** 2)
b = (sum(p.y for p in points) / n) - m * sum(p.x for p in points) / n
print(m, b)
# 1.9393939393939394 4.7333333333333325

# Inverse Matrix Techniques
# Import points
df = pd.read_csv('https://bit.ly/3goOAnt', delimiter=",")
# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1].flatten()
# Add placeholder "1" column to generate intercept
X_1 = np.vstack([X, np.ones(len(X))]).T
# Extract output column (all rows, last column)
Y = df.values[:, -1]
# Calculate coefficents for slope and intercept
b = inv(X_1.transpose() @ X_1) @ (X_1.transpose() @ Y)
print(b)  # [1.93939394, 4.73333333]
# Predict against the y-values
y_predict = X_1.dot(b)

# Import points
df = pd.read_csv('https://bit.ly/3goOAnt', delimiter=",")
# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1].flatten()
# Add placeholder "1" column to generate intercept
X_1 = np.vstack([X, np.ones(len(X))]).transpose()
# Extract output column (all rows, last column)
Y = df.values[:, -1]
# calculate coefficents for slope and intercept
# using QR decomposition
Q, R = qr(X_1)
b = inv(R).dot(Q.transpose()).dot(Y)
print(b)  # [1.93939394, 4.73333333]


# Gradient Descent
def f(x):
    return (x - 3) ** 2 + 4


def dx_f(x):
    return 2 * (x - 3)


# The learning rate
L = 0.001
# The number of iterations to perform gradient descent
iterations = 100_000
# start at a random x
x = random.randint(-15, 15)
for i in range(iterations):
    # get slope
    d_x = dx_f(x)
    # update x by subtracting the (learning rate) * (slope)
    x -= L * d_x
print(x, f(x))  # prints 2.999999999999889 4.0

# Import points from CSV
points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())
# Building the model
m = 0.0
b = 0.0
# The learning Rate
L = .001
# The number of iterations
iterations = 100_000
n = float(len(points))  # Number of elements in X
# Perform Gradient Descent
for i in range(iterations):
    # slope with respect to m
    D_m = sum(2 * p.x * ((m * p.x + b) - p.y) for p in points)
    # slope with respect to b
    D_b = sum(2 * ((m * p.x + b) - p.y) for p in points)
    # update m and b
    m -= L * D_m
    b -= L * D_b
print("y = {0}x + {1}".format(m, b))
# y = 1.9393939393939548x + 4.733333333333227

m, b, i, n = sympy.symbols('m b i n')
x, y = sympy.symbols('x y', cls=sympy.Function)
sum_of_squares = sympy.Sum((m * x(i) + b - y(i)) ** 2, (i, 0, n))
d_m = sympy.diff(sum_of_squares, m)
d_b = sympy.diff(sum_of_squares, b)
print(d_m)
print(d_b)
# OUTPUTS
# Sum(2*(b + m*x(i) - y(i))*x(i), (i, 0, n))
# Sum(2*b + 2*m*x(i) - 2*y(i), (i, 0, n))

# Import points from CSV
points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())
m, b, i, n = sympy.symbols('m b i n')
x, y = sympy.symbols('x y', cls=sympy.Function)
sum_of_squares = sympy.Sum((m * x(i) + b - y(i)) ** 2, (i, 0, n))
d_m = sympy.diff(sum_of_squares, m) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)
d_b = sympy.diff(sum_of_squares, b) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)
# compile using lambdify for faster computation
d_m = sympy.lambdify([m, b], d_m)
d_b = sympy.lambdify([m, b], d_b)
# Building the model
m = 0.0
b = 0.0
# The learning Rate
L = .001
# The number of iterations
iterations = 100_000
# Perform Gradient Descent
for i in range(iterations):
    # update m and b
    m -= d_m(m, b) * L
    b -= d_b(m, b) * L
print("y = {0}x + {1}".format(m, b))
# y = 1.939393939393954x + 4.733333333333231

points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())
m, b, i, n = sympy.symbols('m b i n')
x, y = sympy.symbols('x y', cls=sympy.Function)
sum_of_squares = sympy.Sum((m * x(i) + b - y(i)) ** 2, (i, 0, n)) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)
plot3d(sum_of_squares)

# Stochastic Gradient Descent
# Input data
data = pd.read_csv('https://bit.ly/2KF29Bd', header=0)
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values
n = data.shape[0]  # rows
# Building the model
m = 0.0
b = 0.0
sample_size = 1  # sample size
L = .0001  # The learning Rate
epochs = 1_000_000  # The number of iterations to perform gradient descent
# Performing Stochastic Gradient Descent
for i in range(epochs):
    idx = np.random.choice(n, sample_size, replace=False)
    x_sample = X[idx]
    y_sample = Y[idx]
    # The current predicted value of Y
    Y_pred = m * x_sample + b
    # d/dm derivative of loss function
    D_m = (-2 / sample_size) * sum(x_sample * (y_sample - Y_pred))
    # d/db derivative of loss function
    D_b = (-2 / sample_size) * sum(y_sample - Y_pred)
    m = m - L * D_m  # Update m
    b = b - L * D_b  # Update b
    # print progress
    if i % 10000 == 0:
        print(i, m, b)
print("y = {0}x + {1}".format(m, b))

# Correlation Coefficient
# Read data into Pandas dataframe
df = pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",")
# Print correlations between variables
correlations = df.corr(method='pearson')
print(correlations)
# OUTPUT:
# x y
# x 1.000000 0.957586
# y 0.957586 1.000000

n = 10
lower_cv = t(n - 1).ppf(.025)
upper_cv = t(n - 1).ppf(.975)
print(lower_cv, upper_cv)
# -2.262157162740992 2.2621571627409915

# sample size
n = 10
lower_cv = t(n - 1).ppf(.025)
upper_cv = t(n - 1).ppf(.975)
# correlation coefficient
# derived from data https://bit.ly/2KF29Bd
r = 0.957586
# Perform the test
test_value = r / sqrt((1 - r ** 2) / (n - 2))
print("TEST VALUE: {}".format(test_value))
print("CRITICAL RANGE: {}, {}".format(lower_cv, upper_cv))
if test_value < lower_cv or test_value > upper_cv:
    print("CORRELATION PROVEN, REJECT H0")
else:
    print("CORRELATION NOT PROVEN, FAILED TO REJECT H0 ")
# Calculate p-value
if test_value > 0:
    p_value = 1.0 - t(n - 1).cdf(test_value)
else:
    p_value = t(n - 1).cdf(test_value)
    # Two-tailed, so multiply by 2
    p_value = p_value * 2
    print("P-VALUE: {}".format(p_value))

# Coefficient of Determination
# Read data into Pandas dataframe
df = pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",")
# Print correlations between variables
coeff_determination = df.corr(method='pearson') ** 2
print(coeff_determination)
# OUTPUT:
# x y
# x 1.000000 0.916971
# y 0.916971 1.000000

# Standard Error of the Estimate
# Load the data
points = list(pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",").itertuples())
n = len(points)
# Regression line
m = 1.939
b = 4.733
# Calculate Standard Error of Estimate
S_e = sqrt((sum((p.y - (m*p.x +b))**2 for p in points))/(n-2))
print(S_e)
# 1.87406793500129

# Prediction Intervals
# Load the data
points = list(pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",").itertuples())
n = len(points)
# Linear Regression Line
m = 1.939
b = 4.733
# Calculate Prediction Interval for x = 8.5
x_0 = 8.5
x_mean = sum(p.x for p in points) / len(points)
t_value = t(n - 2).ppf(.975)
standard_error = sqrt(sum((p.y - (m * p.x + b)) ** 2 for p in points) / (n - 2))
margin_of_error = t_value * standard_error * \
 sqrt(1 + (1 / n) + (n * (x_0 - x_mean) ** 2) / \
 (n * sum(p.x ** 2 for p in points) - \
 sum(p.x for p in points) ** 2))
predicted_y = m*x_0 + b
# Calculate prediction interval
print(predicted_y - margin_of_error, predicted_y + margin_of_error)
# 16.462516875955465 25.966483124044537

# Train/Test Splits
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Load the data
df = pd.read_csv('https://bit.ly/3cIH97A', delimiter=",")
# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]
# Extract output column (all rows, last column)
Y = df.values[:, -1]
# Separate training and testing data
# This leaves a third of the data out for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)
model = LinearRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("r^2: %.3f" % result)

df = pd.read_csv('https://bit.ly/3cIH97A', delimiter=",")
# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]
# Extract output column (all rows, last column)\
Y = df.values[:, -1]
# Perform a simple linear regression
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(results)
print("MSE: mean=%.3f (stdev-%.3f)" % (results.mean(), results.std()))

df = pd.read_csv('https://bit.ly/38XwbeB', delimiter=",")
# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]
# Extract output column (all rows, last column)\
Y = df.values[:, -1]
# Perform a simple linear regression
kfold = ShuffleSplit(n_splits=10, test_size=.33, random_state=7)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(results)
print("mean=%.3f (stdev-%.3f)" % (results.mean(), results.std()))

# Multiple Linear Regression
# Load the data
df = pd.read_csv('https://bit.ly/2X1HWH7', delimiter=",")
# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]
# Extract output column (all rows, last column)\
Y = df.values[:, -1]
# Training
fit = LinearRegression().fit(X, Y)
# Print coefficients
print("Coefficients = {0}".format(fit.coef_))
print("Intercept = {0}".format(fit.intercept_))
print("z = {0} + {1}x + {2}y".format(fit.intercept_, fit.coef_[0], fit.coef_[1]))
