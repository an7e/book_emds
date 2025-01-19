import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.linalg import qr, inv
import numpy as np
import random
import sympy as sympy
from sympy.plotting import plot3d

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
