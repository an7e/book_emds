from sympy import *
from sympy.plotting import plot3d
from math import exp

# Functions
x, y = symbols('x y')
f = 2 * x + 3 * y
# plot3d(f)


# Summations
i, n = symbols('i n')
summation = Sum(2 * i, (i, 1, n))
up_to_5 = summation.subs(n, 5)
print(up_to_5.doit())

# Exponents
x = symbols('x')
expr = x ** 7 / x ** 7
print(expr)

# Logarithms
x = log(8, 2)
print(x)

# Euler’s Number
print('e = (1+1/n)**n')
e1 = (1+1/1)**1
e10 = (1+1/10)**10
e100 = (1+1/100)**100
print(e1)
print(e10)
print(e100)

n = symbols('n')
f = (1 + (1/n))**n
result = limit(f, n, oo)
print(result)
print(result.evalf())

# Limits
x = symbols('x')
f = 1 / x
result = limit(f, x, oo)
print(result)

# Derivatives
x = symbols('x')
f = x**2
dx_f = diff(f)
print(dx_f)

# Integrals
x = symbols('x')
f = x**2 + 1
area = integrate(f, (x, 0, 1))
print(area)
