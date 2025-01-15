import numpy as np
import sympy as sympy

# Vector
v = [3, 2]
print(v)
v = np.array([3, 2])
print(v)
v = np.array([4, 1, 2])
print(v)
v = np.array([6, 1, 5, 8, 3])
print(v)
v = np.array([3, 2])
w = np.array([2, -1])
# sum the vectors
v_plus_w = v + w
# display summed vector
print(v_plus_w)
v = np.array([3, 1])
# scale the vector
scaled_v = 2.0 * v
# display scaled vector
print(scaled_v)  # [6 2]

# Matrix Vector Multiplication
# compose basis matrix with i-hat and j-hat
basis = np.array(
    [[3, 0],
     [0, 2]]
)
# declare vector v
v = np.array([1, 1])
# create new vector
# by transforming v with dot product
new_v = basis.dot(v)
print(new_v)

# Declare i-hat and j-hat
i_hat = np.array([2, 0])
j_hat = np.array([0, 3])
# compose basis matrix using i-hat and j-hat
# also need to transpose rows into columns
basis = np.array([i_hat, j_hat]).transpose()
# declare vector v
v = np.array([1, 1])
# create new vector
# by transforming v with dot product
new_v = basis.dot(v)
print(new_v)

# Transformation 1
i_hat1 = np.array([0, 1])
j_hat1 = np.array([-1, 0])
transform1 = np.array([i_hat1, j_hat1]).transpose()
# Transformation 2
i_hat2 = np.array([1, 0])
j_hat2 = np.array([1, 1])
transform2 = np.array([i_hat2, j_hat2]).transpose()
# Combine Transformations
combined = transform2 @ transform1
# Test
print("COMBINED MATRIX:\n {}".format(combined))
v = np.array([1, 2])
print(combined.dot(v))

# Determinants
i_hat = np.array([3, 0])
j_hat = np.array([0, 2])
basis = np.array([i_hat, j_hat]).transpose()
determinant = np.linalg.det(basis)
print(determinant)
i_hat = np.array([1, 0])
j_hat = np.array([1, 1])
basis = np.array([i_hat, j_hat]).transpose()
determinant = np.linalg.det(basis)
print(determinant)
i_hat = np.array([-2, 1])
j_hat = np.array([1, 2])
basis = np.array([i_hat, j_hat]).transpose()
determinant = np.linalg.det(basis)
print(determinant)
i_hat = np.array([-2, 1])
j_hat = np.array([3, -1.5])
basis = np.array([i_hat, j_hat]).transpose()
determinant = np.linalg.det(basis)
print(determinant)

# Systems of Equations and Inverse Matrices
# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72
A = sympy.Matrix([
    [4, 2, 4],
    [5, 3, 7],
    [9, 3, 6]
])
# dot product between A and its inverse
# will produce identity function
inverse = A.inv()
identity = inverse * A
# prints Matrix([[-1/2, 0, 1/3], [11/2, -2, -4/3], [-2, 1, 1/3]])
print("INVERSE: {}".format(inverse))
# prints Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print("IDENTITY: {}".format(identity))

# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72
A = np.array([
    [4, 2, 4],
    [5, 3, 7],
    [9, 3, 6]
])
B = np.array([
    44,
    56,
    72
])
X = np.linalg.inv(A).dot(B)
print(X)

# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72
A = sympy.Matrix([
    [4, 2, 4],
    [5, 3, 7],
    [9, 3, 6]
])
B = sympy.Matrix([
    44,
    56,
    72
])
X = A.inv() * B
print(X)

# Eigenvectors and Eigenvalues
A = np.array([
 [1, 2],
 [4, 5]
])
eigenvals, eigenvecs = np.linalg.eig(A)
print("EIGENVALUES")
print(eigenvals)
print("\nEIGENVECTORS")
print(eigenvecs)
print("\nREBUILD MATRIX")
Q = eigenvecs
R = np.linalg.inv(Q)
L = np.diag(eigenvals)
B = Q @ L @ R
print(B)