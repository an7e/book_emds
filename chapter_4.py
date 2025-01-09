import numpy as np

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

