# Coding

# 1. Create two vectors and compute their dot product: - From scratch - Using NumPy

import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print(A @ B)
# or
print(np.dot(A, B))

# from scratch
dot = 0

for i in range(len(A)):
    dot += A[i] * B[i]

print(dot)

# 2. Multiply two matrices:

A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]

result = [[0, 0], [0, 0]]

for i in range(len(A)):
    for j in range(len(B[0])):
        for k in range(len(B)):
            result[i][j] += A[i][k] * B[k][j]

print(result)
