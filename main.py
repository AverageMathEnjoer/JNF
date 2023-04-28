from JNF_and_basis import *

A = np.array([[3, -1, -1, 1],
      [1, 2, -1, -1]])
print(orthonormalize(A))
print(orthonormalize(orthogonalize(A)))
