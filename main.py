import numpy as np
from GausFunction import *


def spec_with_multiple(phi):
    gammas, v = np.linalg.eigh(phi)
    counter = 1
    spec = []
    cur = gammas[0]
    for i in np.arange(1, len(gammas)):
        if cur != gammas[i]:
            spec.append((gammas[i - 1], counter))
            cur = gammas[i]
            counter = 1
        else:
            counter += 1
    spec.append((gammas[i - 1], counter))
    return spec


def linear_independence(vectors):
    B = np.array(vectors).astype(np.float64)
    m = B.shape[0]
    n = B.shape[1]
    if m > n:
        return False
    B = ladder(B)
    for i in range(m):
        if all(t == 0 for t in B[i]):
            return False
    else:
        return True


def new_find_all_basis(matrix, multiplicity):
    M = np.eye(len(matrix))
    basises = []
    M = np.dot(M, matrix)
    prev = find_basis_u(M)
    basises.append(prev.copy())
    while len(M) - np.linalg.matrix_rank(M) < multiplicity:
        M = np.dot(M, matrix)
        cur = find_basis_u(M)
        independ_b = []
        for i in cur:
            if linear_independence(prev + [i]):
                independ_b.append(i)
                prev.append(i)
        basises.append(independ_b.copy())
        prev = cur.copy()
    for i in basises:
        print(i)
    return basises


A = np.array([[3, -1, -1, 1],
      [1, 2, -1, -1]])
print(ladder(A))
print(find_basis_u(A))

