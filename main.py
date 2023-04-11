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


MyVar = np.array([[-7.0, 0, 0, 1, 4, 1],
                     [0, -7, -4, 0, 1, 0],
                     [0, 0, -1, 0, 1, 2],
                     [0, 0, 0, -1, 0, 0],
                     [0, 0, 0, 0, -3, 0],
                     [0, 0, 0, 0, 0, -7]])

find_all_basis(count_phi(MyVar, -7), 3)
new_find_all_basis(count_phi(MyVar, -7), 3)
print(spec_with_multiple(MyVar))
# # print(count_phi(MyVar, -7))
# print(new_find_all_basis(count_phi(MyVar, -7)))
# # find_all_basis(count_phi(MyVar, -7))
# # print(collinear(np.array([0., 1., 0., 0., 0., 0.]), np.array([ 0.,  1., -0., -0., -0.,  0.])))
# w1 = np.array([0., 0., 0., 0., 0., 1.])
# w1 = np.dot(count_phi(MyVar, -7), w1)
# print(w1)
# w1 = np.dot(count_phi(MyVar, -7), w1)
# print(w1)

# w2 = np.array([0., 0., 0., 1., 0., 0.])
# w2 = np.dot(count_phi(MyVar, -7), w2)
# print(w2)
#
# S = np.array([[ 0.,  -4.,   0.,   0.,   0.,   0. ],
# [ 4.,   1.,   1.,   0.,   0.,   0. ],
# [ 0.,   0.,   0.,   0.,   1.,   0. ],
# [ 1.,   0.,   0.,   0.,   0.,   0. ],
# [ 0.,   0.,   0.,   1.,   0.,   0. ],
# [ 0.,   0.,  -0.5,  7.,  -2.,   1. ]])
# S = S.T
# print(S)
# S_1 = np.linalg.inv(S)
# print(S_1)
# J = np.array([[-7, 1, 0, 0, 0, 0],
#               [0, -7, 1, 0, 0, 0],
#                [0, 0, -7, 0, 0, 0],
#                [0, 0, 0, -7, 1, 0],
#                [0, 0, 0, 0, -7, 0],
#                [0, 0, 0, 0, 0, -7]])
# R = np.dot(np.dot(S, J), S_1)
# print(R)

