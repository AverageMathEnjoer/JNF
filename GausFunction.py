import numpy as np


def line_swap(matrix, a, b):
    temp = matrix[a].copy()
    matrix[a] = matrix[b]
    matrix[b] = temp
    return matrix


def ladder(matrix):
    matrix = matrix.astype(np.float64)
    m = matrix.shape[0]
    n = matrix.shape[1]
    if m == 0 or n == 0:
        return matrix
    if m == 1 and n == 1:
        if matrix[0, 0] != 0:
            matrix = matrix / matrix[0, 0]
    else:
        first_non_zero = -1
        i = 0
        while first_non_zero == -1 and i < m:
            if matrix[i, 0] != 0:
                first_non_zero = i
            i += 1
        if first_non_zero == -1:
            matrix[0:m, 1:n] = ladder(matrix[0:m, 1:n])
        else:
            matrix = line_swap(matrix, 0, first_non_zero)
            matrix[0] = matrix[0] / matrix[0, 0]
            for k in range(1, m):
                matrix[k] = matrix[k] - (matrix[k, 0] * matrix[0])
            matrix[1:m, 0:n] = ladder(matrix[1:m, 0:n])
    return matrix


def find_basis_u(matrix):
    m = matrix.shape[0]
    n = matrix.shape[1]
    not_main = []
    u = []
    main = []
    matrix = ladder(matrix)
    cur_line = 0
    for i in range(m):
        if matrix[cur_line, i] != 0:
            not_main.append((i, cur_line))
            cur_line += 1
        else:
            main.append(i)
    not_main.reverse()
    for i in main:
        v = np.array([0, 0, 0, 0, 0, 0])
        v[i] = 1
        u.append(count_v(matrix, v, not_main))
    return u


def count_x(line, v, number):
    a = line * v
    a = a[number+1:len(a)]
    return -a.sum() / line[number]


def count_v(matrix, v, not_main):
    v = v.astype(np.float64)
    for i in range(len(not_main)):
        v[not_main[i][0]] = count_x(matrix[not_main[i][1]], v, not_main[i][0])
    return v


def count_phi(psi, gamma):
    return psi - np.eye(len(psi)) * gamma


def find_all_basis(matrix):
    B = np.eye(len(matrix))
    while np.linalg.matrix_rank(B) != 0:
        B = np.dot(B, matrix)
        print("Текущая матрица:")
        print(B)
        print("Кол-во векторов(m - rk(B)):")
        print(len(B) - np.linalg.matrix_rank(B))
        print("Векторы:", find_basis_u(B))