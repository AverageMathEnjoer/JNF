import numpy as np
from typing import NoReturn
""" 
The module contains reduction to a step form by the Gauss algorithm
And an older version of the function that was used to find solution bases
"""


def swap(matrix: np.array, a: int, b: int) -> np.array:
    """ Simple function for swap two lines in matrix.
    Matrix, number of first line, number of second line -> changed matrix"""
    temp = matrix[a].copy()
    matrix[a] = matrix[b]
    matrix[b] = temp
    return matrix


def ladder(matrix: np.array) -> np.array:
    """ Recursive function to reduce matrix to a step form"""
    matrix: np.array = matrix.astype(np.float64)
    m = matrix.shape[0]
    n = matrix.shape[1]
    # If at recursion step we get null-matrix, we end the recursion
    if m == 0 or n == 0:
        return matrix
    # If at recursion step we get only number, we end the recursion
    if m == 1 and n == 1:
        if matrix[0, 0] != 0:
            matrix = matrix / matrix[0, 0]
    else:
        first_non_zero = -1
        i = 0
        # Finding any non-zero element
        while first_non_zero == -1 and i < m:
            if matrix[i, 0] != 0:
                first_non_zero = i
            i += 1
        # If all zeros, working with smaller matrix
        if first_non_zero == -1:
            matrix[0:m, 1:n] = ladder(matrix[0:m, 1:n])
        # If we have any non-zero first element we
        else:
            # Swap strings and divide the new first string by the first element in it
            matrix = swap(matrix, 0, first_non_zero)
            matrix[0] = matrix[0] / matrix[0, 0]
            # Subtract the first line from the bottom lines
            for k in range(1, m):
                matrix[k] = matrix[k] - (matrix[k, 0] * matrix[0])
            # Working with smaller matrix
            matrix[1:m, 0:n] = ladder(matrix[1:m, 0:n])
    return matrix


def find_basis_u(matrix: np.array) -> np.array:
    """ Function for find basis of solutions"""
    m = matrix.shape[0]
    n = matrix.shape[1]
    not_main = []
    u = []
    main = []
    matrix = ladder(matrix)
    cur_line = 0
    # Search for principal and non-principal variables
    for i in range(n):
        if cur_line >= m:
            main += [j for j in range(i, n)]
            break
        if matrix[cur_line, i] != 0:
            not_main.append((i, cur_line))
            cur_line += 1
        else:
            main.append(i)
    not_main.reverse()
    # Finding vector from basis of solutions for every principal variable
    for i in main:
        v = np.zeros(n)
        v[i] = 1
        u.append(count_v(matrix, v, not_main))
    return u                                                        #list необходим т.к np.array не содержит append


def count_x(line: np.array, v: np.array, number: int):
    """ Simple function which finding value of principal variable by its number"""
    a = line * v
    a = a[number+1:len(a)]
    return -a.sum() / line[number]


def count_v(matrix: np.array, v: np.array, not_main: list) -> np.array:
    """ Function that turns a vector of zeros and only ones into a vector from a decision base"""
    v = v.astype(np.float64)
    for i in range(len(not_main)):
        v[not_main[i][0]] = count_x(matrix[not_main[i][1]], v, not_main[i][0])
    return v


def count_phi(psi: np.array, gamma) -> np.array:
    return psi - np.eye(len(psi)) * gamma


def find_all_basis(matrix: np.array, multiplicity: int) -> NoReturn:
    """ Just a loop to find the basis of solution for each possible power of the matrix"""
    B = np.eye(len(matrix))
    while len(B) - np.linalg.matrix_rank(B) < multiplicity:
        B = np.dot(B, matrix)
        print("Текущая матрица:")
        print(B)
        print("Кол-во векторов(m - rk(B)):")
        print(len(B) - np.linalg.matrix_rank(B))
        print("Векторы:", find_basis_u(B))
