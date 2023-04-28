from GausFunction import *


def spec_with_multiple(phi: np.array) -> list:
    """ The function of counting the degrees of numbers in spector"""
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
    spec.append((gammas[len(gammas) - 1], counter))
    return spec


def linear_independence(vectors: list) -> bool:
    """ Bool function for checking linear independence of vectors"""
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


def new_find_all_basis(matrix: np.array, multiplicity: int) -> list:
    """ New version of function of finding complementary bases"""
    M = np.eye(len(matrix))
    basises = []
    M = np.dot(M, matrix)
    # Finding basis of first power of matrix
    prev = find_basis_u(M)
    basises.append(prev.copy())
    while len(M) - np.linalg.matrix_rank(M) < multiplicity:
        # Finding basis for higher powers of matrix
        M = np.dot(M, matrix)
        cur = find_basis_u(M)
        independent_b = []
        # Save only linearly independent with previous
        for i in cur:
            if linear_independence(prev + [i]):
                independent_b.append(i)
                prev.append(i)
        basises.append(independent_b.copy())
        prev = cur.copy()
    return basises


def orthogonalize(basis: np.array) -> np.array:
    """ Gram-Shmidt orthogonalize without orthonormalize"""
    basis = basis.astype(np.float64)
    f = [basis[0]]
    m = basis.shape[0]
    # Just the sum of the previous vectors with coefficients according to the formula
    for i in range(1, m):
        e_i = basis[i]
        f_i = e_i
        for f_j in f:
            f_i += -(np.dot(e_i, f_j) / np.dot(f_j, f_j)) * f_j
        f.append(f_i)
    return np.array(f)


def orthonormalize(basis: np.array) -> np.array:
    """ Orthonormalize without Gram-Shmidt orthogonalize"""
    # Just divide by the length
    basis = basis.astype(np.float64)
    for i in range(len(basis)):
        basis[i] /= np.sqrt(np.dot(basis[i], basis[i]))
    return basis
