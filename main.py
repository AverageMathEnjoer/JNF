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
    spec.append((gammas[len(gammas) - 1], counter))
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
        independent_b = []
        for i in cur:
            if linear_independence(prev + [i]):
                independent_b.append(i)
                prev.append(i)
        basises.append(independent_b.copy())
        prev = cur.copy()
    return basises


def orthogonalize(basis):
    basis = basis.astype(np.float64)
    f = [basis[0]]
    m = basis.shape[0]
    n = basis.shape[1]
    for i in range(1, m):
        e_i = basis[i]
        f_i = e_i
        for f_j in f:
            f_i += -(np.dot(e_i, f_j) / np.dot(f_j, f_j)) * f_j
        f.append(f_i)
    return np.array(f)


def orthonormalize(basis):
    basis = orthogonalize(basis)
    for i in range(len(basis)):
        basis[i] /= np.sqrt(np.dot(basis[i], basis[i]))
    return basis


A = np.array([[3, -1, -1, 1],
      [1, 2, -1, -1], np.zeros(4), np.zeros(4)])
print(new_find_all_basis(A, 2))
print(spec_with_multiple(A))

