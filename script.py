from JNF_and_basis import *
from numpy import array as M


A = M([[9, 0, 0, 0, 0, 0], [0, 9, 0, 0, 0, 0], [4, 0, 9, 0, 0, 0], [0, 1, -3, 9, 0, 0], [0, -3, 1, 0, 9, 0],
       [0, 0, 1, 0, -2, 9]])

S, J = JNF(A)
print(S)
print(J)

# 4 2 8
# 5 2 4
# 2 6 2
# 3 0 8

