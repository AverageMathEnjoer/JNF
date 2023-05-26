from JNF_and_basis import *
from numpy import array as M

from Integrals import *

f = lambda x: 1 + x + x**2 + x**3 + x**4
print(*square_integral(f, -1, 1, 0.001))
print(*trap_integral(f, -1, 1, 0.001))
print(simp_integral(f, -1, 1, 2))
