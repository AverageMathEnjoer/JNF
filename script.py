import numpy as np

from JNF_and_basis import *
from itertools import combinations
from scipy.special import comb
from numpy import array as M


A = M([[4, 2], [5, 2], [2, 6], [3, 0]])
f = M([8, 4, 2, 8])
# 4 2 8
# 5 2 4
# 2 6 2
# 3 0 8
print(mnk(A, f))
