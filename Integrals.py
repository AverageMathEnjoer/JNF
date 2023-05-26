import numpy as np


def __square_left_integral(f, a, b):
    return f(a) * (b - a)


def __square_mid_integral(f, a, b):
    return f((b + a) / 2) * (b - a)


def __square_right_integral(f, a, b):
    return f(b) * (b - a)


def __simp_integral(f, a, b):
    return (b - a) * (f(a) + 4 * f((a + b) / 2) + f(b)) / 6


def rec_square_integral(f, a, b, n):
    delta = (b - a) / n
    l_sum = sum([__square_left_integral(f, i, i + delta) for i in np.arange(a, b, delta)])
    m_sum = sum([__square_mid_integral(f, i, i + delta) for i in np.arange(a, b, delta)])
    r_sum = sum([__square_right_integral(f, i, i + delta) for i in np.arange(a, b, delta)])
    return l_sum, m_sum, r_sum


def square_integral(f, a, b, eps):
    n = 16
    l, m, r = rec_square_integral(f, a, b, n)
    while abs(max(l, m, r) - min(l, m, r)) - eps > 0:
        n *= 2
        l, m, r = rec_square_integral(f, a, b, n)
    return min(l, m, r), (l, m, r), max(l, m, r), n


def trap_integral(f, a, b, eps):
    n = 16
    l, m, r = rec_square_integral(f, a, b, n)
    t_ = (l + r) / 2
    n *= 2
    l, m, r = rec_square_integral(f, a, b, n)
    t = (l + r) / 2
    while abs(t - t_) > eps:
        t_ = t
        n *= 2
        l, m, r = rec_square_integral(f, a, b, n)
        t = (l + r) / 2
    return l, t, r, n


def simp_integral(f, a, b, h):
    return sum([__simp_integral(f, i, i + h) for i in np.arange(a, b, h)])


def eps_simp_integral(f, a, b, eps):
    h = 1
    i_ = simp_integral(f, a, b, h)
    h /= 2
    i = simp_integral(f, a, b, h)
    while abs(i - i_) > eps:
        i_ = i
        h /= 2
        i = simp_integral(f, a, b, h)
    return i, h
