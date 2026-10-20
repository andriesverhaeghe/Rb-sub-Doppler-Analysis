import numpy as np


def poly1(x, a, b):
    x = np.array(x)
    return a * x + b


def poly2(x, a, b, c):
    x = np.array(x)
    return a * x**2 + b * x + c


def poly3(x, a, b, c, d):
    x = np.array(x)
    return a * x**3 + b * x**2 + c * x + d


def poly4(x, a, b, c, d, e):
    x = np.array(x)
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def poly5(x, a, b, c, d, e, f):
    x = np.array(x)
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f


def poly6(x, a, b, c, d, e, f, g):
    x = np.array(x)
    return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g


def poly7(x, a, b, c, d, e, f, g, h):
    x = np.array(x)
    return a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h


def poly8(x, a, b, c, d, e, f, g, h, i):
    x = np.array(x)
    return a * x**8 + b * x**7 + c * x**6 + d * x**5 + e * x**4 + f * x**3 + g * x**2 + h * x + i


def poly9(x, a, b, c, d, e, f, g, h, i, j):
    x = np.array(x)
    return (
        a * x**9
        + b * x**8
        + c * x**7
        + d * x**6
        + e * x**5
        + f * x**4
        + g * x**3
        + h * x**2
        + i * x
        + j
    )


def poly10(x, a, b, c, d, e, f, g, h, i, j, k):
    x = np.array(x)
    return (
        a * x**10
        + b * x**9
        + c * x**8
        + d * x**7
        + e * x**6
        + f * x**5
        + g * x**4
        + h * x**3
        + i * x**2
        + j * x
        + k
    )


def poly11(x, a, b, c, d, e, f, g, h, i, j, k, l):
    x = np.array(x)
    return (
        a * x**11
        + b * x**10
        + c * x**9
        + d * x**8
        + e * x**7
        + f * x**6
        + g * x**5
        + h * x**4
        + i * x**3
        + j * x**2
        + k * x
        + l
    )


def poly12(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
    x = np.array(x)
    return (
        a * x**12
        + b * x**11
        + c * x**10
        + d * x**9
        + e * x**8
        + f * x**7
        + g * x**6
        + h * x**5
        + i * x**4
        + j * x**3
        + k * x**2
        + l * x
        + m
    )
