#
#   Pulse models implemented using Cython.
#   2022 Fabian Jankowski
#

import cython
cimport libc.math as cmath
import numpy as np


def gaussian_normed(
    double[:] x,
    double fluence,
    double center,
    double sigma,
):
    """
    A normed Gaussian function.

    Parameters
    ----------
    x: ~np.array
        The running variable (time).
    fluence: float
        The fluence of the pulse, i.e. the area under it.
    center: float
        The mean of the Gaussian, i.e. its location.
    sigma: float
        The Gaussian standard deviation.

    Returns
    -------
    res: ~np.array
        The profile data.
    """

    cdef int i
    cdef int N = len(x)
    res = np.zeros(N, dtype=np.double)
    cdef double[:] res_view = res
    cdef double A

    A = fluence / (sigma * cmath.sqrt(2.0 * cmath.M_PI))

    for i in range(N):
        res_view[i] = A * cmath.exp(-0.5 * cmath.pow((x[i] - center) / sigma, 2))

    return res


def scattered_gaussian_pulse(
    double[:] x,
    double fluence,
    double center,
    double sigma,
    double taus,
    double dc,
):
    """
    A scattered Gaussian pulse. Analytical approach, assuming thin screen scattering.

    This implements Equation 4 from McKinnon 2014.

    Parameters
    ----------
    x: ~np.array
        The running variable (time).
    fluence: float
        The fluence of the pulse, i.e. the area under it.
    center: float
        The mean of the Gaussian, i.e. its location.
    sigma: float
        The Gaussian standard deviation.
    taus: float
        The scattering time.
    dc: float
        The vertical offset of the profile from the baseline.

    Returns
    -------
    res: ~np.array
        The profile data.
    """

    cdef int i
    cdef int N = len(x)
    cdef double A, B, C, D
    res = np.zeros(N, dtype=np.double)
    cdef double[:] res_view = res

    if sigma / taus >= 10.0:
        for i in range(N):
            res_view[i] = dc + gaussian_normed(x, fluence, center, sigma)[i]
    else:
        A = 0.5 * (fluence / taus)

        B = cmath.exp(0.5 * cmath.pow(sigma / taus, 2))

        for i in range(N):
            C = 1 + cmath.erf(
                (x[i] - (center + cmath.pow(sigma, 2) / taus))
                / (sigma * cmath.sqrt(2.0))
            )

            arg_D = -(x[i] - center) / taus

            if C == 0:
                arg_D = 0.0

            D = cmath.exp(arg_D)

            res_view[i] = dc + A * B * C * D

    return res
