#
#   Pulse models implemented using Cython.
#   2022 Fabian Jankowski
#

import cython
import cython.cimports.libc.math as cmath
import numpy as np


@cython.cfunc
@cython.exceptval(check=True)
def gaussian_normed(
    x: cython.double[:],
    fluence: cython.double,
    center: cython.double,
    sigma: cython.double,
) -> cython.double[:]:
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

    i: cython.int
    N: cython.int = len(x)
    res = np.zeros(N, dtype=cython.double)
    res_view: cython.double[:] = res
    A: cython.double

    A = fluence / (sigma * cmath.sqrt(2.0 * cmath.M_PI))

    for i in range(N):
        res_view[i] = A * cmath.exp(-0.5 * cmath.pow((x[i] - center) / sigma, 2))

    return res


@cython.cfunc
@cython.exceptval(check=True)
def scattered_gaussian_pulse(
    x: cython.double[:],
    fluence: cython.double,
    center: cython.double,
    sigma: cython.double,
    taus: cython.double,
    dc: cython.double,
) -> cython.double[:]:
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

    i: cython.int
    N: cython.int = len(x)
    A: cython.double
    B: cython.double
    C: cython.double
    D: cython.double
    res = np.zeros(N, dtype=cython.double)
    res_view: cython.double[:] = res

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

    return res_view
