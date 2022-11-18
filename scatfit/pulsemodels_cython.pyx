#
#   Pulse models implemented using Cython.
#   2022 Fabian Jankowski
#

import cython
cimport libc.math as cmath
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
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


@cython.boundscheck(False)
@cython.wraparound(False)
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
    cdef double A, B, C, arg_D, D
    cdef double[:] gauss_tmp
    res = np.zeros(N, dtype=np.double)
    cdef double[:] res_view = res

    if sigma / taus >= 10.0:
        gauss_tmp = gaussian_normed(x, fluence, center, sigma)

        for i in range(N):
            res_view[i] = dc + gauss_tmp[i]
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


@cython.boundscheck(False)
@cython.wraparound(False)
def bandintegrated_model(
    double[:] x,
    double fluence,
    double center,
    double sigma,
    double taus,
    double dc,
    double f_lo,
    double f_hi,
    int nfreq
):
    """
    A true frequency band-integrated profile model.

    The total (sub-)band-integrated profile is the superposition (weighted sum or
    weighted mean) of several profiles that evolve with frequency across the bandwidth
    of the frequency (sub-)band, one for each frequency channel. Namely, the individual
    profiles evolve with frequency (scattering, pulse width, fluence). For large
    fractional bandwidths or at low frequencies (< 1 GHz), the profile evolution across
    the band cannot be neglected, i.e. the narrow-band approximation fails.

    We compute the frequency evolution across the band between `f_lo` and `f_hi` at
    `nfreq` centre frequencies. The total profile is then the weighted sum over the
    finite frequency grid. Ideally, one would use an infinitesimally narrow grid here.

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
    f_lo: float
        The centre frequency of the lowest channel in the sub-band.
    f_hi: float
        The centre frequency of the highest channel in the sub-band.
    nfreq: int
        The number of centre frequencies to evaluate.

    Returns
    -------
    res: ~np.array
        The profile data.
    """

    cdef double band_cfreq = 0.5 * (f_lo + f_hi)
    cdef int i
    cdef int N = len(x)
    cdef double[:] scatpulse_tmp

    # the low-frequency profiles dominate the total band-integrated
    # profile because of the strong fluence power law scaling
    # use finer steps towards the low-frequency band edge
    cdef double[:] cfreqs = np.geomspace(f_lo, f_hi, num=nfreq)

    profile = np.zeros(N, dtype=np.double)
    cdef double[:] profile_view = profile

    for i in range(nfreq):
        taus_i = taus * cmath.pow(cfreqs[i] / band_cfreq, -4.0)
        fluence_i = fluence * cmath.pow(cfreqs[i] / band_cfreq, -1.5)

        scatpulse_tmp = scattered_gaussian_pulse(x, fluence_i, center, sigma, taus_i, 0.0)

        # accumulate
        for j in range(N):
            profile_view[j] = profile_view[j] + scatpulse_tmp[j]

    # normalise to match input fluence
    cdef double tot_fluence = 0.0
    for j in range(N):
        tot_fluence = tot_fluence + profile_view[j]

    tot_fluence = tot_fluence * cmath.fabs(x[0] - x[1])

    for j in range(N):
        profile_view[j] = dc + (fluence / tot_fluence) * profile_view[j]

    return profile