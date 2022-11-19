#
#   Pulse models implemented using Python and Numpy.
#   2022 Fabian Jankowski
#

import numpy as np
from scipy import special


def gaussian_normed(x, fluence, center, sigma):
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

    invsigma = 1.0 / sigma
    invsqrt = 1.0 / np.sqrt(2.0 * np.pi)
    A = fluence * invsigma * invsqrt

    res = A * np.exp(-0.5 * np.power((x - center) * invsigma, 2))

    return res


def scattered_gaussian_pulse(x, fluence, center, sigma, taus, dc):
    """
    A scattered Gaussian pulse. Analytical approach, assuming thin screen scattering.

    We use a standard implementation of an exponentially modified Gaussian here, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html

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

    # treat the following special cases
    # 1) invK >> 1, i.e. sigma >> taus
    # -> function becomes a regular gaussian

    invsigma = 1.0 / sigma
    K = taus * invsigma
    invK = 1.0 / K
    y = (x - center) * invsigma

    if invK >= 10.0:
        mu_gauss = center + taus
        res = dc + gaussian_normed(x, fluence, mu_gauss, sigma)
    else:
        argexp = 0.5 * invK**2 - y * invK

        # prevent numerical overflows
        mask = argexp >= 300.0
        argexp[mask] = 0.0

        exgaussian = (
            0.5
            * invK
            * invsigma
            * np.exp(argexp)
            * special.erfc(-(y - invK) / np.sqrt(2.0))
        )

        res = dc + fluence * exgaussian

    return res


def bandintegrated_model(x, fluence, center, sigma, taus, dc, f_lo, f_hi, nfreq):
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

    band_cfreq = 0.5 * (f_lo + f_hi)

    # the low-frequency profiles dominate the total band-integrated
    # profile because of the strong fluence power law scaling
    # use finer steps towards the low-frequency band edge
    cfreqs = np.geomspace(f_lo, f_hi, num=nfreq)

    taus_s = taus * np.power(cfreqs / band_cfreq, -4.0)
    fluence_s = fluence * np.power(cfreqs / band_cfreq, -1.5)

    profiles = np.zeros(shape=(nfreq, len(x)))

    for i in range(nfreq):
        profiles[i, :] = scattered_gaussian_pulse(
            x, fluence_s[i], center, sigma, taus_s[i], 0.0
        )

    # sum, weighted by fluence above
    res = np.sum(profiles, axis=0)

    # normalise to match input fluence
    tot_fluence = np.sum(res) * np.abs(x[0] - x[1])
    res = dc + (fluence / tot_fluence) * res

    return res
