#
#   Pulse models.
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

    res = (
        fluence
        / (sigma * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * np.power((x - center) / sigma, 2))
    )

    return res


def scattered_gaussian_pulse(x, fluence, center, sigma, tau, dc):
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
    tau: float
        The scattering time.
    dc: float
        The vertical offset of the profile from the baseline.

    Returns
    -------
    res: ~np.array
        The profile data.
    """

    if sigma / tau >= 10.0:
        res = dc + gaussian_normed(x, fluence, center, sigma)
    else:
        A = 0.5 * (fluence / tau)

        B = np.exp(0.5 * np.power(sigma / tau, 2))

        C = np.exp(-(x - center) / tau)

        D = 1 + special.erf(
            (x - (center + np.power(sigma, 2) / tau)) / (sigma * np.sqrt(2.0))
        )

        mask = D == 0
        C[mask] = 1.0

        res = dc + A * B * C * D

    return res


def gaussian_scattered_afb_instrumental(
    x, fluence, center, sigma, taus, taui, taud, dc
):
    """
    A Gaussian pulse scattered in the ISM and affected by analogue
    (single-sided exponential) instrumental effects from DM-smearing and
    the detector/signal chain.

    This implements Eq. 7 from McKinnon 2014.

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
        The scattering time in the ISM.
    taui: float
        The scattering time due to instrumental effects in the receiver or signal chain
        (e.g. integration time constant).
    taud: float
        The scattering time due to intra-channel dispersive smearing.
    dc: float
        The vertical offset of the profile from the baseline.

    Returns
    -------
    res: ~np.array
        The profile data.
    """

    A = (
        np.power(taus, 2)
        * scattered_gaussian_pulse(x, fluence, center, sigma, taus, 0.0)
        / ((taus - taui) * (taus - taud))
    )

    B = (
        np.power(taui, 2)
        * scattered_gaussian_pulse(x, fluence, center, sigma, taui, 0.0)
        / ((taus - taui) * (taui - taud))
    )

    C = (
        np.power(taud, 2)
        * scattered_gaussian_pulse(x, fluence, center, sigma, taud, 0.0)
        / ((taus - taud) * (taui - taud))
    )

    res = dc + A - B + C

    return res


def boxcar(x, width):
    """
    A simple normalised boxcar function.

    Parameters
    ----------
    x: ~np.array
        The running variable (time).
    width: float
        The width of the boxcar function.

    Returns
    -------
    res: ~np.array
        The boxcar data.
    """

    res = np.zeros(len(x))

    mask = np.abs(x) <= 0.5 * width
    res[mask] = 1.0

    res = res / np.sum(res)

    return res


def gaussian_scattered_dfb_instrumental(x, fluence, center, sigma, taus, taud, dc):
    """
    A Gaussian pulse scattered in the ISM and affected by digital (boxcar-like) instrumental effects.
    Convolving approach. We neglect instumental receiver or signal chain effects.

    This implements Eq. 2 from Loehmer et al. 2001.

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
        The scattering time in the ISM.
    taud: float
        The scattering time due to intra-channel dispersive smearing.
    dc: float
        The vertical offset of the profile from the baseline.

    Returns
    -------
    res: ~np.array
        The profile data.
    """

    A = scattered_profile(x, fluence, center, sigma, taus, 0.0)

    B = boxcar(x, taud)

    res = dc + np.convolve(A, B, mode="same")

    return res


def gaussian_fwhm(sigma):
    """
    The full width at half maximum (W50) of a Gaussian.

    Parameters
    ----------
    sigma: float
        The Gaussian standard deviation.

    Returns
    -------
    res: float
        The Gaussian W50.
    """

    res = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

    return res


def gaussian_fwtm(sigma):
    """
    The full width at tenth maximum (W10) of a Gaussian.

    Parameters
    ----------
    sigma: float
        The Gaussian standard deviation.

    Returns
    -------
    res: float
        The Gaussian W10.
    """

    res = 2.0 * np.sqrt(2.0 * np.log(10)) * sigma

    return res


def broadening_function(x, tau):
    """
    A broadening function for isotropic scattering.

    Parameters
    ----------
    x: ~np.array
        The running variable (time).
    tau: float
        The scattering timescale.

    Returns
    -------
    res: ~np.array
        The profile data.
    """

    res = np.zeros(len(x))

    mask = x >= 0.0
    res[mask] = (1 / tau) * np.exp(-x[mask] / tau)

    return res


def scattered_profile(x, fluence, center, sigma, tau, dc):
    """
    A scattered pulse profile.

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
    tau: float
        The scattering time.
    dc: float
        The vertical offset of the profile from the baseline.

    Returns
    -------
    res: ~np.array
        The profile data.
    """

    scattered = dc + np.convolve(
        gaussian_normed(x, fluence, center, sigma),
        broadening_function(x, tau),
        mode="same",
    )

    return scattered
