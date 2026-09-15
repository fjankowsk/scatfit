#
#   Pulse models.
#   2022 - 2025 Fabian Jankowski
#

import numpy as np
from scipy import signal

try:
    from scatfit.pulsemodels_cython import (
        gaussian_normed,
        scattered_gaussian_pulse,
        bandintegrated_model,
    )
except ImportError:
    print(
        "Could not import Cython pulse model implementations. Falling back to the Python versions."
    )
    from scatfit.pulsemodels_python import (
        gaussian_normed,
        scattered_gaussian_pulse,
        bandintegrated_model,
    )


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

    assert fluence > 0
    assert sigma > 0
    assert taus > 0
    assert taui > 0
    assert taud > 0
    assert taus != taui
    assert taus != taud

    A = np.power(taus, 2) * scattered_gaussian_pulse(x, 1.0, center, sigma, taus, 0.0)

    B = np.power(taui, 2) * scattered_gaussian_pulse(x, 1.0, center, sigma, taui, 0.0)

    C = np.power(taud, 2) * scattered_gaussian_pulse(x, 1.0, center, sigma, taud, 0.0)

    D = (taus - taui) * (taus - taud)

    res = (A - B + C) / D

    # normalise to match input fluence
    tot_fluence = np.sum(res) * np.abs(x[0] - x[1])
    res = dc + (fluence / tot_fluence) * res

    return res


def boxcar(x, width):
    """
    A simple boxcar function.

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

    assert width > 0

    res = np.zeros(len(x))

    mask = np.abs(x) <= 0.5 * width
    res[mask] = 1.0

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

    assert fluence > 0
    assert sigma > 0
    assert taus > 0
    assert taud > 0

    A = scattered_gaussian_pulse(x, fluence, center, sigma, taus, 0.0)

    B = boxcar(x, taud)

    res = dc + signal.oaconvolve(A, B, mode="same") / np.sum(B)

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

    res = 2.0 * np.sqrt(2.0 * np.log(10.0)) * sigma

    return res


def equivalent_width(x, amp, sigma_noise=None):
    """
    Compute the boxcar equivalent width.

    Parameters
    ----------
    x: ~np.array
        The running variable (time).
    amp: ~np.array
        The pulse amplitude.
    sigma_noise: float, optional
        1-sigma Gaussian baseline noise level. If given, used for thresholding.
        Otherwise, 1 % of peak amplitude is used.

    Returns
    -------
    weq: float
        The equivalent width.
    """

    assert x.dtype == np.float64
    assert amp.dtype == np.float64
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(amp))
    assert len(x) == len(amp)

    # rectify, clip at zero
    amp = np.maximum(amp, 0.0)

    max_amp = np.max(amp)
    if max_amp <= 0.0:
        return np.nan

    # thresholding
    if sigma_noise is not None:
        # remove 95 % of noise fluctuations
        _thresh = 2.0 * sigma_noise
    else:
        _thresh = 0.01 * max_amp

    mask = amp >= _thresh
    dx = np.abs(x[1] - x[0])
    fluxsum = np.sum(amp[mask]) * dx
    weq = fluxsum / max_amp

    if not np.isfinite(weq):
        return np.nan

    return weq


def full_width_post(x, amp, level):
    """
    Compute the full pulse width post scattering numerically.

    Parameters
    ----------
    x: ~np.array
        The running variable (time).
    amp: ~np.array
        The pulse amplitude.
    level: float
        The level at which to evaluate the pulse width.

    Returns
    -------
    width: float
        The full pulse width at the given level.
    """

    assert x.dtype == np.float64
    assert amp.dtype == np.float64
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(amp))
    assert len(x) == len(amp)
    assert 0.0 < level <= 1.0

    # rectify, clip at zero
    amp = np.maximum(amp, 0.0)

    max_amp = np.max(amp)
    if max_amp <= 0.0:
        return np.nan

    threshold = level * max_amp
    i_peak = int(np.argmax(amp))

    above = amp >= threshold

    # find transitions
    edges = np.diff(above.view(np.int8))
    rising = np.where(edges == 1)[0]
    falling = np.where(edges == -1)[0]

    # left crossing
    left_idx = rising[rising < i_peak]
    if left_idx.size == 0:
        return np.nan

    # below bin
    k = left_idx[-1]
    y0, y1 = amp[k], amp[k + 1]
    # interpolate
    left = x[k] + (threshold - y0) / (y1 - y0) * (x[k + 1] - x[k])

    # right crossing
    right_idx = falling[falling >= i_peak]
    if right_idx.size == 0:
        return np.nan

    # above bin
    k = right_idx[0]
    y0, y1 = amp[k], amp[k + 1]
    # interpolate
    right = x[k] + (threshold - y0) / (y1 - y0) * (x[k + 1] - x[k])

    width = right - left

    if not (np.isfinite(width) and width > 0.0):
        return np.nan

    return width


def d4sigma_width(x, amp, sigma_noise=None):
    """
    Compute the D4Sigma pulse width based on the second-moment (variance) width.
    The function assumes that the signal is centred.

    Parameters
    ----------
    x: ~np.array, float64
        The running variable (time).
    amp: ~np.array, float64
        The pulse amplitude.
    sigma_noise: float, optional
        1-sigma Gaussian baseline noise level. If given, used for thresholding.
        Otherwise, 1 % of peak amplitude is used.

    Returns
    -------
    wd4s: float
        The D4sigma pulse width.
    """

    assert x.dtype == np.float64
    assert amp.dtype == np.float64
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(amp))
    assert len(x) == len(amp)

    # rectify, clip at zero
    amp = np.maximum(amp, 0.0)

    max_amp = np.max(amp)
    if max_amp <= 0.0:
        return np.nan

    # thresholding
    if sigma_noise is not None:
        # remove 95 % of noise fluctuations
        _thresh = 2.0 * sigma_noise
    else:
        _thresh = 0.01 * max_amp

    mask = amp >= _thresh
    x_valid = x[mask]
    amp_valid = amp[mask]

    area = np.sum(amp_valid)
    if not np.isfinite(area) or area <= 0.0:
        return np.nan

    # second moment
    x_mean = np.sum(amp_valid * x_valid) / area
    if not np.isfinite(x_mean):
        return np.nan

    x_diff = x_valid - x_mean
    variance = np.sum(amp_valid * x_diff**2) / area

    if not np.isfinite(variance) or variance < 0.0:
        return np.nan

    wd4s = 4.0 * np.sqrt(variance)

    return wd4s


def pbf_isotropic(plot_range, taus):
    """
    A pulse broadening function for isotropic scattering.

    Parameters
    ----------
    plot_range: ~np.array
        The evaluation variable (time) in ms.
    taus: float
        The scattering time in ms.

    Returns
    -------
    res: ~np.array
        The profile data.

    Raises
    ------
    RuntimeError
        If the window array is too short to fit the vast
        majority of the exponential sweep, i.e. taus is
        too large for the given time span.
    """

    assert taus > 0

    N = len(plot_range)
    tsamp = np.abs(plot_range[0] - plot_range[1])

    x = np.arange(N) * tsamp
    x -= x[N // 2]

    assert x[N // 2] == 0

    # ensure that we capture the vast majority of the sweep
    # make the window array long enough
    # exp(-t0 / taus) !<= 0.001
    t0 = -taus * np.log(0.001)

    if np.max(x) < t0:
        raise RuntimeError(
            "The window array is too short: {0}, {1}".format(np.max(x), t0)
        )

    res = np.zeros(len(x))

    invtaus = 1.0 / taus

    mask = x >= 0.0
    res[mask] = invtaus * np.exp(-x[mask] * invtaus)

    return res


def scattered_profile(x, fluence, center, sigma, taus, dc):
    """
    A scattered pulse profile.

    Implemented using numerical convolution with a pulse broadening function.

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

    A = gaussian_normed(x, fluence, center, sigma)

    B = pbf_isotropic(x, taus)

    scattered = dc + signal.oaconvolve(A, B, mode="same") / np.sum(B)

    return scattered
