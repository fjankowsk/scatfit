#
#   Statistics functions.
#   2025 Fabian Jankowski
#

import numpy as np


def get_iqr(data, axis):
    """
    Compute the interquartile range.

    Parameters
    ----------
    data: ~np.array
        The input data as numpy float array.
    axis: int
        The data axis to compute the statistic over.

    Returns
    -------
    iqr: float
        The interquartile range.
    """

    q25, q75 = np.nanpercentile(data, [25.0, 75.0], axis=axis)

    iqr = q75 - q25

    return iqr


def get_robust_std(data, axis, mode="Gaussian"):
    """
    Compute the robust standard deviation using the IQR.

    Parameters
    ----------
    data: ~np.array
        The input data.
    axis: int
        The data axis to compute the statistic over.
    mode: str
        The underlying distribution to assume when computing the robust standard deviation.

    Returns
    -------
    sr: float
        The robust standard deviation.

    Raises
    ------
    NotImplementedError
        If the `mode` is not implemented.
    """

    iqr = get_iqr(data, axis=axis)

    if mode == "Gaussian":
        fact = 0.7413
    elif mode == "Rice":
        fact = 0.9183
    elif mode == "Exponential":
        fact = 0.9102
    else:
        raise NotImplementedError(f"Distribution unknown: {mode}")

    sr = fact * iqr

    return sr
