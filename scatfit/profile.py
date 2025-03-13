#
#   Profile functions.
#   2025 Fabian Jankowski
#

import numpy as np


def get_snr_weq(on, off):
    """
    Compute the S/N of a profile using its equivalent width.
    This algorithm is based on Pulsar handbook page 167.

    Parameters
    ----------
    on: ~np.array
        The on-pulse profile data.
    off: ~np.array
        The off-pulse profile data.

    Returns
    -------
    snr: float
        The S/N of the profile.
    """

    off_mean = np.mean(off)
    off_quantiles = np.quantile(off, q=[0.25, 0.75], axis=None)
    off_std = 0.7413 * np.abs(off_quantiles[1] - off_quantiles[0])
    w_eq = np.sum(on) / np.max(on)

    energy = np.sum(on - off_mean)

    if energy > 0 and off_std > 0 and w_eq > 0:
        snr = energy / (off_std * np.sqrt(w_eq))
    else:
        snr = 0.0

    return snr
