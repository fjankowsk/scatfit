#
#   Profile functions.
#   2025 Fabian Jankowski
#

import numpy as np

from scatfit.stats import get_robust_std


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
    off_std = get_robust_std(off, axis=None)
    w_eq = np.sum(on) / np.max(on)

    energy = np.sum(on - off_mean)

    if energy > 0 and off_std > 0 and w_eq > 0:
        snr = energy / (off_std * np.sqrt(w_eq))
    else:
        snr = 0.0

    return snr
