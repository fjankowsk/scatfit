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
    off_var = np.var(off)
    w_eq = np.sum(on) / np.max(on)

    energy = np.sum(on - off_mean)

    snr = energy / np.sqrt(off_var * w_eq)

    # treat special cases
    if energy < 0 or w_eq <= 0:
        snr = 0

    return snr
