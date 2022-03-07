#
#   DM functions.
#   2022 Fabian Jankowski
#


def get_dm_smearing(f_lo, f_hi, dm):
    """
    Compute the intra-channel dispersive smearing.

    Parameters
    ----------
    f_lo: float
        The low frequency edge of the channel in GHz.
    f_hi: float
        The high frequency edge of the channel in GHz.
    dm: float
        The dispersion measure of the source in pc/cm^3.

    Returns
    -------
    dt: float
        The intra-channel dispersive smearing in ms.
    """

    dt = 4.148808 * (f_lo ** (-2) - f_hi ** (-2)) * dm

    return dt
