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
        The low frequency edge of the channel in MHz.
    f_hi: float
        The high frequency edge of the channel in MHz.
    dm: float
        The dispersion measure of the source in pc/cm^3.

    Returns
    -------
    dt: float
        The intra-channel dispersive smearing in ms.
    """

    # use the inverse dispersion constant rounded to three digits that
    # is in common use in pulsar astronomy to be consistent throughout
    # software tools (see page 129 of Manchester and Taylor 1977)
    kdm = 1.0 / 2.41e-4

    dt = kdm * (f_lo ** (-2) - f_hi ** (-2)) * dm

    # we want ms
    dt = 1.0e3 * dt

    return dt
