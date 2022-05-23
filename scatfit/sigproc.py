#
#   SIGPROC filterbank functions.
#   2022 Fabian Jankowski
#

from mtcutils import Candidate


def load_frb_data(filename, dm, fscrunch, tscrunch):
    """
    Load the FRB data from SIGPROC filterbank file.

    Parameters
    ----------
    filname: str
        The filterbank file to load.
    dm: float
        The dispersion measure to use to dedisperse the data.
    fscrunch: int
        The number of frequency channels to sum.
    tscrunch: int
        The number of time samples to sum.

    Returns
    -------
    cand: ~mtcutils.Candidate
        The candidate FRB data.
    """

    cand = Candidate.from_filterbank(filename)
    cand.normalise()

    # calculates and applies both IQRM and ACC1 masks
    mask = cand.apply_chanmask()
    print(
        "Channels masked based on stddev (via IQRM) and acc1: {} / {} ({:.2%})".format(
            mask.sum(), cand.nchans, mask.sum() / cand.nchans
        )
    )

    num_masked_chans = mask.sum()
    print(
        "Channels masked in total: {} / {} ({:.2%})".format(
            num_masked_chans, cand.nchans, num_masked_chans / cand.nchans
        )
    )

    # z-dot filter
    print("Applying z-dot filter")
    cand.zdot()

    # dedisperse
    cand.set_dm(dm)

    dynspec = cand.scrunched_data(f=fscrunch, t=tscrunch) / fscrunch**0.5
    cand.dynspec = dynspec

    return cand
