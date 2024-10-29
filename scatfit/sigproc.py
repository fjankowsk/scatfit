#
#   SIGPROC filterbank functions.
#   2022 Fabian Jankowski
#

from mtcutils import Candidate
import numpy as np


def load_frb_data(filename, dm, fscrunch, tscrunch, norfi):
    """
    Load the FRB data from a SIGPROC filterbank file.

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
    norfi: bool
        Do not perform RFI excision.

    Returns
    -------
    cand: ~mtcutils.Candidate
        The candidate FRB data.
    """

    cand = Candidate.from_filterbank(filename)
    cand.normalise()

    # XXX: move the rfi excision methods outside the specific data loader
    if not norfi:
        # calculates and applies both IQRM and ACC1 masks
        mask = cand.apply_chanmask()
        print(
            "Channels masked based on stddev (via IQRM) and acc1: {} / {} ({:.2%})".format(
                mask.sum(), cand.nchans, mask.sum() / cand.nchans
            )
        )

        # z-dot filter
        print("Applying z-dot filter")
        cand.zdot()

    # dedisperse
    cand.set_dm(dm)

    dynspec = cand.scrunched_data(f=fscrunch, t=tscrunch, select="left") / fscrunch**0.5
    cand.dynspec = dynspec
    times = np.arange(cand.nsamp // tscrunch) * cand.tsamp * tscrunch
    cand.tval = times
    freqs = cand.fch1 + np.arange(cand.nchans // fscrunch) * cand.foff * fscrunch
    cand.fval = freqs

    return cand
