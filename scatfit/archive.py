#
#   Functions to read a PSRCHIVE file.
#   2023 Ines Pastor-Marazuela
#   2024 Fabian Jankowski
#

import numpy as np
import os.path

try:
    import psrchive as ps
except ModuleNotFoundError:
    print("Could not import PSRCHIVE python bindings.")

from mtcutils.dedisp import dispersion_shifts, roll2d
from mtcutils.core import scrunch, zdot, spectral_acc1, tukey_mask
from iqrm import iqrm_mask


class Candidate(object):
    """
    Psrchive Candidate class.

    Parameters
    ----------
    fname: str
        Path to file
    """

    def __init__(self, fname):
        self._fname = os.path.realpath(fname)
        self._arch = self._load_arch()
        self._data = self._load_data()

        # Calculate original channel standard deviations,
        # before normalise() is called
        # NOTE: cast to np.float32 matters, otherwise normalising the data
        # will automatically upcast it to float64
        self._bpmean = self._data.mean(axis=1, dtype=float).astype(np.float32)
        self._bpstd = self._data.std(axis=1, dtype=float).astype(np.float32)
        self._bpacc1 = spectral_acc1(self._data)
        self._normalised = False

        # Track current DM and channel shifts so the dedispersion
        # operation can be reversed
        self._dm = 0.0
        self._dispersion_shifts = np.zeros(self.nchans, dtype=int)

    def _load_arch(self):
        arch = ps.Archive_load(self._fname)
        arch.remove_baseline()
        return arch

    def _load_data(self):
        data = np.squeeze(self._arch.get_data())
        # in case of having Stokes IQUV data
        # if self._arch.get_npol() == 4:
        if len(data.shape) == 3:
            data = data[0]
        # Arranging frequencies in correct order
        data = np.flip(data, axis=0)
        return data

    @property
    def nsamp(self):
        """Number of samples in the data"""
        return self._arch.get_nbin()

    @property
    def nchans(self):
        """Number of channels in the data"""
        return self._arch.get_nchan()

    @property
    def nsubint(self):
        """Number of subintegrations of the archive"""
        return self._arch.get_nsubint()

    @property
    def period(self):
        """Topocentric folding period in seconds (from first subint)"""
        return self._arch[0].get_folding_period()

    @property
    def tsamp(self):
        """Bin width of profile samples in seconds"""
        return self.period / self.nsamp

    @property
    def fcen(self):
        """Central frequency in MHz"""
        return self._arch.get_centre_frequency()

    @property
    def bw(self):
        """Observing bandwidth in MHz"""
        return self._arch.get_bandwidth()

    @property
    def fch1(self):
        """Frequency of the first channel (MHz)"""
        return self.fcen + self.bw / 2

    @property
    def foff(self):
        """Frequency offset between consecutive channels (MHz). Can be negative."""
        return self.bw / self.nchans

    @property
    def fchn(self):
        """Frequency of the last channel (MHz)"""
        return self.fcen - self.bw / 2

    @property
    def freqs(self):
        """Channel frequencies in MHz, in the same order as they appear in the data"""
        return self.fch1 + np.arange(self.nchans) * self.foff

    @property
    def times(self):
        """Time offset of all samples from the start of the data, in seconds"""
        return np.arange(self.nsamp) * self.tsamp

    @property
    def tstart(self):
        """Start MJD of first time sample at the top frequency channel"""
        return self._arch.get_Integration(0).get_start_time().in_days()

    @property
    def normalised(self):
        return self._normalised

    @property
    def bandpass_mean(self):
        """Mean of the data along the time axis, BEFORE normalisation"""
        return self._bpmean

    @property
    def bandpass_std(self):
        """Standard deviation of the data along the time axis, BEFORE normalisation"""
        return self._bpstd

    @property
    def bandpass_acc1(self):
        """Autocorrelation coefficient with a lag of 1 time sample along the time axis"""
        return self._bpacc1

    @property
    def dm(self):
        """Current dispersion measure of the data in pc cm^{-3}"""
        return self._dm

    def normalise(self):
        """
        Normalize all channels to zero mean and unit variance, in place. An exception is constant
        channels, which are just set to zero.
        """
        if self._normalised:
            return

        m = self.bandpass_mean
        s = self.bandpass_std.copy()  # leave original bandpass_std untouched
        s[s == 0] = 1.0
        self._data = (self._data - m.reshape(-1, 1)) / s.reshape(-1, 1)
        self._normalised = True

    def set_dm(self, dm):
        """
        Set the DM of the data to specified value, by circularly shifting the
        channels appropriately.
        """
        new_shifts = dispersion_shifts(self.freqs, dm, self.tsamp)
        delta_shifts = new_shifts - self._dispersion_shifts
        self._data = roll2d(self._data, -delta_shifts)
        self._dispersion_shifts = new_shifts
        self._dm = dm

    def zerodm(self):
        """
        Apply the zero-DM filter, that is subtract the mean of the data along
        the frequency axis from every frequency channel. This method takes
        care of setting the DM of the data to zero before this operation,
        and then sets it back to the original DM afterwards.

        Raises
        ------
        ValueError : if the data are not normalised yet
        """
        if not self._normalised:
            raise ValueError(
                "Refusing to apply zero-DM filter on non-normalised data, call normalise() first"
            )

        dm = self.dm
        self.set_dm(0.0)
        self._data -= self._data.mean(axis=0, dtype=float)
        self.set_dm(dm)

    def zdot(self):
        """
        Apply the Z-dot filter to the data, in place. This is a much better
        variant of the zero-DM filter.
        See Men et al. 2019:
        https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3957M/abstract

        Raises
        ------
        ValueError : if the data are not normalised yet
        """
        if not self._normalised:
            raise ValueError(
                "Refusing to apply zero-DM filter on non-normalised data, call normalise() first"
            )

        dm = self.dm
        self.set_dm(0.0)
        self._data = zdot(self._data)
        self.set_dm(dm)

    def apply_chanmask(self, mask=None):
        """
        Apply given channel mask, where 'True' elements represent a bad channel.
        If no mask is specified, this function will instead infer a mask from
        the data directly, following what is currently done inside the MeerTRAP
        pipeline (as of late Oct. 2021):

        1. Run IQRM on the spectral standard deviation of the data, with
           radius = 0.1 x nchan, threshold = 3.0
        2. Find channels with abnormal ACC1 (autocorrelation coefficient with a
           lag of 1 sample).

        A channel that registers as an outlier in either step is replaced by zeros.
        The function returns the boolean channel mask that was applied.

        Parameters
        ----------
        mask: ndarray or None
            Behaviour of the function depends on argument type:
            - boolean ndarray: mask channels for which the mask is True
            - None: infer using IQRM + ACC1 outlier masking

        Returns
        -------
        mask_array: ndarray
            The boolean mask that was applied
        """
        if not self.normalised:
            raise ValueError("Must normalise data first")

        if mask is None:
            radius = max(2, int(0.1 * self.nchans))
            miqrm, __ = iqrm_mask(self.bandpass_std, radius=radius, threshold=3.0)
            # NOTE: Using |acc1| would work better, but currently (Nov 2021)
            # the MeerTRAP pipeline uses acc1 without the absolute value
            # NOTE 2: Using percentiles below 50-th to determine robust stddev
            # would be better, because there are often cases where more than
            # 25% of the channels have excessive |acc1|. But here again we
            # follow what is currently inside the pipeline.
            macc1 = tukey_mask(self.bandpass_acc1, threshold=2.0)
            mask = miqrm | macc1
        elif isinstance(mask, np.ndarray):
            pass
        else:
            raise ValueError("mask should be a boolean array or None")

        self._data[mask] = 0.0
        return mask

    def scrunched_data(self, t=1, f=1, select="left"):
        """
        Returns a copy of the data scrunched in time and frequency

        Parameters
        ----------
        t : int
            Time scrunching factor. If 't' does not divide 'nsamp',
            then some samples will be excluded from the scrunched data.
            Which samples are excluded depend on the 'select' argument.
        f : int
            Freq scrunching factor. If 'f' does not divide 'nchan',
            an error is raised.
        select : str
            Time sample selection policy, either 'left' or 'centre'. The number
            of samples selected from scrunching is Ns = nsamp - nsamp % t
            If 'left', select the first Ns samples.
            If 'centre', select the middle Ns samples.

        Returns
        -------
        scrunched : ndarray
            The scrunched data
        """
        return scrunch(self._data, t=t, f=f, select=select)


def load_frb_data(filename, dm, fscrunch, tscrunch):
    """
    Load the FRB data from a psrchive file.

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

    cand = Candidate(filename)
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
    print("Applying z-dot filter if DM > 20 pc cm^-3.")
    if dm > 20:
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
