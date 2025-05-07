#
#   Simulate scattered pulses.
#   2022 - 2025 Fabian Jankowski
#

import matplotlib.pyplot as plt
import numpy as np
from your import Your
from your.formats.filwriter import make_sigproc_object

from scatfit.dm import KDM
import scatfit.pulsemodels as pulsemodels


class Pulse(object):
    def __init__(self, dm, sigma, taus_1ghz):
        """
        A pulse.
        """

        self.dm = dm
        self.dm_index = -2.0
        self.scatindex = -4.0
        self.spectral_index = -1.5
        self.fluence_1ghz = 10.0
        # ms
        self.sigma = sigma
        self.toa_highest_freq = 100.0
        self.taus_1ghz = taus_1ghz
        self.dc = 0.0

        # noise
        self.sigma_noise = 0.1

    def generate_data(self, instrument, osfact=32):
        """
        Generate data of a scattered Gaussian pulse in a dynamic
        spectrum.

        This works by oversampling the pulse's frequency - time
        behaviour and matching it to the instrumental parameters
        (time samples and frequency channels). We map a high-resolution
        description of the dispersed and scattered pulse data onto the
        2D array recorded by the instrument at lower resolution.

        Coherent dedispersion and the lack thereof are reflected in the
        mapping or transfer function between the high and low-resolution
        data.

        Oversample only in the frequency domain, as the pulse width and
        intrachannel dispersive smearing is usually much greater than the
        sampling time.

        Parameters
        ----------
        instrument: Instrument
            An Instrument instance.
        osfact: int
            The oversampling factor to use for the high-resolution data.
        """

        self.instrument = instrument

        # oversample the pulse in the frequency domain
        freqs = np.copy(instrument.freqs)
        times = np.copy(instrument.times)
        foff = np.copy(instrument.foff)
        data = np.zeros(shape=(len(freqs), len(times)), dtype=np.float64)
        temp = np.zeros(shape=(osfact, len(times)), dtype=np.float64)

        rng = np.random.default_rng(seed=42)

        for i, ifreq in enumerate(freqs):
            top_freq = ifreq - 0.5 * foff
            bot_freq = ifreq + 0.5 * foff
            sub_foff = foff / float(osfact)
            subfreqs = np.linspace(
                top_freq + 0.5 * sub_foff, bot_freq - 0.5 * sub_foff, num=osfact
            )

            for j, jfreq in enumerate(subfreqs):
                dm_shift = (
                    1.0e3
                    * KDM
                    * (jfreq**self.dm_index - freqs[0] ** self.dm_index)
                    * self.dm
                )
                fluence = self.fluence_1ghz * (jfreq / 1000.0) ** self.spectral_index
                center = self.toa_highest_freq + dm_shift
                taus = self.taus_1ghz * (jfreq / 1000.0) ** self.scatindex
                print(
                    f"Cfreq, subfreq, fluence, center, taus: {ifreq:.2f} MHz, {jfreq:.2f} MHz, {fluence:.2f} a.u., {center:.2f} ms, {taus:.2f} ms"
                )
                temp[j, :] = pulsemodels.scattered_profile(
                    times, fluence, center, self.sigma, taus, self.dc
                )

            # this is for incoherent dedispersion only. for coherent dedispersion, we need
            # to straighten the signal in each frequency channel before averaging
            # XXX: for coherent dedispersion, create the data at zero dm, i.e.
            # centre = 0 and the channel centre frequencies (do not oversample).
            # then shift/roll the data by the appropriate amount to disperse the pulse.
            mean_profile = np.mean(temp, axis=0)

            # add some gaussian radiometer noise
            noise = rng.normal(loc=0.0, scale=self.sigma_noise, size=len(times))
            data[i, :] = mean_profile + noise

        assert data.shape[0] == len(instrument.freqs)
        assert data.shape[1] == len(instrument.times)

        self.data = data
        self.times = times
        self.freqs = freqs

    def plot_data(self, data):
        """
        Plot the data.

        Parameters
        ----------
        data: ~np.array
            The input data.
        """

        # waterfall plot
        fig = plt.figure()
        ax = fig.add_subplot()

        time_step = np.diff(self.times)[0]
        freq_step = np.diff(self.freqs)[0]

        ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            extent=(
                self.times[0] - 0.5 * time_step,
                self.times[-1] + 0.5 * time_step,
                self.freqs[-1] + 0.5 * freq_step,
                self.freqs[0] - 0.5 * freq_step,
            ),
        )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")

        fig.tight_layout()

        # line plot
        fig = plt.figure()
        ax = fig.add_subplot()

        freqs = np.copy(self.freqs)
        times = np.copy(self.times)

        for i in range(len(freqs)):
            ax.plot(times, (len(freqs) - i) + data[i, :], color="black", lw=0.5)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Offset")

        fig.tight_layout()

    def convert_to_integer(self, data, nbit=8):
        """
        Convert the float data to unsigned integers for storing in SIGPROC
        filterbank files or PSRFITS ones.

        Note that SIGPROC does not implement scales and offset storage.

        Parameters
        ----------
        data: ~np.array of float
            The input data.
        nbit: int
            The number of bits in the output data.

        Returns
        -------
        data_int: ~np.array of int
            The output data in integer form.

        Raises
        ------
        NotImplementedError
            If the requested bit depth is not supported.
        """

        mapping = {8: np.uint8, 16: np.uint16, 32: np.uint32}

        if nbit not in mapping:
            raise NotImplementedError(f"Requested bit depth not implemented: {nbit}")

        # normalise the data before integer conversion
        # we follow the psrfits convention here
        # https://www.atnf.csiro.au/research/pulsar/psrfits_definition/PsrfitsDocumentation.html
        # real_value = (data_value - zero_offset) * data_scale + data_offset
        # => data_value = (real_value - data_offset) / data_scale + zero_offset
        zero_offset = 2 ** (nbit - 1) - 0.5
        data_offset = np.mean(data)
        # add some extra dynamic range
        data_scale = (np.max(data) - data_offset) / float(zero_offset - 0.5)

        print(
            f"Zero offset, data offset, data scale: {zero_offset}, {data_offset}, {data_scale}"
        )
        scaled_data = (data - data_offset) / data_scale + zero_offset

        print(
            "Scaled data: {0}, {1}, {2}".format(
                np.min(scaled_data), np.mean(scaled_data), np.max(scaled_data)
            )
        )

        # convert to unsigned integer
        int_type = mapping[nbit]
        min_int = 0
        max_int = 2**nbit - 1
        data_int = np.clip(np.rint(scaled_data), min_int, max_int).astype(int_type)

        print(
            "Integer data: {0}, {1}, {2}".format(
                np.min(data_int), np.mean(data_int), np.max(data_int)
            )
        )

        return data_int

    def write_to_sigproc_file(self, filename):
        # convert to integers
        nbit = self.instrument.nbit
        data_int = self.convert_to_integer(self.data, nbit=nbit)
        print(data_int.shape)
        self.plot_data(data_int)

        sigproc_obj = make_sigproc_object(
            rawdatafile=filename,
            source_name="FAKE",
            # this is the *centre* frequency of the first filterbank channel
            fch1=self.instrument.fch1,  # MHz
            foff=self.instrument.foff,  # MHz
            nchans=self.instrument.nchan,
            tsamp=self.instrument.tsamp * 1e-3,  # seconds
            tstart=60000.0,  # MJD
            src_raj=112233.44,  # HHMMSS.SS
            src_dej=112233.44,  # DDMMSS.SS
            machine_id=0,
            nbeams=0,
            ibeam=0,
            nbits=nbit,
            nifs=1,
            barycentric=0,
            pulsarcentric=0,
            telescope_id=6,
            data_type=0,
            az_start=-1,
            za_start=-1,
        )

        sigproc_obj.write_header(filename)
        sigproc_obj.append_spectra(data_int.T, filename)

        your_obj = Your(filename)
        print(your_obj.your_header)


class Instrument(object):
    def __init__(self):
        """
        An observing instrument.
        """

        # ms
        self.tsamp = 0.30624
        self.time_range = 7000.0
        # mhz
        self.fch1 = 1711.58203125  # centre frequency of first channel
        self.bandwidth = -856.0
        self.nchan = 1024
        self.nbit = 8
        # coherent dedispersion
        self.coherent = False

    @property
    def freqs(self):
        """
        The channel centre frequencies in MHz.
        """

        freqs = self.fch1 + np.arange(self.nchan) * self.foff

        return freqs

    @property
    def foff(self):
        """
        The channel offset in MHz.
        """

        foff = self.bandwidth / float(self.nchan)

        return foff

    @property
    def times(self):
        """
        The bin times in ms.
        """

        nsamp = np.ceil(self.time_range / self.tsamp) + 1
        times = -0.5 * self.time_range + np.arange(nsamp) * self.tsamp

        return times

    @property
    def nsamp(self):
        """
        The number of time samples.
        """

        return len(self.times)


class MeerKAT_Lband(Instrument):
    def __init__(self):
        """
        MeerKAT L-band 1024 channel data.
        """

        super().__init__()

        # ms
        self.tsamp = 0.30624
        self.time_range = 7000.0
        # mhz
        self.fch1 = 1711.58203125  # centre frequency of first channel
        self.bandwidth = -856.0
        self.nchan = 1024
        self.nbit = 8
        # coherent dedispersion
        self.coherent = False


class NenuFAR(Instrument):
    def __init__(self):
        """
        NenuFAR central band 192 channel data.
        """

        super().__init__()

        # ms
        self.tsamp = 0.65536
        self.time_range = 400000.0
        # mhz
        self.fch1 = 73.73046875  # centre frequency of first channel
        self.bandwidth = -37.5
        self.nchan = 192
        self.nbit = 8
        # coherent dedispersion
        self.coherent = True
