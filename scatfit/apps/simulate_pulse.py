#
#   Simulate scattered pulses.
#   2022 - 2023 Fabian Jankowski
#

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate
from your import Your
from your.formats.filwriter import make_sigproc_object

from scatfit.dm import KDM
import scatfit.pulsemodels as pulsemodels


def parse_args():
    """
    Parse the commandline arguments.

    Returns
    -------
    args: populated namespace
        The commandline arguments.
    """

    parser = argparse.ArgumentParser(
        description="Simulate scattered pulses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    args = parser.parse_args()

    return args


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

    def generate_data(self, instrument, osfact=4):
        """
        Generate data of a scattered Gaussian pulse in a dynamic
        spectrum.

        This works by oversampling the pulse's frequency - time
        behaviour and matching it to the instrumental parameters
        (time samples and frequency channels). We map a high-resolution
        description of the dispersed and scattered pulse data onto the
        2D array recorded by the instrument at lower resolution.

        Coherent dedispersion and the lack thereof are reflected in the
        mapping or transfer function between the high and low resolution
        data.

        Parameters
        ----------
        instrument: Instrument
            And Instrument instance.
        osfact: int
            The oversampling factor to use for the high-resolution data.
        """

        self.instrument = instrument

        # oversample the pulse
        original = {"nchan": instrument.nchan}
        instrument.nchan *= osfact
        freqs = np.copy(instrument.freqs)
        instrument.nchan = original["nchan"]

        times = np.linspace(
            instrument.times[0],
            instrument.times[-1],
            num=osfact * len(instrument.times),
            endpoint=False,
        )
        data_high = np.zeros(shape=(len(freqs), len(times)), dtype=np.float32)

        rng = np.random.default_rng(seed=42)

        for i, ifreq in enumerate(freqs):
            dm_shift = (
                1.0e3
                * KDM
                * (ifreq**self.dm_index - freqs[0] ** self.dm_index)
                * self.dm
            )
            fluence = self.fluence_1ghz * (ifreq / 1000.0) ** self.spectral_index
            center = self.toa_highest_freq + dm_shift
            taus = self.taus_1ghz * (ifreq / 1000.0) ** self.scatindex
            print(
                "Cfreq, fluence, center, taus: {0:.2f} MHz, {1:.2f} a.u., {2:.2f} ms, {3:.2f} ms".format(
                    ifreq, fluence, center, taus
                )
            )
            profile = pulsemodels.scattered_profile(
                times, fluence, center, self.sigma, taus, self.dc
            )

            # add some gaussian radiometer noise
            noise = rng.normal(loc=0.0, scale=self.sigma_noise, size=len(times))
            data_high[i, :] = profile + noise

        # match high-resolution data to instrument
        freqs = np.copy(instrument.freqs)
        times = np.copy(instrument.times)
        data_low = np.zeros(shape=(len(freqs), len(times)), dtype=np.float32)

        assert data_high.shape[0] % osfact == 0
        assert data_high.shape[1] % osfact == 0

        # this is incoherent dedispersion only
        # we need to straigthen the signal in each frequency channel for
        # coherent dedispersion
        data_low = decimate(data_high, osfact, ftype="fir", axis=1)
        data_low = decimate(data_low, osfact, ftype="fir", axis=0)

        # free memory
        del data_high

        # plot
        fig = plt.figure()
        ax = fig.add_subplot()

        for i, ifreq in enumerate(freqs):
            ax.plot(times, (len(freqs) - i) + data_low[i, :], color="black", lw=0.5)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Offset")

        fig.tight_layout()

        self.data = data_low
        self.times = times
        self.freqs = freqs

    def plot_data(self, data):
        """
        Parameters
        ----------
        data: ~np.array
            The input data.
        times: ~np.array of float
            The times of the bins.
        freqs: ~np.array of float
            The values of the high-frequency channel edges.
        """

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

    def convert_to_integer(self, data, nbit=8):
        """
        Convert the float data to integers for storing in SIGPROC filterbank
        files or PSRFITS ones.

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
        """

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
            "Zero offset, data offset, data scale: {0}, {1}, {2}".format(
                zero_offset, data_offset, data_scale
            )
        )
        scaled_data = (data - data_offset) / data_scale + zero_offset

        print(
            "Scaled data: {0}, {1}, {2}".format(
                np.min(scaled_data), np.mean(scaled_data), np.max(scaled_data)
            )
        )

        # convert to uint8
        data_int = np.rint(scaled_data).astype(np.uint8)

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


#
# MAIN
#


def main():
    parse_args()

    pulse = Pulse(dm=500.0, sigma=2.5, taus_1ghz=20.0)
    instrument = MeerKAT_Lband()

    pulse.generate_data(instrument, osfact=10)
    pulse.plot_data(pulse.data)

    pulse.write_to_sigproc_file("test_fake_meerkat.fil")

    pulse = Pulse(dm=70.0, sigma=2.5, taus_1ghz=0.01)
    instrument = NenuFAR()

    pulse.generate_data(instrument, osfact=4)
    pulse.plot_data(pulse.data)

    pulse.write_to_sigproc_file("test_fake_nenufar.fil")

    plt.show()


if __name__ == "__main__":
    main()
