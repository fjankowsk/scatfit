#
#   Simulate scattered pulses.
#   2022 Fabian Jankowski
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

    def generate_data(self, instrument):
        self.instrument = instrument
        times = instrument.times
        freqs = instrument.freqs

        data = np.zeros(shape=(len(freqs), len(times)))

        fig = plt.figure()
        ax = fig.add_subplot()

        rng = np.random.default_rng(seed=42)

        for i, ifreq in enumerate(freqs):
            cfreq = ifreq + 0.5 * instrument.foff
            dm_shift = (
                1.0e3
                * KDM
                * (ifreq**self.dm_index - freqs[0] ** self.dm_index)
                * self.dm
            )
            fluence = self.fluence_1ghz * (cfreq / 1000.0) ** self.spectral_index
            center = self.toa_highest_freq + dm_shift
            taus = self.taus_1ghz * (cfreq / 1000.0) ** self.scatindex
            print(
                "Cfreq, fluence, center, taus: {0:.2f} MHz, {1:.2f} a.u., {2:.2f} ms, {3:.2f} ms".format(
                    cfreq, fluence, center, taus
                )
            )
            profile = pulsemodels.scattered_profile(
                times, fluence, center, self.sigma, taus, self.dc
            )

            # add some gaussian radiometer noise
            noise = rng.normal(loc=0.0, scale=self.sigma_noise, size=len(times))

            data[i, :] = profile + noise

            ax.plot(times, (len(freqs) - i) + profile, color="black", lw=0.5)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Offset")

        fig.tight_layout()

        self.data = data
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
                self.freqs[-1] + freq_step,
                self.freqs[0],
            ),
        )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")

        fig.tight_layout()

    def convert_to_integer(self, data, nbit=8):
        """
        Convert the float data to integers for storing in SIGPROC filterbank
        files or PSRFITS ones.

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
            source_name="fake",
            nchans=self.instrument.nchan,
            foff=self.instrument.foff,  # MHz
            fch1=self.instrument.fch1,  # MHz
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
        self.fch1 = 1711.58203125
        self.bandwidth = -856.0
        self.nchan = 1024
        self.nbit = 8

    @property
    def freqs(self):
        """
        The values of the high-frequency channel edges in MHz.
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


#
# MAIN
#


def main():
    pulse = Pulse(dm=500.0, sigma=2.5, taus_1ghz=20.0)
    instrument = Instrument()

    pulse.generate_data(instrument)
    pulse.plot_data(pulse.data)

    pulse.write_to_sigproc_file("test_fake.fil")

    plt.show()


if __name__ == "__main__":
    main()
