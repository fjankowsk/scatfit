import matplotlib.pyplot as plt
import numpy as np
from your import Your
from your.formats.filwriter import make_sigproc_object

from scatfit.dm import KDM
import scatfit.pulsemodels as pulsemodels


def plot_data(data, times, cfreqs):
    fig = plt.figure()
    ax = fig.add_subplot()

    time_step = np.diff(times)[0]
    freq_step = np.diff(cfreqs)[0]

    ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        extent=(
            times[0] - 0.5 * time_step,
            times[-1] + 0.5 * time_step,
            cfreqs[-1] - 0.5 * freq_step,
            cfreqs[0] + 0.5 * freq_step,
        ),
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (MHz)")

    fig.tight_layout()


def convert_to_integer(data, nbit=8):
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
    # we follow the psrfits method here
    # data_value =  zero_offset + (real_value - data_offset) / data_scale
    zero_offset = 2 ** (nbit - 1) - 0.5
    data_offset = np.mean(data)
    scale = np.max(data)

    print(zero_offset, data_offset, scale)
    scaled_data = zero_offset + zero_offset * (data - data_offset) / scale

    print(np.min(scaled_data), np.mean(scaled_data), np.max(scaled_data))

    # convert to uint8
    data_int = np.rint(scaled_data).astype(np.uint8)

    print(np.min(data_int), np.mean(data_int), np.max(data_int))

    return data_int


#
# MAIN
#


def main():
    dm = 500.0
    dm_index = -2.0
    scatindex = -4.0
    spectral_index = -1.5
    fluence_1ghz = 10.0
    # ms
    toa_highest_freq = 100.0
    taus_1ghz = 20.0
    sigma = 2.5
    dc = 0.0
    # noise
    sigma_noise = 0.1

    # ms
    times = np.linspace(-4000.0, 4000.0, num=10 * 1024)
    # mhz
    cfreqs = np.linspace(1600.0, 856.0, num=1024)

    data = np.zeros(shape=(len(cfreqs), len(times)))

    fig = plt.figure()
    ax = fig.add_subplot()

    rng = np.random.default_rng(seed=42)

    for i, ifreq in enumerate(cfreqs):
        dm_shift = 1.0e3 * KDM * (ifreq**dm_index - cfreqs[0] ** dm_index) * dm
        fluence = fluence_1ghz * (ifreq / 1000.0) ** spectral_index
        center = toa_highest_freq + dm_shift
        taus = taus_1ghz * (ifreq / 1000.0) ** scatindex
        print(
            "Cfreq, fluence, center, taus: {0:.2f} MHz, {1:.2f} a.u., {2:.2f} ms, {3:.2f} ms".format(
                ifreq, fluence, center, taus
            )
        )
        profile = pulsemodels.scattered_profile(times, fluence, center, sigma, taus, dc)

        # add some gaussian radiometer noise
        noise = rng.normal(loc=0.0, scale=sigma_noise, size=len(times))

        data[i, :] = profile + noise

        ax.plot(times, (len(cfreqs) - i) + profile, color="black", lw=0.5)

    plot_data(data, times, cfreqs)

    data = convert_to_integer(data)
    print(data.shape)

    plot_data(data, times, cfreqs)

    time_step = np.diff(times)[0]
    freq_step = np.diff(cfreqs)[0]
    output_filename = "test_fake.fil"

    sigproc_obj = make_sigproc_object(
        rawdatafile=output_filename,
        source_name="test",
        nchans=len(cfreqs),
        foff=freq_step,  # MHz
        fch1=cfreqs[0] - 0.5 * freq_step,  # MHz
        tsamp=time_step * 1e-3,  # seconds
        tstart=60000.0,  # MJD
        src_raj=112233.44,  # HHMMSS.SS
        src_dej=112233.44,  # DDMMSS.SS
        machine_id=0,
        nbeams=0,
        ibeam=0,
        nbits=8,
        nifs=1,
        barycentric=0,
        pulsarcentric=0,
        telescope_id=6,
        data_type=0,
        az_start=-1,
        za_start=-1,
    )

    sigproc_obj.write_header(output_filename)
    sigproc_obj.append_spectra(data.T, output_filename)

    your_obj = Your(output_filename)
    print(your_obj.your_header)

    plt.show()


if __name__ == "__main__":
    main()
