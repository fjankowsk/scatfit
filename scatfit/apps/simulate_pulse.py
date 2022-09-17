import matplotlib.pyplot as plt
import numpy as np

from scatfit.dm import KDM
import scatfit.pulsemodels as pulsemodels


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
    times = np.linspace(-4000.0, 4000.0, num=4 * 1024)
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
            cfreqs[-1] + 0.5 * freq_step,
            cfreqs[0] - 0.5 * freq_step,
        ),
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (MHz)")

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
