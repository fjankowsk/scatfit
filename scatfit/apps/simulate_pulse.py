#
#   Simulate scattered pulses.
#   2022 - 2025 Fabian Jankowski
#

import argparse
import matplotlib.pyplot as plt

from scatfit.simulate import Pulse, MeerKAT_Lband, NenuFAR


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


#
# MAIN
#


def main():
    parse_args()

    # single component pulse
    # meerkat l-band
    instrument = MeerKAT_Lband()
    pulse = Pulse(dm=500.0, fluence=10.0, center=100.0, sigma=2.5, taus=20.0)

    pulse.generate_data(instrument, osfact=64)
    pulse.plot_data(pulse.data)

    pulse.write_to_sigproc_file("test_fake_meerkat.fil")

    # nenufar
    instrument = NenuFAR()
    # place the pulse start at 1/4 into the file
    offset = -0.25 * instrument.time_range
    pulse = Pulse(dm=10.0, fluence=10.0, center=offset + 100.0, sigma=2.5, taus=0.01)

    pulse.generate_data(instrument, osfact=32)
    pulse.plot_data(pulse.data)

    pulse.write_to_sigproc_file("test_fake_nenufar.fil")

    # two component pulse
    # meerkat l-band
    instrument = MeerKAT_Lband()
    pulse = Pulse(dm=500.0, fluence=10.0, center=100.0, sigma=2.5, taus=20.0)

    # add secondary pulse component
    pulse.add_component(fluence=3.0, center=120.0, sigma=3.5)

    pulse.generate_data(instrument, osfact=64)
    pulse.plot_data(pulse.data)

    pulse.write_to_sigproc_file("test_fake_meerkat_two_component.fil")

    # three component pulse
    # nenufar
    instrument = NenuFAR()
    # place the pulse start at 1/4 into the file
    offset = -0.25 * instrument.time_range
    pulse = Pulse(
        dm=10.0,
        fluence=10.0,
        center=offset + 0.0,
        sigma=4.0,
        taus=1.5,
        scatindex=-0.25,
        spectral_index=-0.5,
    )

    # add second pulse component
    pulse.add_component(
        fluence=2.0, center=offset + 45.0, sigma=4.0, spectral_index=-0.3
    )

    # add third pulse component
    pulse.add_component(
        fluence=6.0, center=offset + 70.0, sigma=6.0, spectral_index=-0.7
    )

    pulse.generate_data(instrument, osfact=64)
    pulse.plot_data(pulse.data)

    pulse.write_to_sigproc_file("test_fake_nenufar_three_component.fil")

    plt.show()


if __name__ == "__main__":
    main()
