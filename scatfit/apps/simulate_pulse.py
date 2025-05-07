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

    pulse = Pulse(dm=500.0, sigma=2.5, taus_1ghz=20.0)
    instrument = MeerKAT_Lband()

    pulse.generate_data(instrument, osfact=64)
    pulse.plot_data(pulse.data)

    pulse.write_to_sigproc_file("test_fake_meerkat.fil")

    pulse = Pulse(dm=70.0, sigma=2.5, taus_1ghz=0.01)
    instrument = NenuFAR()

    pulse.generate_data(instrument, osfact=32)
    pulse.plot_data(pulse.data)

    pulse.write_to_sigproc_file("test_fake_nenufar.fil")

    plt.show()


if __name__ == "__main__":
    main()
