import numpy as np

import matplotlib.pyplot as plt

import scatfit.pulsemodels as pulsemodels


def plot_comparison(plot_range, curve1, curve2):
    """
    Make a profile comparison plot.
    """

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex="col",
        gridspec_kw={"height_ratios": [1, 0.3], "hspace": 0},
    )

    ax1.step(plot_range, curve1, where="mid", zorder=4)

    ax1.step(plot_range, curve2, where="mid", zorder=5)

    ax1.grid()
    ax1.set_ylabel("Flux (a.u.)")

    # hide bottom ticks
    ax1.tick_params(bottom=False)

    # residuals
    residual = curve1 - curve2

    ax2.step(plot_range, residual, where="mid", zorder=4)

    ax2.grid()
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Residual")

    ax2.set_xlim(-5.0, 8.0)

    fig.tight_layout()

    return fig


def test_compare_models():
    """
    Compare the models visually.
    """

    plot_range = np.linspace(-200.0, 200.0, num=20000)

    fluence = 10.0
    center = 0.0
    dc = 0.0

    full_model = pulsemodels.scattered_profile
    analytical_model = pulsemodels.scattered_gaussian_pulse

    # enable interactive mode
    plt.ion()

    for sigma in np.geomspace(0.1, 50.0, num=10):
        for taus in np.geomspace(0.1, 50.0, num=10):
            print("Sigma, tau: {0:.2f}, {1:.2f}".format(sigma, taus))

            curve_full = full_model(plot_range, fluence, center, sigma, taus, dc)
            curve_analytical = analytical_model(
                plot_range, fluence, center, sigma, taus, dc
            )

            fig = plot_comparison(plot_range, curve_full, curve_analytical)
            plt.draw()

            input("Press any key.")
            plt.close(fig)


if __name__ == "__main__":
    test_compare_models()
