import numpy as np

import scatfit.pulsemodels as pulsemodels


def test_normalisation_gaussian():
    """
    Check that the Gaussian is correctly normalised, i.e.
    has the correct fluence.
    """

    plot_range = np.linspace(-200.0, 200.0, num=2000)

    center = 10.0
    sigma = 2.5

    for fluence in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        res = np.trapz(
            pulsemodels.gaussian_normed(plot_range, fluence, center, sigma),
            x=plot_range,
        )

        assert np.allclose(res, fluence)


def test_conservation_of_fluence():
    """
    Check that the energy, i.e. fluence, is conserved during
    scattering.
    """

    plot_range = np.linspace(-1000.0, 1000.0, num=8000)

    center = 10.0
    sigma = 2.5
    dc = 0.0

    models = [pulsemodels.scattered_profile, pulsemodels.scattered_gaussian_pulse]

    for model in models:
        for fluence in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
            for taus in [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
                res = np.trapz(
                    model(plot_range, fluence, center, sigma, taus, dc),
                    x=plot_range,
                )

                assert np.allclose(res, fluence)


if __name__ == "__main__":
    import nose2

    nose2.main()
