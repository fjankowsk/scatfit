import numpy as np

import scatfit.pulsemodels as pulsemodels


def test_normalisation_gaussian():
    """
    Check that the Gaussian is correctly normalised, i.e.
    has the correct fluence.
    """

    plot_range = np.linspace(-400.0, 400.0, num=4000)

    for fluence in np.geomspace(0.1, 1000.0, num=10):
        for center in np.linspace(-100.0, 100.0, num=10):
            for sigma in np.geomspace(2.0, 50.0, num=10):
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

    sigma = 2.5
    dc = 0.0

    models = [pulsemodels.scattered_profile, pulsemodels.scattered_gaussian_pulse]

    for model in models:
        for fluence in np.geomspace(0.1, 1000.0, num=10):
            for center in np.linspace(-100.0, 100.0, num=10):
                for taus in np.geomspace(3.0, 50.0, num=10):
                    res = np.trapz(
                        model(plot_range, fluence, center, sigma, taus, dc),
                        x=plot_range,
                    )

                    assert np.allclose(res, fluence)


def test_agreement_analytical_and_full_convolution_model():
    """
    Check that the analytical model agrees with the full numerical
    convolution model.
    """

    plot_range = np.linspace(-1000.0, 1000.0, num=8000)

    dc = 0.0

    full_model = pulsemodels.scattered_profile
    analytical_model = pulsemodels.scattered_gaussian_pulse

    for fluence in np.geomspace(0.1, 1000.0, num=10):
        for center in np.linspace(-100.0, 100.0, num=10):
            for sigma in np.geomspace(2.0, 50.0, num=10):
                for taus in np.geomspace(1.0, 50.0, num=10):
                    curve_full = full_model(
                        plot_range, fluence, center, sigma, taus, dc
                    )
                    curve_analytical = analytical_model(
                        plot_range, fluence, center, sigma, taus, dc
                    )

                    # ensure that the energy, i.e. fluence, is the same
                    fluence_full = np.trapz(curve_full, x=plot_range)
                    fluence_analytical = np.trapz(curve_analytical, x=plot_range)

                    rel_error = np.abs(fluence_analytical - fluence_full) / fluence_full
                    # print(fluence_full, fluence_analytical, rel_error)
                    assert rel_error < 0.01

                    # ensure that curves differ little
                    residual = curve_analytical - curve_full
                    # print(np.max(residual))
                    assert np.allclose(curve_analytical, curve_full)


if __name__ == "__main__":
    import nose2

    nose2.main()
