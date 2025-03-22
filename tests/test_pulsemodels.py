import numpy as np
import pytest

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

    for fluence in np.geomspace(0.1, 1000.0, num=5):
        for center in np.linspace(-100.0, 100.0, num=5):
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
                    assert rel_error < 0.01

                    # ensure that curves differ little
                    rel_residual = np.abs(curve_analytical - curve_full) / np.max(
                        curve_full
                    )
                    assert np.median(rel_residual) < 1e-8
                    assert np.mean(rel_residual) < 0.01
                    assert np.all(rel_residual < 0.07)


def test_invalid_parameters():
    """
    Check the handling of invalid input parameters.
    """

    plot_range = np.linspace(-1000.0, 1000.0, num=8000)

    fluence_valid = 10.0
    center = 0.0
    sigma_valid = 2.5
    taus_valid = 5.0
    dc = 0.0
    f_lo_valid = 55.0
    f_hi_valid = 65.0
    nfreq_valid = 3

    # normed gaussian
    model = pulsemodels.gaussian_normed

    for fluence in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(plot_range, fluence, center, sigma_valid)

    for sigma in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(plot_range, fluence_valid, center, sigma)

    # scattered gaussian pulse
    model = pulsemodels.scattered_gaussian_pulse

    for fluence in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(plot_range, fluence, center, sigma_valid, taus_valid, dc)

    for sigma in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(plot_range, fluence_valid, center, sigma, taus_valid, dc)

    for taus in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(plot_range, fluence_valid, center, sigma_valid, taus, dc)

    # bandintegrated model
    model = pulsemodels.bandintegrated_model

    for fluence in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(
                plot_range,
                fluence,
                center,
                sigma_valid,
                taus_valid,
                dc,
                f_lo_valid,
                f_hi_valid,
                nfreq_valid,
            )

    for sigma in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(
                plot_range,
                fluence_valid,
                center,
                sigma,
                taus_valid,
                dc,
                f_lo_valid,
                f_hi_valid,
                nfreq_valid,
            )

    for taus in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(
                plot_range,
                fluence_valid,
                center,
                sigma_valid,
                taus,
                dc,
                f_lo_valid,
                f_hi_valid,
                nfreq_valid,
            )

    for f_lo in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(
                plot_range,
                fluence_valid,
                center,
                sigma_valid,
                taus_valid,
                dc,
                f_lo,
                f_hi_valid,
                nfreq_valid,
            )

    for f_hi in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(
                plot_range,
                fluence_valid,
                center,
                sigma_valid,
                taus_valid,
                dc,
                f_lo_valid,
                f_hi,
                nfreq_valid,
            )

    for nfreq in [1, 0, -1e-5, -1, -2, -10]:
        with pytest.raises(AssertionError):
            model(
                plot_range,
                fluence_valid,
                center,
                sigma_valid,
                taus_valid,
                dc,
                f_lo_valid,
                f_hi_valid,
                nfreq,
            )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
