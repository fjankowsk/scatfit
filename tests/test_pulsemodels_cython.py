import numpy as np
import pytest

import scatfit.pulsemodels_python as pm_python
import scatfit.pulsemodels_cython as pm_cython


def test_agreement_python_and_cython_models_gaussian():
    """
    Check that the Python and Cython implementations match.
    Normed Gaussian.
    """

    plot_range = np.linspace(-1000.0, 1000.0, num=8000)

    # gaussian
    python_model = pm_python.gaussian_normed
    cython_model = pm_cython.gaussian_normed

    for fluence in np.geomspace(0.1, 1000.0, num=10):
        for center in np.linspace(-100.0, 100.0, num=10):
            for sigma in np.geomspace(2.0, 50.0, num=10):
                curve_python = python_model(plot_range, fluence, center, sigma)
                curve_cython = cython_model(plot_range, fluence, center, sigma)

                # ensure that curves differ little
                assert np.allclose(curve_python, curve_cython)


def test_agreement_python_and_cython_models_scattered_pulse():
    """
    Check that the Python and Cython implementations match.
    Scattered Gaussian pulse.
    """

    plot_range = np.linspace(-1000.0, 1000.0, num=8000)

    # scattered gaussian pulse
    python_model = pm_python.scattered_gaussian_pulse
    cython_model = pm_cython.scattered_gaussian_pulse

    for fluence in np.geomspace(0.1, 1000.0, num=5):
        for center in np.linspace(-100.0, 100.0, num=5):
            for sigma in np.geomspace(2.0, 50.0, num=5):
                for taus in np.geomspace(1.0, 50.0, num=5):
                    for dc in np.linspace(-0.5, 0.5, num=3):
                        curve_python = python_model(
                            plot_range, fluence, center, sigma, taus, dc
                        )
                        curve_cython = cython_model(
                            plot_range, fluence, center, sigma, taus, dc
                        )

                        # ensure that curves differ little
                        assert np.allclose(curve_python, curve_cython)


def test_agreement_python_and_cython_models_bandintegrated():
    """
    Check that the Python and Cython implementations match.
    Band-integrated model.
    """

    plot_range = np.linspace(-1000.0, 1000.0, num=8000)

    # bandintegrated model
    python_model = pm_python.bandintegrated_model
    cython_model = pm_cython.bandintegrated_model

    f_lo = 856.0
    f_hi = 1712.0
    nfreq = 9

    for fluence in np.geomspace(0.1, 1000.0, num=5):
        for center in np.linspace(-100.0, 100.0, num=5):
            for sigma in np.geomspace(2.0, 50.0, num=5):
                for taus in np.geomspace(1.0, 50.0, num=5):
                    for dc in np.linspace(-0.5, 0.5, num=3):
                        curve_python = python_model(
                            plot_range,
                            fluence,
                            center,
                            sigma,
                            taus,
                            dc,
                            f_lo,
                            f_hi,
                            nfreq,
                        )
                        curve_cython = cython_model(
                            plot_range,
                            fluence,
                            center,
                            sigma,
                            taus,
                            dc,
                            f_lo,
                            f_hi,
                            nfreq,
                        )

                        # ensure that curves differ little
                        assert np.allclose(curve_python, curve_cython)


def test_c_contiguous_input_output():
    """
    Check that C-contiguous numpy arrays work as input
    and that the functions return C-contiguous arrays.
    """

    plot_range = np.linspace(-1000.0, 1000.0, num=8000)
    assert plot_range.flags.c_contiguous

    fluence = 10.0
    center = 0.0
    sigma = 1.0
    taus = 5.0
    dc = 0.0
    f_lo = 856.0
    f_hi = 1712.0
    nfreq = 9

    # normed gaussian
    res = pm_cython.gaussian_normed(plot_range, fluence, center, sigma)
    assert res.flags.c_contiguous

    # scattered gaussian pulse
    res = pm_cython.scattered_gaussian_pulse(
        plot_range, fluence, center, sigma, taus, dc
    )
    assert res.flags.c_contiguous

    # bandintegrated model
    res = pm_cython.bandintegrated_model(
        plot_range, fluence, center, sigma, taus, dc, f_lo, f_hi, nfreq
    )
    assert res.flags.c_contiguous


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
    model = pm_cython.gaussian_normed

    for fluence in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(plot_range, fluence, center, sigma_valid)

    for sigma in [0.0, -1e-5, -1.0, -10.0]:
        with pytest.raises(AssertionError):
            model(plot_range, fluence_valid, center, sigma)

    # scattered gaussian pulse
    model = pm_cython.scattered_gaussian_pulse

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
    model = pm_cython.bandintegrated_model

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
