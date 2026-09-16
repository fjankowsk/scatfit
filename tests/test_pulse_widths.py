import numpy as np
import pytest

import scatfit.pulsemodels as pulsemodels


def test_gaussian_fwhm_and_fwtm():
    """
    Check that the FWHM & FWTM are computed correctly.
    """

    plot_range = np.linspace(-500.0, 500.0, num=100000)

    bin_width = abs(plot_range[1] - plot_range[0])
    print(f"Bin width: {bin_width}")

    model = pulsemodels.gaussian_normed

    for fluence in np.geomspace(0.1, 1000.0, num=10):
        for center in np.linspace(-50.0, 50.0, num=10):
            for sigma in np.geomspace(1.0, 50.0, num=10):
                analytic_fwhm = pulsemodels.gaussian_fwhm(sigma)
                analytic_fwtm = pulsemodels.gaussian_fwtm(sigma)

                amps = model(plot_range, fluence, center, sigma)
                numeric_fwhm = pulsemodels.full_width_post(plot_range, amps, 0.5)
                numeric_fwtm = pulsemodels.full_width_post(plot_range, amps, 0.1)

                print(numeric_fwhm, analytic_fwhm)
                print(numeric_fwtm, analytic_fwtm)

                assert np.isclose(numeric_fwhm, analytic_fwhm)
                assert np.isclose(numeric_fwtm, analytic_fwtm)


def test_gaussian_boxcar_equivalent_width():
    """
    Check that the boxcar equivalent width Weq is computed correctly.
    """

    plot_range = np.linspace(-200.0, 500.0, num=100000)

    bin_width = abs(plot_range[1] - plot_range[0])
    print(f"Bin width: {bin_width}")

    model = pulsemodels.gaussian_normed

    for fluence in np.geomspace(0.1, 1000.0, num=10):
        for center in np.linspace(-50.0, 50.0, num=10):
            for sigma in np.geomspace(1.0, 50.0, num=10):
                analytic_weq = np.sqrt(2.0 * np.pi) * sigma

                amps = model(plot_range, fluence, center, sigma)
                numeric_weq = pulsemodels.equivalent_width(plot_range, amps)

                print(analytic_weq, numeric_weq)

                assert np.isclose(numeric_weq, analytic_weq, rtol=1e-2)


def test_gaussian_boxcar_width_independent_of_fluence():
    """
    The boxcar equivalent width of a Gaussian depends only on sigma,
    not on fluence or center. Verify this explicitly.
    """

    plot_range = np.linspace(-200.0, 500.0, num=100000)

    model = pulsemodels.gaussian_normed
    sigma = 10.0

    weq1 = pulsemodels.equivalent_width(
        plot_range, model(plot_range, 0.5, -20.0, sigma)
    )
    weq2 = pulsemodels.equivalent_width(
        plot_range, model(plot_range, 500.0, 30.0, sigma)
    )

    assert np.isclose(weq1, weq2)


def test_gaussian_weq_vs_w50_ratio():
    """
    The ratio Weq / W50 ~= 1.0645 for any Gaussian. Test this.
    """

    plot_range = np.linspace(-200.0, 500.0, num=200000)

    bin_width = abs(plot_range[1] - plot_range[0])
    print(f"Bin width: {bin_width}")

    model = pulsemodels.gaussian_normed

    analytic_ratio = np.sqrt(np.pi / (4.0 * np.log(2.0)))

    for fluence in np.geomspace(0.1, 1000.0, num=10):
        for center in np.linspace(-50.0, 50.0, num=10):
            for sigma in np.geomspace(0.1, 50.0, num=10):
                amps = model(plot_range, fluence, center, sigma)

                weq = pulsemodels.equivalent_width(plot_range, amps)
                w50 = pulsemodels.full_width_post(plot_range, amps, 0.5)
                ratio = weq / w50

                print(ratio, analytic_ratio)
                assert np.isclose(ratio, analytic_ratio, rtol=3e-3)


def test_wd4s_gaussian():
    """
    Check WD4s against the analytical 4 sigma for a Gaussian.
    We effectively disable the thresholding here.
    """

    plot_range = np.linspace(-200.0, 500.0, num=100000)

    bin_width = abs(plot_range[1] - plot_range[0])
    print(f"Bin width: {bin_width}")

    model = pulsemodels.gaussian_normed

    for fluence in np.geomspace(0.1, 1000.0, num=10):
        for center in np.linspace(-50.0, 50.0, num=10):
            for sigma in np.geomspace(0.1, 50.0, num=10):
                analytic_wd4s = 4.0 * sigma

                amps = model(plot_range, fluence, center, sigma)
                numeric_wd4s = pulsemodels.d4sigma_width(
                    plot_range, amps, sigma_noise=1e-10
                )

                assert np.isclose(numeric_wd4s, analytic_wd4s, rtol=7e-3)


def test_wd4s_independent_of_fluence():
    """
    The WD4s width depends only on sigma, not on fluence or center.
    """

    plot_range = np.linspace(-200.0, 500.0, num=100000)

    model = pulsemodels.gaussian_normed
    sigma = 10.0

    w1 = pulsemodels.d4sigma_width(
        plot_range,
        model(plot_range, 0.5, -20.0, sigma),
        sigma_noise=1e-10,
    )
    w2 = pulsemodels.d4sigma_width(
        plot_range,
        model(plot_range, 500.0, 30.0, sigma),
        sigma_noise=1e-10,
    )

    assert np.isclose(w1, w2)


def test_wd4s_vs_w50_ratio():
    """
    WD4s / FWHM ~= 1.6651 for an untruncated Gaussian. Check this.
    """

    plot_range = np.linspace(-200.0, 500.0, num=100000)

    model = pulsemodels.gaussian_normed
    fluence, center, sigma = 1.0, 0.0, 5.0

    amps = model(plot_range, fluence, center, sigma)
    wd4s = pulsemodels.d4sigma_width(plot_range, amps, sigma_noise=1e-10)
    w50 = pulsemodels.full_width_post(plot_range, amps, 0.5)

    ratio = wd4s / w50
    analytic_ratio = np.sqrt(2.0 / np.log(2.0))

    print(ratio, analytic_ratio)
    assert np.isclose(ratio, analytic_ratio, rtol=1e-3)


def test_wd4s_vs_weq_ratio():
    """
    WD4s / Weq ~= 1.5958 for an untruncated Gaussian. Check this.
    """

    plot_range = np.linspace(-200.0, 500.0, num=100000)

    model = pulsemodels.gaussian_normed
    fluence, center, sigma = 1.0, 0.0, 5.0

    amps = model(plot_range, fluence, center, sigma)
    wd4s = pulsemodels.d4sigma_width(plot_range, amps, sigma_noise=1e-10)
    weq = pulsemodels.equivalent_width(plot_range, amps)

    ratio = wd4s / weq
    analytic_ratio = np.sqrt(8.0 / np.pi)

    print(ratio, analytic_ratio)
    assert np.isclose(ratio, analytic_ratio, rtol=7e-3)


def test_wd4s_default_threshold_close_to_4sigma():
    """
    With the default 1 % threshold (no sigma_noise), the result
    should be within 2 % of 4 sigma. This verifies the thresholding code.
    """

    plot_range = np.linspace(-200.0, 500.0, num=100000)

    model = pulsemodels.gaussian_normed
    sigma = 10.0

    amps = model(plot_range, 1.0, 0.0, sigma)
    wd4s = pulsemodels.d4sigma_width(plot_range, amps)

    assert abs(wd4s - 4.0 * sigma) / (4.0 * sigma) < 0.02


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
