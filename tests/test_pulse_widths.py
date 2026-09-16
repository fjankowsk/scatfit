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


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
