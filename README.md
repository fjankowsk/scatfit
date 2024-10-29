# Scatfit: Scattering fits of time domain radio signals (Fast Radio Bursts or pulsars) #

[![PyPI latest release](https://img.shields.io/pypi/v/scatfit.svg)](https://pypi.org/project/scatfit/)
[![Documentation](https://readthedocs.org/projects/scatfit/badge/?version=latest)](https://scatfit.readthedocs.io/en/latest/)
[![GitHub issues](https://img.shields.io/badge/issue_tracking-GitHub-blue.svg)](https://github.com/fjankowsk/scatfit/issues/)
[![License - MIT](https://img.shields.io/pypi/l/scatfit.svg)](https://github.com/fjankowsk/scatfit/blob/master/LICENSE)
[![Paper link](https://img.shields.io/badge/paper-10.1093/mnras/stad2041-blue.svg)](https://doi.org/10.1093/mnras/stad2041)
[![arXiv link](https://img.shields.io/badge/arXiv-2302.10107-blue.svg)](https://arxiv.org/abs/2302.10107)

This repository contains code to fit Fast Radio Burst or pulsar profiles to measure scattering and other parameters. The code is mainly developed for Python 3, but Python 2 from version 2.7 onwards should work fine.

## Author ##

The software is primarily developed and maintained by Fabian Jankowski. For more information feel free to contact me via: fabian.jankowski at cnrs-orleans.fr.

## Paper ##

The corresponding paper (Jankowski et al. 2023, MNRAS) is available via this [NASA ADS link](https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.4275J/abstract).

## Citation ##

If you make use of the software, please add a link to this repository and cite our corresponding paper. See above and the CITATION and CITATION.bib files.

The code is also listed in the [Astrophysics Source Code Library (ASCL)](https://ascl.net/code/v/3366).

## Installation ##

The easiest and recommended way to install the software is via the Python command `pip` directly from the `scatfit` GitHub software repository. For instance, to install the master branch of the code, use the following command:

`pip install git+https://github.com/fjankowsk/scatfit.git@master`

This will automatically install all dependencies. Depending on your Python installation, you might want to replace `pip` with `pip3` in the above command.

Please verify that your installation works as expected by downloading a pre-generated `SIGPROC` filterbank file with synthetic data that comes bundled with the GitHub repository:

`wget https://github.com/fjankowsk/scatfit/raw/master/extra/fake_burst_500_DM.fil`

Then run the main analysis on the filterbank data file like this:

`scatfit-fitfrb fake_burst_500_DM.fil 500.0 --fitscatindex --fscrunch 128 --fast --norfi`

You should see several diagnostic windows open. The terminal output should show an updated DM close to 500 pc cm$^{-3}$, a scattering index near -4.0, and a scattering time at 1 GHz of about 20 ms.

## Documentation ##

Further documentation of the software is available on our dedicated [Read the docs website](https://scatfit.readthedocs.io/en/latest/).

## Usage ##

```console
$ scatfit-fitfrb -h
usage: scatfit-fitfrb [-h] [--compare] [--binburst bin] [--fscrunch factor] [--tscrunch factor] [--fast] [--fitrange start end] [--fitscatindex] [--norfi]
                      [--smodel {unscattered,scattered_isotropic_analytic,scattered_isotropic_convolving,scattered_isotropic_bandintegrated,scattered_isotropic_afb_instrumental,scattered_isotropic_dfb_instrumental}] [--showmodels] [--snr snr]
                      [--publish] [-z start end]
                      filename dm

Fit a scattering model to FRB data.

positional arguments:
  filename              The name of the input filterbank file.
  dm                    The dispersion measure of the FRB.

optional arguments:
  -h, --help            show this help message and exit
  --compare             Fit an unscattered Gaussian model for comparison. (default: False)
  --binburst bin        Specify the burst location bin manually. (default: None)
  --fscrunch factor     Integrate this many frequency channels. (default: 256)
  --tscrunch factor     Integrate this many time samples. (default: 1)
  --fast                Enable fast processing. This reduces the number of MCMC steps drastically. (default: False)
  --fitrange start end  Consider only this time range of data in the fit. Increase the region for wide or highly-scattered bursts. Ensure that most of the scattering tail is included in the fit. (default: [-150.0, 150.0])
  --fitscatindex        Fit the scattering times and determine the scattering index. (default: False)
  --norfi               Disable all internal RFI excision methods and use the input data as provided (aside from scaling). This is useful for synthetic input data or if you have cleaned the data already using external tools. (default: False)
  --smodel {unscattered,scattered_isotropic_analytic,scattered_isotropic_convolving,scattered_isotropic_bandintegrated,scattered_isotropic_afb_instrumental,scattered_isotropic_dfb_instrumental}
                        Use the specified scattering model. (default: scattered_isotropic_analytic)
  --showmodels          Show comparison plot of scattering models. (default: False)
  --snr snr             Only consider sub-bands above this S/N threshold. (default: 3.8)
  --publish             Output plots suitable for publication. (default: False)
  -z start end, --zoom start end
                        Zoom into this time region. (default: [-50.0, 50.0])
```

```console
$ scatfit-simpulse -h
usage: scatfit-simpulse [-h]

Simulate scattered pulses.

options:
  -h, --help  show this help message and exit
```

## Profile scattering models ##

Several profile scattering models, i.e. pulse broadening functions and instrumental contributions, are implemented and others can easily be added. The image below shows a selection of them.

![Implemented profile scattering models](https://github.com/fjankowsk/scatfit/raw/master/docs/profile_models.png "Implemented profile scattering models")

## Example output ##

The images below show some example output from the program obtained when fitting simulated filterbank data.

![Profile fit](https://github.com/fjankowsk/scatfit/raw/master/docs/profile_fit.png "Profile fit")

![Width scaling](https://github.com/fjankowsk/scatfit/raw/master/docs/width_scaling.png "Width scaling")

![Correlations](https://github.com/fjankowsk/scatfit/raw/master/docs/corner.png "Correlations")

## Data verification ##

We have verified the `scatfit` software with simulated and real FRB and pulsar data from the following radio telescopes.

*MeerKAT*

* TUSE total intensity
* TUSE high time-resolution voltage buffer data full polarisation

*NenuFAR*

* Folded dispersion-corrected data
