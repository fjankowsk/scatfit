# Scatfit: Scattering fits of time domain radio signals (Fast Radio Bursts or pulsars) #

[![PyPI latest release](https://img.shields.io/pypi/v/scatfit.svg)](https://pypi.org/project/scatfit/)
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

The code is also listed in the Astrophysics Source Code Library (ASCL): https://ascl.net/code/v/3366

## Installation ##

The easiest and recommended way to install the software is through `pip` from the central PyPI index by running:

`pip install scatfit`

This will install the latest release and all its dependencies. If you need a more recent version of the software, install it directly from its GitHub software repository. For instance, to install the master branch of the code, use the following command:

`pip install git+https://github.com/fjankowsk/scatfit.git@master`

This will also automatically install all dependencies.

Please verify that your installation works as expected by downloading a pre-generated `SIGPROC` filterbank file with synthetic data that comes bundled with the GitHub repository:

`wget https://github.com/fjankowsk/scatfit/raw/master/extra/fake_burst_500_DM.fil`

Then run the main analysis on the filterbank data file like this:

`scatfit-fitfrb fake_burst_500_DM.fil 500.0 --fitscatindex --fscrunch 128 --fast`

You should see several diagnostic windows open. The terminal output should show an updated DM close to 500 pc cm^-3, a scattering index near -4.0, and a scattering time at 1 GHz of about 20 ms.

## Usage ##

```
$ scatfit-fitfrb -h
usage: scatfit-fitfrb [-h] [--compare] [--binburst bin] [--fscrunch factor] [--tscrunch factor] [--fast] [--fitrange start end]
                      [--fitscatindex]
                      [--smodel {unscattered,scattered_isotropic_analytic,scattered_isotropic_convolving,scattered_isotropic_bandintegrated,scattered_isotropic_afb_instrumental,scattered_isotropic_dfb_instrumental}]
                      [--showmodels] [--snr snr] [--publish] [-z start end]
                      filename dm

Fit a scattering model to FRB data.

positional arguments:
  filename              The name of the input filterbank file.
  dm                    The dispersion measure of the FRB.

options:
  -h, --help            show this help message and exit
  --compare             Fit an unscattered Gaussian model for comparison. (default: False)
  --binburst bin        Specify the burst location bin manually. (default: None)
  --fscrunch factor     Integrate this many frequency channels. (default: 256)
  --tscrunch factor     Integrate this many time samples. (default: 1)
  --fast                Enable fast processing. This reduces the number of MCMC steps drastically. (default: False)
  --fitrange start end  Consider only this time range of data in the fit. Increase the region for wide or highly-scattered bursts.
                        Ensure that most of the scattering tail is included in the fit. (default: [-200.0, 200.0])
  --fitscatindex        Fit the scattering times and determine the scattering index. (default: False)
  --smodel {unscattered,scattered_isotropic_analytic,scattered_isotropic_convolving,scattered_isotropic_bandintegrated,scattered_isotropic_afb_instrumental,scattered_isotropic_dfb_instrumental}
                        Use the specified scattering model. (default: scattered_isotropic_analytic)
  --showmodels          Show comparison plot of scattering models. (default: False)
  --snr snr             Only consider sub-bands above this S/N threshold. (default: 3.8)
  --publish             Output plots suitable for publication. (default: False)
  -z start end, --zoom start end
                        Zoom into this time region. (default: [-50.0, 50.0])
```

```
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

## FAQ ##

### What scattering model should I use? ###

The answer depends on the data set at hand. However, the following advice is generally accurate. For frequency (sub-)bands of small fractional bandwidth or at high frequencies, i.e. where the narrow bandwidth approximation roughly holds, it is OK to use the default `scattered_isotropic_analytic` model. At low frequencies (< 1 GHz) or for wide (sub-)bands, use the much more complex and, therefore, slower `scattered_isotropic_bandintegrated` model.

### Why do I get the following warning: `Could not import Cython pulse model implementations. Falling back to the Python versions.`? ###

This could happen if you run `scatfit` from its software repository git checkout, where the current working directory takes import preference. Install the software as above and run it as `$ scatfit-fitfrb` in a different path.

### What data formats are supported? ###

`scatfit` supports loading data from `SIGPROC` filterbank files. Alternative loaders can be implemented relatively easily.

### Does it work on ARM-based Macs? ###

Yes, `scatfit` works on ARM-based Macs with M1 or M2 processors. In fact, it runs blazing fast on them! The immense single-core performance results in a speedup of > 3x in comparison with similar x64-based systems, in my experience.
