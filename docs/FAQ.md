# FAQ #

## How do I fit FRB data? ##

`scatfit` reads FRB data from `SIGPROC` filterbank files and various data formats supported by `PSRCHIVE`, including PSRFITS and Timer. The data must be a single dynamic spectrum stream with frequency and time dimension, typically at zero-DM, i.e. *not* dedispersed. `scatfit` includes various methods to RFI clean the data. Thus, it can be used directly on the raw input data, assuming that you already know the rough *DM* and location of the FRB. A typical command line run looks like this:  
`$ scatfit-fitfrb filename.fil 503.6 --fscrunch 256 --fitscatindex --snr 5.0`

Tweak the sub-band *S/N* threshold as required for your data. If `scatfit` does not find the burst location automatically, you can set it manually using the *binburst* command line option. Run `scatfit` several times while changing the input DM to the best-determined one reported in the `scatfit` output until the DM converges (no significant DM change). Use the converged best-determined DM as input for a final `scatfit` run with the large MCMC chains, i.e. without the *fast* option.

## How do I fit folded pulsar profile data? ##

`scatfit` works well with folded pulsar profile data. It can load full-polarisation and total intensity data in most pulsar data formats, including PSRFITS and Timer. It also supports already dedispersed data, as produced by single-pulse pipelines.

The input data should be fully integrated in time (tscrunched) but must contain frequency and, eventually, polarisation information, in addition to the phase bin dimension. The most common pre-processing workflow is to RFI clean the data in the integration (subint) and frequency (channel) dimensions and then integrate them in time and polarisation. You might also want to reduce the number of phase bins (nbin) at the same time. This can be achieved most easily by running `PSRCHIVE`'s `pam` command like so:  
`$ pam -Tp -e Tp filename.fits`

If required, rotate the pulse profile so that the scattering tail does not wrap around in phase. A good approach is to place the peak of the profile at around 0.2 phase. You can do this by running `pam` with the appropriate value for the phase rotation like so:  
`$ pam -r 0.3 filename.Tp -m`

You can then run `scatfit` on the time and polarisation integrated data like this:  
`$ scatfit-fitfrb filename.Tp 57.2 --fscrunch 48 --fitrange -200 200 -z -50 200 --fitscatindex --snr 15.0 --norfi`

Select a good initial *DM* from the ATNF pulsar catalogue or from running `PSRCHIVE`'s `pdmp`. Use an *fscrunch* value appropriate for your data and their total number of channels. Adjust the fit and zoom range to fit your use case. Make sure that most of the on-pulse profile phase bins or time samples are used in the fit. Adjust the minimum sub-band S/N as required. As we have cleaned the data before, we turned off all further RFI excision methods within `scatfit`.

## What should I do if some distributions in the MCMC corner plots look artificially cut off? ##

This should not happen in a correct `scatfit` run. However, it might happen if some fit values are outside the allowed ranges predefined for each fit parameter (e.g. amplitude, centre, sigma, or taus). What happens is that the Markov chain sampler hits one or multiple boundaries of the predefined ranges. As a result, the MCMC samples pile up at the boundary, and the parameter range gets artificially cut off. This is visible as correlation ellipses being truncated or halved.

The solution is simple but requires editing the `scatfit` code, particularly the parameter hints in the `fit_frb.py` Python file in your local installation. Look for lines similar to `model.set_param_hint("taus", value=1.5, min=5.0e-5)` or `model.set_param_hint("center", value=0.0, min=-20.0, max=20.0)` and adjust the *min* or *max* boundaries as required for your data. Ensure that the updated allowed parameter ranges are in effect by rerunning the fit.

## What scattering model should I use? ##

The answer depends on the data set at hand. However, the following advice is generally accurate. For frequency (sub-)bands of small fractional bandwidth or at high frequencies, i.e. where the narrow bandwidth approximation roughly holds, it is OK to use the default `scattered_isotropic_analytic` model. At low frequencies (< 1 GHz) or for wide (sub-)bands, use the much more complex and, therefore, slower `scattered_isotropic_bandintegrated` model.

## Why do I get the warning: "Could not import Cython pulse model implementations. Falling back to the Python versions."? ##

This could happen if you run `scatfit` from its software repository git checkout, where the current working directory takes import preference. Install the software as above and run it as `$ scatfit-fitfrb` in a different path.

## What data formats are supported? ##

`scatfit` supports loading data from `SIGPROC` filterbank files and all the data formats readable by `PSRCHIVE`. The latter includes standard pulsar astronomy "archive" data formats, such as PSRFITS or Timer. You must have the `PSRCHIVE` Python bindings installed for this to work. Alternative loaders can be implemented relatively easily.

## Does it work on ARM-based Macs? ##

Yes, `scatfit` works on ARM-based Macs with M1 or M2 processors. In fact, it runs blazing fast on them! The immense single-core performance results in a speedup of > 3x in comparison with similar x64-based systems, in my experience.

## I get a mtcutils dependency error when trying to install the software. ##

If you encounter a dependency error like the following,

```bash
ERROR: Could not find a version that satisfies the requirement mtcutils (from scatfit==0.2.21) (from versions: none)
ERROR: No matching distribution found for mtcutils (from scatfit==0.2.21)
```

first install the `mtcutils` package manually like this:  
`$ pip3 install git+https://bitbucket.org/jankowsk/mtcutils.git@master`

before installing `scatfit`. Then  
`$ pip3 install git+https://github.com/fjankowsk/scatfit.git@master`

should work fine. Maybe replace `pip3` by `pip` depending on your Python installation.

## How do I estimate the scattering-corrected DM reliably? ##

There are several ways to estimate a pulse or burst's true dispersion measure (DM). For instance, one can optimise the burst profile for the total band-averaged S/N (S/N-maximising DM) or profile structure (structure-maximising DM). Scattering introduces a typically single-sided exponential tail that interferes with the classical DM-estimation techniques by shifting the profile centres in a frequency and DM-dependent way. Hence, one needs to account for scattering by estimating a scattering-corrected DM, which usually differs from the above DM values, sometimes quite significantly.

`scatfit` blindly dedisperses the data at the fiducial DM you provide in its command line call. This behaviour is advantageous, as it gives the user the flexibility to measure the optimum DM using external software, such as `DM_phase` and do the scattering fit based on this. `scatfit` then performs its sub-banded scattering fits, after which it analyses the fit residuals to determine a scattering-corrected DM.

To measure the scattering-corrected DM reliably, one must run `scatfit` iteratively several times by inputting the last-determined scattering-corrected DM in the new call until the output DM converges to a stable value. Convergence occurs when the difference between input and output DM changes less than its fit uncertainty or an externally pre-defined value.

In other words, start by giving the initial DM (e.g. S/N-optimizing DM) in the first `scatfit` call. Copy the output scattering-corrected DM and run `scatfit` again with the output DM as input. Repeat this process until the difference between input DM and output scattering-corrected DM is less than the reported DM uncertainty or a pre-determined value.

If you have access to the voltage data of the burst, you should also iterate on the coherent DM at the same time as the post-detection DM. You ideally want to coherently dedisperse, optimise the DM, coherently dedisperse the data again with the new output DM, and repeat the process several times until convergence.

The leading edge of the burst profile should be as sharp as possible (in the absence of DM-smearing), and the tail well-described by the scattering model. Three to four iterations should be enough to reach a stable scattering-corrected DM if the initial DM was already close to the corrected value. Something must be wrong if the DM does not stabilise after 5-6 iterations.
