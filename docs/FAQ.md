# FAQ #

## What scattering model should I use? ##

The answer depends on the data set at hand. However, the following advice is generally accurate. For frequency (sub-)bands of small fractional bandwidth or at high frequencies, i.e. where the narrow bandwidth approximation roughly holds, it is OK to use the default `scattered_isotropic_analytic` model. At low frequencies (< 1 GHz) or for wide (sub-)bands, use the much more complex and, therefore, slower `scattered_isotropic_bandintegrated` model.

## Why do I get the warning: "Could not import Cython pulse model implementations. Falling back to the Python versions."? ##

This could happen if you run `scatfit` from its software repository git checkout, where the current working directory takes import preference. Install the software as above and run it as `$ scatfit-fitfrb` in a different path.

## What data formats are supported? ##

`scatfit` supports loading data from `SIGPROC` filterbank files. Alternative loaders can be implemented relatively easily.

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
`$ pip3 install scatfit`

should work fine. Maybe replace `pip3` by `pip` depending on your Python installation.

## How do I estimate the scattering-corrected DM reliably? ##

There are several ways to estimate a pulse or burst's true dispersion measure (DM). For instance, one can optimise the burst profile for the total band-averaged S/N (S/N-maximising DM) or profile structure (structure-maximising DM). Scattering introduces a typically single-sided exponential tail that interferes with the classical DM-estimation techniques by shifting the profile centres in a frequency and DM-dependent way. Hence, one needs to account for scattering by estimating a scattering-corrected DM, which usually differs from the above DM values, sometimes quite significantly.

`scatfit` blindly dedisperses the data at the fiducial DM you provide in its command line call. This behaviour is advantageous, as it gives the user the flexibility to measure the optimum DM using external software, such as `DM_phase`. `scatfit` then performs its sub-banded scattering fits, after which it analyses the fit residuals to determine a scattering-corrected DM.

To measure the scattering-corrected DM reliably, one must run `scatfit` iteratively several times by inputting the last determined scattering-corrected DM in the new call until the output DM converges to a stable value. This occurs when the output DM changes less than its fit uncertainty or an externally pre-defined value. The leading edge of the burst profile should be as sharp as possible (in the absence of DM-smearing) and the tail well-described by the scattering model.

Three to four iterations should be enough to reach a stable scattering-corrected DM if the initial DM was already close to the corrected value. Something must be wrong if the DM does not stabilise after 5-6 iterations.
