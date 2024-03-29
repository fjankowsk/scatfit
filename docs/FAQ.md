# FAQ #

## What scattering model should I use? ##

The answer depends on the data set at hand. However, the following advice is generally accurate. For frequency (sub-)bands of small fractional bandwidth or at high frequencies, i.e. where the narrow bandwidth approximation roughly holds, it is OK to use the default `scattered_isotropic_analytic` model. At low frequencies (< 1 GHz) or for wide (sub-)bands, use the much more complex and, therefore, slower `scattered_isotropic_bandintegrated` model.

## Why do I get the warning: "Could not import Cython pulse model implementations. Falling back to the Python versions."? ##

This could happen if you run `scatfit` from its software repository git checkout, where the current working directory takes import preference. Install the software as above and run it as `$ scatfit-fitfrb` in a different path.

## What data formats are supported? ##

`scatfit` supports loading data from `SIGPROC` filterbank files. Alternative loaders can be implemented relatively easily.

## Does it work on ARM-based Macs? ##

Yes, `scatfit` works on ARM-based Macs with M1 or M2 processors. In fact, it runs blazing fast on them! The immense single-core performance results in a speedup of > 3x in comparison with similar x64-based systems, in my experience.
