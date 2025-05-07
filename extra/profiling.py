#
#   Performance profiling functions.
#   2025 Fabian Jankowski
#

import cProfile
import pstats

import numpy as np

from scatfit.pulsemodels_cython import bandintegrated_model


def benchmark_model():
    x = np.linspace(-2000, 2000, num=10 * 1024)
    fluence = 10.0
    center = 0.0
    sigma = 5.0
    taus = 7.5
    f_lo = 55.0
    f_hi = 60.0
    nfreq = 9
    dc = 0.0

    for _ in range(1000):
        _ = bandintegrated_model(x, fluence, center, sigma, taus, dc, f_lo, f_hi, nfreq)


import pyximport

pyximport.install()

cProfile.run("benchmark_model()", filename="profiled.dat")

s = pstats.Stats("profiled.dat")
s.strip_dirs().sort_stats("time").print_stats()
