import os.path
from setuptools import Extension, find_packages, setup


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "scatfit", "version.py")

    with open(version_file, "r") as f:
        raw = f.read()

    items = {}
    exec(raw, None, items)

    return items["__version__"]


def get_long_description():
    with open("README.md", "r") as fd:
        long_description = fd.read()

    return long_description


setup(
    name="scatfit",
    version=get_version(),
    author="Fabian Jankowski",
    author_email="fjankowsk@gmail.com",
    description="Scattering fits of time domain radio signals (Fast Radio Bursts or pulsars).",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/fjankowsk/scatfit",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "astropy",
        "corner",
        "emcee",
        "lmfit",
        "matplotlib",
        "mtcutils @ git+https://bitbucket.org/vmorello/mtcutils.git@master",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "your",
    ],
    entry_points={
        "console_scripts": [
            "scatfit-fitfrb = scatfit.apps.fit_frb:main",
            "scatfit-simpulse = scatfit.apps.simulate_pulse:main",
        ],
    },
    extras_require={
        "develop": [
            "black",
            "cibuildwheel",
            "Cython >= 3.0.0",
            "pytest",
            "pytest-cov",
            "twine",
        ],
        "docs": ["myst-parser", "sphinx", "sphinx_rtd_theme"],
    },
    ext_modules=[
        Extension(
            name="scatfit.pulsemodels_cython",
            sources=["scatfit/pulsemodels_cython.pyx"],
        ),
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
