import os.path
from setuptools import find_packages, setup


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
    author_email="fjankowsk at gmail.com",
    description="Fast Radio Burst scattering fits.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/jankowsk/scatfit",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "astropy",
        "corner",
        "emcee",
        "iqrm @ git+https://github.com/v-morello/iqrm.git@master",
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
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=True,
)
