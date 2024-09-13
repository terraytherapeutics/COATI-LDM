#!/usr/bin/env python

from setuptools import find_packages, setup

__version__ = "0.1.0"

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as req_file:
    # Exclude GitHub dependencies for `install_requires`
    requirements = [
        line for line in req_file.read().splitlines() if not line.startswith("git+")
    ]

setup(
    author="Ben Kaufman, Edward Williams, Carl Underkoffler, Ryan Pederson, Miles Wang-Henderson, John Parkhill",
    author_email="bkaufman@terraytx.com",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 1 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    description="COATI Diffusion",
    install_requires=requirements,
    packages=find_packages(),
    long_description=readme,
    include_package_data=True,
    keywords="diffusion",
    name="coatiLDM",
    version=__version__,
    zip_safe=False,
)
