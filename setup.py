# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# Setting up
setup(
    name="missmdatools",
    version="0.0.1",
    author="Duverier DJIFACK ZEBAZE",
    author_email="djifacklab@gmail.com",
    description="Python library for missing values imputation using Multivariate Data Analysis (MDA) methods",
    long_description_content_type="text/markdown",
    long_description=open("README.md", "r",encoding="utf-8").read(),
    url = "https://github.com/enfantbenidedieu/missmdatools",
    packages=find_packages(),
    classifiers = [
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
    python_requires = ">=3.11",
    install_requires = [
            "numpy>=1.21",
            "pandas>=1.4",
            "scikit-learn>=1.2",
            "statsmodels>=0.14.6",
            "pyreadr>=0.5.4"
        ],
    include_package_data = True,
    package_data = {"": ["data/*"]},
    keywords = "missing values, principal component analysis, single imputation, multiple imputation, multi-table data, mixed data, multiple correspondence analysis, multiple factor analysis",
    project_urls = {
        "Bug Reports": "https://github.com/enfantbenidedieu/missmdatools/issues",
        "Source": "https://github.com/enfantbenidedieu/missmdatools",
        "Documentation": "https://missmdatools.readthedocs.io",
    }
)