import setuptools

with open("README.md", "r",encoding="utf-8") as fh:
    long_description = fh.read()

# Setting up
setuptools.setup(
    name="missmdatools",
    version="0.0.1",
    author="Duverier DJIFACK ZEBAZE",
    author_email="duverierdjifack@gmail.com",
    description="Python library for Handling Missing Values in Multivariate Data Analysis",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.26.4",
                      "matplotlib>=3.8.4",
                      "scikit-learn>=1.2.2",
                      "pandas>=2.2.2",
                      "polars>=0.19.2",
                      "mapply>=0.1.21",
                      "plotnine>=0.10.1"],
    python_requires=">=3.10",
    include_package_data=True,
    package_data={"": ["data/*.xlsx",
                       "data/*.xls",
                       "data/*.txt",
                       "data/*.csv",
                       "data/*.rda"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)