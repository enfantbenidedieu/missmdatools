<p align="center">
    <img src="./docs/source/_static//missmdatools.svg" height=300></img>
</p>

<div align="center">

[![GitHub](https://shields.io/badge/license-MIT-informational)](https://github.com/enfantbenidedieu/missmdatools/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/missmdatools.svg?color=dark-green)](https://pypi.org/project/missmdatools/)
[![Downloads](https://static.pepy.tech/badge/missmdatools)](https://pepy.tech/project/missmdatools)
[![Downloads](https://static.pepy.tech/badge/missmdatools/month)](https://pepy.tech/project/missmdatools)
[![Downloads](https://static.pepy.tech/badge/missmdatools/week)](https://pepy.tech/project/missmdatools)

</div>

# missmdatools : Missing Data Imputation with Multivariate Data Analysis Methods

# Contents

**1. [Overview](#overview)**

**2. [Installation](#installation)**

* [2.1 Global environment](#genv)
* [2.2 Virtual environment](#venv)
* [2.3 Version](#version)
* [2.4 Dependencies](#dependencies)

**3. [Example](#example)**

**4. [Documentation](#doc)**

**5. [About us](#about_us)**

* [5.1 Authors](#authors)
* [5.2 Feedbacks](#authors)
* [5.3 Citing missmdatools](#citing)

## Overview <a name="overview"></a>

missmdatools is a python package for missing values imputation using Multivariate Data Analysis (MDA) methods.

missmdatools provides functions for :

1. **Simple imputation :**

    * Impute contingency table using Correspondence Analysis - **CAimputer**
    * Impute mixed dataset using Factor Analysis of Mixed Data - **FAMDimputer**
    * Impute categorical dataset using Multiple Correspondence Analysis - **MCAimputer**
    * Impute dataset with variables structured into groups of variables using Multiple Factor Analysis - **MFAimputer**
    * Impute continuous dataset using Principal Component Analysis - **PCAimputer**

2. **Estimation of the number of components :**

    * Estimate the number of dimensions for the Factorial Analysis of Mixed Data by cross-validation - **FAMDestimncp**
    * Estimate the number of dimensions for the Multiple Correspondence Analysis by cross-validation - **MCAestimncp**
    * Estimate the number of dimensions for the Principal Component Analysis by cross-validation - **PCAestimncp**

## Installation <a name="installation"></a>

### Global environment <a name="genv"></a>

You can directly install missmdatools using pip :

```bash
pip install missmdatools
```

or set a virtual environment.

### Virtual environment <a name="venv"></a>

Install the 64-bit version of Python 3, for instance from the [official website](https://www.python.org/). Now create a [virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html) and install missmdatools.

The virtual environment is optional but strongly recommended, in order to avoid potential conflicts with other packages.

```bash
PS C:\> python -m venv missmdatools-env # create virtual env
PS C:\> missmdatools-env\Scripts\activate  # activate
PS C:\> pip install -U missmdatools  # install missmdatools
```

### Version <a name="version"></a>

In order to check your installation, you can use.

```python
>>> import missmdatools
>>> print(missmdatools.__version__)
0.0.1
```

Using an isolated environment such as *pip venv* or *conda* makes it possible to install a specific version of missmdatools with pip and conda and its dependencies independently of any previously installed Python packages.

You should always remember to activate the environment of your choice prior to running any Python command whenever you start a new terminal session.

### Dependencies <a name="dependencies"></a>

missmdatools is compatible with python version which supports both dependencies :

| Packages          |  Version |
| :---------------- | :------: |
| numpy             | 1.21     |
| pandas            | 1.4      |
| scikit-learn      | 1.2      |
| statsmodels       | 0.14.6   |
| pyreadr           | 0.5.4    | 

## Example <a name="example"></a>

1. **Loading data**

```python
>>> from missmdatools.datasets import orange
```

2. **Estimation of the number of components**

```python
>>> from missmdatools import PCAestimncp
>>> clf = PCAestimncp(method="loo")
>>> clf.fit(orange)
PCAestimncp(method="loo")
>>> # number of components retained for the PCA
>>> ncp = clf.ncp_
```

3. **Impute using PCA**

```python
>>> from missmdatools import PCAimputer
>>> clf = PCAimputer(ncp=ncp,method="regularized")
>>> clf.fit(orange)
PCAimputer(method="regularized")
>>> # impute dataset
>>> clf.complete_
```

## Documentation <a name="doc"></a>

The official documentation is hosted on [https://missmdatools.readthedocs.io](https://missmdatools.readthedocs.io).

## About Us <a name="about_us"></a>

### Authors <a name="authors"></a>

missmdatools is developed and maintained by [Duvérier DJIFACK ZEBAZE](https://www.linkedin.com/in/duv%C3%A9rier-djifack-z-030097118/), the founder of djifacklab (*Djifack Laboratory of Mathematics, Statistics and Economics books and packages production using Python Programming Language*).

The djifacklab laboratory maintains others python librairies such as [scientisttools](https://pypi.org/project/scientisttools/), [discrimintools](https://pypi.org/project/discrimintools/), [clustvartools](https://pypi.org/project/clustvartools/), [scientistmetrics](https://pypi.org/project/scientistmetrics/), [scientistshiny](https://pypi.org/project/scientistshiny/), [scientisttseries](https://pypi.org/project/scientistshiny/) and [ggcorrplot]( https://pypi.org/project/ggcorrplot/).

### Feedbacks <a name="feedbacks"></a>

If you have found missmdatools useful in your work, research, or company, please let us know by writing to email [djifacklab@gmail.com](mailto:djifacklab@gmail.com).

### Citing scientisttools <a name="citing"></a>

If missmdatools has been significant in your research, and you would like to acknowledge the project in your academic publication, we suggest citing it using the following *BibTeX format*:

```
@misc{DJIFACK ZEBAZE_2026, 
    url = {https://github.com/enfantbenidedieu/missmdatools}, 
    title = {missmdatools: Missing Data Imputation with Multivariate Data Analysis Methods}
    author = {DJIFACK ZEBAZE, Duvérier}, 
    year = {2026}
}
```