# -*- coding: utf-8 -*-
from numpy import array,random,nan
from pandas import DataFrame, Series

def kfold_NA(X,
             pctNA = 0.05,
             random_state = None):
    """
    Random missing values

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data.
    pctNA : float, default = 0.05
        Percentage of missing values added in the data set.
    
    Returns
    -------
    X : DataFrame of shape (n_samples, n_columns)
        Output data.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of pandas DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame."
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # shape of X
    nr, nc = X.shape
    NA_ls = Series([False]*(nr * nc),index=list(range(nr * nc)))
    # generator
    rng = random.default_rng(seed=random_state)
    sample = NA_ls.sample(frac=pctNA,random_state=rng,replace=True).index
    NA_ls[sample] = True
    arr = array(NA_ls).reshape(-1, nc)
    X[arr] = nan
    return X