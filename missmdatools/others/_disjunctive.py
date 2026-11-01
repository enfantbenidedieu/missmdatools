# -*- coding: utf-8 -*-
from numpy import array
from pandas import Series, DataFrame, concat

def disjunctive(X,
                prefix = False,
                sep = "_"):
    """
    Convert categorical variable into dummy/indicator variables

    Parameters
    ----------
    X : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_columns)
        Input data.
    
    prefix : bool, default = False
        If True, append DataFrame with columns names.

    sep : str, default = "_"
        If appending prefix, separator/delimiter to use.

    Returns
    -------
    DataFrame : DataFrame of shape (n_samples, n_categories)
        Dummy-coded data.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # convert to DataFrame if Series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame.",
                        "For more information see:https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if any missing values in X
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    if X.isnull().any().any():
        raise TypeError("X must not contains missing values.")

    # encoder
    def encoder(x,
                prefix=False,
                sep="_"):
        # unique values
        levels = sorted(x.unique())
        # dummy/indicator variables
        y = concat((Series(array(x==k).astype("int")) for k in levels),axis=1)
        # set columns
        y.columns = [f"{x.name}{sep}{k}" for k in levels] if prefix else levels
        return y
    # concatenate
    Y = concat((encoder(x=X[q],prefix=prefix,sep=sep) for q in X.columns),axis=1)
    return Y