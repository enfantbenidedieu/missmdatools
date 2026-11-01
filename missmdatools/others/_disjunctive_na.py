# -*- coding: utf-8 -*-
from pandas import DataFrame, Series, concat, get_dummies, NA

def disjunctive_NA(X,
                   prefix=False,
                   sep="_"):
    """
    Convert categorical variable into dummy/indicator variables

    Parameters
    ----------
    X : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_columns)
        Input data, with at least one missing values.
        
    prefix : bool, default = False
        If True, append DataFrame with columns names.

    sep : str, default = "_"
        If appending prefix, separator to use.

    Returns
    -------
    Y : DataFrame of shape (n_samples, n_levels)
        Dummy-coded data.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # convert to pandas DataFrame if pandas Series
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame.",
                        "For more information see:https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    # disjunctive table
    Y = DataFrame(index=X.index).astype("int")
    for q in X.columns:
        if prefix:
            y = get_dummies(X[q],prefix=q,prefix_sep=sep,dtype="int",dummy_na=False)
        else:
            y = get_dummies(X[q],dtype="int",dummy_na=False)
        # replace 0 with NA in rows with NA
        if X[q].isnull().any():
            levels = y.columns
            mask_na = X[q].isna()
            y.loc[mask_na,levels] = NA
        # concatenate
        Y = concat((Y,y),axis=1)
    return Y
