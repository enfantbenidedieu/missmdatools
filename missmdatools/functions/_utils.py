# -*- coding: utf-8 -*-
from pandas import DataFrame, CategoricalDtype
from pandas.api.types import is_numeric_dtype

def is_object_or_category_dtype(X):
    """
    Performs is_object_or_category_dtype validation

    Parameters
    ----------
    X : 1d array-like of shape (n_samples,)
        Input data for which check should be done
    """
    if not (X.dtype in ("object","category","str")):
        return False
    else:
        return True
    
def check_X_valid(X):
    """
    Input validation
    
    Parameters
    ----------
    X : DataFrame of shape (n_sample,s n_columns)
        Input data, where ``n_samples`` is the number of samples and
        ``n_columns`` is the number of columns.
    
    Returns
    -------
    None or TypeError
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame.",
                        "For more information see:https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X contains columns 
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if any(not (is_numeric_dtype(X[c]) or is_object_or_category_dtype(X[c])) for c in X.columns):
        raise TypeError("Columns in X must be either numeric, object, string or category.")
    
def check_param_valid(ncp,
                      method,
                      lambda_):
    """
    Impute Parameters Validation
    
    Parameters
    ----------
    ncp : int, default = 2
        The number of components used to predict the missing values.
    
    method : {"regularized","em"}, default = "em"
        The method to used. Possible values are:
        
        * "regularized" for regularized iterative algorithm
        * "em" for expectation - maximization algorithm
    
    lambda_ : int, default = 1
        1 by default to perform the regularized PCAImputer algorithm; useful only if method = "regularized". 
        Other regularization terms can be implemented by setting the value to less than 1 in order to regularized less 
        (to get closer to the results of the em method) or more than 1 to regularized more (to get closer to the results of the mean imputation).
        
    Returns
    -------
    None or TypeError
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if ncp is not negative
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if ncp < 0: 
        raise ValueError("ncp must be positive")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if lambda_ is not negative
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if lambda_ <= 0: 
        raise ValueError("lambda_ must be strictly positive")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if method in list
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if method.lower() not in ("em","regularized"):
        raise ValueError("method should be one of 'em', 'regularized'")
    
def to_category(X,
                ordered=True):
    """
    Convert String and Object to Category

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data.
    ordered : bool or None, default = True
        Whether or not this categorical is treated as an ordered categorical.
        None can be used to maintain the ordered value of existing categoricals when
        used in operations that combine categoricals, e.g. astype, and will resolve to
        False if there is no existing ordered to maintain.
        
    Returns
    -------
    X : DataFrame of shape (n_samples, n_columns)
        Output data.
    """
    cols = X.columns[X.dtypes.isin(["object","string","category"])].tolist()
    for k in cols:
        categories = sorted(X[k].dropna().unique())
        X[k] = X[k].astype(CategoricalDtype(categories=categories,ordered=ordered))
    return X