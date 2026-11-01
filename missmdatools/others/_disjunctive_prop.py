# -*- coding: utf-8 -*-
from numpy import ones, array, ndarray, random,isnan
import numpy.ma as npma
from pandas import DataFrame, Series, get_dummies,concat, NA

def disjunctive_prop(X,
                     prefix = False,
                     sep = "_",
                     row_w = None,
                     random_state = None):
    """
    Disjunctive table when missing values are present
    
    Create the disjunctive table. The missing values are replace by the proportions of the category or by random values.
    
    Parameters
    ----------
    X : Series of shape (n_samples,) or DataFrame of shape (n_samples, n_columns)
        Input data, with at least one missing values.
    
    prefix : bool, default = False
        If True, append DataFrame with columns names.
    
    sep : str, default = "_"
        If appending prefix, separator to use.
        
    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.
        
    random_state : int, default = None 
        If None, implies that missing values are initially imputed by the proportion of the category for 
        the categorical variables coded with disjunctive matrix. Other values leads to a random initialization.
            
    Returns
    -------
    Y : DataFrame of shape (n_samples, n_levels)
        Dummy-coded data, where missing values are replace by the proportions of the category or by random values.
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
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # number of samples
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    n_rows = X.shape[0]
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set rows weights
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if row_w is None: 
        row_w = ones(n_rows)/n_rows
    elif not isinstance(row_w,(list,tuple,ndarray,Series)): 
        raise TypeError("row_w must be a 1d array-like of individuals weights.")
    elif len(row_w) != n_rows: 
        raise ValueError(f"row_w must be a 1d array-like of shape ({n_rows},).")
    else: 
        row_w = array(row_w)/sum(row_w)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # dummies with proportions
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # disjunctive table
    Y = DataFrame(index=X.index).astype("int")
    for q in X.columns:
        if prefix:
            y = get_dummies(X[q],prefix=q,prefix_sep=sep,dtype="int",dummy_na=False)
        else:
            y = get_dummies(X[q],dtype="int",dummy_na=False)
        # replace where rows are NA
        if X[q].isnull().any():
            levels = y.columns
            mask = X[q].isna()
            y.loc[mask,levels] = NA
            # fill with weighted average
            if random_state is None:
                y_mask = npma.masked_array(y,mask=isnan(y)).mask 
                value = npma.average(npma.array(y, mask=y_mask), weights=row_w, axis=0)
            else:
                rng = random.default_rng(seed=random_state)
                # fill NA with random values
                value = random.uniform(low=0.0,high=1.0,size=len(levels))
                # normalize such sum equal to 1
                value = value/sum(value)
            # fill NA
            y.loc[mask,levels] = value
        # concatenate
        Y = concat((Y,y),axis=1)
    return Y