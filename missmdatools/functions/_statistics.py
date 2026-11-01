# -*- coding: utf-8 -*-
from numpy import average, cov, sqrt, isnan
import numpy.ma as npma
from pandas import DataFrame, Series,concat

def nanweightstats(X, w=None):
    """
    Weighted Average and Standard Deviation with missing values
    
    Performns weighted average and standard deviation with missing values.
    
    Parameters
    ----------
    X : DataFrame of shape (n_samples,n_columns)
        Input data.
    
    w : 1d array-like of shape (n_samples,), default = None
        An optional samples weights.
    
    Returns
    -------
    center : float
        The weighted average
    scale : float
        The weighted standard deviation.
    """
    # where NA 
    mask = npma.masked_array(X,mask=isnan(X)).mask
    # weighted average
    mu = npma.average(npma.array(X, mask=mask), weights=w, axis=0)
    # weighted standard deviation
    sigma = sqrt(npma.average(npma.array((X - mu)**2, mask=mask), weights=w, axis=0))
    # convert to Series
    mu = Series(mu,index=X.columns,name="center")
    sigma = Series(sigma,index=X.columns, name="scale")
    return mu, sigma

# intern functions
def wmean(X, w=None):
    """
    Weighted average

    Compute the weighted average of all columns in X.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_columns)
        Input data containing the data to be averaged. 
    
    w : 1d array-like of shape (n_samples,) default = None
         Weights associated with the values in X.

    Returns
    -------
    wmean : Series of shape (n_columns,)
        The weighted average of X.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()
        
     #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    return Series(average(X,axis=0,weights=w),index=X.columns,name="center")

def wstd(X, w=None,ddof=0):  
    """
    Weighted standard deviation

    Compute the weighted standard deviation of all columns in X.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_columns)
        Input data containing the data to be averaged.
    
    w : 1d array-like of shape (n_samples,) default = None
        Weights associated with the values in X.

    ddof : int, default = 0
        If not None the default value implied by bias is overridden. Note that ``ddof=1`` will return the unbiased estimate, 
        even if both fweights and aweights are specified, and ``ddof=0`` will return the simple average. 
        The default value is :math:`0`.

    Returns
    -------
    wstd : Series of shape (n_columns,)
        The weighted standard deviation of X.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    return Series(sqrt([cov(m=X.iloc[:,j],aweights=w,ddof=ddof) for j in range(X.shape[1])]),index=X.columns,name="scale").astype("float")

def summary_cat(X):
    """
    Summary statistics for categorical variables
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    # number of missing values
    nbNA = X.isnull().sum().to_frame(name = 'NAs')
    # number of non missing values
    count = X.notnull().sum().to_frame("count")
    # number of unique elements
    nunique = X.nunique().to_frame(name = 'nunique')
    # number of blank
    nblank = Series([len(X[k][X[k]==""]) for k in X.columns],index=X.columns,name="nblank")
    # minimum character
    minmaxchar = X.apply(lambda x: x.astype("str").str.strip().str.len().agg(['min', 'max'])).T
    minmaxchar.columns = ["minchar","maxchar"]
    # concatenate
    summary = concat((count,nunique,nblank,minmaxchar,nbNA),axis=1).T
    summary.index.name = None
    # convert to int
    summary = summary.apply(lambda x : x.astype("int"),axis=0)
    return summary