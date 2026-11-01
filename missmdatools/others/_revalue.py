# -*- coding: utf-8 -*-
from pandas import CategoricalDtype, DataFrame, Series

def revalue(X,
            sep="_"):
    """
    Revalue Categorical Variables

    Check if two categoricals variables have same categories and replace with new values.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns) or Series of shape (n_samples,)
        Input data.
    sep : str, default = "_"
        If appending prefix, separator/delimiter to use.

    Returns
    -------
    Y : DataFrame of shape (n_samples, n_columns)
        Input data or revaluated data.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #convert pd.Series to pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if isinstance(X,Series):
        X = X.to_frame()

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if X is an object of class pd.DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pandas.DataFrame.",
            "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #check if all columns are categorics
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not all(X[k].dtype in ("object","category","str") for k in X.columns):
        raise TypeError("All columns in X must be categorics.")
    
    #find columns with at least one element in common
    def find_intersection(X):
        i, res = 0, {}
        for k in range(X.shape[1]-1):
            for l in range(k+1,X.shape[1]):
                categories1 = X.iloc[:,k].dropna().unique()
                categories2 = X.iloc[:,l].dropna().unique()
                intersect = list(set(categories1) & set(categories2))
                if len(intersect) > 0:
                    res[i] = intersect
                    i +=1
        return res

    #check if shape greater than 1:
    if X.shape[1] > 1:
        all_levels = find_intersection(X)
        while len(all_levels) != 0:
            y = X.isin(all_levels[0]).any()
            cols = list(y[y==True].index)
            if len(cols) > 1:
                for q in cols:
                    categories = X[q].dropna().unique()
                    X[q] = X[q].map({x : f"{q}{sep}{x}" for x in categories})
            all_levels = find_intersection(X)
          
    # convert to ordered categorical
    for q in X.columns:
        categories = sorted(X[q].dropna().unique())
        X[q] = X[q].astype(CategoricalDtype(categories=categories,ordered=True))
    return X