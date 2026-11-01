# -*- coding: utf-8 -*-
from numpy import where
from pandas import DataFrame

def replace_na(X,
               rows,
               cols,
               values,
               by_row=True):
    """
    Replace missing values in a DataFrame with specified values.
    
    Parameters
    ----------
    X : DataFrame of shape (n_rows, n_columns)
        Input data with missing values.
    rows : array-like of shape (len(missings),)
        Row indices of the missing values to be replaced.
    cols : array-like of shape (len(missings),)
        Column indices of the missing values to be replaced.
    values : array-like of shape (len(rows),
        Values to replace the missing values.
    by_row : bool, default=False
        If True, replace missing values by row; if False, replace by column.
    
    Returns
    -------
    X : DataFrame of shape (n_rows, n_columns)
        DataFrame with missing values replaced.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an instance of pd.DataFrame class
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame.",
                        "For more information see:https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if the lengths of rows, cols, and values are the same
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (len(rows) != len(cols)) or (len(rows) != len(values)):
        raise ValueError("Length of rows, cols, and values must be the same.")
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if by_row is a boolean
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(by_row, bool):
        raise TypeError("by_row must be a boolean.")

    # convert DataFrame to numpy array for efficient indexing
    arr = X.to_numpy().copy()

    if by_row:
        k = 0
        for i in range(arr.shape[1]):
            # index of missing values in this column
            mask = where(cols == i)[0]
            # number of missing values in this column
            n = len(mask)
            if n > 0:
                arr[rows[mask], i] = values[k : k + n]
                k += n
        X.iloc[:, :] = arr
    else:
        X.to_numpy()[rows, cols] = values
    return X