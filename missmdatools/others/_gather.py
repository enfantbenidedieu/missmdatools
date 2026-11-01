# -*- coding: utf-8 -*-
from pandas import DataFrame

def gather(X,
           value_to="value"):
    """
    Stack the prescribed level(s) from columns to index

    Return a reshaped DataFrame or Series having a multi-level index with one or more 
    new inner-most levels compared to the current DataFrame. The new inner-most levels 
    are created by pivoting the columns of the current dataframe:

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data.
    
    value_to : str, default = "value"
        The name of the column to create from the data stored in cell values.
    
    Returns
    -------
    Y : DataFrame of shape (n_samples*n_columns,3)
        Stacked data.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of pandas DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame."
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    # pivot longer
    Y = X.stack().reset_index()
    Y.columns = ["Var1","Var2",value_to]
    return Y