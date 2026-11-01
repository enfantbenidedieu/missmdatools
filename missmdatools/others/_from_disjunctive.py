# -*- coding: utf-8 -*-
from pandas import concat, CategoricalDtype, DataFrame

def from_disjunctive(X,
                     sep = "_",
                     ordered=True):
    """
    Create a categorical ``DataFrame`` from ``DataFrame`` of dummy variables

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_levels)
        Data which contains dummy-coded variables
    sep : str, default = "_"
        Separator used in the column names of the dummy categories they are character 
        indicating the separation of the categorical names from the prefixes. For example, 
        if your column names are "col_A" and "col_B", you can strip the underscore by specifying sep="_".
    ordered : bool or None, default = True
        Whether or not this categorical is treated as an ordered categorical.
        None can be used to maintain the ordered value of existing categoricals when
        used in operations that combine categoricals, e.g. astype, and will resolve to
        False if there is no existing ordered to maintain.
        
    Returns
    -------
    data : DataFrame of shape (n_samples, n_columns)
        Categorical data decoded from the dummy input-data.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # check if X is an object of pandas DataFrame
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not isinstance(X,DataFrame):
        raise TypeError(f"{type(X)} is not supported. Please convert to a DataFrame with pd.DataFrame."
                        "For more information see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    
    def predict(X,q,sep="_"):
        df = X[[c for c in X.columns if c.startswith(f"{q}")]]
        df.columns = [x.replace(f"{q}{sep}","") for x in df.columns]
        # find index and convert to categorical
        values = df.idxmax(axis=1).astype(CategoricalDtype(categories=df.columns,ordered=ordered))
        return values
    # find unique prefix
    columns = []
    for col in X.columns:
        prefix = col.split(sep)[0]
        if len(prefix) == len(col):
            raise ValueError(f"Separator not specified for column: {col}")
        if prefix not in columns:
            columns.append(prefix)
    # create categorical data
    data = concat((predict(X=X,q=q,sep=sep).to_frame(q) for q in columns),axis=1)
    return data