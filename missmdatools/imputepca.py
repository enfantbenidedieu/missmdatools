# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.weightstats import DescrStatsW

from .get_melt import get_melt
from .svd_triplet import svd_triplet

def imputePCA(X,
              n_components=5, 
              standardize=True,
              method = "regularized",
              ind_weights = None,
              ind_sup = None,
              quanti_sup = None,
              quali_sup = None,
              ridge_coef = 1,
              threshold = 1e-6,
              random_state = None,
              nb_init = 1,
              max_iter = 1000):
    """
    Impute dataset with Principal Components Analysis (imputePCA)
    -------------------------------------------------------------

    Description
    -----------
    impute the missing values of a dataset with the Principal Components Analysis (PCA) model.
    It can be used as a preliminary step before performing a PCA on an complete dataset.
    The output of the algorithm can be used as an input of the PCA function of the scientisttools 
    package in order to perform PCA on an incomplete dataset.

    Parameters
    ----------
    X : pandas/polars dataframe with continuous variables of shape (n_rows, n_columns) containing missing values

    n_components : an integer corresponding to the number of components used to predict the missing values (default is 5)

    standardize : a boolean, default = True
        - If True : the data are scaled to unit variance.
        - If False : the data are not scaled to unit variance.
    
    method : method to used. Either 'regularized' or 'em'
        - 'regularized' for regularized iterative algorithm
        - 'em' for expectation - maximization algorithm
    
    ind_weights : an optional individuals weights (by default, a list/tuple of 1/(number of active individuals) for uniform individuals weights),
                    the weights are given only for active individuals.
    
    ind_sup : an integer or a list/tuple indicating the indexes of the supplementary individuals

    quanti_sup : an integer or a list/tuple indicating the indexes of the quantitative supplementary variables

    quali_sup : an integer or a list/tuple indicating the indexes of the categorical supplementary variables

    ridge_coef : 1 by default to perform the regularized imputePCA algorithm; useful only if method="regularized". 
                Other regularization terms can be implemented by setting the value to less than 1 in order to regularized less 
                (to get closer to the results of the em method) or more than 1 to regularized more (to get closer to the results of the mean imputation)

    threshold : the threshold for assessing convergence

    random_state : integer, by default random_state = None implies that missing values are initially imputed by the mean of each variable. 
                    Other values leads to a random initialization
    
    nb_init : integer corresponding to the number of random initializations; the first initialization is the initialization with the mean imputation.

    max_iter : integer, maximum number of iteration for the algorithm (default = 1000).

    Return
    ------
    a dictionary containing two elements
    completeObs : the imputed dataset; the observed values are kept for the non-missing entries and the missing values are replaced by the predicted ones.

    fittedX : the reconstructed data

    References
    ----------
    Josse, J, Husson, F & Pagès J (2009), Gestion des données manquantes en Analyse en Composantes Principales, Journal de la SFDS. 150 (2), 28-51

    Josse, J & Husson, F. (2013). Handling missing values in exploratory multivariate data analysis methods. Journal de la SFdS. 153 (2), pp. 79-99.

    Josse, J. and Husson, F. missMDA (2016). A Package for Handling Missing Values in Multivariate Data Analysis. Journal of Statistical Software, 70 (1), pp 1-31 <doi:10.18637/jss.v070.i01>

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    # check if X is an instance of polars dataframe
    if isinstance(X,pl.DataFrame):
        X = X.to_pandas()
    
    # Check if X is an instance of pd.DataFrame class
    if not isinstance(X,pd.DataFrame):
        raise TypeError(
        f"{type(X)} is not supported. Please convert to a DataFrame with "
        "pd.DataFrame. For more information see: "
        "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html")
    # Weighted average and standard deviation with missing values
    def weighted_avg_and_std(X, weights):
        average = np.ma.average(np.ma.array(X.values, mask=X.isnull().values), weights=weights, axis=0)
        # Compute the différence
        Y = (X - average.reshape(1,-1))**2
        variance = np.ma.average(np.ma.array(Y.values, mask=Y.isnull().values), weights=weights, axis=0)
        return (average, np.sqrt(variance))
    
    def impute(X,
               n_components=5,
               standardize=True,
               method=None,
               threshold=1e-6,
               random_state = None,
               init = 1,
               max_iter = 1000,
               ind_weights = None,
               ind_sup = None,
               quanti_sup = None,
               quali_sup = None,
               ridge_coef = 1,
               n_rowX = None,
               n_colX = None):
        #####
        nb_iter = 1

        old = math.inf
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)

        # Define number of components
        ncp = min(n_components, X.shape[1], X.shape[0] - 1)

        # Find indice where missing
        missings = [x for x, y in enumerate(get_melt(X, dropna=False)["value"]) if pd.isna(y)]

        # Compute average mean and average standard error
        d1 = weighted_avg_and_std(X=X,weights=ind_weights)

        # Initializations - scale data
        means = d1[0].reshape(1,-1)

        if standardize:
            std = d1[1].reshape(1,-1)
        else:
            std = np.ones(X.shape[1]).reshape(1,-1)
        # Z = (X - mu)/sigma
        Z = (X - means)/std

        ######## Replace supplementary quantitatives columns by a value approx 0
        if quanti_sup is not None:
            Z.loc[:,quanti_sup] = Z.loc[:,quanti_sup]*1e-08
        
        ##### Replace NA values by 0
        if X.isnull().any().any():
            Z = Z.fillna(0)

        #### Initialize missing values by normal distribution
        if init > 1:
            fill_value = np.random.normal(size=len(missings))
            # Apply pivot longer
            Z = get_melt(Z, dropna=False)
            Z.iloc[missings,2] = fill_value
            # Apply pivot wider
            Z = Z.pivot(index="Var1",columns="Var2",values="value")
            # Reordered using index and columns
            Z = Z.loc[X.index,X.columns]
        
        ## Store data
        fittedX = Z.copy()

        if ncp == 0:
            nb_iter = 0
        
        while nb_iter > 0:
            ## melt
            Z = get_melt(Z, dropna=False)
            fittedX = get_melt(fittedX, dropna=False)
            Z.iloc[missings,2] = fittedX.iloc[missings,2]
            # Apply pivot wider
            Z = Z.pivot(index="Var1",columns="Var2",values="value")
            fittedX = fittedX.pivot(index="Var1",columns="Var2",values="value")
            # Reordered using index and columns
            Z = Z.loc[X.index,X.columns]
            fittedX = fittedX.loc[X.index,X.columns]
            
            if quanti_sup is not None:
                Z.loc[:,quanti_sup] = Z.loc[:,quanti_sup]*1e+08
            # Multiply by standard deviation
            if standardize:
                Z = Z*std
            # Add means
            Z = Z + means

            #### Update means and std
            d1 = DescrStatsW(Z,weights=ind_weights,ddof=0)
            means = d1.mean.reshape(1,-1)
            if standardize:
                std = d1.std.reshape(1,-1)
            else:
                std = np.ones(Z.shape[1]).reshape(1,-1)
            Z = (Z - means)/std

            if quanti_sup is not None:
                Z.loc[:,quanti_sup] = Z.loc[:,quanti_sup]*1e-08
            
            ### Singular Value Decomposition
            svd = svd_triplet(Z,row_weights=ind_weights,n_components=ncp)
            sigma2 = ((n_rowX*n_colX)/min(n_colX, n_rowX - 1))*np.sum((svd["vs"][ncp:]**2)/((n_rowX-1)*(n_colX-ncp) - n_colX*ncp + ncp**2))
            # Update sigma2
            sigma2 = min(sigma2*ridge_coef,svd["vs"][ncp]**2)
            # Update sigma2 using expectation maximizatiopn
            if method == "em":
                sigma2 = 0
            # Shrinkage lambda
            shrinked_lambda = (svd["vs"][:ncp]**2 - sigma2)/svd["vs"][:ncp]
            # Update fitted dataframe
            U = np.apply_along_axis(func1d=lambda x : x*ind_weights,axis=0,arr=svd["U"][:,:ncp])
            U = np.apply_along_axis(func1d=lambda x : x*shrinked_lambda,axis=1,arr=U)
            fittedX = pd.DataFrame(np.dot(U,svd['V'][:,:ncp].T),index = X.index,columns=X.columns)
            ####
            fittedX = fittedX.apply(lambda x : x/ind_weights, axis=0)
            # Compute the difference
            diff = Z - fittedX
            ########### Replace index of differnce by zero
            diff = get_melt(diff, dropna=False)
            diff.iloc[missings,2] = 0
            # Apply pivot wider
            diff = diff.pivot(index="Var1",columns="Var2",values="value")
            # Reordered using index and columns
            diff = diff.loc[X.index,X.columns]
            ##### Compute objective
            objective = diff.apply(lambda x : (x**2)*ind_weights, axis=0).sum().sum()
            ###### Define criterion
            criterion   = np.abs(1 - objective/old)
            old = objective
            nb_iter = nb_iter + 1
            if not pd.isna(criterion):
                if criterion < threshold and (nb_iter > 5):
                    nb_iter = 0
                if objective < threshold and (nb_iter > 5):
                    nb_iter = 0
            if nb_iter > max_iter:
                nb_iter = 0
                print(F"Stopped after {max_iter} itérations")
        
        # End of loop
        if quanti_sup is not None:
            Z.loc[:,quanti_sup] = Z.loc[:,quanti_sup]*1e+08
        # Multiply by standard deviation
        if standardize:
            Z = Z*std
        # Add means
        Z = Z + means

        ##### Complete observations
        completeObs = X.copy()
        completeObs = get_melt(completeObs, dropna=False)
        Z = get_melt(Z, dropna=False)
        completeObs.iloc[missings,2] = Z.iloc[missings,2]
        # Apply pivot wider
        completeObs = completeObs.pivot(index="Var1",columns="Var2",values="value")
        Z = Z.pivot(index="Var1",columns="Var2",values="value")
        # Reordered using index and columns
        completeObs = completeObs.loc[X.index,X.columns]
        Z = Z.loc[X.index,X.columns] 

        # Update supplementary quantitative columns
        if quanti_sup is not None:
            fittedX.loc[:,quanti_sup] = fittedX.loc[:,quanti_sup]*1e+08
        if standardize:
            fittedX = fittedX*std
        # Add means
        fittedX = fittedX + means

        return {"completeObs" : completeObs, "fittedX" : fittedX}

    ############################
    # Check is quali sup
    if quali_sup is not None:
        if (isinstance(quali_sup,int) or isinstance(quali_sup,float)):
            quali_sup_idx = [int(quali_sup)]
        elif ((isinstance(quali_sup,list) or isinstance(quali_sup,tuple))  and len(quali_sup)>=1):
            quali_sup_idx = [int(x) for x in quali_sup]
    else:
        quali_sup_idx = quali_sup

    #  Check if quanti sup
    if quanti_sup is not None:
        if (isinstance(quanti_sup,int) or isinstance(quanti_sup,float)):
            quanti_sup_idx = [int(quanti_sup)]
        elif ((isinstance(quanti_sup,list) or isinstance(quanti_sup,tuple))  and len(quanti_sup)>=1):
            quanti_sup_idx = [int(x) for x in quanti_sup]
    else:
        quanti_sup_idx = quanti_sup

    # Check if individuls supplementary
    if ind_sup is not None:
        if (isinstance(ind_sup,int) or isinstance(ind_sup,float)):
            ind_sup_idx = [int(ind_sup)]
        elif ((isinstance(ind_sup,list) or isinstance(ind_sup,tuple)) and len(ind_sup)>=1):
            ind_sup_idx = [int(x) for x in ind_sup]
    else:
        ind_sup_idx = ind_sup
    
     # Set individuals weight
    if ind_weights is None:
        ind_weights = np.ones(X.shape[0])/X.shape[0]
    elif not isinstance(ind_weights,list):
        raise ValueError("'ind_weights' must be a list of individuals weights.")
    elif len(ind_weights) != X.shape[0]:
        raise ValueError(f"'ind_weights' must be a list with length {X.shape[0]}.")
    else:
        ind_weights = np.array([x/np.sum(ind_weights) for x in ind_weights])
    
    ################################################ Store original data #############################
    Xtot = X.copy()

    # Check if 
    if quali_sup is not None:
        # Remove qualitative columns
        X = X.drop(columns=[name for i, name in enumerate(Xtot.columns.tolist()) if i in quali_sup_idx])
        X = X.astype("float")
    
    # Find Supplementary quantitatives labels
    if quanti_sup is not None:
        quanti_sup_label = [name for i, name in enumerate(Xtot.columns.tolist()) if i in quanti_sup_idx]
    else:
        quanti_sup_label = None
    
    # Number of active rows
    if ind_sup is not None:
        n_row = X.shape[0] - len(ind_sup_idx)
    else:
        n_row = X.shape[0]
    
    # Number of number
    if quanti_sup is not None: 
        n_col = X.shape[1] - len(quanti_sup_idx)
    else:
        n_col = X.shape[1]
    
    # Check if method in list
    if method not in ["em","regularized"]:
        raise ValueError("'method' should be one of 'em', 'regularized'")
    
    # Check if n_components too large
    if n_components > min(X.shape[0] - 2, X.shape[1] - 1 ):
        raise ValueError("ncp is too large")
    # define individuals weights
    if ind_sup is not None:
        ind_weights[ind_sup_idx] = ind_weights[ind_sup_idx]*1e-8

    ###
    obj = math.inf
    for i in range(nb_init):
        if not X.isnull().any().any():
            X = X
        res_impute = impute(X=X,
                            n_components=n_components,
                            standardize=standardize,
                            method=method,
                            threshold=threshold,
                            random_state = random_state,
                            init = i+1,
                            max_iter = max_iter,
                            ind_weights = ind_weights,
                            ind_sup = ind_sup_idx,
                            quanti_sup = quanti_sup_label,
                            quali_sup = quali_sup_idx,
                            ridge_coef = ridge_coef,
                            n_rowX = n_row,
                            n_colX = n_col)
        #####
        
        X1, X2 = get_melt(res_impute["fittedX"],dropna=False), get_melt(X,dropna=False)
        missings = [x for x, y in enumerate(X2["value"]) if pd.isna(y)]
        # Update 
        X1, X2 = X1.loc[X1.index.difference(missings), :], X2.loc[X2.index.difference(missings), :]
        if np.mean((X1["value"] - X2["value"])**2)<obj:
            obj = np.mean((X1["value"] - X2["value"])**2)
    # Update global data
    if quali_sup is not None:
        Xtot.loc[res_impute["completeObs"].index,res_impute["completeObs"].columns] = res_impute["completeObs"]
        res_impute["completeObs"] = Xtot
    return res_impute