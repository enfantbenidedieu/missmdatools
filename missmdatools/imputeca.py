# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import polars as pl
import random

from .get_melt import get_melt

def imputeCA(X,
             n_components=5, 
             row_sup = None,
             col_sup = None,
             quanti_sup = None,
             quali_sup = None,
             random_state = None,
             threshold = 1e-08,
             max_iter = 1000):
    """
    Impute contingency table with Correspondence Analysis (imputeCA)
    ----------------------------------------------------------------

    Description
    -----------
    Impute the missing entries of a contingency table using Correspondence Analysis (CA). 
    Can be used as a preliminary step before performing CA on an incomplete dataset.
    
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    
    # Cree a function to used
    def skrinkageCA(X,n_components=5,row_sup=None,col_sup = None):

        #### Apply
        if row_sup is not None:
            X.loc[row_sup,:] = X.loc[row_sup,:]*1e-08
        if col_sup is not None:
            X.loc[:,col_sup] = X.loc[:,col_sup]*1e-08

        # Compute joint distribution
        joint_dist = X/X.sum().sum()
        
        # Rows sums
        row_sum = joint_dist.sum(axis=1)
        # Columns sums
        col_sum = joint_dist.sum(axis=0)
        #
        S = joint_dist - np.dot(row_sum.values.reshape(len(row_sum),1), col_sum.values.reshape(1,len(col_sum)))
        S = S.apply(lambda x : x/row_sum,axis=0).apply(lambda x : x/col_sum,axis=1)
        # Singular value decomposition
        svd = np.linalg.svd(S)

        # Define number of rows and number of columns
        if row_sup is not None:
            n_rows = X.shape[0] - len(row_sup)
        else:
            n_rows = X.shape[0]
        
        if col_sup is not None:
            n_cols = X.shape[1] - len(col_sup)
        else:
            n_cols = X.shape[1]

        sigma2 = np.sum(svd[1][n_components:]**2)/((n_rows - 1)*(n_cols - 1) - (n_rows - 1)*n_components - (n_cols - 1)*n_components + n_components**2)
        shrinked_lambda = (svd[1][:n_components]**2 - n_rows*(n_cols/min(n_cols,(n_rows-1)))*sigma2)/svd[1][:n_components]
        if n_components == 1:
            U = svd[0][:,0]*shrinked_lambda[0]
            V = svd[2].T[:,0]
            recon = np.dot(U.reshape(len(U),1),V.reshape(1,len(V)))
        else:
            V = np.apply_along_axis(lambda x : x*shrinked_lambda,axis=1,arr=svd[2].T[:,:n_components])
            recon = np.dot(svd[0][:,:n_components],V.T)
        # Update 
        recon = np.apply_along_axis(func1d=lambda x : x*np.sqrt(row_sum.values),axis=0,arr=recon)
        recon = np.apply_along_axis(func1d=lambda x : x*np.sqrt(col_sum.values),axis=1,arr=recon)
        recon = X.sum().sum()*(recon + np.dot(row_sum.values.reshape(len(row_sum),1), col_sum.values.reshape(1,len(col_sum))))

        #### Transform to pandas dataframe
        recon = pd.DataFrame(recon,index=X.index,columns=X.columns)
        if row_sup is not None:
            row_sup.loc[row_sup,:] = row_sup.loc[row_sup,:]*1e+08
        if col_sup is not None:
            recon.loc[:,col_sup] = recon.loc[:,col_sup]*1e+08

        return recon

    ##############################################################################################
    # Extract supplementary rows
    ##############################################################################################
    if row_sup is not None:
        if (isinstance(row_sup,int) or isinstance(row_sup,float)):
            row_sup_idx = [int(row_sup)]
        elif (isinstance(row_sup,list) or isinstance(row_sup,tuple)) and len(row_sup) >=1:
            row_sup_idx = [int(x) for x in row_sup]
        row_sup_label= X.index[row_sup_idx]
    else:
        row_sup_label = row_sup
    
    ##############################################################################################
    # Extract supplementary columns
    ##############################################################################################
    if col_sup is not None:
        if (isinstance(col_sup,int) or isinstance(col_sup,float)):
            col_sup_idx = [int(col_sup)]
        elif (isinstance(col_sup,list) or isinstance(col_sup,tuple)) and len(col_sup) >=1:
            col_sup_idx = [int(x) for x in col_sup]
        col_sup_label = X.columns[col_sup_idx]
    else:
        col_sup_label = col_sup

    ##############################################################################################
    # Extract supplementary qualitatives and put in list
    ##############################################################################################
    if quali_sup is not None:
        if (isinstance(quali_sup,int) or isinstance(quali_sup,float)):
            quali_sup_idx = [int(quali_sup)]
        elif (isinstance(quali_sup,list) or isinstance(quali_sup,tuple)) and len(quali_sup) >=1:
            quali_sup_idx = [int(x) for x in quali_sup]
        quali_sup_label = X.columns[quali_sup_idx]

    ##############################################################################################
    # Extract supplementary quantitatives and put in list
    ##############################################################################################
    if quanti_sup is not None:
        if (isinstance(quanti_sup,int) or isinstance(quanti_sup,float)):
            quanti_sup_idx = [int(quanti_sup)]
        elif (isinstance(quanti_sup,list) or isinstance(quanti_sup,tuple)) and len(quanti_sup) >=1:
            quanti_sup_idx = [int(x) for x in quanti_sup]
        quanti_sup_label = X.columns[quanti_sup_idx]
    
    #####################################################################################################
    # Store data - Save the base in a variables
    #####################################################################################################
    Xtot = X.copy()
    
    ################################# Drop supplementary quantitatives variables ###############################
    if quanti_sup is not None:
        X = X.drop(columns=quanti_sup_label)
    
        ################################# Drop supplementary qualitatives variables ###############################
    if quali_sup is not None:
        X = X.drop(columns=quali_sup_label)
    
    ###### Check if missing values
    if not X.isnull().any().any():
        raise TypeError("No value is value")
    
    # Find indice where missing
    missings = [x for x, y in enumerate(get_melt(X, dropna=False)["value"]) if pd.isna(y)]
    
    # Make a copy
    Z = X.copy()
    # Replace missing values by a random sample
    if random_state is not None:
        np.random.seed(random_state)
    fill_value = np.array(random.sample(get_melt(X,dropna=True)["value"].values.tolist(),len(missings)))+1
    # Apply pivot longer
    Z = get_melt(Z, dropna=False)
    Z.iloc[missings,2] = fill_value
    # Apply pivot wider
    Z = Z.pivot(index="Var1",columns="Var2",values="value")
    # Reordered using index and columns
    Z = Z.loc[X.index,X.columns]

    ###########################################
    nb_iter = 1
    old = math.inf
    objective = 0
    recon  = Z.copy()
    while nb_iter > 0:
        Z = get_melt(Z, dropna=False)
        recon = get_melt(recon, dropna=False)
        Z.iloc[missings,2] = recon.iloc[missings,2]
        # Replace negative value by zero
        Z.iloc[missings,2][Z.iloc[missings,2] <= 0] = 0
        # Apply pivot wider
        Z = Z.pivot(index="Var1",columns="Var2",values="value")
        recon = recon.pivot(index="Var1",columns="Var2",values="value")
        # Reordered using index and columns
        Z = Z.loc[X.index,X.columns]
        recon = recon.loc[X.index,X.columns]

        ### Compute row sum and colums sum
        row_sum = Z.sum(axis=1)
        col_sum = Z.sum(axis=0)

        if (np.sum(np.where(row_sum > 1e-06,1,0)) == X.shape[0]) & (np.sum(np.where(col_sum > 1e-06,1,0)) == X.shape[1]):
            recon = skrinkageCA(X=Z,n_components=n_components,row_sup=row_sup_label,col_sup=col_sup_label)
        
        # Compute the difference
        diff = Z - recon
        ########### Replace index of differnce by zero
        diff = get_melt(diff, dropna=False)
        diff.iloc[missings,2] = 0
        # Apply pivot wider
        diff = diff.pivot(index="Var1",columns="Var2",values="value")
        # Reordered using index and columns
        diff = diff.loc[X.index,X.columns]
        ##### Compute objective
        objective = (diff**2).sum().sum()
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

    return completeObs