# -*- coding: utf-8 -*-
from math import inf
import numpy.ma as npma
from warnings import warn
from numpy import where, isnan, abs, sum, outer,sqrt,random
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from sklearn.base import BaseEstimator,TransformerMixin

# intern functions
from ..functions._get_sup_label import get_sup_label
from ..functions._utils import check_X_valid, to_category
from ..others._replace_na import replace_na
from ..others._gsvd import gSVD

class CAimputer(BaseEstimator,TransformerMixin):
    """
    Impute contingency table with Correspondence Analysis (CAimputer)

    Impute the missing entries of a contingency table using Correspondence Analysis (CA). 
    It can be used as a preliminary step before performing a CA on an complete dataset. 
    The output of the algorithm can be used as an input of the CA function of the `scientisttools <https://scientisttools.readthedocs.io/en/latest/>`_ 
    package in order to perform CA on an incomplete dataset.
    
    Parameters
    ----------
    ncp : int, default = 2
        The number of components used to predict the missing values.
        
    row_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary rows.
        
    col_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary columns.

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary variables (continuous and/or categorical).
    
    threshold : float, default = 1e-8
        The threshold for assessing convergence.

    max_iter : int, default = 1000
        The maximum number of iteration for the regularized iterative CA algorithm.
        
    random_state : int, default = None 
        Random initialization.

    warn_message : bool, default = True
        If True, then raise a warning.
        
    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.
        
    Attributes
    ----------
    call_ : call
        An object containing the summary called parameters with the following attributes:

        Xtot : DataFrame of shape (n_samples + n_samples_sup, n_columns + n_columns_sup)
            Input data.
        X : DataFrame of shape (n_samples, n_columns)
            Active data.
        ncp : int
            The number of components kepted.
        row_sup : None, list
            The names of the supplementary rows.
        col_sup : None, list
            The names of the supplementary columns.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).

    complete_ : DataFrame of shape (n_rows + n_rows_sup, n_columns + n_columns_sup)
        The imputed dataset; the observed values are kept for the non-missing entries and the missing values 
        are replaced by the predicted ones in active dataset.
        
    fitted_ : DataFrame of shape (n_rows, n_columns)
        The predicted dataset.

    References
    ----------
    [1] Anne Gégout-Petit, Myriam Maumy-Bertrand, Gilbert Saporta, Christine Thomas-Agnan (Dir.). `Données manquantes <https://www.editionstechnip.com/en/catalogue-detail/2344/donnees-manquantes.html>`_. Editions Technip, 2022, 978-2-7108-1195-4. ⟨hal-03696270⟩
    
    [2] Josse, J, Husson, F & Pagès J (2009). `Gestion des données manquantes en Analyse en Composantes Principales <https://www.numdam.org/item/JSFS_2009__150_2_28_0/>`_. Journal de la société française de la statistique, Volume 150 (2009). no. 2. pp 28-51

    [3] Josse, J & Husson, F. (2013). `Handling missing values in exploratory multivariate data analysis methods <https://www.numdam.org/item/JSFS_2012__153_2_79_0/>`_. Journal de la société française de la statistique. Volume 153 (2012), no. 2. pp. 79-99.

    [4] Josse, J. and Husson, F. missMDA (2016). missMDA : A Package for Handling Missing Values in Multivariate Data Analysis. Journal of Statistical Software, 70 (1), pp 1-31 `<doi:10.18637/jss.v070.i01> <https://www.jstatsoft.org/article/view/v070i01>`_.

    Examples
    --------
    >>> from missmdatools.datasets import children
    >>> from missmdatools import CAimputer
    >>> clf = CAimputer()
    >>> clf.fit(children)
    CAimputer()
    """
    def __init__(
        self,
        ncp = 2, 
        row_sup = None,
        col_sup = None,
        sup_var = None,
        threshold = 1e-8,
        max_iter = 1000,
        random_state = None,
        warn_message = True,
        tol = 1e-7
    ):
        self.ncp = ncp
        self.row_sup = row_sup
        self.col_sup = col_sup
        self.sup_var = sup_var
        self.threshold = threshold
        self.max_iter = max_iter
        self.random_state = random_state
        self.warn_message = warn_message
        self.tol = tol

    def fit(self,X,y=None):
        """Fit the imputer to X
        
        Parameters
        ----------
        X : DataFrame of shape (n_rows, n_columns),
            Training data, where ``n_rows`` in the number of rows 
            and ``n_columns`` is the number of columns.
            ``X`` is a contingency table containing absolute frequencies.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if valid input data and parameters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_X_valid(X=X)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if ncp is not negative
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ncp < 0: 
            raise ValueError("ncp must be positive")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # select object, string and convert to category
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = to_category(X=X)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        row_sup_label = get_sup_label(X=X,indexes=self.row_sup,axis=0)
        col_sup_label = get_sup_label(X=X,indexes=self.col_sup,axis=1)
        sup_var_label = get_sup_label(X=X,indexes=self.sup_var,axis=1)
        
        # make a copy of input data
        Xtot = X.copy()
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop supplementary variables
        if self.sup_var is not None:
            X = X.drop(columns=sup_var_label)
            
        # drop supplementary columns
        if self.col_sup is not None:
            X = X.drop(columns=col_sup_label)
        
        # drop supplementary rows
        if self.row_sup is not None:
            X = X.drop(index=row_sup_label)
                
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if any missing values in X
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
        if not X.isnull().any().any():
            raise TypeError("X must contains missing values.")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if all elements in X are numerics
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not all(is_numeric_dtype(X[c]) for c in X.columns):
            raise TypeError("X must contains only numerics")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # impute contingency table
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set shape of X
        n_rows, n_cols = X.shape
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # find missings values
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # rows and columns indexes of missing values
        rows, cols = where(X.isna())
        # make a copy
        Xc = X.copy()
        # replace missing values by a random sample
        rng = random.default_rng(seed=self.random_state)
        value = Series(X.dropna().values.flatten()).sample(n=len(rows),random_state=rng).values
        # replace missing values by value
        Xc = replace_na(X=Xc,rows=rows,cols=cols,values=value,by_row=True)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # impute
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # random imputation when ncp = 0
        if self.ncp == 0:
            # store number of components
            ncp = 0
            # imputation with random choice
            completeX = Xc.copy()
            # reconstructed X
            fittedX = Xc.copy()
        else:
            # where NA 
            mask = npma.masked_array(X,mask=isnan(X)).mask
            iter = 1
            old_obj = inf
            obj = 0
            fittedX = Xc.copy()
            while iter > 0:
                # make a numpy copy of Xc
                arr = Xc.to_numpy().copy()
                arr[mask] = fittedX.values[mask]
                # check if negative replacement
                neg_mask = arr[mask] <= 0
                # replace negative value by zero
                arr[mask][neg_mask] = 0
                # replace values where missing with reconstructed (fitted) values
                Xc.iloc[:,:] = arr
                
                # compute row sum and colums sum
                row_s, col_s = Xc.sum(axis=1), Xc.sum(axis=0)
                # check criteria
                if ((sum(where(row_s > 1e-06,1,0)) == n_rows) and
                    (sum(where(col_s > 1e-06,1,0)) == n_cols)):
                    
                    ## Applied Correspondence Analysis on X where missing values are replaced by fitted values
                    # total - sum of all elements
                    total = Xc.sum().sum()
                    # table of frequencies
                    P = Xc/total
                    # rows and columns margins
                    row_m, col_m = P.sum(axis=1).to_numpy(), P.sum(axis=0).to_numpy()
                    # scale the data
                    Z = (((P - outer(row_m,col_m))/sqrt(col_m))/sqrt(row_m[:,None]))
                    # generalized singular value decomposition (gSVD)
                    svd = gSVD(X=Z,ncp=self.ncp,tol=self.tol)
                    # set number of components to keep
                    ncp = svd.ncp
                    # noise variance
                    sigma2 = (svd.sumvp - sum(svd.vp[:ncp]))/((n_rows - 1 - ncp)*(n_cols - 1 - ncp)) #
                    # shrinkage parameter
                    shrinkage = (svd.vp[:ncp] - n_rows*(n_cols/min(n_cols,n_rows - 1))*sigma2)/svd.vs[:ncp]
                    if ncp == 1:
                        U = svd.U[:,0][:,None] * shrinkage
                        V = svd.V[:,0][:,None]
                    else:
                        U = svd.U[:,:ncp]
                        V = svd.V[:,:ncp] * shrinkage
                    # rescale => reconstruction of X => fitted
                    fittedX = total * (((U.dot(V.T) * sqrt(col_m)) * sqrt(row_m[:,None])) + outer(row_m, col_m))
                    # convert to DataFrame
                    fittedX = DataFrame(fittedX,index=X.index,columns=X.columns)
                                
                # compute the difference
                diff = Xc - fittedX
                # replace missing index of differnce by zero
                arr_diff = diff.to_numpy().copy()
                arr_diff[mask] = 0
                # compute objective - sum of squared differences
                obj = sum(sum(arr_diff**2))
                # define criteria
                crit = abs(1 - obj/old_obj)
                old_obj = obj
                # update number of iterations
                iter += 1
                # check if convergence
                if not isnan(crit):
                    if crit < self.threshold and (iter > 5):
                        iter = 0
                    if obj < self.threshold and (iter > 5):
                        iter = 0
                if iter > self.max_iter:
                    iter = 0
                    if self.warn_message:
                        warn(F"Stopped after {self.max_iter} itérations")
                    
            # active dataset
            completeX = X.copy()
            arr = completeX.to_numpy().copy()
            arr[mask] = fittedX.values[mask]
            completeX.iloc[:,:] = arr

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # complete dataset
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # complete Xtot
        completeXtot = Xtot.copy()
        completeXtot.loc[X.index,X.columns] = completeX.values
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set model attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ssummary called informations
        call_ = {"Xtot": Xtot, "X": X, "ncp": ncp,"row_sup": row_sup_label, "col_sup": col_sup_label, "sup_var": sup_var_label}
        self.call_ = namedtuple("call", call_.keys())(*call_.values())
        # complete dataset
        self.complete_ = completeXtot
        # complete active dataset
        self.fitted_ = fittedX
        return self