# -*- coding: utf-8 -*-
import numpy.ma as npma
from numpy import ndarray, array, ones, repeat,isnan,average
from pandas import DataFrame, Series
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

# interns function
from ..functions._get_sup_label import get_sup_label
from ..functions._utils import check_X_valid, check_param_valid,to_category
from ..others._disjunctive_na import disjunctive_NA
from ..others._disjunctive_prop import disjunctive_prop
from ..others._gsvd import gSVD
from ..others._from_disjunctive import from_disjunctive

class MCAimputer(BaseEstimator,TransformerMixin):
    """
    Impute dataset with Multiple Correspondence Analysis (MCAimputer)
    
    Impute the missing values of a categorical dataset using Multiple Correspondence Analysis (MCA).
    
    Parameters
    ----------
    ncp : int, default = 2
        The number of components used to predict the missing values.
    
    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.
        
    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columns weights. The weights are given only for the active columns.
        
    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary variables (continuous and/or categorical).
        
    method : {"em","regularized"}, default = "em"
        The algorithm to used. Possible values are:
        
        * "em" for expectation - maximization algorithm
        * "regularized" for regularized iterative algorithm
        
    lambda_ : int, default = 1
        1 by default to perform the regularized MCAimputer algorithm; useful only if ``method`` = "regularized". 
        Other regularization terms can be implemented by setting the value to less than 1 in order to regularized less 
        (to get closer to the results of the em method) or more than 1 to regularized more (to get closer to the results of the proportion imputation).
    
    threshold : float, default = 1e-6
        The threshold for assessing convergence.

    max_iter : int, default = 1000
        The maximum number of iteration for the algorithm.
    
    random_state : int, default = None 
        If None, implies that missing values are initially imputed by the proportion of the category for 
        the categorical variables coded with disjunctive matrix. Other values leads to a random initialization.
    
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
        row_w : Series of shape (n_samples,)
            The samples weights.
        var_w : Series of shape (n_columns,)
            The variables weights.
        col_w : Series of shape (n_levels,)
            The columns (levels) weights.
        ncp : int
            The number of components kepted.
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).

    complete_ : DataFrame of shape (n_samples + n_samples_plus, n_columns + n_columns_sup)
        The categorical imputed dataset; the observed values are kept for the non-missing entries and the missing values 
        are replaced by the predicted ones. Missing values are imputed with the most plausible categories according to the 
        values in the ``dummies_`` output.

    dummies_ : DataFrame of shape (n_samples, n_levels)
        The imputed indicator matrix. the observed values are kept for the non-missing entries and the missing values 
        are replaced by the predicted ones. The imputed values are real numbers but they met the constraint 
        that the sum of the entries corresponding to one individual and one variable is equal to one. 
        
    References
    ----------
    [1] Anne Gégout-Petit, Myriam Maumy-Bertrand, Gilbert Saporta, Christine Thomas-Agnan (Dir.). `Données manquantes <https://www.editionstechnip.com/en/catalogue-detail/2344/donnees-manquantes.html>`_. Editions Technip, 2022, 978-2-7108-1195-4. ⟨hal-03696270⟩
        
    [2] Josse, J, Husson, F & Pagès J (2009). `Gestion des données manquantes en Analyse en Composantes Principales <https://www.numdam.org/item/JSFS_2009__150_2_28_0/>`_. Journal de la société française de la statistique, Volume 150 (2009). no. 2. pp 28-51
    
    [3] Josse, J & Husson, F. (2013). `Handling missing values in exploratory multivariate data analysis methods <https://www.numdam.org/item/JSFS_2012__153_2_79_0/>`_. Journal de la société française de la statistique. Volume 153 (2012), no. 2. pp. 79-99.
    
    [4] Josse, J. and Husson, F. missMDA (2016). missMDA : A Package for Handling Missing Values in Multivariate Data Analysis. Journal of Statistical Software, 70 (1), pp 1-31 `<doi:10.18637/jss.v070.i01> <https://www.jstatsoft.org/article/view/v070i01>`_.
    
    Examples
    --------
    >>> from missmdatools.datasets import vnf
    >>> from missmdatools import MCAimputer
    >>> clf = MCAimputer(ncp=4)
    >>> clf.fit(vnf)
    MCAimputer(ncp=4)
    """
    def __init__(
            self,
            ncp = 2,
            row_w = None,
            col_w = None,
            ind_sup = None,
            sup_var = None,
            method = "em",
            lambda_ = 1,
            threshold = 1e-6,
            max_iter = 1000,
            random_state = None,
            tol = 1e-7
    ):
        self.ncp = ncp
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.sup_var = sup_var
        self.method = method
        self.lambda_ = lambda_
        self.threshold = threshold
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def fit(self,X,y=None):
        """Fit the imputer to X
        
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Input data, where ``n_samples`` is the number of samples and
            ``n_columns`` is the number of columns.

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
        check_param_valid(ncp=self.ncp,method=self.method,lambda_=self.lambda_)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # select object, string and convert to category
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        X = to_category(X=X,ordered=True)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_sup_label = get_sup_label(X=X,indexes=self.ind_sup,axis=0)
        sup_var_label = get_sup_label(X=X,indexes=self.sup_var,axis=1)
        
        # make a copy of input data
        Xtot = X.copy()
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop supplementary variables
        if self.sup_var is not None:
            X = X.drop(columns=sup_var_label)
        
        # drop supplementary individuals
        if self.ind_sup is not None:
            X = X.drop(index=ind_sup_label)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if any missing values in X
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
        if not X.isnull().any().any():
            raise TypeError("X must contains missing values.")
        
        # set shape of X
        n_rows, n_cols = X.shape
        
        # set individuals
        if self.row_w is None: 
            row_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("row_w must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"row_w must be a 1d array-like of shape ({n_rows},).")
        else: 
            row_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")
            
        #set variables weights
        if self.col_w is None: 
            var_w = Series(ones(n_cols)/n_cols,index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("'col_w' must be a 1d array-like of variables weights.")
        elif len(self.col_w) != n_cols: 
            raise ValueError(f"'col_w' must be a 1d array-like of shape ({n_cols},).")
        else: 
            var_w = Series(array(self.col_w)/sum(self.col_w),index=X.columns,name="weight")
        # repeat according to number of unique modlaities
        m_k = repeat(var_w.values,X.nunique())   
        
        # disjunctive table with NA
        Y = disjunctive_NA(X=X,prefix=True,sep="_")
        # where NA in disjunctive table
        mask = npma.masked_array(Y,mask=isnan(Y)).mask 
        # weighted average - proportion of levels
        p_k = npma.average(npma.array(Y, mask=mask), weights=row_w, axis=0)
        # weights of levels : w_kq = p_kq * m_kq
        col_w = p_k * m_k

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # imputation algorithm
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ncp == 0:
            # store number of components
            ncp = 0
            # fill NA with weigted average (proportion of each categories)
            completeY = Y.fillna(Series(p_k,index=Y.columns)) 
        else:
            # convert row weights to numpy array
            roww = row_w.to_numpy()
            passed = True
            iter = 0
            # initialize disjunctive matrix with - proportion or random
            fittedY = disjunctive_prop(X=X,prefix=True,random_state=self.random_state,row_w=row_w)
            Yold = fittedY.copy()
            while passed:
                # proportion of levels : n_kq/n
                p_k = average(a=fittedY,axis=0,weights=row_w)
                # check if any weight is negative
                if any(x < 0 for x in p_k):
                    raise ValueError(f"The algorithm fails to converge. Choose a number of components (ncp) less or equal than {ncp-1}",
                                     f"or a number of iterations (max_iter) less or equal than {self.max_iter-1}")
                # weights of levels : w_kq = p_kq * m_kq
                col_w = p_k * m_k
                # standardized data : z_ikq = (y_ikq/p_kq) - 1
                Z = (fittedY/p_k) - 1
                # generalized singular values decomposition (GSVD)
                svd = gSVD(X=Z,ncp=self.ncp,row_w=row_w,col_w=col_w,tol=self.tol)
                # set number of components to keep
                ncp = svd.ncp
                # noise variance - p127
                if self.method.lower() == "em":
                    # for expectation maximization
                    sigma2 = 0
                else:
                    # for regularized iterative algorithm
                    sigma2 = (svd.sumvp - sum(svd.vp[:ncp]))/(svd.rank - ncp) #p 128
                    # update sigma2
                    sigma2 = min(sigma2 * self.lambda_,svd.vp[ncp])
            
                # skrinkage parameter
                shrinkage = (svd.vp[:ncp] - sigma2)/svd.vs[:ncp] # p127
                # conver to 2D numpy array
                if ncp == 1:
                    U, V = svd.U[:,0][:,None], svd.V[:,0][:,None]
                else:
                    U, V = svd.U[:,:ncp], svd.V[:,:ncp]
                # reconstructed matrix
                recon = DataFrame(((U * shrinkage).dot(V.T) + 1) * p_k,index=Y.index,columns=Y.columns)
                # compute the difference - resid
                diff = recon - Yold
                # replace missing index of differnce by zero
                arr = diff.to_numpy().copy()
                arr[mask] = 0
                diff.iloc[:,:] = arr
                # compute objective
                obj = ((diff ** 2) * roww[:,None]).sum().sum()
                # replace by reconstructed
                Yold = recon.copy()
                # updated fitted values with reconstructed matrix
                arr = fittedY.to_numpy().copy()
                arr[mask] = recon.values[mask]
                fittedY.iloc[:,:] = arr
                # update number of iterations
                iter += 1
                # check if criterion is respected
                if not ((obj > self.threshold) & (iter < self.max_iter)):
                    passed = False
                    
            # complete Y
            completeY = Y.copy()       
            arr = completeY.to_numpy().copy()
            arr[mask] = fittedY.values[mask]
            completeY.iloc[:,:] = arr
        # convert to Series
        col_w = Series(col_w,index=Y.columns,name="weight")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # complete dataset
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # complete X
        completeX = from_disjunctive(X=completeY,sep="_",ordered=True)
        # complete Xtot
        completeXtot = Xtot.copy()
        completeXtot.loc[completeX.index,completeX.columns] = completeX.values
        # convert to category
        completeXtot = to_category(X=completeXtot)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set model attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # summary called informations                                                                                                                                                                                                                                rmations
        call_ = {"Xtot": Xtot, "X": X, "row_w": row_w, "var_w": var_w,"col_w": col_w, "ncp": ncp, 
                 "ind_sup": ind_sup_label, "sup_var": sup_var_label}
        self.call_ = namedtuple("call", call_.keys())(*call_.values())
        # complete dataset
        self.complete_ = completeXtot
        # complete disjunctive table
        self.dummies_ = completeY
        return self