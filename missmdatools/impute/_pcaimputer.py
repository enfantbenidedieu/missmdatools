# -*- coding: utf-8 -*-
from math import inf
import numpy.ma as npma
from warnings import warn
from numpy import ndarray,array,ones,where,random,isnan,sqrt,average
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

# interns functions
from ..functions._get_sup_label import get_sup_label
from ..functions._utils import check_X_valid, check_param_valid, to_category
from ..others._replace_na import replace_na
from ..others._gsvd import gSVD

class PCAimputer(BaseEstimator,TransformerMixin):
    """
    Impute continuous dataset with Principal Component Analysis (PCAimputer)
        
    Impute the missing values of a dataset of continuous variables with the Principal Components Analysis (PCA) model. 
    It can be used as a preliminary step before performing a PCA on an complete dataset. 
    The output of the algorithm can be used as an input of the PCA function of the `scientisttools <https://scientisttools.readthedocs.io/en/latest/>`_ package in order to perform PCA on an incomplete dataset.

    Parameters
    ----------
    scale_unit : bool, default = True
        If True, then the data are scaled to unit variance.
    
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
        1 by default to perform the regularized PCAimputer algorithm; useful only if method = "regularized". 
        Other regularization terms can be implemented by setting the value to less than 1 in order to regularized less 
        (to get closer to the results of the "em" method) or more than 1 to regularized more (to get closer to the results of the mean imputation)

    threshold : float, default = 1e-6
        The threshold for assessing convergence.

    max_iter : int, default = 1000
        The maximum number of iteration for the algorithm.
    
    random_state : int, default = None 
        If None, implies that missing values are initially imputed by the mean of each variable. Other values leads to a random initialization
            
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
        row_w : Series of shape (n_samples,)
            The samples weights.
        col_w : Series of shape (n_columns,)
            The columns weights.
        ncp : int
            The number of components kepted.
        center : Series of shape (n_columns,)
            The variables weighted average.
        scale : Series of shape (n_columns,)
            The variables weighted standard deviation.
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).

    complete_ : DataFrame of shape (n_samples + n_samples_plus, n_columns + n_columns_sup)
        The imputed dataset; the observed values are kept for the non-missing entries and the missing values 
        are replaced by the predicted ones.
        
    fitted_ : DataFrame of shape (n_samples, n_columns)
        Reconstructed data.

    References
    ----------
    [1] Anne Gégout-Petit, Myriam Maumy-Bertrand, Gilbert Saporta, Christine Thomas-Agnan (Dir.). `Données manquantes <https://www.editionstechnip.com/en/catalogue-detail/2344/donnees-manquantes.html>`_. Editions Technip, 2022, 978-2-7108-1195-4. ⟨hal-03696270⟩
        
    [2] Josse, J, Husson, F & Pagès J (2009). `Gestion des données manquantes en Analyse en Composantes Principales <https://www.numdam.org/item/JSFS_2009__150_2_28_0/>`_. Journal de la société française de la statistique, Volume 150 (2009). no. 2. pp 28-51
    
    [3] Josse, J & Husson, F. (2013). `Handling missing values in exploratory multivariate data analysis methods <https://www.numdam.org/item/JSFS_2012__153_2_79_0/>`_. Journal de la société française de la statistique. Volume 153 (2012), no. 2. pp. 79-99.
    
    [4] Josse, J. and Husson, F. missMDA (2016). missMDA : A Package for Handling Missing Values in Multivariate Data Analysis. Journal of Statistical Software, 70 (1), pp 1-31 `<doi:10.18637/jss.v070.i01> <https://www.jstatsoft.org/article/view/v070i01>`_.
    
    Examples
    --------
    >>> from missmdatools.datasets import orange
    >>> from missmdatools import PCAimputer
    >>> clf = PCAimputer(method="regularized")
    >>> clf.fit(orange)
    PCAimputer(method="regularized")
    """
    def __init__(
            self,
            scale_unit = True,
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
            warn_message = True, 
            tol = 1e-7
    ):
        self.ncp = ncp
        self.scale_unit = scale_unit
        self.method = method
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.sup_var = sup_var
        self.lambda_ = lambda_
        self.threshold = threshold
        self.max_iter = max_iter
        self.random_state = random_state
        self.warn_message = warn_message
        self.tol = tol
        
    def fit(self,X,y=None):
        """Fit the imputer to X.

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
        X = to_category(X=X)
        
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
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if all elements in X are numerics
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not all(is_numeric_dtype(X[c]) for c in X.columns):
            raise TypeError("X must contains only numerics")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # data preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set shape of X
        n_rows, n_cols = X.shape
            
        # set individuals weights
        if self.row_w is None: 
            row_w = Series(ones(n_rows)/n_rows,index=X.index,name="weight")
        elif not isinstance(self.row_w,(list,tuple,ndarray,Series)): 
            raise TypeError("row_w must be a 1d array-like of individuals weights.")
        elif len(self.row_w) != n_rows: 
            raise ValueError(f"row_w must be a 1d array-like of shape ({n_rows},).")
        else: 
            row_w = Series(array(self.row_w)/sum(self.row_w),index=X.index,name="weight")
            
        # set columns (variables) weights
        if self.col_w is None:
            col_w = Series(ones(n_cols),index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("col_w must be a 1d array-like of columns weights.")
        elif len(self.col_w) != n_cols: 
            raise ValueError(f"col_w must be a 1d array-like of shape ({n_cols},).")
        else:
            col_w = Series(array(self.col_w),index=X.columns,name="weight")
            
        # rows and columns indexes of missing values
        rows, cols = where(X.isna())
        # where NA 
        mask = npma.masked_array(X,mask=isnan(X)).mask
        
        # initializations -weighted average
        center = npma.average(npma.array(X, mask=mask), weights=row_w, axis=0)
        # weighted standard deviation
        if self.scale_unit:
            scale = sqrt(npma.average(npma.array((X - center)**2, mask=mask), weights=row_w, axis=0))
        else:
            scale = ones(n_cols)
    
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # imputation missing values using principal component analysis
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # mean imputation when ncp = 0
        if self.ncp == 0:
            # store number of components
            ncp = 0
            # fill NA with weigted average
            completeX = X.fillna(Series(center,index=X.columns))
            # fitted X
            fittedX = completeX.copy()
        else:
            # convert weights to numpy array
            roww, colw = row_w.to_numpy(), col_w.to_numpy()
            # standardization : z_ik = (x_ik - m_k)/s_k
            Z = (X - center)/scale
            # initialization with mean
            if self.random_state is None:
                # fill missing values with 0
                Z = Z.fillna(0)
            # initialization with random normal values
            else:
                # generator
                rng = random.default_rng(seed=self.random_state)
                # random values for missing values
                values = random.normal(size=len(rows),loc=0,scale=1)
                # fill missing values with random values
                Z = replace_na(X=Z,rows=rows,cols=cols,values=values,by_row=True)
                
            # multiply by squared of weight of columns
            Z *= sqrt(col_w)
            
            # initial fitted values
            old_obj, obj = inf, 0
            fittedZ = Z.copy()
            iter = 1
            while iter > 0:
                # replace values where missing with reconstructed (fitted) values
                arr = Z.to_numpy().copy()
                arr[mask] = fittedZ.values[mask]
                Z.iloc[:,:] = arr
                # reconstruct the standardized data
                Z = (Z * scale) + center
                # update weighted average and weighted standard deviation  
                center = average(a=Z,axis=0,weights=row_w)
                if self.scale_unit:
                    scale = sqrt(average(a=(Z - center)**2,axis=0,weights=row_w))
                else:
                    scale = ones(n_cols)
                # standardization and multiply by the columns weights : z_ik = ((x_ik - m_k)/s_k) * sqrt(w_k)
                Z = ((Z - center) / scale) * sqrt(col_w)
                # generalized singular value decomposition (gSVD)
                svd = gSVD(X=Z,row_w=row_w,ncp=self.ncp,tol=self.tol)
                # set number of components to keep
                ncp = svd.ncp
                # noise variance - p108
                if self.method.lower() == "em":
                    # for expectation maximization
                    sigma2 = 0
                else:
                    # for regularized iterative algorithm
                    sigma2 = ((n_rows * n_cols)/min(n_cols, n_rows - 1))*(svd.sumvp - sum(svd.vp[:ncp]))/((n_rows - 1 - ncp)*(n_cols - ncp)) #p 112
                    # regularized sigma2
                    sigma2 = min(sigma2 * self.lambda_,svd.vp[ncp])
                
                # shrinkage parameter
                shrinkage = (svd.vp[:ncp] - sigma2)/svd.vs[:ncp] # pp 112 - 113
                # update fitted dataframe
                if ncp == 1:
                    U, V = svd.U[:,0][:,None], svd.V[:,0][:,None]
                else:
                    U, V = svd.U[:,:ncp], svd.V[:,:ncp]
                # reconstructed standardized data
                fittedZ = (((U * roww[:,None]) * shrinkage).dot(V.T) / roww[:,None]) / sqrt(colw)
                # convert to DataFrame
                fittedZ = DataFrame(fittedZ,index = Z.index,columns=Z.columns)
                #print(fittedZ.head(6))
                # compute the difference : standardized - reconstructed standardized
                diff = Z - fittedZ
                # replace missing index of difference by zero
                arr = diff.to_numpy().copy()
                arr[mask] = 0
                diff.iloc[:,:] = arr
                # compute objective
                obj = ((diff ** 2) * roww[:,None]).sum().sum()
                # define criterion
                crit = abs(1 - obj/old_obj)
                old_obj = obj
                # update number of iterations
                iter += 1
                # check if convergence
                if not isnan(crit):
                    if (crit < self.threshold) and (iter > 5):
                        iter = 0
                    if (obj < self.threshold) and (iter > 5):
                        iter = 0
                if iter > self.max_iter:
                    iter = 0
                    if self.warn_message:
                        warn(F"Stopped after {self.max_iter} itérations")
                        
            # reconstructed fittedX and Z
            fittedX = (fittedZ / sqrt(colw)) * scale + center
            Z = (Z / sqrt(colw)) * scale + center
            
            # complete X
            completeX = X.copy()         
            arr = completeX.to_numpy().copy()
            arr[mask] = Z.values[mask]
            completeX.iloc[:,:] = arr
            
        # convert to Series
        center = Series(center,index=X.columns,name="center")
        scale = Series(scale,index=X.columns,name="scale")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # complete dataset
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # complete Xtot
        complete = Xtot.copy()
        complete.loc[completeX.index,completeX.columns] = completeX.values
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set model attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # summary called informations
        call_ = {"Xtot" : Xtot, "X" : X, "row_w" : row_w, "col_w" : col_w, "ncp": ncp, "center" : center, "scale" : scale,
                 "ind_sup": ind_sup_label, "sup_var": sup_var_label}
        self.call_ = namedtuple("call", call_.keys())(*call_.values())
        # complete dataset
        self.complete_ = complete
        # complete recoded dataset
        self.fitted_ = fittedX
        return self