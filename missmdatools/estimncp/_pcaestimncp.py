# -*- coding: utf-8 -*-
import numpy.ma as npma
from numpy import ndarray, array, ones,where,diff,argmin,nan,arange,c_,isnan
from pandas import DataFrame, Series, isna
from collections import namedtuple
from itertools import product
from sklearn.base import BaseEstimator,TransformerMixin

# intern functions
from ..impute._pcaimputer import PCAimputer
from ..functions._utils import check_X_valid, to_category
from ..functions._get_sup_label import get_sup_label
from ..others._kfold_na import kfold_NA

class PCAestimncp(BaseEstimator,TransformerMixin):
    """
    Estimate the number of dimensions for the Principal Component Analysis by cross-validation (PCAestimncp)
        
    Parameters
    ----------
    scale_unit : bool, default = True
        If True, then the data are scaled to unit variance.
    
    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.
        
    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columns weights. The weights are given only for the active columns.
        
    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.

    sup_var : int, str, list, tuple or range, default = None 
        The indexes or names of the supplementary variables (continuous and/or categorical).
        
    mincp : int, default = 0
        The minimum number of components to test.
        
    maxcp : int, default = 5
        The maximum number of components to test.
            
    method : {"em","regularized"}, default = "em"
        The algorithm to used. Possible values are:
        
        * "em" for expectation - maximization algorithm
        * "regularized" for regularized iterative algorithm
    
    cv : {"gcv","loo","kfold"}, default = "gcv"
        The cross-validation splitting strategy. Possible inputs are :
        
        * "gcv" for generalized cross-validation
        * "loo" for leave-one-out cross-validation
        * "kfold" for KFold cross-validation
        
    nbsim : int, default = 100
        Number of simulations, useful only if ``cv`` = "kfold".
        
    pctNA : float, default = 0.05
        Percentage of missing values added in the data set, useful only if ``cv`` = "kfold".

    threshold : float, default = 1e-4
        The threshold for assessing convergence.
        
    max_iter : int, default = 1000
        The maximum number of iteration for the algorithm.
        
    random_state : int, default = None
        The seed of the random number generator. Used whem ``cv`` = "kfold".
        
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
        mincp : int
            The minimum number of components to test.
        maxcp : int
            The maximum number of components to test.
        ncp : int
            The number of components retained.
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).

    ncp_ : int
        The number of components retained from the PCA.
        
    criterion_ : DataFrame of shape (range(mincp, maxcp + 1), 2)
        The criterion (MSEP, Mean Square Error of Prediction) calculated for each number of components.

    References
    ----------
    [1] Anne Gégout-Petit, Myriam Maumy-Bertrand, Gilbert Saporta, Christine Thomas-Agnan (Dir.). `Données manquantes <https://www.editionstechnip.com/en/catalogue-detail/2344/donnees-manquantes.html>`_. Editions Technip, 2022, 978-2-7108-1195-4. ⟨hal-03696270⟩
    
    [2] Bro, R., Kjeldahl, K. Smilde, A. K. and Kiers, H. A. L. (2008) Cross-validation of component models: A critical look at current methods. Analytical and Bioanalytical Chemistry, 5, 1241-1251.
    
    [3] Josse, J, Husson, F & Pagès J (2009). `Gestion des données manquantes en Analyse en Composantes Principales <https://www.numdam.org/item/JSFS_2009__150_2_28_0/>`_. Journal de la société française de la statistique, Volume 150 (2009). no. 2. pp 28-51
    
    [4] Josse, J. and Husson, F. (2011). Selecting the number of components in PCA using cross-validation approximations. Computational Statistics and Data Analysis. 56 (6), pp. 1869-1879.
    
    [5] Josse, J & Husson, F. (2013). `Handling missing values in exploratory multivariate data analysis methods <https://www.numdam.org/item/JSFS_2012__153_2_79_0/>`_. Journal de la société française de la statistique. Volume 153 (2012), no. 2. pp. 79-99.
    
    [6] Josse, J. and Husson, F. missMDA (2016). missMDA : A Package for Handling Missing Values in Multivariate Data Analysis. Journal of Statistical Software, 70 (1), pp 1-31 `<doi:10.18637/jss.v070.i01> <https://www.jstatsoft.org/article/view/v070i01>`_.
    
    Examples
    --------
    >>> from missmdatools.datasets import orange
    >>> from missmdatools import PCAestimncp
    >>> clf = PCAestimncp(method="regularized")
    >>> clf.fit(orange)
    PCAestimncp(method="regularized")
    """
    def __init__(
        self,
        scale_unit = True,
        row_w = None,
        col_w = None,
        ind_sup = None,
        sup_var = None,
        mincp = 0,
        maxcp = 5,
        method = "em",
        cv = "gcv",
        nbsim = 100,
        pctNA = 0.05,
        threshold = 1e-4,
        max_iter = 1000,
        random_state = None,
        warn_message = True,
        tol = 1e-7
    ):
        self.scale_unit = scale_unit
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.sup_var = sup_var
        self.mincp = mincp
        self.maxcp = maxcp
        self.method = method
        self.cv = cv
        self.nbsim = nbsim
        self.pctNA = pctNA
        self.threshold = threshold
        self.max_iter = max_iter
        self.random_state = random_state
        self.warn_message = warn_message
        self.tol = tol
        
    def fit(self,X,y=None):
        """Fit the estimate to X.
        
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
        # check if valid cross validation method
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not (self.cv.lower() in ("gcv","loo","kfold")):
            raise ValueError("cv should be one of 'gcv', 'loo', 'kfold'")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if valid percentage of NA
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.pctNA is None:
            pctNA = 5e-2
        elif not isinstance(self.pctNA,float):
            raise TypeError("{} is not supported".format(type(self.pctNA)))
        elif self.pctNA < 0 or self.pctNA > 1:
            raise ValueError("the 'pct_NA' value {} is not within the required range of 0 and 1.".format(self.pctNA))
        else:
            pctNA = self.pctNA
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if valid input data and parameters
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_X_valid(X=X)
        
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
        
        # set shape of X
        n_rows, n_cols = X.shape
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # individuals and columns weights
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # minimum and maximum number of components to test
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # maximum number of components to test
        if self.maxcp is None:
            maxcp = min(n_cols - 1,n_rows - 2)
        elif self.maxcp < 0:
            raise ValueError("maxcp must be positive")
        else:
            maxcp = min(self.maxcp, n_cols - 1,n_rows - 2)
        
        # minimum number of components to test
        if self.mincp is None:
            mincp = 0
        elif self.mincp < 0:
            raise ValueError("mincp must be positive")
        else:
            mincp = int(self.mincp)
        
        # check if mincp is less than maxcp
        if mincp > maxcp:
            raise ValueError("mincp must be less than maxcp")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # optimal number of components
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # rows indexes of missing values
        rows = where(X.isna())[0]
        dof = n_rows * n_cols - len(rows)
        # generalized cross-validation
        if self.cv == "gcv":
            crit = []
            if mincp == 0:
                mask = npma.masked_array(X,mask=isnan(X)).mask
                mu = npma.average(npma.array(X, mask=mask), weights=row_w, axis=0)
                val = ((X - mu)**2).sum(axis=0).sum()/dof
                # append
                crit.append(float(val))
            for k in range(max(mincp,1),maxcp+1):
                clf = PCAimputer(scale_unit=self.scale_unit,ncp=k,row_w=row_w,col_w=col_w,method=self.method,
                                 max_iter=self.max_iter,warn_message=self.warn_message,tol=self.tol).fit(X)
                Xfit = clf.fitted_
                den = (n_rows - 1) * n_cols - len(rows) - k * (n_rows + n_cols - k - 1)
                val = ((dof * (X - Xfit)/den)**2).sum().sum()/dof
                # append
                crit.append(float(val))
            # convert to array
            crit = array(crit)
            if any(diff(crit)>0):
                ncp = min(where(diff(crit)>0))[0]
            else:
                ncp = argmin(crit)
            # update ncp
            ncp = ncp + mincp
        # leave - one - out cross-validation
        elif self.cv == "loo":
            crit = []
            for k in range(mincp,maxcp+1):
                Xhat = X.copy()
                for i, j in product(range(n_rows),range(n_cols)):
                    if not isna(X.iloc[i,j]):
                        XNA = X.copy()
                        XNA.iloc[i,j] = nan
                        if k == 0:
                            Xhat.iloc[i,j] = XNA.iloc[:,j].mean()
                        else:
                            # instanciation and training
                            clf = PCAimputer(scale_unit=self.scale_unit,ncp=k,method=self.method,threshold=self.threshold,
                                             max_iter=self.max_iter,warn_message=self.warn_message,tol=self.tol).fit(XNA)
                            Xhat.iloc[i,j] = clf.complete_.iloc[i,j]
                # mean squared error for prediction
                val = ((Xhat - X)**2).sum().sum()/dof
                crit.append(float(val))
            # optimal number of components
            ncp = argmin(crit) + mincp
        # kfold cross - validation
        else:
            # initialization
            crit = DataFrame(index=list(range(self.nbsim)),columns=list(range(maxcp - mincp + 1))).astype("float")
            for s in range(self.nbsim):
                XNA = X.copy()
                # add random NA and convert to category
                XNA = kfold_NA(X=XNA,pctNA=pctNA,random_state=self.random_state).astype("float")  
                for k in range(mincp, maxcp + 1):
                    if k == 0:
                        Xhat = XNA.copy()
                        # fill NA with weighted average
                        mask = npma.masked_array(Xhat,mask=isnan(Xhat)).mask
                        mu = npma.average(npma.array(Xhat, mask=mask), weights=row_w, axis=0)
                        Xhat = Xhat.fillna(Series(mu,index=Xhat.columns))
                    else:
                        # instanciation and training
                        clf = PCAimputer(scale_unit=self.scale_unit,ncp=k,method=self.method,threshold=self.threshold,
                                         max_iter=self.max_iter,warn_message=self.warn_message,tol=self.tol).fit(XNA)
                        Xhat = clf.complete_
                    # sum squared error of prediction
                    crit.iloc[s,k - mincp] = ((Xhat - X)**2).sum().sum()
            # average
            crit = crit.mean(axis=0).values
            # optimal number of components
            ncp = argmin(crit) + mincp
        
        # convert to DataFrame
        crit = DataFrame(c_[arange(mincp,maxcp+1),crit],columns=["ncp","criterion"])
        crit["ncp"] = crit["ncp"].astype("int")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set model attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # summary called informations
        call_ = {"Xtot": Xtot, "X": X, "row_w": row_w, "col_w": col_w,"mincp":mincp,"maxcp":maxcp, "ncp": ncp,
                 "ind_sup": ind_sup_label, "sup_var": sup_var_label}
        self.call_ = namedtuple("call", call_.keys())(*call_.values())
        # optimal number of components
        self.ncp_ = int(ncp)
        # criterion
        self.criterion_ = crit
        return self