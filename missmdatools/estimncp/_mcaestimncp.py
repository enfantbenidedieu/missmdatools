# -*- coding: utf-8 -*-
from numpy import ndarray, array, ones,where,repeat,argmin,arange,c_
from pandas import DataFrame, Series, isna,NA
from collections import namedtuple
from itertools import product
from sklearn.base import BaseEstimator,TransformerMixin

# intern functions
from ..impute._mcaimputer import MCAimputer
from ..functions._utils import check_X_valid, to_category
from ..functions._get_sup_label import get_sup_label
from ..functions._statistics import summary_cat
from ..others._disjunctive_na import disjunctive_NA
from ..others._kfold_na import kfold_NA

class MCAestimncp(BaseEstimator,TransformerMixin):
    """
    Estimate the number of dimensions for the Multiple Correspondence Analysis by cross-validation (MCAestimncp)
    
    Parameters
    ----------
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
    
    cv : {"loo","kfold"}, default = "kfold"
        The cross-validation splitting strategy. Possible inputs are :
        
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
            The levels weights.
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
        The number of components retained from the MCA.
        
    criterion_ : DataFrame of shape (range(mincp, maxcp + 1), 2)
        The criterion (MSEP, Mean Square Error of Prediction) calculated for each number of components.
    
    References
    ----------
    [1] Anne Gégout-Petit, Myriam Maumy-Bertrand, Gilbert Saporta, Christine Thomas-Agnan (Dir.). `Données manquantes <https://www.editionstechnip.com/en/catalogue-detail/2344/donnees-manquantes.html>`_. Editions Technip, 2022, 978-2-7108-1195-4. ⟨hal-03696270⟩
    
    [2] Josse, J., Chavent, M., Liquet, B. and Husson, F. (2010). `Handling missing values with Regularized Iterative Multiple Correspondence Analysis <https://hal.science/hal-00763227v3>`_, Journal of Clcassification, 29 (1), pp. 91-116.
    
    [3] Josse, J & Husson, F. (2013). `Handling missing values in exploratory multivariate data analysis methods <https://www.numdam.org/item/JSFS_2012__153_2_79_0/>`_. Journal de la société française de la statistique. Volume 153 (2012), no. 2. pp. 79-99.
    
    [4] Josse, J. and Husson, F. missMDA (2016). missMDA : A Package for Handling Missing Values in Multivariate Data Analysis. Journal of Statistical Software, 70 (1), pp 1-31 `<doi:10.18637/jss.v070.i01> <https://www.jstatsoft.org/article/view/v070i01>`_.
    
    Examples
    --------
    >>> from missmdatools.datasets import subvoting
    >>> from missmdatools import MCAestimncp
    >>> clf = MCAestimncp()
    >>> clf.fit(subvoting)
    MCAestimncp()
    """
    def __init__(
        self,
        row_w = None,
        col_w = None,
        ind_sup = None,
        sup_var = None,
        mincp = 0,
        maxcp = 5,
        method = "em",
        cv = "kfold",
        nbsim = 100,
        pctNA = 0.05,
        threshold = 1e-4,
        max_iter = 1000,
        random_state = None,
        tol = 1e-7
    ):
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
        if not (self.cv.lower() in ("loo","kfold")):
            raise ValueError("cv should be one of 'loo', 'kfold'")
        
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
        
        # disjunctive table with missing values
        Y = disjunctive_NA(X=X,prefix=True,sep="_")
        # set shape of X
        n_rows, n_cols = X.shape
        # number of categories
        n_levels = Y.shape[1]

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # individuals, variables and columns weights
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
            
        # set variables weights
        if self.col_w is None:
            var_w = Series(ones(n_cols)/n_cols,index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("col_w must be a 1d array-like of columns weights.")
        elif len(self.col_w) != n_cols: 
            raise ValueError(f"col_w must be a 1d array-like of shape ({n_cols},).")
        else:
            var_w = Series(array(self.col_w)/sum(self.col_w),index=X.columns,name="weight")
            
        # set columns weights
        col_w = (Y * row_w.to_numpy()[:,None]).sum(axis=0) * repeat(var_w.values,X.nunique())
        col_w.name = "weight"
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # minimum and maximum number of components to test
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # maximum number of components to test
        if self.maxcp is None:
            maxcp = int(min(n_rows - 2, n_levels - n_cols - 1))
        elif self.maxcp < 0:
            raise ValueError("maxcp must be positive")
        else:
            maxcp = int(min(self.maxcp,n_rows - 2,n_levels - n_cols - 1))
            
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
        # true indexes of missing values
        rows = where(Y.isna())[0]
        if self.cv.lower() == "loo":
            Yhat = Y.copy().astype("float")
            crit = []
            for k in range(mincp,maxcp + 1):
                for i, q in product(X.index, X.columns):
                    if not isna(X.loc[i,q]):
                        XNA = X.copy()
                        XNA.loc[i,q] = NA
                        # convert to category
                        XNA = to_category(X=XNA)
                        df_sum = summary_cat(X=XNA)
                        if not any(x == 0 for x in df_sum.to_numpy().ravel()):
                            # instanciation and training
                            clf = MCAimputer(ncp=k,row_w=row_w,col_w=var_w,method=self.method,threshold=self.threshold,max_iter=self.max_iter,tol=self.tol).fit(XNA)
                            # fitted disjunctive table
                            Yfit = clf.dummies_
                            # find columns start with
                            cols = [c for c in Yfit.columns if c.startswith(f"{q}")]
                            # replace
                            Yhat.loc[i,cols] = Yfit.loc[i,cols]
                # mean square error for prediction
                val = ((Yhat - Y)**2).sum().sum()/(n_rows * n_levels - len(rows))
                # append
                crit.append(float(val))
        else:
            # initialization
            crit = DataFrame(index=list(range(self.nbsim)),columns=list(range(maxcp - mincp + 1))).astype("float")
            for s in range(self.nbsim):
                XNA = X.copy()
                # add random NA and convert to category
                XNA = kfold_NA(X=XNA,pctNA=pctNA,random_state=self.random_state).astype("category")  
                # disjunctive table
                YNA = disjunctive_NA(X=XNA,prefix=True,sep="_")
                # indexes of missing values with random missing
                rowsNA = where(YNA.isna())[0]
                for k in range(mincp, maxcp + 1):
                    # instanciation and training
                    clf = MCAimputer(ncp=k,row_w=row_w,col_w=var_w,method=self.method,threshold=self.threshold,
                                     max_iter=self.max_iter,tol=self.tol).fit(XNA)
                    # fitted disjunctive table
                    Yhat = clf.dummies_
                    # difference in number of missing values
                    n_len = len(rowsNA) - len(rows)
                    if n_len != 0:
                        crit.iloc[s,k - mincp] = ((Yhat - Y)**2).sum().sum()/n_len
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
        call_ = {"Xtot": Xtot, "X": X, "row_w": row_w, "var_w": var_w, "col_w" : col_w,"mincp":mincp,"maxcp":maxcp, "ncp": ncp,
                 "ind_sup": ind_sup_label, "sup_var": sup_var_label}
        self.call_ = namedtuple("call", call_.keys())(*call_.values())
        # optimal number of components
        self.ncp_ = ncp
        # criterion (mean square error of prediction)
        self.criterion_ = crit
        return self