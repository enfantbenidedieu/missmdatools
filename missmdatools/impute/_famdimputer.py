# -*- coding: utf-8 -*-
from math import inf
import numpy.ma as npma
from warnings import warn
from numpy import ndarray, array, ones, sqrt, where,random,isnan
from pandas import DataFrame, Series, concat
from pandas.api.types import is_numeric_dtype
from collections import namedtuple
from sklearn.base import BaseEstimator, TransformerMixin

# interns function
from ..functions._get_sup_label import get_sup_label
from ..functions._statistics import nanweightstats, wmean, wstd
from ..functions._utils import check_X_valid, check_param_valid, to_category
from ..others._concat_empty import concat_empty
from ..others._from_disjunctive import from_disjunctive
from ..others._replace_na import replace_na
from ..others._disjunctive import disjunctive
from ..others._disjunctive_na import disjunctive_NA
from ..others._disjunctive_prop import disjunctive_prop
from ..others._gsvd import gSVD

class FAMDimputer(BaseEstimator,TransformerMixin):
    """
    Impute mixed dataset with Factor Analysis of Mixed Data (FAMDimputer)
    
    Impute the missing values of a mixed dataset (with continuous and categorical variables) 
    using the principal component method "factorial analysis for mixed data" (FAMD).
    
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
        The method to used. Possible values are:
        
        * "em" for expectation - maximization algorithm
        * "regularized" for regularized iterative algorithm
    
    lambda_ : int, default = 1
        1 by default to perform the regularized FAMDimputer algorithm; useful only if ``method`` = "regularized". 
        Other regularization terms can be implemented by setting the value to less than 1 in order to regularized less 
        (to get closer to the results of the em method) or more than 1 to regularized more (to get closer to the results of the mean imputation).

    threshold : float, default = 1e-6
        The threshold for assessing convergence.

    max_iter : int, default = 1000
        The maximum number of iteration for the algorithm.
        
    random_state : int, default = None 
        If None, implies that missing values are initially imputed by the mean of each variable. Other values leads to a random initialization.
        
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
        ind_sup : None, list
            The names of the supplementary individuals.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).

    complete_ : DataFrame of shape (n_samples + n_samples_plus, n_columns + n_columns_sup)
        The imputed dataset; the observed values are kept for the non-missing entries and the missing values 
        are replaced by the predicted ones.
        
    fitted_ : DataFrame of shape (n_samples, n_columns)
        The imputed matrix; the observed values are kept for the non-missing entries and the missing values 
        are replaced by the predicted ones. The categorical variables are coded with the indicator matrix of dummy variables. 
        In this indicator matrix, the imputed values are real numbers but they met the constraint 
        that the sum of the entries corresponding to one individual and one variable is equal to one. 
        Consequently they can be seen as degree of membership to the corresponding category
    
    References
    ----------
    [1] Anne Gégout-Petit, Myriam Maumy-Bertrand, Gilbert Saporta, Christine Thomas-Agnan (Dir.). `Données manquantes <https://www.editionstechnip.com/en/catalogue-detail/2344/donnees-manquantes.html>`_. Editions Technip, 2022, 978-2-7108-1195-4. ⟨hal-03696270⟩
        
    [2] Josse, J, Husson, F & Pagès J (2009). `Gestion des données manquantes en Analyse en Composantes Principales <https://www.numdam.org/item/JSFS_2009__150_2_28_0/>`_. Journal de la société française de la statistique, Volume 150 (2009). no. 2. pp 28-51
    
    [3] Josse, J & Husson, F. (2013). `Handling missing values in exploratory multivariate data analysis methods <https://www.numdam.org/item/JSFS_2012__153_2_79_0/>`_. Journal de la société française de la statistique. Volume 153 (2012), no. 2. pp. 79-99.
    
    [4] Audigier, V., Husson, F., & Josse (2016). A principal components method to impute missing values for mixed data. *Advances in Data Analysis and Classification*, 2016, 10 (1), pp.5-26. `⟨10.1007/s11634-014-0195-1⟩. ⟨hal-00832935⟩ <https://hal.science/hal-00832935v1>`_.
    
    [5] Josse, J. and Husson, F. missMDA (2016). missMDA : A Package for Handling Missing Values in Multivariate Data Analysis. Journal of Statistical Software, 70 (1), pp 1-31 `<doi:10.18637/jss.v070.i01> <https://www.jstatsoft.org/article/view/v070i01>`_.
    
    Examples
    --------
    >>> from missmdatools.datasets import ozone
    >>> from missmdatools import FAMDimputer
    >>> clf = FAMDimputer(ncp=2,method="regularized")
    >>> clf.fit(ozone)
    FAMDimputer(method="regularized")
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
            warn_message = True,
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
            self.warn_message = warn_message
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
        # imputation of missing values using factor analysis of mixed data (FAMD)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set number of rows and columns
        n_rows, n_cols = X.shape

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set individuals weights and columns weights
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
            var_w = Series(ones(n_cols),index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)): 
            raise TypeError("col_w must be a 1d array-like of variables weights")
        elif len(self.col_w) != n_cols: 
            raise TypeError(f"col_w must be a 1d array-like of shape ({n_cols},).")
        else:
            var_w = Series(array(self.col_w),index=X.columns,name="weight")
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # data preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # find group type
        type_group = ["s" if is_numeric_dtype(X[k]) else "n" for k in X.columns]
        # name group
        name_group = [f"Gr{i+1}" for i in range(n_cols)]
        # group dict
        group_dict = dict(zip(name_group,X.columns))
        # concatenate
        Y, columns_dict = None, {}
        for g, q in group_dict.items():
            if is_numeric_dtype(X[q]):
                y = X[q].to_frame(q)  
            else: 
                y = disjunctive_NA(X=X[q],prefix=True,sep="_")
            columns_dict[g] = y.columns
            # concatenate
            Y = concat_empty(Y,y,axis=1)
        # number of recoded columns
        n_ycols = Y.shape[1]
        # where NA in recoded dataset
        mask = npma.masked_array(Y,mask=isnan(Y)).mask
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # imputation algorithm
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # mean imputation when ncp = 0
        if self.ncp == 0:
            # store number of components
            ncp = 0
            # weighted average
            mu = npma.average(npma.array(Y, mask=mask), weights=row_w, axis=0)
            # fill NA with weigted average
            completeY = Y.fillna(Series(mu,index=Y.columns))
            # columns weights
            col_w = Series(ones(n_ycols),index=Y.columns,name="weight")
        else:
            # convert to array
            roww = row_w.to_numpy()
            Z = None
            vsgroup = Series(index=name_group,dtype="float")
            mncols = Series(index=name_group,dtype="int")
            center, scale, gcol_w, dummies = {}, {}, {}, {}
            for g, c in group_dict.items():
                x = X[c]
                # convert to DataFrame if Series
                if isinstance(x,Series):
                    x = x.to_frame()
                if type_group[name_group.index(g)] == "s":
                    # find missing values index
                    xrows, xcols = where(x.isna())
                    # weighted average and weighted standard deviation with missing values
                    mu, sigma = nanweightstats(X=x,w=row_w)
                    # standardization: z_ik = (x_ik - m_k)/s_k
                    z = (x - mu)/sigma
                    # weight of variables
                    colw = var_w[c]
                    if isinstance(colw,float):
                        colw = Series(colw,index=[c],name="weight")
                    # replace NA
                    if len(xrows) > 0:
                        if self.random_state is None:
                            # replace NA values with 0
                            z = z.fillna(0)
                        else:
                            # replace NA values with random values
                            rng = random.default_rng(seed=self.random_state)
                            # random values for missing values
                            values = random.normal(loc=0,scale=1,size=len(xrows))
                            # fill missing values with random values
                            z = replace_na(X=z,rows=xrows,cols=xcols,values=values)
                    # multiply by squared of columns weights
                    z *= sqrt(colw)
                    # add to dictionary
                    center[g], scale[g] = mu, sigma
                else:
                    if x.isnull().any().any():
                        # disjunctive table where missing values are replace by mean or random values
                        y = disjunctive_prop(X=x,prefix=True,sep="_",random_state=self.random_state,row_w=row_w)
                    else:
                        y = disjunctive(X=x,prefix=True,sep="_")
                    # proportion of levels
                    p_k = (y * roww[:,None]).sum(axis=0)
                    # weight of columns
                    colw = p_k * var_w[c]
                    # standardization: z_ikq = ((y_ikq/p_kq) - 1) * sqrt(w_kq)
                    z = ((y / p_k) - 1) * sqrt(colw)
                    # replace value ~ 1e-15 by 0
                    z = z.where(abs(z) > 1e-15,0)
                    # add to dictionary
                    center[g], dummies[g] = colw, y
                
                # generalized singular values decomposition (gSVD)
                vs = gSVD(X=z,ncp=self.ncp,row_w=row_w,col_w=None,tol=self.tol).vs[0]
                # concatenate
                Z = concat_empty(Z,z/vs,axis=1)  
                # append
                gcol_w[g], vsgroup.loc[g], mncols.loc[g] = colw, float(vs), int(z.shape[1])
            
            # number of final rows and columns
            nr, nc = Z.shape
            # update nc (remove number of categorical columns)
            nc = nc - len(where(array(type_group)=="n")[0])
            iter = 1
            old_obj = inf
            while iter > 0:
                for g, c in columns_dict.items():
                    z = (Z[c] * vsgroup.loc[g])/sqrt(gcol_w[g])
                    if type_group[name_group.index(g)] == "s":
                        y = (z * scale[g]) + center[g]
                        # weighted average and weighted standard deviation
                        mu, sigma = wmean(X=y,w=row_w),wstd(X=y,w=row_w,ddof=0)
                        # standardization: z_ik = ((x_ik - m_k)/s_k) * sqrt(w_k)
                        z = ((y - mu)/sigma) * sqrt(gcol_w[g]) 
                        # update values
                        center[g], scale[g] = mu, sigma
                    else:
                        # reconstructed disjunctive table
                        y = (z + 1) * (dummies[g] * roww[:,None]).sum(axis=0)
                        # replace value ~ 1e-15 by 0
                        y = y.where(abs(y) > 1e-15,0)
                        # proportion of levels
                        p_k = (y * roww[:,None]).sum(axis=0)
                        # check if any weight is negative
                        if any(x < 0 for x in p_k.values):
                            raise ValueError(f"The algorithm fails to converge. Choose a number of components (ncp) less or equal than {self.ncp-1}",
                                             f"or a number of iterations (maxiter) less or equal than {self.max_iter-1}")
                        # weight of columns
                        colw = p_k/y.shape[1]
                        # standardization : z_ikq = ((y_ikq/p_kq) - 1) * sqrt(w_kq)
                        z = ((y/p_k) - 1) * sqrt(colw)
                        # replace value ~ 1e-15 by 0
                        z = z.where(abs(z) > 1e-15,0)
                        # update values
                        gcol_w[g], center[g], dummies[g] = colw, colw, y
                    
                    # generalized singular values decomposition (gSVD)
                    vs = gSVD(X=z,ncp=self.ncp,row_w=row_w,col_w=None,tol=self.tol).vs[0]
                    # update singular value and standardized data
                    vsgroup.loc[g] = float(vs)
                    # divide by first singular value
                    Z[c] = z/vs
                
                # generalized singular values decomposition (gSVD)
                svd = gSVD(X=Z,ncp=self.ncp,row_w=row_w,col_w=None,tol=self.tol)
                # update number of components
                ncp = svd.ncp
                # noise variance
                if self.method.lower() == "em":
                    # for expectation maximization 
                    sigma2 = 0
                else:
                    # for regularized iterative algorithm
                    sigma2 = ((nr * nc)/min(nc,nr - 1))*(svd.sumvp - sum(svd.vp[:ncp]))/((nr - 1 - ncp)*(nc - ncp)) #p 135
                    # update sigma2
                    sigma2 = min(sigma2 * self.lambda_,svd.vp[ncp])
                
                # shrinkage parameter
                shrinkage = (svd.vp[:ncp] - sigma2)/svd.vs[:ncp] # pp 112 - 
                # update fitted dataframe
                if ncp == 1:
                    U, V = svd.U[:,0][:,None], svd.V[:,0][:,None]
                else:
                    U, V = svd.U[:,:ncp], svd.V[:,:ncp]
                # reconstructed standardized data
                recon = ((U * roww[:,None]) * shrinkage).dot(V.T) / roww[:,None]
                # convert to DataFrame
                recon = DataFrame(recon,index = Z.index,columns=Z.columns)
                # compute the difference : standardized - reconstructed standardized
                diff = Z - recon
                # replace missing index of difference by zero
                arr = diff.to_numpy().copy()
                arr[mask] = 0
                diff.iloc[:,:] = arr
                # compute objective
                obj = ((diff ** 2) * roww[:,None]).sum().sum()
                # define criterion
                crit = abs(1 - obj/old_obj)
                old_obj = obj
                iter += 1
                # replace missing index of Z by reconstructed standardized data
                arr = Z.to_numpy().copy()
                arr[mask] = recon.values[mask]
                Z.iloc[:,:] = arr
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
            
            # columns weights
            col_w = concat((gcol_w[g] for g, _ in columns_dict.items()),axis=0)
            col_w.name = "weight"
            # fitted Y
            fittedY = None
            for g, c in columns_dict.items():
                z = (Z[c] * vsgroup.loc[g])/sqrt(gcol_w[g])
                if type_group[name_group.index(g)] == "s":
                    y = (z * scale[g]) + center[g]
                elif type_group[name_group.index(g)] == "n":      
                    y = (z + 1) * (dummies[g] * roww[:,None]).sum(axis=0)
                fittedY = concat_empty(fittedY,y,axis=1)
            
            # complete Y
            completeY = Y.copy()         
            arr = completeY.to_numpy().copy()
            arr[mask] = fittedY.values[mask]
            completeY.iloc[:,:] = arr

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # complete dataset
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # complete X
        completeX = None
        for g, c in columns_dict.items():
            if type_group[name_group.index(g)] == "s":
                x = completeY[c]
            elif type_group[name_group.index(g)] == "n":
                x = from_disjunctive(X=completeY[c],sep="_",ordered=True)
            # concatenate
            completeX = concat_empty(completeX,x,axis=1)
        
        # complete Xtot
        completeXtot = Xtot.copy()
        completeXtot.loc[completeX.index,completeX.columns] = completeX.values
        # convert to category
        completeXtot = to_category(X=completeXtot)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set model attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # summary called informations
        call_ = {"Xtot": Xtot, "X": X, "row_w": row_w, "col_w":col_w,"ncp": ncp,
                 "ind_sup": ind_sup_label, "sup_var": sup_var_label}
        self.call_ = namedtuple("call", call_.keys())(*call_.values())  
        # complete
        self.complete_ = completeXtot
        # complete recoded dataset
        self.fitted_ = completeY
        return self