# -*- coding: utf-8 -*-
from math import inf
import numpy.ma as npma
from warnings import warn
from numpy import ndarray, where, array, ones,inf,sqrt,random,repeat, sqrt, isnan
from pandas import Series, DataFrame, concat
from pandas.api.types import is_numeric_dtype
from itertools import chain
from collections import namedtuple
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# intern functions
from ..functions._get_sup_label import get_sup_label
from ..functions._statistics import nanweightstats, wmean, wstd
from ..functions._utils import check_X_valid, check_param_valid, to_category
from ..others._concat_empty import concat_empty
from ..others._from_disjunctive import from_disjunctive
from ..others._replace_na import replace_na
from ..others._disjunctive_na import disjunctive_NA
from ..others._disjunctive_prop import disjunctive_prop
from ..others._disjunctive import disjunctive
from ..others._splitmix import splitmix
from ..others._gsvd import gSVD

class MFAimputer(BaseEstimator,TransformerMixin):
    """
    Impute dataset with variables structured into groups of variables with Multiple Factor Analysis (MFAimputer)
    
    Impute the missing values of a dataset with Multiple Factor Analysis (MFA). 
    The variables are structured a priori into groups of variables. 
    The variables can be continuous, categorical or mixed. 
    
    Parameters
    ----------
    ncp : int, default = 2
        The number of components used to predict the missing values.
        
    group : list, tuple
        The number of variables in each group.

    name_group : list, tuple, default = None
        The name of the groups. If None, the group are named Gr1, Gr2 and so on.

    type_group : list, tuple
        The type of variables in each group. Possible values are: 

        * "c" or "s" for continuous variables (the difference is that for "s" variables are scaled to unit variance)
        * "n" for categorical variables
        * "m" for mixed variables (continuous and categorical variables)
        
    row_w : 1d array-like of shape (n_rows,), default = None
        An optional rows weights. The weights are given only for the active rows.
        
    col_w : 1d array-like of shape (n_columns,), default = None
        An optional columns weights. The weights are given only for the active columns.
        
    ind_sup : int, str, list, tuple or range, default = None
        The indexes or names of the supplementary individuals.
        
    num_group_sup : list, tuple
        The indexes of the illustrative groups (by default, None and no group are illustrative)

    method : {"em","regularized"}, default = "em"
        The algorithm to used. Possible values are:
        
        * "em" for expectation - maximization algorithm
        * "regularized" for regularized iterative algorithm
    
    lambda_ : int, default = 1
        1 by default to perform the regularized MFAimputer algorithm; useful only if ``method`` = "regularized". 
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
        num_group_sup : None, list
            The indexes of the illustrative groups.
        sup_var : None, list
            The names of the supplementary variables (continuous and/or categorical).

    complete_ : DataFrame of shape (n_samples + n_samples_plus, n_columns + n_columns_sup)
        The imputed dataset; the observed values are kept for the non-missing entries and the missing values are 
        replaced by the predicted ones. For the continuous variables, the values are the same as in the ``fitted_`` output; 
        for the categorical variables missing values are imputed with the most plausible categories according to the values 
        in the ``fitted_`` output.

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
    
    [3] Josse, J & Husson, F. (2013). `Handling missing values in multiple factor analysis <https://hal.science/hal-01070888v1>`_.  *Food Quality and Preference*, 2013, 30 (2), pp.77-85.
    
    [4] Josse, J. and Husson, F. missMDA (2016). missMDA : A Package for Handling Missing Values in Multivariate Data Analysis. Journal of Statistical Software, 70 (1), pp 1-31 `<doi:10.18637/jss.v070.i01> <https://www.jstatsoft.org/article/view/v070i01>`_.
    
    Examples
    --------
    >>> from missmdatools.datasets import orange, vnf
    >>> from missmdatools import MFAimputer
    >>> clf = MFAimputer(ncp=2,group=(5,3),type_group=("s","s"),method="regularized")
    >>> clf.fit(orange)
    MFAimputer(group=(5,3),method="regularized",type_group=("s","s"))
    >>> clf = MFAimputer(ncp=2,group=(6,5,3),type_group=("n","n","n"),method="regularized")
    >>> clf.fit(vnf)
    MFAimputer(group=(6,5,3),method="regularized",type_group=("n","n","n"))
    """
    def __init__(
        self,
        ncp = 2,
        group = None, 
        type_group = None,
        name_group = None,
        row_w = None, 
        col_w = None,
        ind_sup = None, 
        num_group_sup = None, 
        method = "em",
        lambda_ = 1,
        threshold = 1e-6,
        max_iter = 1000,
        random_state = None,
        warn_message = True,
        tol = 1e-7
    ):
        self.ncp = ncp
        self.group = group
        self.type_group = type_group
        self.name_group = name_group
        self.row_w = row_w
        self.col_w = col_w
        self.ind_sup = ind_sup
        self.num_group_sup = num_group_sup
        self.method = method
        self.lambda_ = lambda_
        self.threshold = threshold
        self.max_iter = max_iter
        self.random_state = random_state
        self.warn_message = warn_message
        self.tol = tol
        
    def fit(self,X,y=None):
        """Fit the model to X

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_columns)
            Training data, where ``n_samples`` in the number of samples 
            and ``n_columns`` is the number of columns.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself
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
        # check if group is None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.group is None:
            raise ValueError("'group' must be assigned.")
        elif not isinstance(self.group, (list,tuple,ndarray,Series)):
            raise ValueError("'group' must be a 1d array-like with the number of variables in each group")
        else:
            group = [int(x) for x in self.group]

        # check if group definition
        if sum(group) != X.shape[1]:
            raise TypeError("Not convenient group definition")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if type_group in not None
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.type_group is None:
            raise ValueError("'type_group' must be assigned")
        elif not isinstance(self.type_group, (list,tuple,ndarray,Series)): 
            raise ValueError("'type_group' must be a 1d array-like with the type of variables in each group")
        else:
            type_group = [f"{x}" for x in self.type_group]

        if any(x not in ("c","m","n","s") for x in type_group):
            raise ValueError("Not convenient type_group definition")
        
        if len(self.group) != len(self.type_group):
            raise TypeError("Not convenient group definition")
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assigned group name
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.name_group is None:
            name_group = [f"Gr{x+1}" for x in range(len(group))]
        elif not isinstance(self.name_group,(list,tuple,ndarray,Series)):
            raise TypeError("'name_group' must be a 1d array-like with name of group")
        else:
            name_group = [f"{x}" for x in self.name_group]
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check if supplementary groups
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.num_group_sup is not None:
            if isinstance(self.num_group_sup,int):
                num_group_sup = [int(self.num_group_sup)]
            elif isinstance(self.num_group_sup,(list,tuple)) and len(self.num_group_sup) >= 1:
                num_group_sup = [int(x) for x in self.num_group_sup]
        else:
            num_group_sup = None
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # assigned group name to label
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        group_dict, k = {}, 0
        for i, g in zip(range(len(group)),name_group):
            group_dict[g] = X.columns[k:(k+group[i])]
            k += group[i]

        if self.num_group_sup is not None:
            group_sup_dict = {g : group_dict[g] for i, g in enumerate(name_group) if i in num_group_sup}
            group_dict = {g : group_dict[g] for i, g in enumerate(name_group) if not i in num_group_sup}

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # get supplementary elements labels
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ind_sup_label = get_sup_label(X=X, indexes=self.ind_sup, axis=0)
        if self.num_group_sup is None:
            sup_var_label  = None  
        else:
            sup_var_label  = list(chain.from_iterable(group_sup_dict.values()))
        
        # make a copy of the original data
        Xtot = X.copy()

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop supplementary elements
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # drop supplementary groups columns
        if self.num_group_sup is not None:
            X = X.loc[:,sup_var_label], X.drop(columns=sup_var_label)
    
        # drop supplementary individuals
        if self.ind_sup is not None:
            X = X.loc[ind_sup_label,:], X.drop(index=ind_sup_label)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # imputation of missing values using multiple factor analysis (MFAImputer)
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # number of rows/columns
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
            
        # set variables weights
        if self.col_w is None:
            var_w = Series(ones(n_cols),index=X.columns,name="weight")
        elif not isinstance(self.col_w,(list,tuple,ndarray,Series)):
            raise TypeError("col_w must be a 1d array-like of variables weights.")
        elif len(self.col_w) != n_cols:
            raise ValueError(f"col_w must be a 1d array-like of shape ({n_cols},).")
        else:
            var_w = Series(array(self.col_w),index=X.columns,name="weight")
            
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # data preparation
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # convert to array
        roww = row_w.to_numpy()
        # data preparation for imputation
        Y, Z = None, None
        mncols = Series(index=list(group_dict.keys()),dtype="int")
        vsgroup = Series(index=list(group_dict.keys()),dtype="float")
        columns_dict, gcol_w, center, scale, dummies = {}, {}, {}, {}, {}
        for g, c in group_dict.items():
            x = X[c]
            # convert to DataFrame if Series
            if isinstance(x,Series):
                x = x.to_frame()
            if self.type_group[name_group.index(g)] in ("c","s"):
                y = x.copy()
                # where NA 
                ymask = npma.masked_array(y,mask=isnan(y)).mask 
                # weighted average
                mu = npma.average(npma.array(y, mask=ymask), weights=row_w, axis=0)
                # standard deviation
                if self.type_group[name_group.index(g)] == "c":
                    sigma = ones(y.shape[1])
                else:
                    sigma = sqrt(npma.average(npma.array((y - mu)**2, mask=ymask), weights=row_w, axis=0))
                # standardization: z_ik = (y_ik - u_k)/s_k with y_ik = x_ik
                z = (y - mu)/sigma
                # weights of columns
                colw = var_w[c]
                # find missing values index
                xrows, xcols = where(y.isna())
                # replace NA
                if len(xrows) > 0:
                    if self.random_state is None:
                        # replace NA values with 0 - mean imputation
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
            elif self.type_group[name_group.index(g)] == "n":
                if x.isnull().any().any():
                    # disjunctive table where missing values are replace
                    y1 = disjunctive_prop(X=x,prefix=True,sep="_",random_state=self.random_state,row_w=row_w)
                    # disjunctive table with NA
                    y = disjunctive_NA(X=x,prefix=True,sep="_")
                else:
                    y1 = disjunctive(X=x,prefix=True,sep="_")
                    y = y1.copy()
                # proportion of levels
                p_k = (y1 * row_w.to_numpy()[:,None]).sum(axis=0)
                # variable weights
                if x.shape[1] == 1:
                    m_k = Series([var_w[c].values[0]]*y1.shape[1],index=y1.columns)
                else:
                    m_k = Series(repeat((var_w[c]/sum(var_w[c])).values,x.nunique()),index=y1.columns)
                # weight of columns
                colw = p_k * m_k
                # standardization: z_ikq = ((y_ikq/p_kq) - 1) * sqrt(w_kq)
                z = ((y1 / p_k) - 1) *  sqrt(colw)
                # replace value ~ 1e-15 by 0
                z = z.where(abs(z) > 1e-15,0)
                # add to dictionary
                center[g], dummies[g] = colw, y1
            elif self.type_group[name_group.index(g)] == "m":
                # split x
                split_x = splitmix(X=x)
                # extract elements
                x_quanti, x_quali = split_x.quanti, split_x.quali
                k1, k2 = split_x.k1, split_x.k2
                # check if valid group type
                if (k1 == 0) or (k2 == 0) :
                    raise TypeError("Invalid group type")
                # check if any missing value in categorical variables
                if x_quali.isnull().any().any():
                    print("Missing values are not allowed in categorical variables for mixed group.",
                          "It has been replaced by the most frequent categorie.")
                    # fill missing values with most frequent
                    clf = SimpleImputer(strategy="most_frequent")
                    x_quali = DataFrame(clf.fit_transform(x_quali),index=x_quali.index,columns=x_quali.columns)
                # find missing values index in quantitative data
                xrows = where(x_quanti.isna())[0]
                # disjunctive table
                y2 = disjunctive(X=x_quali,prefix=True,sep="_")
                # proportion of levels
                p_k = (y2 * roww[:,None]).sum(axis=0)
                # concatenate continuous data with disjunctive
                y = concat((x_quanti,y2),axis=1)
                # weighted average and weighted standard deviation 
                mu1, sigma1 = nanweightstats(X=x_quanti,w=row_w)
                mu2, sigma2 = p_k, sqrt(p_k)
                # standarization: z_ik = (x_ik - m_k)/s_k, z_ikq = (y_ikq - p_k)/sqrt(p_kq)
                z1, z2 = (x_quanti - mu1)/sigma1, (y2 - mu2)/sigma2
                
                # replace NA
                if len(xrows) > 0:
                    # replace NA values with 0
                    if self.random_state is None:
                        z1 = z1.fillna(0)
                    else:
                        # replace NA values by random values
                        rng = random.default_rng(self.random_state)
                        # random values for missing values
                        values = random.normal(loc=0,scale=1,size=len(rows))
                        # fill missing values with random values
                        z1 = replace_na(X=z1,rows=rows,cols=cols,values=values)
                
                # weight of columns
                if k1 == 1:
                    colw1 = Series(var_w[x_quanti.columns],index=[x_quanti.columns],name="weight")
                else:
                    colw1 = var_w[x_quanti.columns]
                if k2 == 1:
                    colw2 = Series([var_w[x_quali.columns].values[0]]*y2.shape[1],index=y2.columns)
                else:
                    colw2 = Series(repeat(var_w[x_quali.columns].values,x_quali.nunique()),index=y2.columns)
                
                # concatenate
                z, colw = concat((z1,z2),axis=1), concat((colw1,colw2),axis=0)
                mu, sigma = concat((mu1,mu2),axis=0), concat((sigma1,sigma2),axis=0)
                # multiply by squared of columns weights
                z *=  sqrt(colw)
                # add to dictionary
                dummies[g] = y2
                center[g], scale[g] = mu, sigma
                
            # generalized singular values decomposition (gSVD)
            vs = gSVD(X=z,ncp=self.ncp,row_w=row_w,col_w=None,tol=self.tol).vs[0]
            # concatenate
            Y, Z  = concat_empty(Y,y,axis=1), concat_empty(Z,z/vs,axis=1)
            gcol_w[g], columns_dict[g] = colw, z.columns
            vsgroup.loc[g], mncols.loc[g] = float(vs), z.shape[1]
            
        # where NA 
        mask = npma.masked_array(Y,mask=isnan(Y)).mask 
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # imputation algorithm
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # imputation with mean if ncp = 0
        if self.ncp == 0:
            # store number of components
            ncp = 0
            # weighted average
            mu = npma.average(npma.array(Y, mask=mask), weights=row_w, axis=0)
            # fill NA with weigted average²
            completeY = Y.fillna(Series(mu,index=Y.columns))
        else:
            # find missing values in recoded values - 
            # original continues values and disjunctive table for categorical
            rows, cols = where(Y.isna())
            # number of final rows and columns
            nr, nc = Z.shape
            # update nc (remove number of categorical columns)
            col_type_group = ["s" if is_numeric_dtype(X[k]) else "n" for k in X.columns]
            nc = nc - len(where(array(col_type_group)=="n")[0])
            iter = 1
            old_obj = inf
            while iter > 0:
                for g, c in columns_dict.items():
                    z = (Z[c] * vsgroup.loc[g])/sqrt(gcol_w[g])
                    if self.type_group[name_group.index(g)] in ("c","s"):
                        # reconstructed
                        y = (z * scale[g]) + center[g]
                        # weighted average
                        mu = wmean(X=y,w=row_w)
                        # standard deviation
                        if self.type_group[name_group.index(g)] == "s":
                            sigma = wstd(X=y,w=row_w,ddof=0)
                        else:
                            sigma = Series(ones(mncols[g]),index=y.columns)
                        # standardization: z_ik = ((x_ik - m_k)/s_k) * sqrt(w_k)
                        z = ((y - mu)/sigma) * sqrt(gcol_w[g])
                        # update values
                        center[g], scale[g] = mu, sigma
                    elif self.type_group[name_group.index(g)] == "n":
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
                        colw = p_k/mncols[g]
                        # standardization
                        z = ((y/p_k) - 1) * sqrt(colw)
                        # replace value ~ 1e-15 by 0
                        z = z.where(abs(z) > 1e-15,0)
                        # update values
                        gcol_w[g], center[g], dummies[g] = colw, colw, y
                    elif self.type_group[name_group.index(g)] == "m":
                        # reconstructed
                        y = (z * scale[g]) + center[g]
                        # weighted average and weighted standard deviation
                        mu, sigma = wmean(X=y,w=row_w),wstd(X=y,w=row_w,ddof=0)
                        # proportion of levels
                        p_k = (dummies[g] * roww[:,None]).sum(axis=0)
                        # update weighted standard deviation for categories
                        mu.loc[dummies[g].columns] = p_k
                        sigma.loc[dummies[g].columns] = sqrt(p_k)
                        # standardization: z_ik = ((x_ik - m_k)/s_k) * sqrt(w_k)
                        z = ((y - mu)/sigma) * sqrt(gcol_w[g])
                        # update values
                        center[g], scale[g] = mu, sigma
                    
                    # generalized singular values decomposition (gSVD)
                    vs = gSVD(X=z,ncp=self.ncp,row_w=row_w,tol=self.tol).vs[0]
                    # update sinular value and standardized data
                    vsgroup.loc[g] = float(vs)
                    Z[c] = z/vs
                
                # generalized singular values decomposition (gSVD)
                svd = gSVD(X=Z,ncp=self.ncp,row_w=row_w,tol=self.tol)
                # update number of components
                ncp = svd.ncp
                # noise variance
                if self.method.lower() == "em":
                    # for expectation maximization
                    sigma2 = 0
                else:
                    # for regularized iterative algorithm
                    sigma2 = ((nr * nc)/min(nc,nr-1))*(svd.sumvp - sum(svd.vp[:ncp]))/((nr - 1 - ncp)*(nc - ncp)) #p 112
                    # update sigma2
                    sigma2 = min(sigma2 * self.lambda_,svd.vp[ncp])
                
                # shrinkage parameter
                shrinkage = (svd.vp[:ncp] - sigma2)/svd.vs[:ncp] # pp 112 - 113
                # update fitted dataframe
                if ncp == 1:
                    U, V = svd.U[:,0][:,None], svd.V[:,0][:,None]
                else:
                    U, V = svd.U[:,:ncp], svd.V[:,:ncp]
                # reconstructed standardized data
                recon = (((U * roww[:,None]) * shrinkage).dot(V.T)) / roww[:,None]
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
                    
            # fitted Y
            fittedY = None
            for g, c in columns_dict.items():
                z = (Z[c] * vsgroup.loc[g])/sqrt(gcol_w[g])
                if self.type_group[name_group.index(g)] in ("c","s","m"):
                    y = (z * scale[g]) + center[g]
                elif self.type_group[name_group.index(g)] == "n":      
                    y = (z + 1) * (dummies[g] * roww[:,None]).sum(axis=0)
                # concatenate
                fittedY = concat_empty(fittedY,y,axis=1)
            
            # complete Y
            completeY = Y.copy()         
            arr = completeY.to_numpy().copy()
            arr[mask] = fittedY.values[mask]
            completeY.iloc[:,:] = arr
        
        # weights of columns
        col_w = concat((gcol_w[g] for g, _ in columns_dict.items()),axis=0)
        col_w.name = "weight"
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # complete dataset
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # complete X
        completeX = None
        for g, c in columns_dict.items():
            if self.type_group[name_group.index(g)] in ("c","f","s"):
                x = completeY[c].copy()
            elif self.type_group[name_group.index(g)] == "n":
                x = from_disjunctive(X=completeY[c],sep="_",ordered=True)
            elif self.type_group[name_group.index(g)] == "m":
                xcod = completeY[c]
                x1 = xcod[[x for x in xcod.columns if x not in dummies[g].columns]]
                x2 = from_disjunctive(X=xcod[dummies[g].columns],sep="_",ordered=True)
                x = concat((x1,x2),axis=1)
            # concatenate
            completeX = concat_empty(completeX,x,axis=1)
        # reorder columns
        completeX = concat((completeX[k] for k in X.columns),axis=1)
        # complete Xtot
        completeXtot = Xtot.copy()
        completeXtot.loc[completeX.index,completeX.columns] = completeX.values
        # convert to category
        completeXtot = to_category(X=completeXtot)
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # set model attributes
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # summary called informations
        call_ = {"Xtot": Xtot, "X": X, "row_w": row_w, "col_w":col_w,"ncp": ncp,"ind_sup": ind_sup_label, 
                 "num_group_sup" : num_group_sup, "sup_var": sup_var_label}
        self.call_ = namedtuple("call", call_.keys())(*call_.values())  
        # complete
        self.complete_ = completeXtot
        # complete recode
        self.fitted_ = completeY
        return self