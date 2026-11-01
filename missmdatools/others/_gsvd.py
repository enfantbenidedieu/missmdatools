# -*- coding: utf-8 -*-
from numpy import ones, array, sqrt,linalg,sign,nan_to_num,set_printoptions, errstate
from collections import namedtuple
set_printoptions(suppress=True)
errstate(invalid='ignore', divide='ignore')

def gSVD(X, 
         ncp = 5, 
         row_w = None, 
         col_w = None, 
         tol = 1e-7):
    """
    Generalized Singular Value Decomposition of a Matrix

    Compute the generalized singular value decomposition (gsvd) of a rectangular matrix with weights for rows and columns

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_columns)
        Input data.

    ncp : int, default = 5
        The number of dimensions kept in the results.

    row_w : 1d array-like of shape (n_samples,), default = None
        The rows weights.

    col_w : 1d array-like of shape (n_columns,), default = None
        The columns weights.

    tol : float, default = 1e-7
        A tolerance threshold to test whether the distance matrix is Euclidean : an eigenvalue is considered positive if it is larger 
        than ``-tol*lambda1`` where ``lambda1`` is the largest eigenvalue.

    Returns
    -------
    result : gSVDResult
        An object containing all the results for the generalized singular value decomposition (GSVD) with the following attributes:
            
        vs : 1d numpy array of shape (rank,)
            The singular values.
        vp : 1d numpy array of shape (rank,)
            The eigen values (= squared of singular values).
        U : 2d numpy array of shape (n_columns, rank)
            The left singular vectors.
        V : 2d numpy array of shape (n_rows, rank)
            The right singular vectors.
        rank : int
            The maximum number of components.
        ncp : int
            The number of components kepted.
        sumvp : float
            The sum of eigen values.
    """
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set number of rows and columns weights
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #set number of rows and columns
    n_rows, n_cols = X.shape

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set rows and columsn weights
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set row weights
    if row_w is None: 
        row_w = ones(n_rows)/n_rows
    else:
        row_w = array(row_w,dtype="float")
    
    # set columns weights
    if col_w is None: 
        col_w = ones(n_cols)
    else:
        col_w = array(col_w,dtype="float")

    # convert to array
    X = array(X,dtype="float")
    
    # multiply X with square root of rows and columns weights
    X *= sqrt(col_w)
    X *= sqrt(row_w[:,None])

    # singular value decomposition
    if n_cols < n_rows:
        svd = linalg.svd(X,full_matrices=False)
        U, V = svd[0], svd[2].T
    else:
        svd = linalg.svd(X.T,full_matrices=False)
        U, V = svd[2].T, svd[0]

    # set maximum number of components
    rank = sum(((svd[1]/svd[1][0])**2) > tol)
    
    # set number of components
    if ncp is None: 
        ncp = rank
    elif ncp < 0: 
        raise ValueError("ncp must be positive")
    else: 
        ncp = int(min(ncp,rank))

    if ncp > 1:
        mult = sign(V.sum(axis=0))
        mult[mult==0] = 1	
        U *= mult
        V *= mult

    # extract singular values decomposition
    vs = svd[1][:rank]
    # eigen values
    vp = vs ** 2 
    # recalibrate U and V using row weight and col weight
    U /= sqrt(row_w[:,None])
    V /= sqrt(col_w[:,None])
    # replace NAN, inf with 1e-15
    U = nan_to_num(U,nan=1e-15,posinf=1e-15,neginf=-1e-15)
    V = nan_to_num(V,nan=1e-15,posinf=1e-15,neginf=-1e-15)
    # select index and values which satisfied the condition
    num = vs < 1e-15
    if num.sum() == 1: 
        U[:,num][:,None] *= vs[num]
        V[:,num][:,None] *= vs[num]
    elif num.sum() > 1: 
        U[:,num] *= vs[num]
        V[:,num] *= vs[num]
    
    # converr to dictionary
    svd_ = {"vs":vs,"vp":vp,"U":U,"V":V,"rank":rank,"ncp":ncp,"sumvp":vp.sum()}
    # convert to namedtuple
    return namedtuple("gSVDResult",svd_.keys())(*svd_.values())
