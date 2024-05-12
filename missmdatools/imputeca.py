# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.weightstats import DescrStatsW

from .get_melt import get_melt
from .svd_triplet import svd_triplet

def imputeCA(X,
             n_components=5, 
             row_sup = None,
             col_sup = None,
             quanti_sup = None,
             quali_sup = None,
             ridge_coef = 1,
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
    pass