# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.weightstats import DescrStatsW

from .get_melt import get_melt
from .svd_triplet import svd_triplet

def imputeMCA(X,n_components=2):
    pass