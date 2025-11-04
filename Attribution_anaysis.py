# coding='utf-8'
import sys


import lytools
import pingouin
import pingouin as pg
import xymap
from fontTools.subset import subset
from matplotlib.mlab import detrend
from matplotlib.pyplot import xticks
from numba.core.compiler_machinery import pass_info
from numba.cuda.libdevice import fdiv_rd
from openpyxl.styles.builtins import percent, total
from scipy.ndimage import label
# from green_driver_trend_contribution import *
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import t
from statsmodels.sandbox.regression.gmm import results_class_dict


from SI_anaysis import climate_variables

version = sys.version_info.major
assert version == 3, 'Python Version Error'
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import re
import xarray

from osgeo import ogr
from osgeo import osr
from tqdm import tqdm
from datetime import datetime
import matplotlib.dates as mdates
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import copyreg
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import types
from scipy.stats import gamma as gam
import math
import copy
import scipy
import sklearn
import random
# import h5py
from netCDF4 import Dataset
import shutil
import requests
from lytools import *
from osgeo import gdal

from osgeo import gdal

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from operator import itemgetter
from itertools import groupby
# import RegscorePy
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from mpl_toolkits.mplot3d import Axes3D
import pickle
from dateutil import relativedelta
from sklearn.inspection import permutation_importance
from pprint import pprint
T=Tools()
D = DIC_and_TIF(pixelsize=0.5)
centimeter_factor = 1/2.54


this_root = 'D:\Project3\\'
data_root = 'D:/Project3/Data/'
result_root = 'D:/Project3/Result/Nov//'
class Attribution_anaysis:
    def __init__(self):
        pass
    def run(self):
        self.detrend_deseasonalized()

        pass
    def detrend_deseasonalized(self):
        fdir=rf'D:\Project3\Data\LAI4g\extract_phenology_monthly\\'
        outdir=rf'D:\Project3\Data\LAI4g\extract_phenology_monthly_detrend_deseason\\'
        T.mk_dir(outdir,force=True)
        dic=T.load_npy_dir(fdir)
        result_dic={}
        for pix in dic:
            vals=dic[pix]
            n_years, n_months = vals.shape
            vals_T = vals.T

            ## reshape 38 years or 39years
            deseason_T = []
            for i in range(n_months):
                month_series = vals_T[i]
                month_mean = np.nanmean(month_series)
                deseason = month_series - month_mean
                detrend_deseason = T.detrend_vals(deseason)
                deseason_T.append(detrend_deseason)

            # 转回 (years, months)
            deseason_arr = np.array(deseason_T).T
            # plt.imshow(deseason_arr,interpolation='nearest',cmap='jet'
            #            )
            # plt.show()

            result_dic[pix] = deseason_arr
        outf=outdir+'extract_phenology_monthly_detrend_deseason.npy'

        T.save_npy(result_dic, outf)



        pass

def main():
    Attribution_anaysis().run()

    pass

if __name__ == '__main__':
    main()

