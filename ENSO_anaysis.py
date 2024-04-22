# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg
# from green_driver_trend_contribution import *

version = sys.version_info.major
assert version == 3, 'Python Version Error'
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import re

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
T=Tools()




this_root = 'D:\Project3\\'
data_root = 'D:/Project3/Data/'
result_root = 'D:/Project3/Result/'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

class ENSO_anaysis():
    def __init__(self):


        self.weak_El_Nino_list = [2004,2005,2006,2007,2014,2015,2018,2019]

        self.weak_La_Nina_list = [1983,1984,1985, 2000,2001,2005, 2006,2008,2009,2016,2017,2018,]

        self.moderate_El_Nino_list = [1986,1987,1994,1995,2002,2003,2009,2010]
        self.moderate_La_Nina_list = [1995,1996,2011,2012,2020,]

        self.strong_El_Nino_list = [1987,1988, 1991,1992,]

        self.strong_La_Nina_list = [1988, 1989,1998,1999, 2000,2007,2008,2010,2011]
        self.very_strong_El_Nino_list = [1982,1983, 1997,1998,2015,2016]


        ### netural year = all years - ENSO years
        self.netural_list = [i for i in range(1982,2021) if i not in self.strong_El_Nino_list+self.strong_La_Nina_list+self.moderate_El_Nino_list+self.moderate_La_Nina_list]
        # print('netural_list',self.netural_list)
        pass
    def run(self):
        # self.ENSO_index_average_retrieval()
        self.ENSO_index_average_retrieval_method2()
        # self.ENSO_index_distance_retrieval()
        # self.ENSO_index_distance_retrieval_lagged()

        # self.ENSO_index_binary_retrieval()
        # self.plt_moving_window()
        # self.extract_data_based_on_ENSO()
        # self.calculate_relatve_change_based_on_ENSO()

        # self.extract_first_year_and_second_year_ENSO()
        # self.calculate_relatve_change_based_on_ENSO_first_second_year()
        # self.plot_spatial_average_ENSO_LAI()


        pass
    def ENSO_index_average_retrieval(self):
        indx='''
        1979     2024
1979     0.47     0.29    -0.05     0.21     0.27    -0.11    -0.11     0.47     0.38     0.23     0.53     0.63
1980     0.33     0.20     0.39     0.51     0.45     0.55     0.47     0.03     0.12     0.02    -0.07    -0.12
1981    -0.36    -0.23     0.33     0.43    -0.24    -0.70    -0.61    -0.34    -0.07    -0.16    -0.27    -0.19
1982    -0.43    -0.49    -0.27    -0.36    -0.12     0.62     1.65     1.91     1.69     1.78     2.14     2.37
1983     2.48     2.68     2.61     2.76     2.86     1.98     0.63    -0.17    -0.49    -0.53    -0.47    -0.50
1984    -0.54    -0.56    -0.16     0.07    -0.45    -0.66    -0.34    -0.20    -0.14    -0.19    -0.58    -0.28
1985    -0.25    -0.58    -0.60    -0.76    -1.23    -0.67    -0.10    -0.49    -0.56    -0.08    -0.06    -0.42
1986    -0.40    -0.34    -0.40    -0.53    -0.26     0.01     0.41     0.97     1.29     0.56     0.64     1.06
1987     1.03     1.16     1.57     1.74     1.96     2.07     1.88     1.46     1.26     1.18     0.90     0.79
1988     0.63     0.33     0.17    -0.01    -0.39    -1.17    -1.78    -1.81    -1.80    -1.59    -1.70    -1.55
1989    -1.12    -1.10    -1.24    -1.09    -1.03    -0.99    -1.08    -0.66    -0.53    -0.49    -0.36    -0.07
1990     0.13     0.45     0.61     0.17     0.08     0.05     0.13     0.02     0.22    -0.07     0.12     0.31
1991     0.19     0.08     0.20     0.20     0.35     0.97     0.91     0.37     0.54     0.97     1.04     1.23
1992     1.68     1.58     1.71     1.97     1.68     1.51     0.62    -0.02     0.40     0.67     0.62     0.73
1993     0.79     0.89     0.77     0.98     1.47     1.51     0.90     0.60     0.61     0.96     0.70     0.27
1994    -0.01    -0.21    -0.23     0.01    -0.03     0.25     0.96     0.87     1.10     1.52     1.01     0.91
1995     0.82     0.54     0.17     0.13     0.14     0.01    -0.29    -0.70    -0.94    -0.74    -0.77    -0.90
1996    -0.85    -0.80    -0.66    -0.76    -0.85    -0.91    -0.83    -0.64    -0.31    -0.37    -0.34    -0.47
1997    -0.65    -0.72    -0.30     0.15     0.71     2.34     2.26     2.26     2.20     2.06     2.14     2.11
1998     2.30     2.45     2.27     2.61     2.33     0.42    -1.53    -1.82    -1.40    -1.29    -1.34    -1.29
1999    -1.28    -1.20    -1.15    -1.16    -1.33    -1.26    -1.18    -1.03    -1.16    -1.33    -1.33    -1.43
2000    -1.25    -1.26    -1.37    -0.90    -0.93    -1.18    -0.61    -0.08    -0.35    -0.52    -0.86    -0.79
2001    -0.83    -0.86    -0.78    -0.55    -0.51    -0.73     0.04     0.38    -0.06    -0.22    -0.28     0.06
2002     0.09    -0.26    -0.20    -0.36    -0.14     0.34     0.44     1.01     0.88     0.84     0.84     0.91
2003     0.82     0.63     0.53    -0.11    -0.61    -0.10     0.01     0.05     0.18     0.28     0.30     0.14
2004     0.19    -0.05    -0.43    -0.23    -0.46    -0.39     0.46     0.77     0.59     0.40     0.56     0.48
2005     0.09     0.61     0.83     0.14     0.19     0.20    -0.00     0.01    -0.02    -0.66    -0.70    -0.70
2006    -0.64    -0.48    -0.62    -0.84    -0.42    -0.18     0.21     0.59     0.65     0.77     1.00     0.64
2007     0.64     0.39    -0.19    -0.32    -0.41    -0.83    -0.75    -0.90    -1.06    -1.13    -1.12    -1.19
2008    -1.06    -1.27    -1.52    -1.11    -0.99    -0.80    -0.78    -1.01    -1.02    -1.08    -0.98    -1.01
2009    -1.01    -0.85    -0.95    -0.81    -0.72    -0.05     0.56     0.56     0.42     0.56     1.04     0.95
2010     0.93     1.28     1.33     0.49    -0.12    -1.29    -2.43    -2.38    -2.26    -2.16    -2.01    -1.86
2011    -1.77    -1.59    -1.75    -1.69    -1.23    -1.02    -0.75    -0.81    -1.08    -1.30    -1.14    -1.18
2012    -1.07    -0.68    -0.58    -0.38    -0.32    -0.28     0.34    -0.02    -0.29    -0.19    -0.03    -0.03
2013    -0.06    -0.08    -0.12    -0.35    -0.68    -1.14    -0.79    -0.45    -0.33    -0.13    -0.16    -0.34
2014    -0.50    -0.42    -0.05    -0.16    -0.17     0.00     0.42     0.23    -0.09     0.12     0.37     0.35
2015     0.23     0.06     0.15     0.31     0.95     1.90     1.79     1.95     2.24     2.15     1.94     1.93
2016     1.94     1.81     1.32     1.33     1.24     0.36    -0.53    -0.27    -0.29    -0.54    -0.48    -0.37
2017    -0.43    -0.42    -0.58    -0.19     0.19    -0.24    -0.63    -0.73    -0.74    -0.59    -0.62    -0.74
2018    -0.79    -0.71    -0.80    -1.32    -0.94    -0.52    -0.05     0.46     0.62     0.52     0.33     0.18
2019     0.09     0.50     0.76     0.30     0.23     0.35     0.30     0.32     0.16     0.31     0.47     0.36
2020     0.25     0.26     0.13    -0.15    -0.23    -0.68    -0.94    -0.96    -1.15    -1.17    -1.13    -1.14
2021    -1.20    -0.96    -0.79    -0.95    -1.07    -1.05    -1.44    -1.29    -1.38    -1.46    -1.39    -1.20
2022    -1.01    -0.98    -1.31    -1.61    -1.63    -1.90    -2.17    -1.75    -1.73    -1.73    -1.53    -1.28
2023    -1.11    -0.91    -0.76    -0.37    -0.05     0.43     0.50     0.51     0.69     0.48     0.91     1.13
2024     0.71     0.70  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00
        '''
        format_indx=indx.split('\n')
        format_indx=format_indx[1:-1]
        result_dic={}
        for line in format_indx:
            line=line.split()
            # print(line)
            year=int(line[0])
            vals=line[1:]
            vals=[float(i) for i in vals]
            result_dic[year]=vals
        # print(result_dic)
        year_list=[i for i in range(1982,2021)]
        # print(year_list)
        #### append all years month data to a list
        vals_list=[]


        for year in year_list:
            vals=result_dic[year]
            vals_list.append(vals)
        vals_list=np.array(vals_list)
        vals_list_flatten=vals_list.flatten()

        growing_season_Northern_dic_average={}
        growing_season_Southern_dic_average={}
        growing_season_tropical_dic_average={}



        for year in year_list:

            ## northern hemisphere April to October

            growing_season_Northern=vals_list_flatten[(year-1982)*12+4:(year-1982)*12+10]
            average_Northern=np.nanmean(growing_season_Northern)

            growing_season_Northern_dic_average[year]=average_Northern


        for year in year_list:
            if year>=2020:
                break

            growing_season_Southern=vals_list_flatten[(year-1982)*12+10:(year-1982+1)*12+4]
            average_Southern=np.nanmean(growing_season_Southern)
            growing_season_Southern_dic_average[year]=average_Southern



        for year in year_list:
            growing_season_tropical=vals_list_flatten[(year-1982)*12:(year-1982+1)*12]
            average_tropical=np.nanmean(growing_season_tropical)
            growing_season_tropical_dic_average[year]=average_tropical




        #### add into df
        df= T.load_df(data_root+rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            index = row.year_range

            pix = row['pix']
            r,c=pix
            if r<120:
                NDVI_list.append(np.nan)
                continue
            elif 120<=r<240:
                ## index type to string
                val=growing_season_Northern_dic_average[index+1982]
            elif 240<=r<480:
                val=growing_season_tropical_dic_average[index+1982]
            elif r>=480:
                val=growing_season_Southern_dic_average[index+1982]
            else:
                raise
            NDVI_list.append(val)

        df['ENSO_index_average'] = NDVI_list
        df = df.dropna(subset=['ENSO_index_average'])
        outf=data_root+rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df,outf)
        # ## xlxs
        T.df_to_excel(df, outf)

        pass

    def ENSO_index_average_retrieval_method2(self):  #### whole year
        indx = '''
            1979     2024
    1979     0.47     0.29    -0.05     0.21     0.27    -0.11    -0.11     0.47     0.38     0.23     0.53     0.63
    1980     0.33     0.20     0.39     0.51     0.45     0.55     0.47     0.03     0.12     0.02    -0.07    -0.12
    1981    -0.36    -0.23     0.33     0.43    -0.24    -0.70    -0.61    -0.34    -0.07    -0.16    -0.27    -0.19
    1982    -0.43    -0.49    -0.27    -0.36    -0.12     0.62     1.65     1.91     1.69     1.78     2.14     2.37
    1983     2.48     2.68     2.61     2.76     2.86     1.98     0.63    -0.17    -0.49    -0.53    -0.47    -0.50
    1984    -0.54    -0.56    -0.16     0.07    -0.45    -0.66    -0.34    -0.20    -0.14    -0.19    -0.58    -0.28
    1985    -0.25    -0.58    -0.60    -0.76    -1.23    -0.67    -0.10    -0.49    -0.56    -0.08    -0.06    -0.42
    1986    -0.40    -0.34    -0.40    -0.53    -0.26     0.01     0.41     0.97     1.29     0.56     0.64     1.06
    1987     1.03     1.16     1.57     1.74     1.96     2.07     1.88     1.46     1.26     1.18     0.90     0.79
    1988     0.63     0.33     0.17    -0.01    -0.39    -1.17    -1.78    -1.81    -1.80    -1.59    -1.70    -1.55
    1989    -1.12    -1.10    -1.24    -1.09    -1.03    -0.99    -1.08    -0.66    -0.53    -0.49    -0.36    -0.07
    1990     0.13     0.45     0.61     0.17     0.08     0.05     0.13     0.02     0.22    -0.07     0.12     0.31
    1991     0.19     0.08     0.20     0.20     0.35     0.97     0.91     0.37     0.54     0.97     1.04     1.23
    1992     1.68     1.58     1.71     1.97     1.68     1.51     0.62    -0.02     0.40     0.67     0.62     0.73
    1993     0.79     0.89     0.77     0.98     1.47     1.51     0.90     0.60     0.61     0.96     0.70     0.27
    1994    -0.01    -0.21    -0.23     0.01    -0.03     0.25     0.96     0.87     1.10     1.52     1.01     0.91
    1995     0.82     0.54     0.17     0.13     0.14     0.01    -0.29    -0.70    -0.94    -0.74    -0.77    -0.90
    1996    -0.85    -0.80    -0.66    -0.76    -0.85    -0.91    -0.83    -0.64    -0.31    -0.37    -0.34    -0.47
    1997    -0.65    -0.72    -0.30     0.15     0.71     2.34     2.26     2.26     2.20     2.06     2.14     2.11
    1998     2.30     2.45     2.27     2.61     2.33     0.42    -1.53    -1.82    -1.40    -1.29    -1.34    -1.29
    1999    -1.28    -1.20    -1.15    -1.16    -1.33    -1.26    -1.18    -1.03    -1.16    -1.33    -1.33    -1.43
    2000    -1.25    -1.26    -1.37    -0.90    -0.93    -1.18    -0.61    -0.08    -0.35    -0.52    -0.86    -0.79
    2001    -0.83    -0.86    -0.78    -0.55    -0.51    -0.73     0.04     0.38    -0.06    -0.22    -0.28     0.06
    2002     0.09    -0.26    -0.20    -0.36    -0.14     0.34     0.44     1.01     0.88     0.84     0.84     0.91
    2003     0.82     0.63     0.53    -0.11    -0.61    -0.10     0.01     0.05     0.18     0.28     0.30     0.14
    2004     0.19    -0.05    -0.43    -0.23    -0.46    -0.39     0.46     0.77     0.59     0.40     0.56     0.48
    2005     0.09     0.61     0.83     0.14     0.19     0.20    -0.00     0.01    -0.02    -0.66    -0.70    -0.70
    2006    -0.64    -0.48    -0.62    -0.84    -0.42    -0.18     0.21     0.59     0.65     0.77     1.00     0.64
    2007     0.64     0.39    -0.19    -0.32    -0.41    -0.83    -0.75    -0.90    -1.06    -1.13    -1.12    -1.19
    2008    -1.06    -1.27    -1.52    -1.11    -0.99    -0.80    -0.78    -1.01    -1.02    -1.08    -0.98    -1.01
    2009    -1.01    -0.85    -0.95    -0.81    -0.72    -0.05     0.56     0.56     0.42     0.56     1.04     0.95
    2010     0.93     1.28     1.33     0.49    -0.12    -1.29    -2.43    -2.38    -2.26    -2.16    -2.01    -1.86
    2011    -1.77    -1.59    -1.75    -1.69    -1.23    -1.02    -0.75    -0.81    -1.08    -1.30    -1.14    -1.18
    2012    -1.07    -0.68    -0.58    -0.38    -0.32    -0.28     0.34    -0.02    -0.29    -0.19    -0.03    -0.03
    2013    -0.06    -0.08    -0.12    -0.35    -0.68    -1.14    -0.79    -0.45    -0.33    -0.13    -0.16    -0.34
    2014    -0.50    -0.42    -0.05    -0.16    -0.17     0.00     0.42     0.23    -0.09     0.12     0.37     0.35
    2015     0.23     0.06     0.15     0.31     0.95     1.90     1.79     1.95     2.24     2.15     1.94     1.93
    2016     1.94     1.81     1.32     1.33     1.24     0.36    -0.53    -0.27    -0.29    -0.54    -0.48    -0.37
    2017    -0.43    -0.42    -0.58    -0.19     0.19    -0.24    -0.63    -0.73    -0.74    -0.59    -0.62    -0.74
    2018    -0.79    -0.71    -0.80    -1.32    -0.94    -0.52    -0.05     0.46     0.62     0.52     0.33     0.18
    2019     0.09     0.50     0.76     0.30     0.23     0.35     0.30     0.32     0.16     0.31     0.47     0.36
    2020     0.25     0.26     0.13    -0.15    -0.23    -0.68    -0.94    -0.96    -1.15    -1.17    -1.13    -1.14
    2021    -1.20    -0.96    -0.79    -0.95    -1.07    -1.05    -1.44    -1.29    -1.38    -1.46    -1.39    -1.20
    2022    -1.01    -0.98    -1.31    -1.61    -1.63    -1.90    -2.17    -1.75    -1.73    -1.73    -1.53    -1.28
    2023    -1.11    -0.91    -0.76    -0.37    -0.05     0.43     0.50     0.51     0.69     0.48     0.91     1.13
    2024     0.71     0.70  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00
            '''
        format_indx = indx.split('\n')
        format_indx = format_indx[1:-1]
        result_dic = {}
        for line in format_indx:
            line = line.split()
            # print(line)
            year = int(line[0])
            vals = line[1:]
            vals = [float(i) for i in vals]
            result_dic[year] = vals
        # print(result_dic)
        year_list = [i for i in range(1982, 2021)]
        # print(year_list)
        #### append all years month data to a list
        vals_list = []

        for year in year_list:
            vals = result_dic[year]
            vals_list.append(vals)
        vals_list = np.array(vals_list)
        vals_list_flatten = vals_list.flatten()

        growing_season_Northern_dic_average = {}
        growing_season_Southern_dic_average = {}
        growing_season_tropical_dic_average = {}

        for year in year_list:
            ## northern hemisphere April to October

            growing_season_Northern = vals_list_flatten[(year - 1982) * 12 :(year - 1982) * 12 + 12]
            average_Northern = np.nanmean(growing_season_Northern)

            growing_season_Northern_dic_average[year] = average_Northern

        for year in year_list:
            if year >= 2020:
                break

            growing_season_Southern = vals_list_flatten[(year - 1982) * 12 :(year - 1982) * 12 + 12]
            average_Southern = np.nanmean(growing_season_Southern)
            growing_season_Southern_dic_average[year] = average_Southern

        for year in year_list:
            growing_season_tropical = vals_list_flatten[(year - 1982) * 12 :(year - 1982) * 12 + 12]
            average_tropical = np.nanmean(growing_season_tropical)
            growing_season_tropical_dic_average[year] = average_tropical

        #### add into df
        df = T.load_df(data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            index = row.year_range

            pix = row['pix']
            r, c = pix
            if r < 120:
                NDVI_list.append(np.nan)
                continue
            if index - 1 < 0:
                NDVI_list.append(np.nan)
                continue
            elif 120 <= r < 240:
                ## index type to string
                val = growing_season_Northern_dic_average[index-1 + 1982]
            elif 240 <= r < 480:
                val = growing_season_tropical_dic_average[index - 1 + 1982]
            elif r >= 480:
                val = growing_season_Southern_dic_average[index - 1 + 1982]
            else:
                raise
            NDVI_list.append(val)

        df['ENSO_index_average_lagged_whole_year'] = NDVI_list
        df = df.dropna(subset=['ENSO_index_average'])
        outf = data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        # ## xlxs
        T.df_to_excel(df, outf)

        pass



    def ENSO_index_average_retrieval_lagged(self):
        indx = '''
            1979     2024
    1979     0.47     0.29    -0.05     0.21     0.27    -0.11    -0.11     0.47     0.38     0.23     0.53     0.63
    1980     0.33     0.20     0.39     0.51     0.45     0.55     0.47     0.03     0.12     0.02    -0.07    -0.12
    1981    -0.36    -0.23     0.33     0.43    -0.24    -0.70    -0.61    -0.34    -0.07    -0.16    -0.27    -0.19
    1982    -0.43    -0.49    -0.27    -0.36    -0.12     0.62     1.65     1.91     1.69     1.78     2.14     2.37
    1983     2.48     2.68     2.61     2.76     2.86     1.98     0.63    -0.17    -0.49    -0.53    -0.47    -0.50
    1984    -0.54    -0.56    -0.16     0.07    -0.45    -0.66    -0.34    -0.20    -0.14    -0.19    -0.58    -0.28
    1985    -0.25    -0.58    -0.60    -0.76    -1.23    -0.67    -0.10    -0.49    -0.56    -0.08    -0.06    -0.42
    1986    -0.40    -0.34    -0.40    -0.53    -0.26     0.01     0.41     0.97     1.29     0.56     0.64     1.06
    1987     1.03     1.16     1.57     1.74     1.96     2.07     1.88     1.46     1.26     1.18     0.90     0.79
    1988     0.63     0.33     0.17    -0.01    -0.39    -1.17    -1.78    -1.81    -1.80    -1.59    -1.70    -1.55
    1989    -1.12    -1.10    -1.24    -1.09    -1.03    -0.99    -1.08    -0.66    -0.53    -0.49    -0.36    -0.07
    1990     0.13     0.45     0.61     0.17     0.08     0.05     0.13     0.02     0.22    -0.07     0.12     0.31
    1991     0.19     0.08     0.20     0.20     0.35     0.97     0.91     0.37     0.54     0.97     1.04     1.23
    1992     1.68     1.58     1.71     1.97     1.68     1.51     0.62    -0.02     0.40     0.67     0.62     0.73
    1993     0.79     0.89     0.77     0.98     1.47     1.51     0.90     0.60     0.61     0.96     0.70     0.27
    1994    -0.01    -0.21    -0.23     0.01    -0.03     0.25     0.96     0.87     1.10     1.52     1.01     0.91
    1995     0.82     0.54     0.17     0.13     0.14     0.01    -0.29    -0.70    -0.94    -0.74    -0.77    -0.90
    1996    -0.85    -0.80    -0.66    -0.76    -0.85    -0.91    -0.83    -0.64    -0.31    -0.37    -0.34    -0.47
    1997    -0.65    -0.72    -0.30     0.15     0.71     2.34     2.26     2.26     2.20     2.06     2.14     2.11
    1998     2.30     2.45     2.27     2.61     2.33     0.42    -1.53    -1.82    -1.40    -1.29    -1.34    -1.29
    1999    -1.28    -1.20    -1.15    -1.16    -1.33    -1.26    -1.18    -1.03    -1.16    -1.33    -1.33    -1.43
    2000    -1.25    -1.26    -1.37    -0.90    -0.93    -1.18    -0.61    -0.08    -0.35    -0.52    -0.86    -0.79
    2001    -0.83    -0.86    -0.78    -0.55    -0.51    -0.73     0.04     0.38    -0.06    -0.22    -0.28     0.06
    2002     0.09    -0.26    -0.20    -0.36    -0.14     0.34     0.44     1.01     0.88     0.84     0.84     0.91
    2003     0.82     0.63     0.53    -0.11    -0.61    -0.10     0.01     0.05     0.18     0.28     0.30     0.14
    2004     0.19    -0.05    -0.43    -0.23    -0.46    -0.39     0.46     0.77     0.59     0.40     0.56     0.48
    2005     0.09     0.61     0.83     0.14     0.19     0.20    -0.00     0.01    -0.02    -0.66    -0.70    -0.70
    2006    -0.64    -0.48    -0.62    -0.84    -0.42    -0.18     0.21     0.59     0.65     0.77     1.00     0.64
    2007     0.64     0.39    -0.19    -0.32    -0.41    -0.83    -0.75    -0.90    -1.06    -1.13    -1.12    -1.19
    2008    -1.06    -1.27    -1.52    -1.11    -0.99    -0.80    -0.78    -1.01    -1.02    -1.08    -0.98    -1.01
    2009    -1.01    -0.85    -0.95    -0.81    -0.72    -0.05     0.56     0.56     0.42     0.56     1.04     0.95
    2010     0.93     1.28     1.33     0.49    -0.12    -1.29    -2.43    -2.38    -2.26    -2.16    -2.01    -1.86
    2011    -1.77    -1.59    -1.75    -1.69    -1.23    -1.02    -0.75    -0.81    -1.08    -1.30    -1.14    -1.18
    2012    -1.07    -0.68    -0.58    -0.38    -0.32    -0.28     0.34    -0.02    -0.29    -0.19    -0.03    -0.03
    2013    -0.06    -0.08    -0.12    -0.35    -0.68    -1.14    -0.79    -0.45    -0.33    -0.13    -0.16    -0.34
    2014    -0.50    -0.42    -0.05    -0.16    -0.17     0.00     0.42     0.23    -0.09     0.12     0.37     0.35
    2015     0.23     0.06     0.15     0.31     0.95     1.90     1.79     1.95     2.24     2.15     1.94     1.93
    2016     1.94     1.81     1.32     1.33     1.24     0.36    -0.53    -0.27    -0.29    -0.54    -0.48    -0.37
    2017    -0.43    -0.42    -0.58    -0.19     0.19    -0.24    -0.63    -0.73    -0.74    -0.59    -0.62    -0.74
    2018    -0.79    -0.71    -0.80    -1.32    -0.94    -0.52    -0.05     0.46     0.62     0.52     0.33     0.18
    2019     0.09     0.50     0.76     0.30     0.23     0.35     0.30     0.32     0.16     0.31     0.47     0.36
    2020     0.25     0.26     0.13    -0.15    -0.23    -0.68    -0.94    -0.96    -1.15    -1.17    -1.13    -1.14
    2021    -1.20    -0.96    -0.79    -0.95    -1.07    -1.05    -1.44    -1.29    -1.38    -1.46    -1.39    -1.20
    2022    -1.01    -0.98    -1.31    -1.61    -1.63    -1.90    -2.17    -1.75    -1.73    -1.73    -1.53    -1.28
    2023    -1.11    -0.91    -0.76    -0.37    -0.05     0.43     0.50     0.51     0.69     0.48     0.91     1.13
    2024     0.71     0.70  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00
            '''
        format_indx = indx.split('\n')
        format_indx = format_indx[1:-1]
        result_dic = {}
        for line in format_indx:
            line = line.split()
            # print(line)
            year = int(line[0])
            vals = line[1:]
            vals = [float(i) for i in vals]
            result_dic[year] = vals
        # print(result_dic)
        year_list = [i for i in range(1982, 2021)]
        # print(year_list)
        #### append all years month data to a list
        vals_list = []

        for year in year_list:
            vals = result_dic[year]
            vals_list.append(vals)
        vals_list = np.array(vals_list)
        vals_list_flatten = vals_list.flatten()

        growing_season_Northern_dic_average = {}
        growing_season_Southern_dic_average = {}
        growing_season_tropical_dic_average = {}

        for year in year_list:
            ## northern hemisphere April to October

            growing_season_Northern = vals_list_flatten[(year - 1982) * 12 + 4:(year - 1982) * 12 + 10]
            average_Northern = np.nanmean(growing_season_Northern)

            growing_season_Northern_dic_average[year] = average_Northern

        for year in year_list:
            if year >= 2020:
                break

            growing_season_Southern = vals_list_flatten[(year - 1982) * 12 + 10:(year - 1982 + 1) * 12 + 4]
            average_Southern = np.nanmean(growing_season_Southern)
            growing_season_Southern_dic_average[year] = average_Southern

        for year in year_list:
            growing_season_tropical = vals_list_flatten[(year - 1982) * 12:(year - 1982 + 1) * 12]
            average_tropical = np.nanmean(growing_season_tropical)
            growing_season_tropical_dic_average[year] = average_tropical

        #### add into df
        df = T.load_df(data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            index = row.year_range

            pix = row['pix']
            r, c = pix
            if r < 120:
                NDVI_list.append(np.nan)
                continue
            if index - 1 < 0:
                NDVI_list.append(np.nan)
                continue
            elif 120 <= r < 240:
                ## index type to string
                val = growing_season_Northern_dic_average[index + 1982 - 1]
            elif 240 <= r < 480:
                val = growing_season_tropical_dic_average[index + 1982 - 1]
            elif r >= 480:
                val = growing_season_Southern_dic_average[index + 1982 - 1]
            else:
                raise
            NDVI_list.append(val)

        df['ENSO_index_average_lagged'] = NDVI_list
        df = df.dropna(subset=['ENSO_index_average'])
        outf = data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        # ## xlxs
        T.df_to_excel(df, outf)

        pass

    pass
    def ENSO_index_distance_retrieval(self):
        indx = '''
            1979     2024
    1979     0.47     0.29    -0.05     0.21     0.27    -0.11    -0.11     0.47     0.38     0.23     0.53     0.63
    1980     0.33     0.20     0.39     0.51     0.45     0.55     0.47     0.03     0.12     0.02    -0.07    -0.12
    1981    -0.36    -0.23     0.33     0.43    -0.24    -0.70    -0.61    -0.34    -0.07    -0.16    -0.27    -0.19
    1982    -0.43    -0.49    -0.27    -0.36    -0.12     0.62     1.65     1.91     1.69     1.78     2.14     2.37
    1983     2.48     2.68     2.61     2.76     2.86     1.98     0.63    -0.17    -0.49    -0.53    -0.47    -0.50
    1984    -0.54    -0.56    -0.16     0.07    -0.45    -0.66    -0.34    -0.20    -0.14    -0.19    -0.58    -0.28
    1985    -0.25    -0.58    -0.60    -0.76    -1.23    -0.67    -0.10    -0.49    -0.56    -0.08    -0.06    -0.42
    1986    -0.40    -0.34    -0.40    -0.53    -0.26     0.01     0.41     0.97     1.29     0.56     0.64     1.06
    1987     1.03     1.16     1.57     1.74     1.96     2.07     1.88     1.46     1.26     1.18     0.90     0.79
    1988     0.63     0.33     0.17    -0.01    -0.39    -1.17    -1.78    -1.81    -1.80    -1.59    -1.70    -1.55
    1989    -1.12    -1.10    -1.24    -1.09    -1.03    -0.99    -1.08    -0.66    -0.53    -0.49    -0.36    -0.07
    1990     0.13     0.45     0.61     0.17     0.08     0.05     0.13     0.02     0.22    -0.07     0.12     0.31
    1991     0.19     0.08     0.20     0.20     0.35     0.97     0.91     0.37     0.54     0.97     1.04     1.23
    1992     1.68     1.58     1.71     1.97     1.68     1.51     0.62    -0.02     0.40     0.67     0.62     0.73
    1993     0.79     0.89     0.77     0.98     1.47     1.51     0.90     0.60     0.61     0.96     0.70     0.27
    1994    -0.01    -0.21    -0.23     0.01    -0.03     0.25     0.96     0.87     1.10     1.52     1.01     0.91
    1995     0.82     0.54     0.17     0.13     0.14     0.01    -0.29    -0.70    -0.94    -0.74    -0.77    -0.90
    1996    -0.85    -0.80    -0.66    -0.76    -0.85    -0.91    -0.83    -0.64    -0.31    -0.37    -0.34    -0.47
    1997    -0.65    -0.72    -0.30     0.15     0.71     2.34     2.26     2.26     2.20     2.06     2.14     2.11
    1998     2.30     2.45     2.27     2.61     2.33     0.42    -1.53    -1.82    -1.40    -1.29    -1.34    -1.29
    1999    -1.28    -1.20    -1.15    -1.16    -1.33    -1.26    -1.18    -1.03    -1.16    -1.33    -1.33    -1.43
    2000    -1.25    -1.26    -1.37    -0.90    -0.93    -1.18    -0.61    -0.08    -0.35    -0.52    -0.86    -0.79
    2001    -0.83    -0.86    -0.78    -0.55    -0.51    -0.73     0.04     0.38    -0.06    -0.22    -0.28     0.06
    2002     0.09    -0.26    -0.20    -0.36    -0.14     0.34     0.44     1.01     0.88     0.84     0.84     0.91
    2003     0.82     0.63     0.53    -0.11    -0.61    -0.10     0.01     0.05     0.18     0.28     0.30     0.14
    2004     0.19    -0.05    -0.43    -0.23    -0.46    -0.39     0.46     0.77     0.59     0.40     0.56     0.48
    2005     0.09     0.61     0.83     0.14     0.19     0.20    -0.00     0.01    -0.02    -0.66    -0.70    -0.70
    2006    -0.64    -0.48    -0.62    -0.84    -0.42    -0.18     0.21     0.59     0.65     0.77     1.00     0.64
    2007     0.64     0.39    -0.19    -0.32    -0.41    -0.83    -0.75    -0.90    -1.06    -1.13    -1.12    -1.19
    2008    -1.06    -1.27    -1.52    -1.11    -0.99    -0.80    -0.78    -1.01    -1.02    -1.08    -0.98    -1.01
    2009    -1.01    -0.85    -0.95    -0.81    -0.72    -0.05     0.56     0.56     0.42     0.56     1.04     0.95
    2010     0.93     1.28     1.33     0.49    -0.12    -1.29    -2.43    -2.38    -2.26    -2.16    -2.01    -1.86
    2011    -1.77    -1.59    -1.75    -1.69    -1.23    -1.02    -0.75    -0.81    -1.08    -1.30    -1.14    -1.18
    2012    -1.07    -0.68    -0.58    -0.38    -0.32    -0.28     0.34    -0.02    -0.29    -0.19    -0.03    -0.03
    2013    -0.06    -0.08    -0.12    -0.35    -0.68    -1.14    -0.79    -0.45    -0.33    -0.13    -0.16    -0.34
    2014    -0.50    -0.42    -0.05    -0.16    -0.17     0.00     0.42     0.23    -0.09     0.12     0.37     0.35
    2015     0.23     0.06     0.15     0.31     0.95     1.90     1.79     1.95     2.24     2.15     1.94     1.93
    2016     1.94     1.81     1.32     1.33     1.24     0.36    -0.53    -0.27    -0.29    -0.54    -0.48    -0.37
    2017    -0.43    -0.42    -0.58    -0.19     0.19    -0.24    -0.63    -0.73    -0.74    -0.59    -0.62    -0.74
    2018    -0.79    -0.71    -0.80    -1.32    -0.94    -0.52    -0.05     0.46     0.62     0.52     0.33     0.18
    2019     0.09     0.50     0.76     0.30     0.23     0.35     0.30     0.32     0.16     0.31     0.47     0.36
    2020     0.25     0.26     0.13    -0.15    -0.23    -0.68    -0.94    -0.96    -1.15    -1.17    -1.13    -1.14
    2021    -1.20    -0.96    -0.79    -0.95    -1.07    -1.05    -1.44    -1.29    -1.38    -1.46    -1.39    -1.20
    2022    -1.01    -0.98    -1.31    -1.61    -1.63    -1.90    -2.17    -1.75    -1.73    -1.73    -1.53    -1.28
    2023    -1.11    -0.91    -0.76    -0.37    -0.05     0.43     0.50     0.51     0.69     0.48     0.91     1.13
    2024     0.71     0.70  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00
            '''
        format_indx = indx.split('\n')
        format_indx = format_indx[1:-1]
        result_dic = {}
        for line in format_indx:
            line = line.split()
            # print(line)
            year = int(line[0])
            vals = line[1:]
            vals = [float(i) for i in vals]
            result_dic[year] = vals
        # print(result_dic)
        year_list = [i for i in range(1982, 2021)]
        # print(year_list)
        #### append all years month data to a list
        vals_list = []

        for year in year_list:
            vals = result_dic[year]
            vals_list.append(vals)
        vals_list = np.array(vals_list)
        vals_list_flatten = vals_list.flatten()


        growing_season_Northern_dic_distance = {}
        growing_season_Southern_dic_distance = {}
        growing_season_tropical_dic_distance = {}

        for year in year_list:
            ## northern hemisphere April to October

            growing_season_Northern = vals_list_flatten[(year - 1982) * 12 + 4:(year - 1982) * 12 + 10]

            distance_north = np.nanmax(growing_season_Northern) - np.nanmin(growing_season_Northern)

            growing_season_Northern_dic_distance[year] = distance_north

        for year in year_list:
            if year >= 2020:
                break

            growing_season_Southern = vals_list_flatten[(year - 1982) * 12 + 10:(year - 1982 + 1) * 12 + 4]

            distance_southern = np.nanmax(growing_season_Southern) - np.nanmin(growing_season_Southern)
            growing_season_Southern_dic_distance[year] = distance_southern

        for year in year_list:
            growing_season_tropical = vals_list_flatten[(year - 1982) * 12:(year - 1982 + 1) * 12]


            distance_tropical = np.nanmax(growing_season_tropical) - np.nanmin(growing_season_tropical)
            growing_season_tropical_dic_distance[year] = distance_tropical

        #### add into df
        df = T.load_df(data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            index = row.year_range

            pix = row['pix']
            r, c = pix
            if r < 120:
                NDVI_list.append(np.nan)
                continue


            elif 120 <= r < 240:
                ## index type to string
                val = growing_season_Northern_dic_distance[index+ 1982]
            elif 240 <= r < 480:
                val = growing_season_tropical_dic_distance[index+ 1982]
            elif r >= 480:
                val = growing_season_Southern_dic_distance[index+ 1982]
            else:
                raise
            NDVI_list.append(val)

        df['ENSO_index_distance'] = NDVI_list
        # df = df.dropna(subset=['ENSO_index_average'])
        outf = data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        # ## xlxs
        T.df_to_excel(df, outf)

        pass

    def ENSO_index_distance_retrieval_lagged(self):
        indx = '''
            1979     2024
    1979     0.47     0.29    -0.05     0.21     0.27    -0.11    -0.11     0.47     0.38     0.23     0.53     0.63
    1980     0.33     0.20     0.39     0.51     0.45     0.55     0.47     0.03     0.12     0.02    -0.07    -0.12
    1981    -0.36    -0.23     0.33     0.43    -0.24    -0.70    -0.61    -0.34    -0.07    -0.16    -0.27    -0.19
    1982    -0.43    -0.49    -0.27    -0.36    -0.12     0.62     1.65     1.91     1.69     1.78     2.14     2.37
    1983     2.48     2.68     2.61     2.76     2.86     1.98     0.63    -0.17    -0.49    -0.53    -0.47    -0.50
    1984    -0.54    -0.56    -0.16     0.07    -0.45    -0.66    -0.34    -0.20    -0.14    -0.19    -0.58    -0.28
    1985    -0.25    -0.58    -0.60    -0.76    -1.23    -0.67    -0.10    -0.49    -0.56    -0.08    -0.06    -0.42
    1986    -0.40    -0.34    -0.40    -0.53    -0.26     0.01     0.41     0.97     1.29     0.56     0.64     1.06
    1987     1.03     1.16     1.57     1.74     1.96     2.07     1.88     1.46     1.26     1.18     0.90     0.79
    1988     0.63     0.33     0.17    -0.01    -0.39    -1.17    -1.78    -1.81    -1.80    -1.59    -1.70    -1.55
    1989    -1.12    -1.10    -1.24    -1.09    -1.03    -0.99    -1.08    -0.66    -0.53    -0.49    -0.36    -0.07
    1990     0.13     0.45     0.61     0.17     0.08     0.05     0.13     0.02     0.22    -0.07     0.12     0.31
    1991     0.19     0.08     0.20     0.20     0.35     0.97     0.91     0.37     0.54     0.97     1.04     1.23
    1992     1.68     1.58     1.71     1.97     1.68     1.51     0.62    -0.02     0.40     0.67     0.62     0.73
    1993     0.79     0.89     0.77     0.98     1.47     1.51     0.90     0.60     0.61     0.96     0.70     0.27
    1994    -0.01    -0.21    -0.23     0.01    -0.03     0.25     0.96     0.87     1.10     1.52     1.01     0.91
    1995     0.82     0.54     0.17     0.13     0.14     0.01    -0.29    -0.70    -0.94    -0.74    -0.77    -0.90
    1996    -0.85    -0.80    -0.66    -0.76    -0.85    -0.91    -0.83    -0.64    -0.31    -0.37    -0.34    -0.47
    1997    -0.65    -0.72    -0.30     0.15     0.71     2.34     2.26     2.26     2.20     2.06     2.14     2.11
    1998     2.30     2.45     2.27     2.61     2.33     0.42    -1.53    -1.82    -1.40    -1.29    -1.34    -1.29
    1999    -1.28    -1.20    -1.15    -1.16    -1.33    -1.26    -1.18    -1.03    -1.16    -1.33    -1.33    -1.43
    2000    -1.25    -1.26    -1.37    -0.90    -0.93    -1.18    -0.61    -0.08    -0.35    -0.52    -0.86    -0.79
    2001    -0.83    -0.86    -0.78    -0.55    -0.51    -0.73     0.04     0.38    -0.06    -0.22    -0.28     0.06
    2002     0.09    -0.26    -0.20    -0.36    -0.14     0.34     0.44     1.01     0.88     0.84     0.84     0.91
    2003     0.82     0.63     0.53    -0.11    -0.61    -0.10     0.01     0.05     0.18     0.28     0.30     0.14
    2004     0.19    -0.05    -0.43    -0.23    -0.46    -0.39     0.46     0.77     0.59     0.40     0.56     0.48
    2005     0.09     0.61     0.83     0.14     0.19     0.20    -0.00     0.01    -0.02    -0.66    -0.70    -0.70
    2006    -0.64    -0.48    -0.62    -0.84    -0.42    -0.18     0.21     0.59     0.65     0.77     1.00     0.64
    2007     0.64     0.39    -0.19    -0.32    -0.41    -0.83    -0.75    -0.90    -1.06    -1.13    -1.12    -1.19
    2008    -1.06    -1.27    -1.52    -1.11    -0.99    -0.80    -0.78    -1.01    -1.02    -1.08    -0.98    -1.01
    2009    -1.01    -0.85    -0.95    -0.81    -0.72    -0.05     0.56     0.56     0.42     0.56     1.04     0.95
    2010     0.93     1.28     1.33     0.49    -0.12    -1.29    -2.43    -2.38    -2.26    -2.16    -2.01    -1.86
    2011    -1.77    -1.59    -1.75    -1.69    -1.23    -1.02    -0.75    -0.81    -1.08    -1.30    -1.14    -1.18
    2012    -1.07    -0.68    -0.58    -0.38    -0.32    -0.28     0.34    -0.02    -0.29    -0.19    -0.03    -0.03
    2013    -0.06    -0.08    -0.12    -0.35    -0.68    -1.14    -0.79    -0.45    -0.33    -0.13    -0.16    -0.34
    2014    -0.50    -0.42    -0.05    -0.16    -0.17     0.00     0.42     0.23    -0.09     0.12     0.37     0.35
    2015     0.23     0.06     0.15     0.31     0.95     1.90     1.79     1.95     2.24     2.15     1.94     1.93
    2016     1.94     1.81     1.32     1.33     1.24     0.36    -0.53    -0.27    -0.29    -0.54    -0.48    -0.37
    2017    -0.43    -0.42    -0.58    -0.19     0.19    -0.24    -0.63    -0.73    -0.74    -0.59    -0.62    -0.74
    2018    -0.79    -0.71    -0.80    -1.32    -0.94    -0.52    -0.05     0.46     0.62     0.52     0.33     0.18
    2019     0.09     0.50     0.76     0.30     0.23     0.35     0.30     0.32     0.16     0.31     0.47     0.36
    2020     0.25     0.26     0.13    -0.15    -0.23    -0.68    -0.94    -0.96    -1.15    -1.17    -1.13    -1.14
    2021    -1.20    -0.96    -0.79    -0.95    -1.07    -1.05    -1.44    -1.29    -1.38    -1.46    -1.39    -1.20
    2022    -1.01    -0.98    -1.31    -1.61    -1.63    -1.90    -2.17    -1.75    -1.73    -1.73    -1.53    -1.28
    2023    -1.11    -0.91    -0.76    -0.37    -0.05     0.43     0.50     0.51     0.69     0.48     0.91     1.13
    2024     0.71     0.70  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00  -999.00
            '''
        format_indx = indx.split('\n')
        format_indx = format_indx[1:-1]
        result_dic = {}
        for line in format_indx:
            line = line.split()
            # print(line)
            year = int(line[0])
            vals = line[1:]
            vals = [float(i) for i in vals]
            result_dic[year] = vals
        # print(result_dic)
        year_list = [i for i in range(1982, 2021)]
        # print(year_list)
        #### append all years month data to a list
        vals_list = []

        for year in year_list:
            vals = result_dic[year]
            vals_list.append(vals)
        vals_list = np.array(vals_list)
        vals_list_flatten = vals_list.flatten()

        growing_season_Northern_dic_distance = {}
        growing_season_Southern_dic_distance = {}
        growing_season_tropical_dic_distance = {}

        for year in year_list:
            ## northern hemisphere April to October

            growing_season_Northern = vals_list_flatten[(year - 1982) * 12 + 4:(year - 1982) * 12 + 10]

            distance_north = np.nanmax(growing_season_Northern) - np.nanmin(growing_season_Northern)

            growing_season_Northern_dic_distance[year] = distance_north

        for year in year_list:
            if year >= 2020:
                break

            growing_season_Southern = vals_list_flatten[(year - 1982) * 12 + 10:(year - 1982 + 1) * 12 + 4]

            distance_southern = np.nanmax(growing_season_Southern) - np.nanmin(growing_season_Southern)
            growing_season_Southern_dic_distance[year] = distance_southern

        for year in year_list:
            growing_season_tropical = vals_list_flatten[(year - 1982) * 12:(year - 1982 + 1) * 12]

            distance_tropical = np.nanmax(growing_season_tropical) - np.nanmin(growing_season_tropical)
            growing_season_tropical_dic_distance[year] = distance_tropical

        #### add into df
        df = T.load_df(data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            index = row.year_range

            pix = row['pix']
            r, c = pix
            if r < 120:
                NDVI_list.append(np.nan)
                continue
            if index - 1 < 0:
                NDVI_list.append(np.nan)
                continue

            elif 120 <= r < 240:
                ## index type to string
                val = growing_season_Northern_dic_distance[index - 1 + 1982]
            elif 240 <= r < 480:
                val = growing_season_tropical_dic_distance[index - 1 + 1982]
            elif r >= 480:
                val = growing_season_Southern_dic_distance[index - 1 + 1982]
            else:
                raise
            NDVI_list.append(val)

        df['ENSO_index_distance_lagged'] = NDVI_list
        # df = df.dropna(subset=['ENSO_index_average'])
        outf = data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        # ## xlxs
        T.df_to_excel(df, outf)

        pass

    def ENSO_index_binary_retrieval(self):
        yearlist=[i for i in range(1982,2021)]
        ENSO_index_binary_dic={}
        for year in yearlist:
            if year in self.strong_El_Nino_list:
                ENSO_index_binary_dic[year]=1
            elif year in self.strong_La_Nina_list:
                ENSO_index_binary_dic[year]=-1
            else:
                ENSO_index_binary_dic[year]=0

        df = T.load_df(data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            index = row.year_range

            pix = row['pix']
            r, c = pix
            if r < 120:
                NDVI_list.append(np.nan)
                continue

            val = ENSO_index_binary_dic[index + 1982]
            NDVI_list.append(val)

        df['ENSO_index_binary'] = NDVI_list
        # df = df.dropna(subset=['ENSO_index_average'])
        outf = data_root + rf'E RA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        # ## xlxs
        T.df_to_excel(df, outf)




        pass
    def extract_data_based_on_ENSO(self):
        variable='tmin'
        f=result_root+rf'\Detrend\detrend_original\\{variable}.npy'
        outdir=result_root+'ENSO\\ENSO_original\\'
        T.mk_dir(outdir,force=True)
        dict=np.load(f,allow_pickle=True).item()



        result_dic_strong_El_Nino = {}
        result_dic_strong_La_Nina = {}

        result_dic_netural = {}

        for pix in tqdm(dict.keys()):
            result_strong_El_Nino = []
            result_strong_La_Nina = []

            result_netural = []

            vals=dict[pix]
            print(len(vals))
            ## get index of ENSO years
            if np.nanmean(vals) == 0:
                continue
            ##1) get number of ENSO years 2) substract 1982 to extract vals
            for year in self.strong_El_Nino_list:
                idx = year-1982
                result_strong_El_Nino.append(vals[idx])
            result_dic_strong_El_Nino[pix]=result_strong_El_Nino
            for year in self.strong_La_Nina_list:
                idx = year-1982
                result_strong_La_Nina.append(vals[idx])
            result_dic_strong_La_Nina[pix]=result_strong_La_Nina


            for year in self.netural_list:
                idx = year-1982
                result_netural.append(vals[idx])
            result_dic_netural[pix]=result_netural


        np.save(outdir+f'strong_El_Nino_{variable}.npy',result_dic_strong_El_Nino)
        np.save(outdir+f'strong_La_Nina_{variable}.npy',result_dic_strong_La_Nina)

        np.save(outdir+f'netural_{variable}.npy',result_dic_netural)



    def calculate_relatve_change_based_on_ENSO(self):
        fdir=result_root+'ENSO\\ENSO_original\\'
        outdir=result_root+'ENSO\\\\ENSO_relative_change\\'
        T.mk_dir(outdir,force=True)
        variable='tmin'
        f1=fdir+f'strong_El_Nino_{variable}.npy'
        f2=fdir+f'strong_La_Nina_{variable}.npy'

        f4=fdir+f'netural_{variable}.npy'
        strong_El_Nino=T.load_npy(f1)
        strong_La_Nina=T.load_npy(f2)

        netural=T.load_npy(f4)

        dic_strong_El_Nino = {}
        dic_strong_La_Nina = {}

        for pix in tqdm(strong_El_Nino.keys()):
            val1=strong_El_Nino[pix]
            val2=strong_La_Nina[pix]

            val4=netural[pix]
            val1=np.array(val1)
            val2=np.array(val2)

            val4=np.array(val4)

            ## relative change
            relative_val1=(val1-np.nanmean(val4))/np.nanmean(val4)*100
            relative_val2=(val2-np.nanmean(val4))/np.nanmean(val4)*100

            dic_strong_La_Nina[pix]=relative_val2
            dic_strong_El_Nino[pix]=relative_val1





            ## Save
        np.save(outdir+f'relative_strong_El_Nino_{variable}.npy',dic_strong_El_Nino)
        np.save(outdir+f'relative_strong_La_Nina_{variable}.npy',dic_strong_La_Nina)
    def calculate_relatve_change_based_on_ENSO_first_second_year(self):

        fdir = result_root + 'ENSO\\ENSO_first_year_second_year\\extraction\\'
        outdir = result_root + 'ENSO\\\\ENSO_first_year_second_year\\relative_change\\'
        T.mk_dir(outdir, force=True)
        variable = 'LAI4g'
        f1 = fdir + f'strong_El_Nino_first_{variable}.npy'
        f2 = fdir + f'strong_La_Nina_first_{variable}.npy'

        f3=fdir+f'strong_El_Nino_second_{variable}.npy'
        f4=fdir+f'strong_La_Nina_second_{variable}.npy'
        f5=fdir+f'netural_{variable}.npy'
        strong_El_Nino_first = T.load_npy(f1)
        strong_La_Nina_first = T.load_npy(f2)
        strong_El_Nino_second = T.load_npy(f3)
        strong_La_Nina_second = T.load_npy(f4)
        netural = T.load_npy(f5)



        dic_strong_El_Nino_first = {}
        dic_strong_La_Nina_first = {}
        dic_strong_El_Nino_second = {}
        dic_strong_La_Nina_second = {}


        for pix in tqdm(strong_El_Nino_first.keys()):
            val1 = strong_El_Nino_first[pix]
            val2 = strong_La_Nina_first[pix]
            val3 = strong_El_Nino_second[pix]
            val4 = strong_La_Nina_second[pix]
            val5 = netural[pix]
            val1 = np.array(val1)
            val2 = np.array(val2)
            val3 = np.array(val3)
            val4 = np.array(val4)
            val5 = np.array(val5)
            ## relative change
            relative_val1 = (val1 - np.nanmean(val5)) / np.nanmean(val5) * 100
            relative_val2 = (val2 - np.nanmean(val5)) / np.nanmean(val5) * 100
            relative_val3 = (val3 - np.nanmean(val5)) / np.nanmean(val5) * 100
            relative_val4 = (val4 - np.nanmean(val5)) / np.nanmean(val5) * 100
            dic_strong_La_Nina_first[pix] = relative_val2
            dic_strong_El_Nino_first[pix] = relative_val1
            dic_strong_La_Nina_second[pix] = relative_val4
            dic_strong_El_Nino_second[pix] = relative_val3


            ## Save
        np.save(outdir + f'relative_strong_El_Nino_first_{variable}.npy', dic_strong_El_Nino_first)
        np.save(outdir + f'relative_strong_La_Nina_first_{variable}.npy', dic_strong_La_Nina_first)
        np.save(outdir + f'relative_strong_El_Nino_second_{variable}.npy', dic_strong_El_Nino_second)
        np.save(outdir + f'relative_strong_La_Nina_second_{variable}.npy', dic_strong_La_Nina_second)




    def extract_first_year_and_second_year_ENSO(self):


        strong_first_El_Nino_list = [1987, 1991,1982,1997,2015]
        strong_second_El_Nino_list = [1988,1992,1983,1998,2016]


        strong_first_La_Nina_list = [1988, 1998, 2000,2007,2010]
        strong_second_La_Nina_list = [1989,1999,2001,2008,2011]

        ### netural year = all years - ENSO years
        netural_list = [i for i in range(1982,2021) if i not in self.strong_El_Nino_list+self.strong_La_Nina_list+self.moderate_El_Nino_list+self.moderate_La_Nina_list]

        variable='LAI4g'
        f=rf'\Detrend\detrend_original\\{variable}.npy'
        dict=np.load(result_root+f,allow_pickle=True).item()
        outdir=result_root+'ENSO\\ENSO_first_year_second_year\\'
        T.mk_dir(outdir,force=True)
        result_dic_strong_El_Nino_first = {}
        result_dic_strong_La_Nina_first = {}
        result_dic_strong_El_Nino_second = {}
        result_dic_strong_La_Nina_second = {}

        result_dic_netural = {}

        for pix in tqdm(dict.keys()):
            result_strong_El_Nino_first = []
            result_strong_La_Nina_first = []
            result_strong_El_Nino_second = []
            result_strong_La_Nina_second = []



            result_netural = []
            vals = dict[pix]
            print(len(vals))
            ## get index of ENSO years
            if np.nanmean(vals) == 0:
                continue
            ##1) get number of ENSO years 2) substract 1982 to extract vals
            for year in strong_first_El_Nino_list:
                idx = year - 1982
                result_strong_El_Nino_first.append(vals[idx])
            result_dic_strong_El_Nino_first[pix] = result_strong_El_Nino_first
            for year in strong_first_La_Nina_list:
                idx = year - 1982
                result_strong_La_Nina_first.append(vals[idx])
            result_dic_strong_La_Nina_first[pix] = result_strong_La_Nina_first
            for year in strong_second_El_Nino_list:
                idx = year - 1982
                result_strong_El_Nino_second.append(vals[idx])
            result_dic_strong_El_Nino_second[pix] = result_strong_El_Nino_second
            for year in strong_second_La_Nina_list:
                idx = year - 1982
                result_strong_La_Nina_second.append(vals[idx])
            result_dic_strong_La_Nina_second[pix] = result_strong_La_Nina_second


            for year in self.netural_list:
                idx = year - 1982
                result_netural.append(vals[idx])
            result_dic_netural[pix] = result_netural

        np.save(outdir + f'strong_El_Nino_first_{variable}.npy', result_dic_strong_El_Nino_first)
        np.save(outdir + f'strong_La_Nina_first_{variable}.npy', result_dic_strong_La_Nina_first)
        np.save(outdir + f'strong_El_Nino_second_{variable}.npy', result_dic_strong_El_Nino_second)
        np.save(outdir + f'strong_La_Nina_second_{variable}.npy', result_dic_strong_La_Nina_second)


        np.save(outdir + f'netural_{variable}.npy', result_dic_netural)

    pass


    def plot_spatial_average_ENSO_LAI(self):

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)


        fdir=result_root+rf'\ENSO\ENSO_extraction_together\relative_change\\'
        result_dic={}
        for f in os.listdir(fdir):
            if not 'tmin' in f:
                continue
            if not '.npy' in f:
                continue

            dic=T.load_npy(fdir+f)
            for pix in dic.keys():
                r,c=pix
                if r<120:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                vals=dic[pix]

                result_dic[pix]=np.nanmean(vals)
            array=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic)
            array=array*array_mask
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(array,fdir+f.replace('.npy','.tif'))

            plt.imshow(array,vmin=0,vmax=0.5,cmap='jet',interpolation='nearest')
            plt.title(f)
            plt.colorbar()
            plt.show()
            plt.close()
        pass
    def plt_moving_window(self):
        ## create 24 windows across 1982-2020
        ## for each window, calculate the ENSO events nunmber
        ## plot the ENSO events number

        window_list = []
        window_size=15
        for i in range(1982, 2021):
            ### extract 15 years
            if i+window_size>2021:
                break

            window_list.append([i + j for j in range(window_size)])

        # print(window_list)
        # exit()
        result_dic = {}
        flag=0

        for year in window_list:


            result_dic[flag] = {}
            number_strong_El_Nino = 0
            number_strong_La_Nina = 0
            number_moderate_El_Nino = 0
            number_moderate_La_Nina = 0

            number_netural = 0
            very_strong_El_Nino = 0

            for year_i in year:


                if year_i in self.strong_El_Nino_list:
                    number_strong_El_Nino += 1
                elif year_i in self.strong_La_Nina_list:
                    number_strong_La_Nina += 1
                elif year_i in self.moderate_El_Nino_list:
                    number_moderate_El_Nino += 1
                elif year_i in self.moderate_La_Nina_list:
                    number_moderate_La_Nina += 1
                elif year_i in self.weak_El_Nino_list:
                    number_netural += 1
                elif year_i in self.weak_La_Nina_list:
                    number_netural += 1
                elif year_i in self.netural_list:
                    number_netural += 1
                elif year_i in self.very_strong_El_Nino_list:
                    very_strong_El_Nino += 1
                else:
                    raise

            result_dic[flag]['strong_El_Nino'] = number_strong_El_Nino
            result_dic[flag]['strong_La_Nina'] = number_strong_La_Nina
            result_dic[flag]['moderate_El_Nino'] = number_moderate_El_Nino
            result_dic[flag]['moderate_La_Nina'] = number_moderate_La_Nina

            result_dic[flag]['netural'] = number_netural
            result_dic[flag]['very_strong_El_Nino'] = very_strong_El_Nino

            flag+=1


                ### plot bar for each window
        plt.figure(figsize=(10, 5))
        ## plot stacked bar
        number_strong_El_Nino = [result_dic[i]['strong_El_Nino'] for i in result_dic.keys()]
        number_strong_La_Nina = [result_dic[i]['strong_La_Nina'] for i in result_dic.keys()]
        number_moderate_El_Nino = [result_dic[i]['moderate_El_Nino'] for i in result_dic.keys()]
        number_moderate_La_Nina = [result_dic[i]['moderate_La_Nina'] for i in result_dic.keys()]
        number_ver_strong_El_Nino = [result_dic[i]['very_strong_El_Nino'] for i in result_dic.keys()]



        number_netural = [result_dic[i]['netural'] for i in result_dic.keys()]
        ##stacked bar
        plt.bar(range(len(number_strong_El_Nino)), number_strong_El_Nino, label='strong_El_Nino', color='r')
        plt.bar(range(len(number_strong_La_Nina)), number_strong_La_Nina, bottom=number_strong_El_Nino, label='strong_La_Nina', color='b')
        plt.bar(range(len(number_moderate_El_Nino)), number_moderate_El_Nino, bottom=np.array(number_strong_El_Nino)+np.array(number_strong_La_Nina), label='moderate_El_Nino', color='g')
        plt.bar(range(len(number_moderate_La_Nina)), number_moderate_La_Nina, bottom=np.array(number_strong_El_Nino)+np.array(number_strong_La_Nina)+np.array(number_moderate_El_Nino), label='moderate_La_Nina', color='y')
        plt.bar(range(len(number_ver_strong_El_Nino)), number_ver_strong_El_Nino, bottom=np.array(number_strong_El_Nino)+np.array(number_strong_La_Nina)+np.array(number_moderate_El_Nino)+np.array(number_moderate_La_Nina), label='very_strong_El_Nino', color='c')
        # plt.bar(range(len(number_weak_El_Nino)), number_weak_El_Nino, bottom=np.array(number_strong_El_Nino)+np.array(number_strong_La_Nina)+np.array(number_moderate_El_Nino)+np.array(number_moderate_La_Nina), label='weak_El_Nino', color='c')
        # plt.bar(range(len(number_weak_La_Nina)), number_weak_La_Nina, bottom=np.array(number_strong_El_Nino)+np.array(number_strong_La_Nina)+np.array(number_moderate_El_Nino)+np.array(number_moderate_La_Nina)+np.array(number_weak_El_Nino), label='weak_La_Nina', color='m')
        plt.bar(range(len(number_netural)), number_netural, bottom=np.array(number_strong_El_Nino)+np.array(number_strong_La_Nina)+np.array(number_moderate_El_Nino)+np.array(number_moderate_La_Nina)+np.array(number_ver_strong_El_Nino), label='netural', color='grey')
        plt.ylabel('frequency of ENSO events')
        plt.ylim(0, 15)
        plt.legend()


        plt.show()




        ## plot









class ENSO_vs_trend():
    def __init__(self):
        self.weak_El_Nino_list = [2004,2005,2006,2007,2014,2015,2018,2019]

        self.weak_La_Nina_list = [1983,1984,1985, 2000,2001,2005, 2006,2008,2009,2016,2017,2018,]

        self.moderate_El_Nino_list = [1986,1987,1994,1995,2002,2003,2009,2010]
        self.moderate_La_Nina_list = [1995,1996,2011,2012,2020,]

        self.strong_El_Nino_list = [1982,1983, 1987,1988, 1991,1992,1997,1998,2015,2016]

        self.strong_La_Nina_list = [1988, 1989,1998,1999, 2000,2007,2008,2010,2011]

        self.very_strong_El_Nino_list = [1982,1983, 1997,1998,2015,2016]

        ### netural year = all years - ENSO years
        self.netural_list = [i for i in range(1982,2021) if i not in self.strong_El_Nino_list+self.strong_La_Nina_list+self.moderate_El_Nino_list+self.moderate_La_Nina_list]
        # print('netural_list',self.netural_list)

    def run (self):
        # self.relationship_between_ENSO_and_trend()
        self.la_nina_vs_el_nino()

    def relationship_between_ENSO_and_trend(self):

        df=T.load_df(result_root+rf'Dataframe\relative_changes\\relative_changes_yearly_new.df')
        df=df[df['landcover_classfication']!='Cropland']
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]
        # df = df[df['LAI4g_p_value'] < 0.05]
        continent_list=['Africa','Australia','Asia','North_America','South_America']
        for continent in continent_list:
            df_continent=df[df['continent']==continent]

            vals_greening=df_continent['LAI4g_trend'].tolist()
            vals_greening=np.array(vals_greening)

            vals_EL_relative_change=df_continent['relative_strong_La_Nina_LAI4g'].tolist()
            vals_LA_relative_change=df_continent['relative_strong_El_Nino_LAI4g'].tolist()
        ### EL+LA = ENSO
            vals_EL_relative_change=np.array(vals_EL_relative_change)
            vals_LA_relative_change=np.array(vals_LA_relative_change)
            vals_EL_relative_change_flatten=vals_EL_relative_change.flatten()
            vals_LA_relative_change_flatten=vals_LA_relative_change.flatten()
            ## sum
            vals_ENSO_all=vals_EL_relative_change_flatten+vals_LA_relative_change_flatten


       ##plot bin

            ## plot scatter
            KDE_plot().plot_scatter(vals_ENSO_all,vals_greening,continent)
            # plt.scatter(vals_ENSO_all,vals_greening)
            plt.xlim(-20,20)

            plt.xlabel('EL+La relative change')
            plt.ylabel('LAI trend')

            plt.title(continent)

            plt.show()



        ### get the relationship between ENSO and trend

    def la_nina_vs_el_nino(self):
        tiff_el= result_root+rf'\ENSO\ENSO_extraction_together\relative_change\\relative_strong_El_Nino_LAI4g.tif'
        tiff_la= result_root+rf'\ENSO\ENSO_extraction_together\relative_change\\relative_strong_La_Nina_LAI4g.tif'
        array_el,originX,originY,pixelWidth,pixelHeight=ToRaster().raster2array(tiff_el)
        array_la,originX,originY,pixelWidth,pixelHeight=ToRaster().raster2array(tiff_la)
        array=array_el+array_la
        array[array<-999]=np.nan
        array[array==0]=np.nan
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(array,result_root+rf'\ENSO\ENSO_extraction_together\relative_change\\relative_strong_El_Nino_add_La_Nina_LAI4g.tif')
        plt.imshow(array,vmin=-10,vmax=10,cmap='jet',interpolation='nearest')
        plt.colorbar()
        plt.show()

        pass


    pass

class plot_ENSO():
    def __init__(self):
        self.weak_El_Nino_list = [2004, 2005, 2006, 2007, 2014, 2015, 2018, 2019]

        self.weak_La_Nina_list = [1983, 1984, 1985, 2000, 2001, 2005, 2006, 2008, 2009, 2016, 2017, 2018, ]

        self.moderate_El_Nino_list = [1986, 1987, 1994, 1995, 2002, 2003, 2009, 2010]
        self.moderate_La_Nina_list = [1995, 1996, 2011, 2012, 2020, ]

        self.strong_El_Nino_list = [1987, 1988, 1991, 1992]
        self.strong_La_Nina_list = [1988, 1989, 1998, 1999, 2000, 2007, 2008, 2010, 2011]
        self.very_strong_El_Nino_list = [1982, 1983, 1997, 1998, 2015, 2016]

        ### netural year = all years - ENSO years



        pass
    def run(self):
        self.plot_anomaly_LAI_based_on_cluster()
        # self.plot_anomaly_LAI_based_on_cluster_test()
        # self.barplot_relative_change()
        pass
    def clean_df(self,df):
        df=df[df['landcover_classfication']!='Cropland']
        df=df[df['row']>120]


        return df

    def plot_anomaly_LAI_based_on_cluster(self):  ##### plot for 4 clusters

        df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_new.df')
        print(len(df))
        df=self.clean_df(df)
        ## get uqniue AI_classfication
        # AI_classfication_unique = T.get_df_unique_val_list(df, 'continent')
        # print(AI_classfication_unique)
        # exit()


        print(len(df))
        T.print_head_n(df)
        # exit()
        df=df[df['row']>120]


        df=df[df['landcover_classfication']!='Cropland']

        df=df[df['Aridity']<0.65]



        color_list=['blue','green','red','orange','aqua','brown','cyan', 'black']
        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=2
        linewidth_list[2]=2

        fig = plt.figure()
        ## size
        fig.set_size_inches(12, 8)
        flag=1
        variable_list=['GLEAM_SMroot',]
        # variable_list=['VPD',]
        # variable_list=['CRU','GPCC']
        variable_list=['LAI4g','GIMMS_AVHRR_LAI','GIMMS3g_LAI']
        variable_list=['LAI4g',]

        region_unique = T.get_df_unique_val_list(df, 'AI_classfication')
        print(region_unique)
        region_val_dict = {
            'Arid': 1,
            'Semi-Arid': 2,
            'Sub-Humid': 3,
        }
        region_val = []
        # for i,row in df.iterrows():
        #     region = row['AI_classfication']
        #     val = region_val_dict[region]
        #     region_val.append(val)
        # df['region_val'] = region_val
        # spatial_dict_region = T.df_to_spatial_dic(df, 'region_val')
        # region_arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_region)
        # plt.imshow(region_arr, cmap='jet', vmin=1, vmax=3,interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        # exit()
        for continent in ['Africa','Australia','Asia','North_America','South_America']:
            ax = fig.add_subplot(2, 3, flag)
            if continent == 'North_America':
                df_NA = df[df['lon'] > -125]
                df_NA = df_NA[df_NA['lon'] < -95]
                df_NA = df_NA[df_NA['lat'] > 0]
                df_region = df_NA[df_NA['lat'] < 45]
            else:
                df_region = df[df['continent'] == continent]

            GPP_all = df_region['FLUXCOM'].tolist()
            GPP_min = np.nanmin(GPP_all)
            GPP_max = np.nanmax(GPP_all)


            for product in variable_list:

                print(product)
                ## creat year list dic
                dic_year={}
                yearlist=list(range(1982,2021))
                for year in yearlist:
                    dic_year[str(year)]=[]



                for i, row in df_region.iterrows():
                    year=row['year']
                    LAI_vals = row[product]
                    LAI_vals = np.array(LAI_vals)
                    gpp_vals = row['FLUXCOM']
                    gpp_vals = np.array(gpp_vals)
                    ## calculated weighted GPP
                    GPP_weights= (gpp_vals-GPP_min)/(GPP_max-GPP_min)
                    vals = np.array(LAI_vals)
                    vals = vals * GPP_weights
                    dic_year[str(year)].append(vals)
                ## calculate mean of each year

                for year in yearlist:
                    vals=dic_year[str(year)]
                    vals_mean = np.nanmean(vals, axis=0)
                    dic_year[str(year)]=vals_mean


            #plot time series
                val_list=[]
                for year in yearlist:
                    vals=dic_year[str(year)]
                    val_list.append(vals)
                val_list=np.array(val_list)
                plt.plot(val_list,label=product,color=color_list[variable_list.index(product)],linewidth=linewidth_list[variable_list.index(product)])
                plt.xticks(range(0, 40, 4), range(1982, 2021, 4), rotation=45)
                plt.xlabel('year')
                plt.ylabel(f'{product}')
                plt.title(f'{continent}')
                plt.ylim(-5, 5)
                plt.grid(which='major', alpha=0.5)


            for j in range(len(self.strong_La_Nina_list)):
                plt.axvspan(self.strong_La_Nina_list[j]-1982, self.strong_La_Nina_list[j]+1-1982, color='pink', alpha=0.5)

            for k in range(len(self.strong_El_Nino_list)):
                plt.axvspan(self.strong_El_Nino_list[k]-1982, self.strong_El_Nino_list[k]+1-1982, color='skyblue', alpha=0.5)
            for m in range(len(self.very_strong_El_Nino_list)):
                plt.axvspan(self.very_strong_El_Nino_list[m]-1982, self.very_strong_El_Nino_list[m]+1-1982, color='blue', alpha=0.5)

            ## show legend of ENSO events
            plt.plot([], [], color='pink', label='Strong La Nina')
            plt.plot([], [], color='skyblue', label='Strong El Nino')
            plt.plot([], [], color='blue', label='Very Strong El Nino')
            flag=flag+1


        plt.legend()
        plt.show()
        plt.close()

    def plot_anomaly_LAI_based_on_cluster_test(self):  ##### plot for 4 clusters

        df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes.df')
        print(len(df))
        df=self.clean_df(df)
        ## get uqniue AI_classfication
        # AI_classfication_unique = T.get_df_unique_val_list(df, 'continent')
        # print(AI_classfication_unique)
        # exit()


        print(len(df))
        T.print_head_n(df)
        # exit()
        df=df[df['row']>120]


        df=df[df['landcover_classfication']!='Cropland']

        df=df[df['Aridity']<0.65]



        color_list=['blue','green','red','orange','aqua','brown','cyan', 'black']
        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=2
        linewidth_list[2]=2

        fig = plt.figure()
        ## size
        fig.set_size_inches(12, 8)
        i = 1
        variable_list=['GLEAM_SMroot',]
        # variable_list=['VPD',]
        # variable_list=['CRU','GPCC']
        variable_list=['LAI4g','GIMMS_AVHRR_LAI','GIMMS3g_LAI']
        variable_list=['LAI4g',]

        region_unique = T.get_df_unique_val_list(df, 'AI_classfication')
        print(region_unique)
        region_val_dict = {
            'Arid': 1,
            'Semi-Arid': 2,
            'Sub-Humid': 3,
        }
        region_val = []
        # for i,row in df.iterrows():
        #     region = row['AI_classfication']
        #     val = region_val_dict[region]
        #     region_val.append(val)
        # df['region_val'] = region_val
        # spatial_dict_region = T.df_to_spatial_dic(df, 'region_val')
        # region_arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_region)
        # plt.imshow(region_arr, cmap='jet', vmin=1, vmax=3,interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        # exit()
        for continent in ['Africa','Australia','Asia','North_America','South_America']:
            ax = fig.add_subplot(2, 3, i)
            if continent == 'North_America':
                df_NA = df[df['lon'] > -125]
                df_NA = df_NA[df_NA['lon'] < -95]
                df_NA = df_NA[df_NA['lat'] > 0]
                df_region = df_NA[df_NA['lat'] < 45]
            else:
                df_region = df[df['continent'] == continent]

            for product in variable_list:


                print(product)
                vals=df_region[product].tolist()

                ### weighted LAI by GPP
                vals = np.array(vals)






                # print(vals)

                vals_nonnan=[]
                for val in vals:
                    if type(val)==float: ## only screening
                        continue
                    if len(val) ==0:
                        continue
                    val[val<-99]=np.nan

                    vals_nonnan.append(list(val))
                    # if not len(val) == 34:
                    #     print(val)
                    #     print(len(val))
                    #     exit()
                    # print(type(val))
                    # print(len(val))
                    # print(vals)

                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)
                val_std=np.nanstd(vals_mean,axis=0)

                # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
                plt.plot(vals_mean,label=product,color=color_list[variable_list.index(product)],linewidth=linewidth_list[variable_list.index(product)])
                # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])


                # plt.scatter(range(len(vals_mean)),vals_mean)
                # plt.text(0,vals_mean[0],product,fontsize=8)
            i=i+1

            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
            # plt.ylim(-0.2, 0.2)
            plt.ylim(-25,25)


            plt.xlabel('year')

            plt.ylabel(f'relative change{product}(%/year)')
            # plt.legend()

            plt.title(f'{continent}')
            plt.grid(which='major', alpha=0.5)


            for j in range(len(self.strong_La_Nina_list)):
                plt.axvspan(self.strong_La_Nina_list[j]-1982, self.strong_La_Nina_list[j]+1-1982, color='pink', alpha=0.5)

            for k in range(len(self.strong_El_Nino_list)):
                plt.axvspan(self.strong_El_Nino_list[k]-1982, self.strong_El_Nino_list[k]+1-1982, color='skyblue', alpha=0.5)
            for m in range(len(self.very_strong_El_Nino_list)):
                plt.axvspan(self.very_strong_El_Nino_list[m]-1982, self.very_strong_El_Nino_list[m]+1-1982, color='blue', alpha=0.5)

            ## show legend of ENSO events
            plt.plot([], [], color='pink', label='Strong La Nina')
            plt.plot([], [], color='skyblue', label='Strong El Nino')
            plt.plot([], [], color='blue', label='Very Strong El Nino')

        plt.legend()
        plt.show()
        plt.close()
    def barplot_relative_change(self):
        df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes.df')
        print(len(df))
        df=self.clean_df(df)
        variable='LAI4g'

        print(len(df))
        T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        df=df[df['landcover_classfication']!='Cropland']
        df=df[df['Aridity']<0.65]


        fig = plt.figure()
        ## size
        fig.set_size_inches(9, 5)
        vals_list_El_Nino = {}
        vals_list_La_Nina = {}
        all_dic={}

        for continent in ['Africa','Australia','Asia','North_America','South_America']:

            if continent == 'North_America':
                df_NA = df[df['lon'] > -125]
                # df_NA = df_NA[df_NA['lon'] < -105]
                df_NA = df_NA[df_NA['lon'] < -95]
                df_NA = df_NA[df_NA['lat'] > 0]
                df_region = df_NA[df_NA['lat'] < 45]
            else:
                df_region = df[df['continent'] == continent]
            # pix_list= df_region['pix'].tolist()
            # spatial_dic= {}
            # for pix in pix_list:
            #     spatial_dic[pix]=1
            # array=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(array, cmap='jet', vmin=0, vmax=1, interpolation='nearest')
            # plt.colorbar()
            # plt.show()


            vals = df_region[f'relative_strong_La_Nina_{variable}'].tolist()
            #### group by lanina and elnino












def main():
    ENSO_anaysis().run()
    # plot_ENSO().run()
    # ENSO_vs_trend().run()


    pass

if __name__ == '__main__':
    main()

