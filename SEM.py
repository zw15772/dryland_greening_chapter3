# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg
from openpyxl.styles.builtins import percent
# from green_driver_trend_contribution import *
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import t
from sympy.codegen.cfunctions import isnan

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
result_root = 'D:/Project3/Result/'

class SEM_wen:
    def __init__(self):
        # This class is used to calculate the structural equation model
        self.this_class_arr = result_root + rf'3mm\SHAP_beta\Dataframe\\'
        self.dff = self.this_class_arr + 'moving_window_zscore.df'
        self.outdir = result_root + '3mm//SEM//'
        T.mkdir(self.outdir, force=True)

        pass

    def run(self):
        df, dff = self.__load_df()
        print(df.columns)
        des = self.model_description_not_detrend_test2()
        # des = self.model_description_not_detrend_new()

        df_clean = self.df_clean(df)
        self.SEM_model(df_clean, des)
        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        # print(df.columns);exit()

        return df, dff
        # return df_early,dff

    def df_clean(self, df):
        # T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        print('original len(df):', len(df))
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        # df = df[df['LC_max'] < 5]
        # df=df[df['composite_LAI_CV_trend'] > 0]
        # df = df[df['extraction_mask'] == 1]

        df = df[df['MODIS_LUCC'] != 12]
        df=df[df['landcover_classfication'] != 'Cropland']
        print('filtered len(df):', len(df))
        # exit()

        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        # df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def model_description_not_detrend_test2(self):
        desc_all_limited_SMroot = '''
                                     # regressions


                                     composite_LAI_CV_zscore~detrended_sum_rainfall_CV_zscore+composite_LAI_beta+fire_ecosystem_year_average_zscore
                                     composite_LAI_beta~sum_rainfall_ecosystem_year_zscore+CO2_zscore+CV_intraannual_rainfall_ecosystem_year_zscore+pi_average_zscore+fire_ecosystem_year_average_zscore+grass_relative_change+trees_relative_change
                                  
                                     # residual correlations
                                     
                                      composite_LAI_CV_zscore~~composite_LAI_CV_zscore
                                      composite_LAI_CV_zscore~~detrended_sum_rainfall_CV_zscore
                            

                                    composite_LAI_CV_zscore~~fire_ecosystem_year_average_zscore
                         
                               

                                         composite_LAI_CV_zscore~~composite_LAI_beta
                                      

                                
                                    
                                 
                                         composite_LAI_beta~~sum_rainfall_ecosystem_year_zscore
                                     
                                         composite_LAI_beta~~CV_intraannual_rainfall_ecosystem_year_zscore
                                         composite_LAI_beta~~pi_average_zscore
                                      

                        
                                     





                                     '''

        return desc_all_limited_SMroot

    def model_description_not_detrend_new(self):
        desc_all_limited_SMroot = '''
                             # regressions

                             

                             composite_LAI_CV_zscore~detrended_sum_rainfall_CV_zscore+composite_LAI_beta+FVC_max_zscore+Burn_area_mean
                             
                             composite_LAI_beta~sum_rainfall_ecosystem_year_zscore+CV_intraannual_rainfall_ecosystem_year_zscore+FVC_max_zscore+CO2_zscore
                             composite_LAI_beta~detrended_sum_rainfall_CV_zscore
                            
                         
                             
                             
                          
                             
                             # residual correlations
                              composite_LAI_CV_zscore~~composite_LAI_CV_zscore
                              
                            
                                 composite_LAI_CV_zscore~~Burn_area_mean
            
                        
                                 composite_LAI_CV_zscore~~composite_LAI_beta
                                 composite_LAI_CV_zscore~~FVC_max_zscore
              
                                
                                 composite_LAI_beta~~FVC_max_zscore
                          
                                 composite_LAI_beta~~CV_intraannual_rainfall_ecosystem_year_zscore
                                
                        
                                
                          
                               
                                 
                                 
                       
                          
                               
                                 
                                 
                        
                       
                                 
                             
                              
                            
                             
                                  
                             
                                 
                                 
                             '''

        return desc_all_limited_SMroot

    def SEM_model(self, df, desc):
        import semopy
        mod = semopy.Model(desc)
        res = mod.fit(df)

        result = mod.inspect()
        T.print_head_n(result)
        outf = self.outdir + f'SEM2'

        T.save_df(result, outf + '.df')
        T.df_to_excel(result, outf + '.xlsx')

        outf = self.outdir + 'SEM2'
        semopy.report(mod, outf)

def main():
    SEM_wen().run()

if __name__ == '__main__':
    main()