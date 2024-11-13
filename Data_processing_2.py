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
from pprint import pprint
T=Tools()
D = DIC_and_TIF(pixelsize=0.25)



this_root = 'D:\Project3\\'
data_root = 'D:/Project3/Data/'
result_root = 'D:/Project3/Result/'
class Data_processing_2:

    def __init__(self):

        pass

    def run(self):

        # self.dryland_mask()
        self.test_histogram()

        pass
    def dryland_mask(self):
        ## here we extract dryland tif
        fpath=rf'D:\Project3\Data\Base_data\aridity_index05.tif\\aridity_index.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        NDVI_mask_f = rf'D:\Greening\Data\Base_data\\NDVI_mask.tif'
        array_NDVI_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)

        array[array_NDVI_mask < 0.1] = np.nan


        array[array >= 0.65] = np.nan


        outf=rf'D:\Project3\Data\Base_data\aridity_index05.tif\\dryland_mask.tif'
        DIC_and_TIF().arr_to_tif(array, outf)

    def test_histogram(self):

        fpath = rf'E:\Project3\Data\ERA5_daily\dict\moving_window_average_anaysis\\detrended_annual_LAI4g_CV.npy'
        spatial_dic=np.load(fpath, allow_pickle=True, encoding='latin1').item()
        data_list=[]

        for pix in spatial_dic:
            val=spatial_dic[pix]
            data_list.append(val)

        data_list=np.array(data_list)
        ## histogram

        plt.hist(data_list,bins=50)

        plt.show()








    def run(self):

        self.dryland_mask()
    pass

class Phenology():
    ## this function is to see phenology of NH, SH and tropical
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass
    def run(self):

        self.phenology()
        pass

    def read_shp(self):
        fpath = r"C:\Users\wenzhang1\Desktop\point2.shp"
        df=T.read_point_shp(fpath, )
        return df

    def phenology(self):
        fdir_all = rf'E:\Project3\Data\LAI4g\\dic\\'
        spatial_dic=T.load_npy_dir(fdir_all)
        result_dic={}
        shp_df=self.read_shp()
        print(shp_df)
        lon_list=shp_df['point_x_pos'].to_list()
        lat_list=shp_df['point_y_pos'].to_list()
        pix_list=DIC_and_TIF().lon_lat_to_pix(lon_list, lat_list)


        for pix in pix_list:
            lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
            r,c=pix
            # if r<60:
            #     continu
            val=spatial_dic[pix]
            if T.is_all_nan(val):
                continue

            vals_reshape=val.reshape(-1,24)
            # plt.imshow(vals_reshape, interpolation='nearest', cmap='jet')
            # plt.colorbar()
            # plt.show()
            multiyear_mean = np.nanmean(vals_reshape, axis=0)
            #
            result_dic[pix]=multiyear_mean
            x=np.arange(0,24)
            xtick = [str(i) for i in x]
            plt.plot(x, multiyear_mean)
            plt.xticks(x, xtick)
            plt.xlabel('biweekly')
            plt.ylabel('LAI4g (m2/m2)')
            plt.title(f'lat:{lat},lon:{lon}')
            plt.show()
        # array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(result_dic)
        # plt.imshow(array, interpolation='nearest', cmap='jet')
        # plt.colorbar()
        #
        # plt.show()
            ## plt.plot(multiyear_mean)
            ## plt.show()





        pass

def main():
    # Data_processing_2().run()
    Phenology().run()



    pass

if __name__ == '__main__':
    main()