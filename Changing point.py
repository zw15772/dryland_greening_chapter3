# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg
# from green_driver_trend_contribution import *
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import t

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
D = DIC_and_TIF(pixelsize=0.5)
centimeter_factor = 1/2.54
import Rbeast as rb
# exit()

this_root = 'D:\Project3\\'
data_root = 'D:/Project3/Data/'
result_root = 'D:/Project3/Result/'

class changing_point():
    def __init__(self):
        pass
    def run(self):
        self.plot()
        # self.beast()
        # self.anaysis_result_of_changing_point()
        pass
    def plot(self):
        fdir=rf'D:\Project3\Data\LAI4g\dic_dryland\\'
        spatial_dict=T.load_npy_dir(fdir)
        for pix in tqdm(spatial_dict):
            vals=spatial_dict[pix]
            print(len(vals))

    def beast(self):
        f=rf'D:\Project3\Result\3mm\relative_change_growing_season\\LAI4g_detrend.npy'
        outf = rf'D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\
        \\LAI4g_detrend_changing_point'
        spatial_dict=np.load(f,allow_pickle=True).item()
        spatial_dict_o = {}
        for pix in tqdm(spatial_dict):

            vals=spatial_dict[pix]
            print(len(vals))
            vals = np.array(vals)
            if True in list(np.isnan(vals)):
                continue
            vals = pd.Series(vals)
            o = rb.beast(vals, quiet=True, start=0,season='none')
            # rb.plot(o, fig=plt.figure(figsize=(15, 10)))
            # plt.show()
            spatial_dict_o[pix] = o
        T.save_dict_to_binary(spatial_dict_o, outf)

    def anaysis_result_of_changing_point(self):

        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        dic=T.load_dict_from_binary(result_root+rf'3mm\changing_point_detection\\LAI4g_cp.pkl')
        result_dic={}

        for pix in dic:
            r,c=pix

            if r < 60:
                continue
            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue
            if dic_modis_mask[pix] == 12:
                continue
            o=dic[pix]
            cp=o.trend.cp
            pro=o.trend.cpPr
            print(cp,pro)

            # #
            if T.is_all_nan(cp):
                continue
            flag=0
            for i in range(len(cp)):
                cp_i=cp[i]
                rb.plot(o, fig=plt.figure(figsize=(10, 5)))
                plt.show()

                # if 21<=cp_i<=23 and pro[i] > 0.5:
                if cp_i == 22 and pro[i] > 0.5:

                    result_dic[pix]=flag+1
        spatial_array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(result_dic)
        plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
        plt.show()
        exit()



        pass

    def anaysis_result_of_changing_point_CV(self):

        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        dic=T.load_dict_from_binary(result_root+rf'3mm\changing_point_detection\\LAI4g_cp.pkl')
        result_dic={}

        for pix in dic:
            r,c=pix

            if r < 60:
                continue
            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue
            if dic_modis_mask[pix] == 12:
                continue
            o=dic[pix]
            cp=o.trend.cp
            pro=o.trend.cpPr
            print(cp,pro)


            # #


            if T.is_all_nan(cp):
                continue
            flag=0
            for i in range(len(cp)):
                cp_i=cp[i]
                # rb.plot(o, fig=plt.figure(figsize=(5, 3)))
                # plt.show()
                #
                # if 21<=cp_i<=23 and pro[i] > 0.5:
                if cp_i == 22 and pro[i] > 0.5:

                    result_dic[pix]=flag+1
        spatial_array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(result_dic)
        plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
        plt.show()
        exit()



        pass


def main():
    changing_point().run()
    pass

if __name__ == '__main__':
    main()










