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


this_root = 'D:\Project3\\Result\\Nov\\'
data_root = 'D:/Project3/Result/Nov/MODIS_LAI_validation//Data//'
result_root = 'D:/Project3/Result/Nov//MODIS_LAI_validation//Result//'
T.mkdir(result_root)

class preprocessing_MODIS_validation():
    def __init__(self):
        pass
    def run(self):
        # self.MVC()
        # self.extract_dryland_tiff()
        # self.tif_to_dic()
        # self.extract_growing_season_monthly()
        # self.extract_growing_season_LAI_mean()
        self.spatial_plot()
        # self.relative_change()
        # self.detrend()
        # self.trend_analysis()

    pass
    def MVC(self):

        fdir = data_root + '/Data/LAI/'
        outdir = data_root + 'Data/MVC/'
        T.mk_dir(outdir, force=True)
        Pre_Process().monthly_compose(fdir, outdir, method='max')

    def extract_dryland_tiff(self):
        self.datadir=rf'D:\Project3\Data\\'
        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask05.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan


        fdir_all = data_root

        for fdir in T.listdir(fdir_all):
            if not 'MVC' in fdir:
                continue

            fdir_i = join(fdir_all, fdir)

            outdir_i = join(fdir_all, 'dryland_tiff')

            T.mk_dir(outdir_i)
            for fi in tqdm(T.listdir(fdir_i), desc=fdir):
                if not fi.endswith('.tif'):
                    continue
                fpath = join(fdir_i, fi)
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr[np.isnan(array_mask)] = np.nan
                # plt.imshow(arr)
                # plt.show()
                outpath = join(outdir_i, fi)

                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)

        pass

    def tif_to_dic(self):

        fdir_all = data_root+'/dryland_tiff//'
        outdir = data_root + '/dic//'
        T.mk_dir(outdir, force=True)

        year_list = list(range(2001, 2025))
        # 作为筛选条件

        all_array = []  #### so important  it should be go with T.mk_dic

        for f in T.listdir(fdir_all):
            print(f)

            if not f.endswith('.tif'):
                continue
            if int(f.split('.')[0][0:4]) not in year_list:
                continue

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_all, f))
            array = np.array(array, dtype=float)

            # array_unify = array[:720][:720,
            #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
            # array_unify = array[:3600][:3600,
            #               :7200]

            array[array < -999] = np.nan
            # array_unify[array_unify > 10] = np.nan
            # array[array ==0] = np.nan

            array[array < 0] = np.nan

            # plt.imshow(array)
            # plt.show()

            # plt.imshow(array)
            # plt.show()

            array_dryland = array
            # plt.imshow(array_dryland)
            # plt.show()

            all_array.append(array_dryland)

        row = len(all_array[0])
        col = len(all_array[0][0])
        key_list = []
        dic = {}

        for r in tqdm(range(row), desc='构造key'):  # 构造字典的键值，并且字典的键：值初始化
            for c in range(col):
                dic[(r, c)] = []
                key_list.append((r, c))
        # print(dic_key_list)

        for r in tqdm(range(row), desc='构造time series'):  # 构造time series
            for c in range(col):
                for arr in all_array:
                    value = arr[r][c]
                    dic[(r, c)].append(value)
                # print(dic)
        time_series = []
        flag = 0
        temp_dic = {}
        for key in tqdm(key_list, desc='output...'):  # 存数据
            flag = flag + 1
            time_series = dic[key]
            time_series = np.array(time_series)
            temp_dic[key] = time_series
            if flag % 10000 == 0:
                # print(flag)
                np.save(outdir + rf'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + rf'per_pix_dic_%03d' % 0, temp_dic)



        pass

    def extract_growing_season_monthly(self):
        fdir = data_root + rf'dic//'

        outdir = data_root + rf'extract_growing_season_monthly/'

        Tools().mk_dir(outdir, force=True)
        f_phenology = rf'D:\Project3\Data\LAI4g\4GST\\4GST.npy'
        phenology_dic = T.load_npy(f_phenology)
        new_spatial_dic = {}
        # for pix in phenology_dic:
            # print(phenology_dic[pix]);exit()
        #     val = phenology_dic[pix]['SeasType']
        #     try:
        #         val = float(val)
        #     except:
        #         continue
        #
        #     new_spatial_dic[pix] = val
        # spatial_array = D.pix_dic_to_spatial_arr(new_spatial_dic)
        # plt.imshow(spatial_array, interpolation='nearest', cmap='jet')
        # plt.show()
        # exit()
        spatial_dict_gs_count = {}

        for f in T.listdir(fdir):

            outf = outdir + f
            #
            # if os.path.isfile(outf):
            #     continue
            # print(outf)
            spatial_dict = dict(np.load(fdir + f, allow_pickle=True, encoding='latin1').item())
            dic_DOY = {15: 1,
                       30: 1,
                       45: 2,
                       60: 2,
                       75: 3,
                       90: 3,
                       105: 4,
                       120: 4,
                       135: 5,
                       150: 5,
                       165: 6,
                       180: 6,
                       195: 7,
                       210: 7,
                       225: 8,
                       240: 8,
                       255: 9,
                       270: 9,
                       285: 10,
                       300: 10,
                       315: 11,
                       330: 11,
                       345: 12,
                       360: 12,
                       }

            result_dic = {}

            for pix in tqdm(spatial_dict):
                if not pix in phenology_dic:
                    continue

                r, c = pix

                SeasType = phenology_dic[pix]['SeasType']
                if SeasType == 2:

                    SOS = phenology_dic[pix]['Onsets']
                    try:
                        SOS = float(SOS)

                    except:
                        continue

                    SOS = int(SOS)
                    SOS_monthly = dic_DOY[SOS]

                    EOS = phenology_dic[pix]['Offsets']
                    EOS = int(EOS)
                    EOS_monthly = dic_DOY[EOS]
                    # print(SOS_monthly,EOS_monthly)
                    # print(SOS,EOS)

                    time_series = spatial_dict[pix]

                    time_series = np.array(time_series)
                    if SOS_monthly > EOS_monthly:  ## south hemisphere
                        time_series_flatten = time_series.flatten()

                        # lon, lat = D.pix_to_lon_lat(pix)
                        #

                        time_series_reshape = time_series_flatten.reshape(-1, 12)
                        # plt.imshow(time_series_reshape)
                        #
                        # plt.title(f'lon:{lon}, lat:{lat},SOS_monthly:{SOS_monthly}, EOS_monthly:{EOS_monthly}')
                        # plt.show()
                        # plt.plot(time_series_reshape[0])
                        # plt.show()
                        time_series_dict = {}
                        for y in range(len(time_series_reshape)):
                            if y + 1 == len(time_series_reshape):
                                break

                            time_series_dict[y] = np.concatenate(
                                (time_series_reshape[y][SOS_monthly - 1:], time_series_reshape[y + 1][:EOS_monthly]))

                    else:
                        time_series_flatten = time_series.flatten()
                        time_series_reshape = time_series_flatten.reshape(-1, 12)
                        time_series_dict = {}
                        for y in range(len(time_series_reshape)):
                            time_series_dict[y] = time_series_reshape[y][SOS_monthly - 1:EOS_monthly]
                    time_series_gs = []
                    for y in range(len(time_series_dict)):
                        time_series_gs.append(time_series_dict[y])
                    time_series_gs = np.array(time_series_gs)

                elif SeasType == 3:
                    time_series = spatial_dict[pix]
                    time_series = np.array(time_series)
                    time_series_gs = np.reshape(time_series, (-1, 12))

                elif SeasType == 1:
                    time_series = spatial_dict[pix]
                    time_series = np.array(time_series)
                    time_series_gs = np.reshape(time_series, (-1, 12))


                else:
                    SeasClss = phenology_dic[pix]['SeasClss']
                    print(SeasType, SeasClss)
                    continue
                spatial_dict_gs_count[pix] = time_series_gs.shape[1]
                result_dic[pix] = time_series_gs
            # print(spatial_dict_gs_count)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_gs_count)
            # arr[arr<6] = np.nan
            # plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=12)
            # plt.colorbar()
            # plt.show()
            np.save(outf, result_dic)


    def extract_growing_season_LAI_mean(self):  ## extract LAI average
        fdir = data_root + rf'/extract_growing_season_monthly/'

        outdir = data_root + r'extract_growing_season_LAI_mean/'

        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            ### annual year

            vals_growing_season = spatial_dic[pix]

            print(vals_growing_season.shape[1])
            # plt.imshow(vals_growing_season)
            # plt.colorbar()
            # plt.show()
            growing_season_mean_list = []

            for val in vals_growing_season:
                if T.is_all_nan(val):
                    continue
                val = np.array(val)
                # if len(vals_growing_season) == 42:
                #     plt.plot(val)
                #     plt.show()

                sum_growing_season = np.nanmean(val)

                growing_season_mean_list.append(sum_growing_season)

            result_dic[pix] = growing_season_mean_list


        outf = outdir + 'growing_season_LAI_mean.npy'

        np.save(outf, result_dic)


    def spatial_plot(self):
        f = result_root + r'\moving_window_extraction_CV\5year\\growing_season_LAI_mean_detrend_CV.npy'
        dic = T.load_npy(f)
        spatial_dic = {}
        for pix in tqdm(dic):
            r, c = pix
            vals_growing_season = dic[pix]
            spatial_dic[pix] = len(vals_growing_season)
        arr = D.pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.show()

        pass
    def relative_change(self):
        self.datadir=rf'D:\Project3\Data\\'
        NDVI_mask_f =  self.datadir + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir_all = data_root
        outdir=result_root + rf'\relative_change\\'

        T.mk_dir(outdir, force=True)


        for fdir in os.listdir(fdir_all):
            if not 'extract_growing_season_LAI_mean' in fdir:
                continue

            outf = outdir + f'\\{fdir}_relative_change.npy'
            fpath=join(fdir_all,fdir,f'growing_season_LAI_mean.npy')

            dic = T.load_npy(fpath)

            relative_change = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue

                time_series = dic[pix]
                # print(time_series)

                # # 检查 time_series 是否为 list 或 array（防止是 float/NaN）

                if not isinstance(time_series, (list, np.ndarray)):
                    print(f"{pix}: invalid time_series (not iterable): {time_series}")
                    continue

                time_series = np.array(time_series, dtype=float)
                # time_series = time_series[3:37]

                print(len(time_series))
                ## exclude nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                # if np.nanmean(time_series) >999:
                #     continue
                if np.nanmean(time_series) < -999:
                    continue
                time_series = time_series
                mean = np.nanmean(time_series)

                relative_change[pix] = (time_series - mean) / mean * 100

                # plt.plot(time_series)
                # plt.legend(['raw'])
                # plt.show()
                #
                # plt.plot(relative_change[pix])
                # plt.legend(['zscore'])
                # # plt.legend(['raw','zscore'])
                # plt.show()

                ## save
            np.save(outf, relative_change)

    def detrend(self):
        self.datadir = rf'D:\Project3\Data\\'
        NDVI_mask_f = self.datadir  + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = self.datadir  + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = self.datadir  + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir_all=data_root+rf'extract_growing_season_LAI_mean\\'
        outdir=result_root + rf'detrend\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir_all):

            fpath = join( fdir_all, f)
            fname=f.split('.')[0]

            outf=outdir + f'\\{fname}_detrend.npy'

            dic = dict(np.load( fpath, allow_pickle=True, ).item())

            detrend_zscore_dic={}

            for pix in tqdm(dic):
                dryland_values=array_mask[pix]
                if np.isnan(dryland_values):
                    continue
                crop_values=crop_mask[pix]
                if crop_values == 16 or crop_values == 17 or crop_values == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue
                r, c= pix
                # print(len(dic[pix]))
                time_series = dic[pix]
                print(len(time_series))
                # print(time_series)
                time_series=np.array(time_series,dtype=float)
                # plt.plot(time_series)
                # plt.show()
                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.std(time_series) == 0:
                    continue
                ##### if count of nan is more than 50%, then skip
                if np.sum(np.isnan(time_series))/len(time_series) > 0.5:
                    continue
                # mean = np.nanmean(time_series)
                # std=np.nanstd(time_series)
                # if std == 0:
                #     continue
                # delta_time_series = (time_series - mean) / std
                # if np.isnan(time_series).any():
                #     continue
                time_series=T.interp_nan(time_series)
                detrend_delta_time_series = T.detrend_vals(time_series)
                # plt.plot(time_series)
                # plt.plot(detrend_delta_time_series)
                # plt.show()

                detrend_zscore_dic[pix] = detrend_delta_time_series

            np.save(outf, detrend_zscore_dic)

    def trend_analysis(self):  ##each window average trend
        self.datadir = rf'D:\Project3\Data\\'

        landcover_f = self.datadir  + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = self.datadir  + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root + rf'\detrend_relative_change\\'
        outdir = result_root + rf'\\trend_analysis\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            outf = outdir + f.split('.')[0]
            # if os.path.isfile(outf + '_trend.tif'):
            #     continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r, c = pix
                if r < 60:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                    ## ignore the last one year


                time_series = dic[pix][10:]
                time_series = np.array(time_series)
                # print(time_series)

                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                # if len(set(time_series)) == 1:
                #     continue
                # print(time_series)

                if np.nanstd(time_series) == 0:
                    continue
                try:

                    # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series)), time_series)
                    trend_dic[pix] = slope
                    p_value_dic[pix] = p_value
                except:
                    continue

            arr_trend = D.pix_dic_to_spatial_arr(trend_dic)

            p_value_arr = D.pix_dic_to_spatial_arr(p_value_dic)

            # plt.imshow(arr_trend, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend_second.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value_first_second.tif')

            np.save(outf + '_trend_second', arr_trend)
            np.save(outf + '_p_value_second', p_value_arr)

class moving_window():
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/MODIS_LAI_validation\Data/'
        self.result_root = 'D:/Project3/Result/Nov/MODIS_LAI_validation/Result/'
        pass
    def run(self):
        # self.moving_window_extraction()
        #
        # self.moving_window_CV_extraction_anaysis_LAI()


        # self.moving_window_max_anaysis()
        # self.moving_window_min_anaysis()
        # self.moving_window_std_anaysis()
        # self.moving_window_trend_anaysis()
        self.trend_analysis()



        pass

    def moving_window_extraction(self):


        fdir_all =result_root+ rf'detrend\\'
        outdir = result_root + rf'\moving_window_extraction\\15year\\'

        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir_all):


            if not f.endswith('.npy'):
                continue
            # if not 'detrend' in f:
            #     continue


            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)


            # if os.path.isfile(outf):
            #     continue
            # if os.path.isfile(outf):
            #     continue

            dic = T.load_npy(fdir_all+f)
            window = 15

            new_x_extraction_by_window = {}
            for pix in tqdm(dic):

                # time_series = dic[pix][mode]
                time_series = dic[pix]
                # plt.plot(time_series)
                # plt.show()


                time_series = np.array(time_series)
                # if T.is_all_nan(time_series):
                #     continue
                if len(time_series) == 0:
                    continue


                # time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    print('error')
                    continue
                # print((len(time_series)))
                ## if all values are identical, then continue
                if np.nanmax(time_series) == np.nanmin(time_series):
                    continue

                # new_x_extraction_by_window[pix] = self.forward_window_extraction_detrend_anomaly(time_series, window)
                new_x_extraction_by_window[pix] = self.forward_window_extraction(time_series, window)

            T.save_npy(new_x_extraction_by_window, outf)





    def forward_window_extraction(self, x, window):
        # 前窗滤波
        # window = window-1
        # 不改变数据长度

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        # new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        new_x_extraction_by_window=[]
        for i in range(len(x)+1):
            if i + window >= len(x)+1:
                continue
            else:
                anomaly = []
                relative_change_list=[]
                x_vals=[]
                for w in range(window):
                    x_val=(x[i + w])
                    x_vals.append(x_val)
                if np.isnan(np.nanmean(x_vals)):
                    continue

                # x_mean=np.nanmean(x_vals)

                # for i in range(len(x_vals)):
                #     if x_vals[0]==None:
                #         continue
                    # x_anomaly=(x_vals[i]-x_mean)
                    # relative_change = (x_vals[i] - x_mean) / x_mean

                    # relative_change_list.append(x_vals)
                new_x_extraction_by_window.append(x_vals)
        return new_x_extraction_by_window

    def forward_window_extraction_detrend_anomaly(self, x, window):
        # 前窗滤波
        # window = window-1
        # 不改变数据长度

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        # new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        new_x_extraction_by_window = []
        for i in range(len(x)):
            if i + window >= len(x):
                continue
            else:
                anomaly = []

                x_vals = []
                for w in range(window):
                    x_val = (x[i + w])
                    x_vals.append(x_val)
                if np.isnan(np.nanmean(x_vals)):
                    continue

                x_mean=np.nanmean(x_vals)

                # for i in range(len(x_vals)):
                #     if x_vals[0]==None:
                #         continue
                #     x_anomaly=x_vals[i]-x_mean
                #
                #     anomaly.append(x_anomaly)
                # if np.isnan(anomaly).any():
                #     continue
                # detrend_anomaly=signal.detrend(anomaly)+x_mean
                detrend_original=signal.detrend(x_vals)+x_mean


                new_x_extraction_by_window.append(detrend_original)
        return new_x_extraction_by_window




    def moving_window_CV_extraction_anaysis_LAI(self):


        fdir = result_root + rf'moving_window_extraction\\15year\\'
        outdir = result_root + rf'\moving_window_extraction_CV\\15year\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            dic = T.load_npy(fdir + f)

            outf = outdir + f.split('.')[0] + f'_CV.npy'
            print(outf)

            # if os.path.isfile(outf):
            #     continue

            new_x_extraction_by_window = {}
            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):
                trend_list = []
                time_series_all = dic[pix]
                # print(len(time_series_all));exit()
                if len(time_series_all)<9:  ##
                    continue
                time_series_all = np.array(time_series_all)
                slides=len(time_series_all)
                for ss in range(slides):
                    if np.isnan(np.nanmean(time_series_all)):
                        print('error')
                        continue
                    # print((len(time_series)))

                    ### if all values are identical, then continue
                    time_series=time_series_all[ss]
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    # print(len(time_series))

                    if np.nanmean(time_series)==0:
                        continue
                    cv=np.nanstd(time_series)/np.nanmean(time_series)*100

                    trend_list.append(cv)
                    # print(trend_list)
                # plt.plot(trend_list)
                # plt.show()
                trend_dic[pix]=trend_list

            np.save(outf, trend_dic)
            T.open_path_and_file(outdir)

            ##tiff
            # arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            #
            # p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')






    def moving_window_min_anaysis(self): ## each window calculating the average
        window_size = 15

        fdir = result_root + rf'\CRU_monthly\extract_annual_growing_season_mean\relative_change\detrend\moving_window_extraction\\'
        outdir = result_root + rf'CRU_monthly\extract_annual_growing_season_mean\relative_change\detrend\\moving_window_extraction_max_min\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'detrend' in f:
                continue



            dic = T.load_npy(fdir + f)
            # outf = outdir + 'GLOBMAP' + f'_min.npy'


            outf = outdir + f.split('.')[0] + f'_min.npy'
            print(outf)

            trend_dic = {}


            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                time_series_all = np.array(time_series_all)
                # print(len(time_series_all));exit()
                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                slides=len(time_series_all)
                print(slides)
                for ss in range(slides):


                    ### if all values are identical, then continue
                    if len(time_series_all)<24:
                        continue


                    time_series = time_series_all[ss]
                    # print(time_series)
                    # if np.nanmax(time_series) == np.nanmin(time_series):
                    #     continue
                    # print(len(time_series))
                    ##average
                    average=np.nanmin(time_series)
                    # print(average)

                    trend_list.append(average)

                trend_dic[pix] = trend_list
                # plt.plot(trend_dic[pix])
                # plt.show()

                ## save
            np.save(outf, trend_dic)


    def moving_window_max_anaysis(self): ## each window calculating the average
        window_size = 15


        fdir =result_root+ rf'\CRU_monthly\extract_annual_growing_season_mean\relative_change\detrend\moving_window_extraction\\'
        outdir = result_root+rf'CRU_monthly\extract_annual_growing_season_mean\relative_change\detrend\\moving_window_extraction_max_min\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'detrend' in f:
                continue



            dic = T.load_npy(fdir + f)



            outf = outdir + f.split('.')[0] + f'_max.npy'
            print(outf)

            trend_dic = {}


            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                time_series_all = np.array(time_series_all)
                # print(time_series_all)
                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                slides=len(time_series_all)
                print(slides)
                for ss in range(slides):


                    ### if all values are identical, then continue
                    if len(time_series_all)<24:
                        continue


                    time_series = time_series_all[ss]
                    # print(time_series)
                    # if np.nanmax(time_series) == np.nanmin(time_series):
                    #     continue
                    # print(len(time_series))
                    ##average
                    average=np.nanmax(time_series)
                    # print(average)

                    trend_list.append(average)

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)






    def moving_window_trend_anaysis(self): ## each window calculating the trend
        window_size = 10

        fdir=rf'D:\Project3\Result\3mm\moving_window_robust_test\moving_window_extraction_average\10_year\\'
        outdir = rf'D:\Project3\Result\3mm\moving_window_robust_test\moving_window_extraction_average\10_year\\trend\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            # if not f.split('.')[0] in ['average_heat_spell', 'heat_event_frequency',
            #    'maxmum_heat_spell']:
            #     continue


            dic = T.load_npy(fdir + f)

            slides = 38 - window_size
            outf = outdir + f.split('.')[0] + f'_trend.npy'
            print(outf)

            trend_dic = {}


            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                time_series_all = np.array(time_series_all)
                # print(time_series_all)
                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                for ss in range(slides):


                    ### if all values are identical, then continue
                    if len(time_series_all)<29:
                        continue


                    time_series = time_series_all[ss]
                    # print(time_series)
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    print(len(time_series))
                    ### calculate slope and intercept

                    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    print(slope)

                    trend_list.append(slope)

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)
    def trend_analysis(self):  ##each window average trend
        self.datadir = rf'D:\Project3\Data\\'

        landcover_f = self.datadir + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = self.datadir + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir =result_root+ rf'\moving_window_extraction_CV\\5year\\'
        outdir =result_root + (rf'\moving_window_extraction_CV\\trend\\5year\\')
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            if not f.endswith('.npy'):
                continue



            outf = outdir + f.split('.')[0]
            if os.path.isfile(outf + '_trend.tif'):
                continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
            # dic=T.load_npy_dir(fdir)

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r, c = pix
                if r < 60:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                    ## ignore the last one year

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
                # print((time_series))
                # exit()
                time_series = np.array(time_series)
                average = np.nanmean(time_series)
                # print(time_series)

                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                # if len(set(time_series)) == 1:
                #     continue
                # print(time_series)

                if np.nanstd(time_series) == 0:
                    continue
                try:

                    # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series)), time_series)
                    trend_dic[pix] = slope
                    p_value_dic[pix] = p_value
                except:
                    continue



            arr_trend = D.pix_dic_to_spatial_arr(trend_dic)


            p_value_arr = D.pix_dic_to_spatial_arr(p_value_dic)

            # plt.imshow(arr_trend, cmap='jet', vmin=1, vmax=1)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

        T.open_path_and_file(outdir)

    pass



class PLOT_result:
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):

        self.plot_histogram()
        # self.weighted_average_LAICV()
        # self.plot_CV_LAI()
        # self.statistic_CV_trend_bar()

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def plot_histogram(self):

        from scipy.stats import gaussian_kde

        dff = result_root + r'\Dataframe\15year\\Dataframe.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        variable = 'growing_season_LAI_mean_detrend_CV_trend'

        vals = np.array(df[variable], dtype=float)
        vals = vals[~np.isnan(vals)]
        ## calculate >0 and <0
        percentage_positive = len(vals[vals > 0])/len(vals)*100
        percentage_negative = len(vals[vals < 0])/len(vals)*100


        plt.figure(figsize=(4,3))

        # histogram PDF
        plt.hist(vals, bins=100, density=True, alpha=0.4, edgecolor='#ff7f0e', color='#ff7f0e' )


        # KDE smooth PDF
        kde = gaussian_kde(vals)
        x = np.linspace(np.nanmin(vals), np.nanmax(vals), 300)
        plt.plot(x, kde(x), linewidth=2,color='#ff7f0e'  )
        plt.axvline(0, color='k', linestyle='-',linewidth=1.5 )
        plt.xlim(-2, 2)
        plt.text(
            0.98, 0.95,
            f'Inc CV: {percentage_positive:.1f}%\n Dec CV: {percentage_negative:.1f}%',
            transform=plt.gca().transAxes,
            ha='right',
            va='top',
            fontsize=10
        )


        # plt.tight_layout()
        # plt.show()
        outdir=result_root+rf'Figure\15year\\'
        T.mk_dir(outdir, force=True)

        plt.savefig(outdir+f'histogram.pdf')
        plt.close()



        ## add x=0


        pass

    def weighted_average_LAICV(self):  ###add weighted average LAI in dataframe
        df = T.load_df(
            result_root + rf'\Dataframe\\Dataframe.df')

        df_clean = self.df_clean(df)
        # print(len(df_clean))

        df_clean['area_weight'] = np.cos(np.deg2rad(df['lat']))

        # plt.figure(figsize=(6, 4))
        #
        # plt.plot(
        #     df_aw_year['year'],
        #     df_aw_year['SNU_LAI_relative_change_area_weighted'],
        #     color='black',
        #     lw=2
        # )
        #
        # plt.xlabel('Year')
        # plt.ylabel('Area-weighted LAI change')
        # plt.title('Dryland vegetation change (area-weighted)')
        # plt.tight_layout()
        # plt.show()

        # df[df['year'] == 1982][
        #     ['SNU_LAI_relative_change_area_weighted',
        #      'LAI4g_relative_change_area_weighted',
        #      'composite_LAI_mean_relative_change_area_weighted',
        #      'GLOBMAP_LAI_relative_change_area_weighted',
        #
        #      ]
        # ].head()
        # T.print_head_n(df)


        outf=result_root+rf'\Dataframe\\Dataframe_area_weight.df'
        T.save_df(df_clean, outf)
        T.df_to_excel(df_clean, outf)
    def plot_CV_LAI(self):  ##### plot for 4 clusters

        df = T.load_df(
            result_root + rf'\Dataframe\\10year\\Dataframe.df')
        print(len(df))
        df = self.df_clean(df)

        print(len(df))
        T.print_head_n(df)
        # exit()

        # create color list with one green and another 14 are grey

        color_list = ['black', 'green', 'blue', 'magenta', 'black', 'purple', 'purple', 'black', 'yellow', 'purple',
                      'pink', 'grey',
                      'brown', 'lime', 'teal', 'magenta']
        linewidth_list = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        variable_list = ['growing_season_LAI_mean',
                        ]
        dic_label = {'growing_season_LAI_mean': 'MODIS LAI',
                    }
        year_list = range(0, 15)

        result_dic = {}

        for var in variable_list:
            mean_dic={}
            for year in year_list:
                df_i = df[df['window'] == year]
                ## scheme1
                vals = np.array(df_i[f'{var}_detrend_CV'].tolist(), dtype=float)
                weight=np.array(df_i['area_weight'].tolist(),dtype=float)
                weighted_mean_values = (
                        np.nansum(vals * weight)
                        / np.nansum(weight * np.isfinite(vals))
                )
                print(year,weighted_mean_values)
                ## scheme2
                # vals = np.array(df_i[f'{var}_detrend_CV_area_weighted'].tolist(), dtype=float)
                # weighted_mean_values = np.nanmean(vals)

                mean_dic[year] = weighted_mean_values

            result_dic[var] = mean_dic


        # 转成 DataFrame
        df_new = pd.DataFrame(result_dic).reset_index()
        # T.print_head_n(df_new);exit()

        flag = 0

        plt.figure(figsize=(self.map_width, self.map_height))

        for var in variable_list:
            plt.plot(
                year_list,
                df_new[var],
                label=dic_label[var],
                linewidth=linewidth_list[flag],
                color=color_list[flag]
            )

            slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, df_new[var])
            print(var, f'{slope:.2f}', f'{p_value:.2f}')
            trend = slope * np.array(year_list) + intercept

            plt.plot(
                year_list,
                trend,
                linestyle='--',
                linewidth=2,
                color=color_list[flag],
                alpha=0.8
            )

            plt.text(
                0.92, 0.95,
                # f'Slope: {slope:.2f}\n P: {p_value:.2f}',
                f'Slope: {slope:.2f}***',
                transform=plt.gca().transAxes,
                ha='right',
                va='top',
                fontsize=10
            )

            ## std

            flag = flag + 1
        ## if var == 'composite_LAI_CV': plot CI bar

        window_size = 10

        # set xticks with 1982-1997, 1998-2013,.. 2014-2020
        year_range = range(2001, 2025)
        year_range_str = []
        for year in year_range:

            start_year = year
            end_year = year + window_size - 1
            if end_year > 2024:
                break
            year_range_str.append(f'{start_year}-{end_year}')

        plt.xticks(range(len(year_range_str))[::3], year_range_str[::3], rotation=45, ha='right')
        plt.yticks(np.arange(13, 15,.5))

        plt.ylabel(f'CVLAI (%/yr)')
        plt.grid(True, ) # 只画竖线（随 x 刻度）



        # plt.show()
        # plt.tight_layout()
        out_pdf_fdir = result_root + rf'\\Figure\10year\\'
        T.mk_dir(out_pdf_fdir, force=True)
        plt.savefig(out_pdf_fdir + 'time_series_CV_mean.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        #
        # plt.legend()
        # plt.show()




    pass



def main():
    # preprocessing_MODIS_validation().run()
    # moving_window().run()
    PLOT_result().run()


    pass

if __name__ == '__main__':
    main()



    pass
