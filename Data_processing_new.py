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

class processing_GLOBMAP():
    def __init__(self):
        pass
    def run (self):
        # self.extract_dryland()
        # self.extract_phenology_monthly_variables()
        # self.extract_annual_growing_season_LAI_mean()
        # self.relative_change()
        # self.trend_analysis()
        self.detrend()
        # self.deseanalized_detrend()
        pass


    def extract_phenology_monthly_variables(self):
        fdir = rf'D:\Project3\Data\GLOBMAP\dic\\'

        outdir = rf'D:\Project3\Data\GLOBMAP\\extract_phenology_monthly\\'

        Tools().mk_dir(outdir, force=True)
        f_phenology = rf'D:\Project3\Data\LAI4g\4GST\\4GST.npy'
        phenology_dic = T.load_npy(f_phenology)
        new_spatial_dic={}
        for pix in phenology_dic:
            val=phenology_dic[pix]['Offsets']
            try:
                val=float(val)
            except:
                continue

            new_spatial_dic[pix]=val
        spatial_array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(new_spatial_dic)
        # plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
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
            dic_DOY={15: 1,
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
                # print(pix)

                r, c = pix

                SeasType=phenology_dic[pix]['SeasType']
                if SeasType==2:

                    SOS=phenology_dic[pix]['Onsets']
                    try:
                        SOS=float(SOS)

                    except:
                        continue

                    SOS=int(SOS)
                    SOS_monthly=dic_DOY[SOS]

                    EOS=phenology_dic[pix]['Offsets']
                    EOS=int(EOS)
                    EOS_monthly=dic_DOY[EOS]
                    # print(SOS_monthly,EOS_monthly)
                    # print(SOS,EOS)

                    time_series = spatial_dict[pix]

                    time_series = np.array(time_series)
                    if SOS_monthly>EOS_monthly:  ## south hemisphere
                        time_series_flatten=time_series.flatten()
                        time_series_reshape=time_series_flatten.reshape(-1,12)
                        time_series_dict={}
                        for y in range(len(time_series_reshape)):
                            if y+1==len(time_series_reshape):
                                break

                            time_series_dict[y]=np.concatenate((time_series_reshape[y][SOS_monthly-1:],time_series_reshape[y+1][:EOS_monthly]))

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

                elif SeasType==3:
                    time_series = spatial_dict[pix]
                    time_series = np.array(time_series)
                    time_series_gs = np.reshape(time_series, (-1, 12))

                else:
                    SeasClss=phenology_dic[pix]['SeasClss']
                    print(SeasType,SeasClss)
                    continue
                spatial_dict_gs_count[pix] = time_series_gs.shape[1]
                result_dic[pix] = time_series_gs
        # print(spatial_dict_gs_count)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_gs_count)
        # # arr[arr<6] = np.nan
        # plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=12)
        # plt.colorbar()
        # plt.show()
            np.save(outf, result_dic)





    def extract_annual_growing_season_LAI_mean(self):  ## extract LAI average
        fdir =rf'D:\Project3\Data\GLOBMAP\\extract_phenology_monthly\\'

        outdir_CV = result_root+rf'\3mm\GLOBMAP\\extract_annual_growing_season_LAI_mean\\'
        # print(outdir_CV);exit()

        T.mk_dir(outdir_CV, force=True)

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
            growing_season_mean_list=[]



            for val in vals_growing_season:
                if T.is_all_nan(val):
                    continue
                val = np.array(val)

                sum_growing_season = np.nanmean(val)

                growing_season_mean_list.append(sum_growing_season)



            result_dic[pix] = {
                             'growing_season':growing_season_mean_list,
                             }

        outf = outdir_CV + 'extract_annual_growing_season_LAI_mean.npy'

        np.save(outf, result_dic)

    def relative_change(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir = result_root + rf'\3mm\GLOBMAP\extract_annual_growing_season_LAI_mean\\'
        outdir = result_root + rf'\3mm\GLOBMAP\relative_change\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue



            outf = outdir + f.split('.')[0]+'_relative_change.npy'
            # if isfile(outf):
            #     continue
            # print(outf)

            dic = T.load_npy(fdir + f)

            relative_change = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue


                time_series = dic[pix]['growing_season']
                print(time_series)

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
                if np.nanmean(time_series) <-999:
                    continue
                time_series = time_series
                mean = np.nanmean(time_series)


                relative_change[pix] = (time_series-mean)/mean * 100


                # plt.plot(time_series)
                # plt.legend(['raw'])
                # # plt.show()
                #
                # plt.plot(relative_change[pix])
                # plt.legend(['zscore'])
                # plt.legend(['raw','zscore'])
                # plt.show()

                ## save
            np.save(outf, relative_change)

    def trend_analysis(self):  ##each window average trend

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'D:\Project3\Result\Nov\LAI4g\relative_change\\'
        outdir = rf'D:\Project3\Result\Nov\LAI4g\relative_change\\trend_analysis\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            # if not f.split('.')[0] in ['seasonal_rainfall_intervals', 'seasonal_rainfall_event_size',
            #                            'rainfall_frequency', 'heavy_rainfall_days', 'rainfall_event_size',

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

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
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

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

    def detrend(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=result_root + rf'\LAI4g\relative_change\\'
        outdir=result_root + rf'\LAI4g\\relative_change\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue


            print(f)

            outf=outdir+f.split('.')[0]+'_detrend.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

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



class processing_LAI4g():
    def __init__(self):
        pass

    def run(self):

        # self.extract_phenology_monthly_variables()
        # self.extract_annual_growing_season_LAI_mean()
        # self.relative_change()
        # self.trend_analysis()
        self.detrend()
        # self.deseanalized_detrend()
        pass

    def extract_phenology_monthly_variables(self):
        fdir = rf'D:\Project3\Data\LAI4g\dic_monthly\\'

        outdir = rf'D:\Project3\Data\LAI4g\\extract_phenology_monthly\\'

        Tools().mk_dir(outdir, force=True)
        f_phenology = rf'D:\Project3\Data\LAI4g\4GST\\4GST.npy'
        phenology_dic = T.load_npy(f_phenology)
        new_spatial_dic = {}
        for pix in phenology_dic:
            val = phenology_dic[pix]['Offsets']
            try:
                val = float(val)
            except:
                continue

            new_spatial_dic[pix] = val
        spatial_array = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(new_spatial_dic)
        # plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
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
                # print(pix)

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
                        time_series_reshape = time_series_flatten.reshape(-1, 12)
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

                else:
                    SeasClss = phenology_dic[pix]['SeasClss']
                    print(SeasType, SeasClss)
                    continue
                spatial_dict_gs_count[pix] = time_series_gs.shape[1]
                result_dic[pix] = time_series_gs
            # print(spatial_dict_gs_count)
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_gs_count)
            # # arr[arr<6] = np.nan
            # plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=12)
            # plt.colorbar()
            # plt.show()
            np.save(outf, result_dic)

    def extract_annual_growing_season_LAI_mean(self):  ## extract LAI average
        fdir = rf'D:\Project3\Data\LAI4g\\extract_phenology_monthly\\'

        outdir_CV = result_root + rf'\3mm\LAI4g\\extract_annual_growing_season_LAI_mean\\'
        # print(outdir_CV);exit()

        T.mk_dir(outdir_CV, force=True)

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

                sum_growing_season = np.nanmean(val)

                growing_season_mean_list.append(sum_growing_season)

            result_dic[pix] = {
                'growing_season': growing_season_mean_list,
            }

        outf = outdir_CV + 'LAI4g_mean.npy'

        np.save(outf, result_dic)

    def relative_change(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir = result_root + rf'\Nov\LAI4g\extract_annual_growing_season_LAI_mean\\'
        outdir = result_root + rf'\Nov\LAI4g\relative_change\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            outf = outdir + f.split('.')[0] + '_relative_change.npy'
            # if isfile(outf):
            #     continue
            # print(outf)

            dic = T.load_npy(fdir + f)

            relative_change = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue

                time_series = dic[pix]['growing_season']
                print(time_series)

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
                # # plt.show()
                #
                # plt.plot(relative_change[pix])
                # plt.legend(['zscore'])
                # plt.legend(['raw','zscore'])
                # plt.show()

                ## save
            np.save(outf, relative_change)

    def trend_analysis(self):  ##each window average trend

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'D:\Project3\Result\Nov\LAI4g\relative_change\\'
        outdir = rf'D:\Project3\Result\Nov\LAI4g\relative_change\\trend_analysis\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            # if not f.split('.')[0] in ['seasonal_rainfall_intervals', 'seasonal_rainfall_event_size',
            #                            'rainfall_frequency', 'heavy_rainfall_days', 'rainfall_event_size',

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

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
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

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

    def detrend(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=result_root + rf'\Nov\LAI4g\extract_annual_growing_season_LAI_mean\\'
        outdir=result_root + rf'Nov\LAI4g\\detrend\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue


            print(f)

            outf=outdir+f.split('.')[0]+'_detrend.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

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
                time_series = dic[pix]['growing_season']
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


class processing_SNU_LAI():
    def __init__(self):
        pass

    def run(self):

        self.extract_phenology_monthly_variables()
        # self.extract_annual_growing_season_LAI_mean()
        # self.relative_change()
        # self.trend_analysis()
        # self.detrend()
        # self.deseanalized_detrend()
        pass

    def extract_phenology_monthly_variables(self):
        fdir = rf'D:\Project3\Data\SNU_LAI\dic\\'

        outdir = rf'D:\Project3\Data\SNU_LAI\\extract_phenology_monthly\\'

        Tools().mk_dir(outdir, force=True)
        f_phenology = rf'D:\Project3\Data\LAI4g\4GST\\4GST.npy'
        phenology_dic = T.load_npy(f_phenology)
        new_spatial_dic = {}
        for pix in phenology_dic:
            val = phenology_dic[pix]['Offsets']
            try:
                val = float(val)
            except:
                continue

            new_spatial_dic[pix] = val
        spatial_array = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(new_spatial_dic)
        # plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
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
                # print(pix)

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
                        time_series_reshape = time_series_flatten.reshape(-1, 12)
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

                else:
                    SeasClss = phenology_dic[pix]['SeasClss']
                    print(SeasType, SeasClss)
                    continue
                spatial_dict_gs_count[pix] = time_series_gs.shape[1]
                result_dic[pix] = time_series_gs
            # print(spatial_dict_gs_count)
            # np.save(outf, result_dic)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_gs_count)
        # arr[arr<6] = np.nan
        plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=12)
        plt.colorbar()
        plt.show()


    def extract_annual_growing_season_LAI_mean(self):  ## extract LAI average
        fdir = rf'D:\Project3\Data\SNU_LAI\\extract_phenology_monthly\\'

        outdir_CV = result_root + rf'\Nov\SNU_LAI\\extract_annual_growing_season_LAI_mean\\'
        # print(outdir_CV);exit()

        T.mk_dir(outdir_CV, force=True)

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

                sum_growing_season = np.nanmean(val)

                growing_season_mean_list.append(sum_growing_season)

            result_dic[pix] = {
                'growing_season': growing_season_mean_list,
            }

        outf = outdir_CV + 'extract_annual_growing_season_LAI_mean.npy'

        np.save(outf, result_dic)

    def relative_change(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir = result_root + rf'\Nov\SNU_LAI\extract_annual_growing_season_LAI_mean\\'
        outdir = result_root + rf'\Nov\SNU_LAI\relative_change\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            outf = outdir + f.split('.')[0] + '_relative_change.npy'
            # if isfile(outf):
            #     continue
            # print(outf)

            dic = T.load_npy(fdir + f)

            relative_change = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue

                time_series = dic[pix]['growing_season']
                print(time_series)

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
                # # plt.show()
                #
                # plt.plot(relative_change[pix])
                # plt.legend(['zscore'])
                # plt.legend(['raw','zscore'])
                # plt.show()

                ## save
            np.save(outf, relative_change)

    def trend_analysis(self):  ##each window average trend

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'D:\Project3\Result\Nov\SNU_LAI\relative_change\\'
        outdir = rf'D:\Project3\Result\Nov\SNU_LAI\relative_change\\trend_analysis\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            # if not f.split('.')[0] in ['seasonal_rainfall_intervals', 'seasonal_rainfall_event_size',
            #                            'rainfall_frequency', 'heavy_rainfall_days', 'rainfall_event_size',

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

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
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

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

    def detrend(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=result_root + rf'\Nov\SNU_LAI\extract_annual_growing_season_LAI_mean\\'
        outdir=result_root + rf'Nov\SNU_LAI\\detrend\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue


            print(f)

            outf=outdir+f.split('.')[0]+'_detrend.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

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
                time_series = dic[pix]['growing_season']
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


class moving_window():
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/Nov/'
        pass
    def run(self):
        # self.moving_window_extraction()

        # self.moving_window_CV_extraction_anaysis_LAI()
        # self.moving_window_CV_extraction_anaysis_rainfall()
        self.moving_window_average_anaysis()
        # self.moving_window_max_min_anaysis()
        # self.moving_window_std_anaysis()
        # self.moving_window_trend_anaysis()
        # self.trend_analysis()

        # self.robinson()

        pass

    def moving_window_extraction(self):


        fdir_all =result_root+ rf'\Composite_LAI\detrend_deseasonal\\'
        outdir = self.result_root + rf'\Composite_LAI\detrend_deseasonal\\\\moving_window_extraction\\'

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

    def moving_window_CV_extraction_anaysis_rainfall(self):
        window_size=15


        fdir = rf'D:\Project3\Result\Nov\CRU_monthly\extract_annual_growing_season_mean\detrend\moving_window_extraction\\'
        outdir = rf'D:\Project3\Result\Nov\CRU_monthly\extract_annual_growing_season_mean\detrend\moving_window_extraction_CV\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'sum' in f:
                continue


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

                if len(time_series_all)<24:  ##
                    continue
                time_series_all = np.array(time_series_all)

                slides = len(time_series_all)
                for ss in range(slides):
                    if np.isnan(np.nanmean(time_series_all)):
                        print('error')
                        continue
                    # print((len(time_series)))

                    ### if all values are identical, then continue
                    time_series=time_series_all[ss]
                    # plt.plot(time_series)
                    # plt.show()
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    # print(len(time_series))

                    if np.nanmean(time_series)==0:
                        continue
                    cv=np.nanstd(time_series)/np.nanmean(time_series)*100

                    trend_list.append(cv)
                print(len(trend_list))

                trend_dic[pix]=trend_list


            np.save(outf, trend_dic)
            T.open_path_and_file(outdir)




    def moving_window_CV_extraction_anaysis_LAI(self):
        window_size=15

        fdir = result_root + rf'\TRENDY\S2\15_year\moving_window_extraction\\'
        outdir = result_root + rf'\\TRENDY\S2\15_year\\\\moving_window_extraction_CV\\'
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
                print(len(time_series_all))
                if len(time_series_all)<24:  ##
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




    def moving_window_average_anaysis(self): ## each window calculating the average
        window_size = 15


        fdir = result_root + rf'\extraction_rainfall_characteristic\moving_window_extraction\\'
        outdir = result_root + rf'\extraction_rainfall_characteristic\moving_window_extraction_average\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):


            dic = T.load_npy(fdir + f)


            outf = outdir + f.split('.')[0] + f'_average.npy'
            print(outf)

            trend_dic = {}


            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                # plt.imshow(time_series_all)
                # plt.show()
                time_series_all = np.array(time_series_all)
                # print(time_series_all)
                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                slides = len(time_series_all)
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

                    average=np.nanmean(time_series)
                    # print(average)

                    trend_list.append(average)
                # plt.plot(trend_list)
                #
                # plt.show()

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)

    def moving_window_max_min_anaysis(self): ## each window calculating the average
        window_size = 15


        fdir =result_root+ rf'\3mm\Fire\moving_window_extraction\\'
        outdir = result_root+rf'\3mm\Fire\moving_window_extraction\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'sum' in f:
                continue



            dic = T.load_npy(fdir + f)

            slides = 36 - window_size+1   ## revise!!
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
                for ss in range(slides):


                    ### if all values are identical, then continue
                    if len(time_series_all)<22:
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


    def moving_window_std_anaysis(self):
        window_size=15
        fdir = rf'D:\Project3\Result\3mm\relative_change_growing_season\moving_window_extraction\\'
        outdir = rf'D:\Project3\Result\3mm\relative_change_growing_season\\\\moving_window_mean_std_anaysis\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'LAI4g_detrend' in f:
                continue

            dic = T.load_npy(fdir + f)
            slides = 38-window_size+1
            outf = outdir + f.split('.')[0] + f'_mean.npy'
            print(outf)

            # if os.path.isfile(outf):
            #     continue

            new_x_extraction_by_window = {}
            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):
                trend_list = []


                time_series_all = dic[pix]
                if len(time_series_all)<24:
                    continue
                time_series_all = np.array(time_series_all)
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
                    # cv=np.nanstd(time_series)
                    cv = np.nanmean(time_series)
                    trend_list.append(cv)

                trend_dic[pix]=trend_list

            np.save(outf, trend_dic)

            ##tiff
            # arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            #
            # p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')

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

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir =result_root+ rf'\TRENDY\S2\15_year\moving_window_extraction_CV\\'
        outdir =result_root + (rf'\TRENDY\S2\15_year\moving_window_extraction_CV\\trend\\')
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

    def trend_analysis_relative_change(self):  ##trend/average

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'E:\Project3\Result\extract_rainfall_phenology_year\CRU-JRA\moving_window_average_anaysis\ecosystem_year\\'
        outdir = rf'E:\Project3\Result\extract_rainfall_phenology_year\CRU-JRA\moving_window_average_anaysis\ecosystem_year\trend\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            if not f.split('.')[0] in ['detrended_sum_rainfall_CV', 'heat_event_frenquency',
                                       'rainfall_intensity','rainfall_frenquency',
                'rainfall_seasonality_all_year']:
                continue
            #     continue
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

            # plt.imshow(arr_trend, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)



    def robinson(self):
        fdir=rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\trend\\'
        temp_root=rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\trend\\'
        out_pdf_fdir=rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\trend\\pdf\\'

        T.mk_dir(out_pdf_fdir,force=True)


        variable='detrended_growing_season_LAI_mean_CV'
        f_trend=fdir+variable+'_trend.tif'

        f_p_value=fdir+variable+'_p_value.tif'
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        m,ret=Plot().plot_Robinson(f_trend, ax=ax, vmin=-1, vmax=1, is_plot_colorbar=True, is_discrete=True,colormap_n=5)

        self.plot_Robinson_significance_scatter(m, f_p_value,temp_root,0.05,s=0.1)
        # m.colorbar(location='bottom',label='(LAI CV(%/window (15 years per window)))')
        # cbar.set_label(fontsize=6, label='(LAI CV(%/window (15 years per window)))')

        # plt.title(f'{variable}_(%/yr2)')
        # m.title(f'LAI CV(%/window (15 years per window)')



        # plt.title(f'{variable}_(day/yr)')
        # plt.title('r')
        # plt.show()
        ## save fig pdf
        #save pdf
        plt.savefig(out_pdf_fdir+variable+'.png', dpi=900, bbox_inches='tight')
        plt.close()
        T.open_path_and_file(out_pdf_fdir)

    def plot_Robinson(self, fpath, ax=None, cmap=None, vmin=None, vmax=None, is_plot_colorbar=True, is_reproj=True,res=25000,is_discrete=False,colormap_n=11):
        '''
        :param fpath: tif file
        :param is_reproj: if True, reproject file from 4326 to Robinson
        :param res: resolution, meter
        '''
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        # Blue represents high values, and red represents low values.
        if ax == None:
            # plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list)
        if not is_reproj:
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        else:
            fpath_robinson = self.Robinson_reproj(fpath, fpath + '_robinson-reproj.tif',res=res)
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_robinson)
            os.remove(fpath_robinson)
        originY1 = copy.copy(originY)
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr_m = ma.masked_where(np.isnan(arr), arr)
        originX = 0
        originY = originY * 2
        lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
        lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        lon_list, lat_list = np.meshgrid(lon_list, lat_list)
        m = Basemap(projection='robin', lon_0=0, lat_0=90., ax=ax, resolution='i')
        ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax,)
        m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        for obj in meridict:
            line = meridict[obj][0][0]
        coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        polys = m.fillcontinents(color='#D1D1D1', lake_color='#EFEFEF',zorder=90)
        if is_plot_colorbar:
            if is_discrete:
                bounds = np.linspace(vmin, vmax, colormap_n)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.5)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation='horizontal')

            else:
                cbar = plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom', pad=0.05)
        return m, ret


    def plot_Robinson_significance_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5, s=5,
                                        c='k', marker='.',
                                        zorder=100, res=2):

        fpath_clip = fpath_p + 'clip.tif'
        fpath_spatial_dict = DIC_and_TIF(tif_template=fpath_p).spatial_tif_to_dic(fpath_p)
        D_clip = DIC_and_TIF(tif_template=fpath_p)
        D_clip_lon_lat_pix_dict = D_clip.spatial_tif_to_lon_lat_dic(temp_root)
        fpath_clip_spatial_dict_clipped = {}
        for pix in fpath_spatial_dict:
            lon, lat = D_clip_lon_lat_pix_dict[pix]
            fpath_clip_spatial_dict_clipped[pix] = fpath_spatial_dict[pix]
        DIC_and_TIF(tif_template=fpath_p).pix_dic_to_tif(fpath_clip_spatial_dict_clipped, fpath_clip)
        fpath_resample = fpath_clip + 'resample.tif'
        ToRaster().resample_reproj(fpath_clip, fpath_resample, res=res)
        fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        self.Robinson_reproj(fpath_resample, fpath_resample_ortho, res=res * 100000)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample_ortho)

        arr = Tools().mask_999999_arr(arr, warning=False)
        arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample_ortho)
        #
        os.remove(fpath_clip)
        os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)

        spatial_dict = D_resample.spatial_arr_to_dic(arr)
        lon_lat_pix_dict = D_resample.spatial_tif_to_lon_lat_dic(temp_root)

        lon_list = []
        lat_list = []
        for pix in spatial_dict:
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            lon, lat = lon_lat_pix_dict[pix]
            lon_list.append(lon)
            lat_list.append(lat)
        lon_list = np.array(lon_list)
        lat_list = np.array(lat_list)
        lon_list = lon_list - originX
        lat_list = lat_list + originY
        lon_list = lon_list + pixelWidth / 2
        lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        m.scatter(lon_list, lat_list, latlon=False, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)

        return m

    def Robinson_reproj(self, fpath, outf, res=50000):
        wkt = self.Robinson_wkt()
        srs = DIC_and_TIF().gen_srs_from_wkt(wkt)
        ToRaster().resample_reproj(fpath, outf, res, dstSRS=srs)
        return outf

    def Robinson_wkt(self):
        wkt = '''
        PROJCRS["Sphere_Robinson",
    BASEGEOGCRS["Unknown datum based upon the Authalic Sphere",
        DATUM["Not specified (based on Authalic Sphere)",
            ELLIPSOID["Sphere",6371000,0,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["Sphere_Robinson",
        METHOD["Robinson"],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Not known."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["ESRI",53030]]'''
        return wkt




        pass



class processing_composite_LAI():
    def __init__(self):
        pass
    def run (self):

        self.average_detrend_deseasonalized_composite_LAI()
        # self.average_LAI_detrend()
        pass



    def average_detrend_deseasonalized_composite_LAI(self):

        f_GLOBMAP=data_root+ rf'\GLOBMAP\extract_phenology_monthly_detrend_deseason\\extract_phenology_monthly_detrend_deseason_relative_change.npy'
        f_SNU_LAI=data_root+rf'\SNU_LAI\extract_phenology_monthly_detrend_deseason\\extract_phenology_monthly_detrend_deseason_relative_change.npy'
        f_LAI4g=data_root+rf'\LAI4g\extract_phenology_monthly_detrend_deseason\\extract_phenology_monthly_detrend_deseason_relative_change.npy'
        outf=result_root+rf'\Composite_LAI\\detrend_deseasonal\\composite_LAI_relative_change_detrend_mean.npy'
        dic_GLOBMAP=T.load_npy(f_GLOBMAP)
        dic_SNU_LAI=T.load_npy(f_SNU_LAI)
        dic_LAI4g=T.load_npy(f_LAI4g)
        result_dic={}

        for pix in dic_GLOBMAP:
            vals_GLOBMAP=dic_GLOBMAP[pix]
            if pix not in dic_SNU_LAI:
                continue
            vals_SNU_LAI=dic_SNU_LAI[pix]
            if pix not in dic_LAI4g:
                continue
            vals_LAI4g=dic_LAI4g[pix]
            vals=np.nanmean([vals_GLOBMAP,vals_SNU_LAI,vals_LAI4g],axis=0)
            # plt.imshow(vals,interpolation='nearest',cmap='jet'
            #            )
            # plt.show()
            result_dic[pix]=vals

        T.save_npy(result_dic,outf)




        pass

    def average_LAI_relative_change(self):

        infdir = result_root
        f_1 = infdir + rf'LAI4g\\anomaly\\\\LAI4g_detrend.npy'
        f_2 = infdir + rf'GLOBMAP\anomaly\\\\GLOBMAP_detrend.npy'
        f_3 = infdir + rf'SNU_LAI\anomaly\\\\SNU_LAI_detrend.npy'
        dic1 = np.load(f_1, allow_pickle=True).item()
        dic2 = np.load(f_2, allow_pickle=True).item()
        dic3 = np.load(f_3, allow_pickle=True).item()
        average_dic = {}

        for pix in tqdm(dic1):
            if not pix in dic2:
                continue
            if not pix in dic3:
                continue
            value1 = dic1[pix]
            value2 = dic2[pix]
            value3 = dic3[pix]

            value1 = np.array(value1)
            value2 = np.array(value2)
            value3 = np.array(value3)
            if len(value1) < 38 or len(value2) < 38 or len(value3) < 38:
                print(pix, len(value1), len(value2), len(value3))
                continue
            print(len(value1), len(value2), len(value3))
            if len(value1) != len(value2) or len(value2) != len(value3):
                print(pix, len(value1), len(value2), len(value3))
                continue


            average_val = np.nanmean([value1, value2, value3], axis=0)

            # print(average_val)
            if np.nanmean(average_val) > 999:
                continue
            if np.nanmean(average_val) < -999:
                continue
            average_dic[pix] = average_val
            #
            # plt.plot(value1,color='blue')
            # plt.plot(value2,color='green')
            # plt.plot(value3,color='orange')
            # plt.plot(average_val,color='red')
            # plt.legend(['GlOBMAP','SNU','LAI4g','average'])
            # plt.show()

        outdir = result_root + rf'\Composite_LAI\\anomaly_detrend\\'
        Tools().mk_dir(outdir, force=True)

        np.save(outdir + 'composite_LAI_anomaly_detrend_mean.npy', average_dic)


    def average_LAI_detrend(self):

        infdir = result_root
        f_1 = infdir + rf'LAI4g\\relative_change\\\\LAI4g_relative_change_detrend.npy'
        f_2 = infdir + rf'GLOBMAP\relative_change\\\\GLOBMAP_LAI_relative_change_detrend.npy'
        f_3 = infdir + rf'SNU_LAI\relative_change\\\\SNU_LAI_relative_change_detrend.npy'
        dic1 = np.load(f_1, allow_pickle=True).item()
        dic2 = np.load(f_2, allow_pickle=True).item()
        dic3 = np.load(f_3, allow_pickle=True).item()
        average_dic = {}

        for pix in tqdm(dic1):
            if not pix in dic2:
                continue
            if not pix in dic3:
                continue
            value1 = dic1[pix]
            value2 = dic2[pix]
            value3 = dic3[pix]

            value1 = np.array(value1)
            value2 = np.array(value2)
            value3 = np.array(value3)
            if len(value1) < 38 or len(value2) < 38 or len(value3) < 38:
                print(pix, len(value1), len(value2), len(value3))
                continue
            print(len(value1), len(value2), len(value3))
            if len(value1) != len(value2) or len(value2) != len(value3):
                print(pix, len(value1), len(value2), len(value3))
                continue


            average_val = np.nanmedian([value1, value2, value3], axis=0)

            # print(average_val)
            if np.nanmean(average_val) > 999:
                continue
            if np.nanmean(average_val) < -999:
                continue
            average_dic[pix] = average_val
            #
            # plt.plot(value1,color='blue')
            # plt.plot(value2,color='green')
            # plt.plot(value3,color='orange')
            # plt.plot(average_val,color='red')
            # plt.legend(['GlOBMAP','SNU','LAI4g','average'])
            # plt.show()

        outdir = result_root + rf'\Composite_LAI\\relative_change_detrend\\'
        Tools().mk_dir(outdir, force=True)

        np.save(outdir + 'composite_LAI_relative_change_detrend_median.npy', average_dic)

    def trend_analysis(self):  ##each window average trend

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root+rf'\Composite_LAI\relative_change\\'
        outdir = result_root+rf'\Composite_LAI\relative_change\\trend_analysis\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            # if not f.split('.')[0] in ['seasonal_rainfall_intervals', 'seasonal_rainfall_event_size',
            #                            'rainfall_frequency', 'heavy_rainfall_days', 'rainfall_event_size',

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

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
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

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)


    def detrend(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=result_root + rf'\Composite_LAI\relative_change\\'
        outdir=result_root + rf'\Composite_LAI\\relative_change\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue


            print(f)

            outf=outdir+f.split('.')[0]+'_detrend.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

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
                time_series = dic[pix]['growing_season']
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


class processing_TRENDY():
    def __init__(self):
        pass

    def run(self):
        # self.extract_phenology_monthly_variables()
        # self.extract_annual_growing_season_LAI_mean()
        # self.relative_change()
        # self.trend_analysis()
        # self.detrend()
        self.TRENDY_ensemble_npy()
        pass

    def extract_phenology_monthly_variables(self):
        fdir_all = rf'D:\Project3\Data\TRENDY\S2\dic\\'
        f_phenology = rf'D:\Project3\Data\LAI4g\4GST\\4GST.npy'
        phenology_dic = T.load_npy(f_phenology)
        for fdir in os.listdir(fdir_all):
            new_spatial_dic = {}
            spatial_dict_gs_count = {}

            outdir = rf'D:\Project3\Data\TRENDY\\S2\\extract_phenology_monthly\\{fdir}\\'

            Tools().mk_dir(outdir, force=True)


            for pix in phenology_dic:
                val=phenology_dic[pix]['Offsets']
                try:
                    val=float(val)
                except:
                    continue

                new_spatial_dic[pix]=val
            spatial_array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(new_spatial_dic)
            # plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
            # plt.show()
            # exit()


            for f in T.listdir(fdir_all+fdir):

                outf = outdir + f
                #
                # if os.path.isfile(outf):
                #     continue
                # print(outf)
                spatial_dict = dict(np.load(join(fdir_all+fdir, f), allow_pickle=True, encoding='latin1').item())
                dic_DOY={15: 1,
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
                    # print(pix)

                    r, c = pix

                    SeasType=phenology_dic[pix]['SeasType']
                    if SeasType==2:

                        SOS=phenology_dic[pix]['Onsets']
                        try:
                            SOS=float(SOS)

                        except:
                            continue

                        SOS=int(SOS)
                        SOS_monthly=dic_DOY[SOS]

                        EOS=phenology_dic[pix]['Offsets']
                        EOS=int(EOS)
                        EOS_monthly=dic_DOY[EOS]
                        # print(SOS_monthly,EOS_monthly)
                        # print(SOS,EOS)

                        time_series = spatial_dict[pix]

                        time_series = np.array(time_series)
                        if SOS_monthly>EOS_monthly:  ## south hemisphere
                            time_series_flatten=time_series.flatten()
                            time_series_reshape=time_series_flatten.reshape(-1,12)
                            time_series_dict={}
                            for y in range(len(time_series_reshape)):
                                if y+1==len(time_series_reshape):
                                    break

                                time_series_dict[y]=np.concatenate((time_series_reshape[y][SOS_monthly-1:],time_series_reshape[y+1][:EOS_monthly]))

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

                    elif SeasType==3:
                        time_series = spatial_dict[pix]
                        time_series = np.array(time_series)
                        time_series_gs = np.reshape(time_series, (-1, 12))

                    else:
                        SeasClss=phenology_dic[pix]['SeasClss']
                        print(SeasType,SeasClss)
                        continue
                    spatial_dict_gs_count[pix] = time_series_gs.shape[1]
                    result_dic[pix] = time_series_gs
            # print(spatial_dict_gs_count)
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_gs_count)
            # # arr[arr<6] = np.nan
            # plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=12)
            # plt.colorbar()
            # plt.show()
                np.save(outf, result_dic)

    def extract_annual_growing_season_LAI_mean(self):  ## extract LAI average
        fdir_all =data_root + rf'\TRENDY\S2\extract_phenology_monthly\\'

        for fdir in T.listdir(fdir_all):
            outdir = result_root + rf'TRENDY\S2\extract_annual_growing_season_LAI_mean\\{fdir}\\'

            T.mk_dir(outdir, force=True)
            spatial_dic = T.load_npy_dir(fdir_all+fdir)
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

                    sum_growing_season = np.nanmean(val)

                    growing_season_mean_list.append(sum_growing_season)

                result_dic[pix] = {
                    'growing_season': growing_season_mean_list,
                }

            outf = outdir + f'{fdir}_LAI.npy'

            np.save(outf, result_dic)

    def relative_change(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir_all = result_root + rf'\TRENDY\S2\extract_annual_growing_season_LAI_mean\\'
        outdir=result_root + rf'\TRENDY\S2\relative_change\\'

        T.mk_dir(outdir, force=True)


        for fdir in os.listdir(fdir_all):

            outf = outdir + f'\\{fdir}_relative_change.npy'
            fpath=join(fdir_all,fdir,f'{fdir}_LAI.npy')

            dic = T.load_npy(fpath)

            relative_change = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue

                time_series = dic[pix]['growing_season']
                print(time_series)

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

    def trend_analysis(self):  ##each window average trend

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'D:\Project3\Result\Nov\TRENDY\S2\relative_change\\'
        outdir = rf'D:\Project3\Result\Nov\TRENDY\S2\relative_change\\trend_analysis\\'
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

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
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

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)


    def detrend(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir_all=result_root + rf'\TRENDY\S2\relative_change\\relative_change\\'
        outdir=result_root + rf'\TRENDY\S2\\relative_change\\detrend_relative_change\\'
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

    def TRENDY_ensemble_npy(self):
        composite_LAI_reference=result_root + rf'\LAI4g\relative_change\\LAI4g_relative_change_detrend.npy'

        composite_LAI = dict(np.load(composite_LAI_reference, allow_pickle=True, ).item())

        model_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',
                      'YIBs_S2_Monthly_lai']

        fdir = result_root + rf'\TRENDY\S2\relative_change\detrend_relative_change\\'

        result_dic={}

        for model in model_list:
            fpath = fdir + model + '_relative_change_detrend.npy'
            dic=T.load_npy(fpath)
            result_dic[model]=dic
        ## for each pixel calculate mean
        all_pixels = list(result_dic[model_list[0]].keys())
        ensemble_mean_dic = {}

        for pix in all_pixels:
            timeseries_list = []
            for model in model_list:
                if pix not in result_dic[model]:
                    continue
                vals = result_dic[model][pix]
                vals_Ref = composite_LAI[pix]

                # print(model, len(result_dic[model][pix]))
                if len(vals)!= len(vals_Ref):
                    continue
                if np.all(np.isnan(vals)):
                    continue

                timeseries_list.append(vals)
            print(len(timeseries_list))

            if timeseries_list:
                # stack 成 (13, T) 然后对 axis=0 平均
                timeseries_list = np.array(timeseries_list)
                for time_series in timeseries_list:
                    plt.plot(time_series)



                mean_ts = np.nanmedian(timeseries_list, axis=0)

                # plt.plot(mean_ts,color='k',label='median')
                # plt.legend()
                #
                # plt.show()

                ensemble_mean_dic[pix] = mean_ts


        outf=result_root + rf'TRENDY\S2\relative_change\detrend_relative_change\TRENDY_ensemble_relative_change_detrend_median.npy'
        T.save_npy(ensemble_mean_dic, outf)




    pass
class processing_climate_variable():
    def __init__(self):
        pass
    def run(self):
        # self.extract_phenology_monthly_variables()
        # self.extract_annual_growing_season_LAI_mean()
        # self.detrend()


        pass
    def extract_phenology_monthly_variables(self):
        fdir = rf'D:\Project3\Data\CRU_monthly\VPD\dic\\'

        outdir = rf'D:\Project3\Data\CRU_monthly\\VPD\\extract_phenology_monthly\\'

        Tools().mk_dir(outdir, force=True)
        f_phenology = rf'D:\Project3\Data\LAI4g\4GST\\4GST.npy'
        phenology_dic = T.load_npy(f_phenology)
        new_spatial_dic = {}

        for pix in phenology_dic:
            val = phenology_dic[pix]['Offsets']
            try:
                val = float(val)
            except:
                continue

            new_spatial_dic[pix] = val
        spatial_array = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(new_spatial_dic)
        # plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
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
                # print(pix)

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
                    print(time_series)

                    time_series = np.array(time_series)
                    if SOS_monthly > EOS_monthly:  ## south hemisphere
                        time_series_flatten = time_series.flatten()
                        time_series_reshape = time_series_flatten.reshape(-1, 12)
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

                else:
                    SeasClss = phenology_dic[pix]['SeasClss']
                    print(SeasType, SeasClss)
                    continue
                spatial_dict_gs_count[pix] = time_series_gs.shape[1]
                result_dic[pix] = time_series_gs
            # print(spatial_dict_gs_count)
            # np.save(outf, result_dic)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_gs_count)
        arr[arr>4] = np.nan
        plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=12)
        plt.colorbar()
        plt.show()


    def extract_annual_growing_season_LAI_mean(self):  ## extract LAI average
        fdir = rf'D:\Project3\Data\CRU_monthly\\Precip\\extract_phenology_monthly\\'

        outdir_CV = result_root + rf'\CRU_monthly\\extract_annual_growing_season_mean\\'
        # print(outdir_CV);exit()

        T.mk_dir(outdir_CV, force=True)

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

                sum_growing_season = np.nansum(val)

                growing_season_mean_list.append(sum_growing_season)

            result_dic[pix] = {
                'growing_season': growing_season_mean_list,
            }

        outf = outdir_CV + 'Precip_sum.npy'

        np.save(outf, result_dic)

    def anomaly(self):  ### anomaly GS

        fdir = rf'D:\Project3\Result\Nov\CRU_monthly\extract_annual_growing_season_mean\\'

        outdir = rf'D:\Project3\Result\Nov\CRU_monthly\\anomaly\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            if not f.endswith('.npy'):
                continue

            outf = outdir + f.split('.')[0] + '_anomaly.npy'
            print(outf)
            # if os.path.isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')

            dic = np.load(fdir + f, allow_pickle=True, ).item()

            anomaly_dic = {}

            for pix in tqdm(dic):

                r, c = pix

                time_series = dic[pix]['growing_season']
                print(len(time_series))

                time_series = np.array(time_series, float)

                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                # plt.plot(time_series)
                # plt.show()

                mean = np.nanmean(time_series)
                std = np.nanstd(time_series)

                delta_time_series = (time_series - mean)
                #
                # plt.plot(delta_time_series)
                # plt.show()

                anomaly_dic[pix] = delta_time_series

            np.save(outf, anomaly_dic)

    def detrend(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=result_root + rf'\CRU_monthly\extract_annual_growing_season_mean\\'
        outdir=result_root + rf'\CRU_monthly\\extract_annual_growing_season_mean\\detrend\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            if not f.endswith('.npy'):
                continue


            print(f)

            outf=outdir+f.split('.')[0]+'_detrend.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

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
                time_series = dic[pix]['growing_season']
                # print(len(time_series));exit()
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
                detrend_delta_time_series = signal.detrend(time_series)+np.nanmean(time_series)
                # plt.plot(time_series,color='blue')
                # plt.plot(detrend_delta_time_series,color='red')
                # plt.show()

                detrend_zscore_dic[pix] = detrend_delta_time_series

            np.save(outf, detrend_zscore_dic)






    pass
class processing_daily_rainfall():
    def __init__(self):
        pass
    def run(self):
        # self.extract_rainfall_CV_daily()
        # self.extract_rainfall_CV_monthly()
        self.calculating_colinearity()
        pass

    def extract_phenology_year_based_daily_rainfall(self):
        fdir_all = rf'D:\Project3\Data\CRU-JRA\Precip\\transform\\'
        outdir = rf'D:\Project3\Data\CRU-JRA\Precip\\extract_phenology_year_based_daily_rainfall\\'
        Tools().mk_dir(outdir, force=True)
        f_phenology = rf'D:\Project3\Data\LAI4g\4GST\\4GST.npy'
        phenology_dic = T.load_npy(f_phenology)
        spatial_dict_gs_count = {}
        for f in T.listdir(fdir_all):
            outf=outdir+f

            spatial_dict = dict(np.load(fdir_all + f, allow_pickle=True, encoding='latin1').item())

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

                    EOS = phenology_dic[pix]['Offsets']
                    EOS = int(EOS)

                    # print(SOS_monthly,EOS_monthly)
                    # print(SOS,EOS)

                    time_series = spatial_dict[pix]
                    # print(time_series)

                    time_series = np.array(time_series)
                    if SOS > EOS:  ## south hemisphere
                        time_series = np.array(time_series)
                        time_series_flatten = time_series.flatten()

                        time_series_flatten_reshape = time_series_flatten.reshape(-1, 365)
                        print(time_series_flatten_reshape.shape)
                        time_series_dict = {}
                        for y in range(len(time_series_flatten_reshape)):
                            if y + 1 == len(time_series_flatten_reshape):
                                break

                            time_series_dict[y] = np.concatenate(
                                (
                                time_series_flatten_reshape[y][SOS - 1:], time_series_flatten_reshape[y + 1][:EOS]))

                    else:
                        time_series_flatten = time_series.flatten()

                        time_series_flatten_reshape = time_series_flatten.reshape(-1, 365)
                        time_series_dict = {}
                        for y in range(len(time_series_flatten_reshape)):
                            time_series_dict[y] = time_series_flatten_reshape[y][SOS - 1:EOS]
                    time_series_gs = []
                    for y in range(len(time_series_dict)):
                        time_series_gs.append(time_series_dict[y])
                    time_series_gs = np.array(time_series_gs)

                elif SeasType == 3:
                    time_series = spatial_dict[pix]
                    time_series = np.array(time_series)
                    time_series_gs = np.reshape(time_series, (-1, 365))

                else:
                    SeasClss = phenology_dic[pix]['SeasClss']
                    print(SeasType, SeasClss)
                    continue
                # print(time_series_gs.shape[1])
                spatial_dict_gs_count[pix] = time_series_gs.shape[1]
                result_dic[pix] = time_series_gs
            # print(spatial_dict_gs_count)
            np.save(outf, result_dic)
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_gs_count)
        # arr[arr > 4] = np.nan
        plt.imshow(arr, interpolation='nearest', cmap='jet', vmin=0, vmax=365)
        plt.colorbar()
        plt.show()

    def extract_rainfall_CV_daily(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'D:\Project3\Data\CRU-JRA\Precip\extract_phenology_year_based_daily_rainfall\\'
        outdir_CV = result_root + rf'\CRU-JRA\extraction_rainfall_characteristic\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            # plt.imshow(vals)
            # plt.show()

            CV_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)
                val=val[val>1]

                CV = np.std(val) / np.mean(val) * 100
                # print(CV)
                CV_list.append(CV)
            result_dic[pix] = CV_list

        outf = outdir_CV + 'CV_daily_rainfall.npy'

        np.save(outf, result_dic)

    def extract_rainfall_CV_monthly(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'D:\Project3\Data\CRU_monthly\Precip\extract_phenology_monthly_detrend_deseason\\'
        outdir_CV = result_root + rf'\CRU-JRA\extraction_rainfall_characteristic\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            # plt.imshow(vals)
            # plt.show()

            CV_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)
                val=val[val>1]

                CV = np.std(val) / np.mean(val) * 100
                # print(CV)
                CV_list.append(CV)
            result_dic[pix] = CV_list

        outf = outdir_CV + 'CV_monthly_rainfall.npy'

        np.save(outf, result_dic)

    def calculating_colinearity(self):
        fdir=r'D:\Project3\Result\Nov\partial_correlation\colinear_test\\'
        for f in os.listdir(fdir):
            if not 'CV_monthly_rainfall_average' in f:
                continue

            spatial_dic=T.load_npy(fdir+f)
            for pix in spatial_dic:
                val=spatial_dic[pix]
                print(len(val))
                plt.plot(val)
                plt.show()

        pass
def main():
    # processing_GLOBMAP().run()
    # processing_LAI4g().run()
    # processing_SNU_LAI().run()
    # moving_window().run()
    # processing_composite_LAI().run()

    # processing_TRENDY().run()
    # processing_climate_variable().run()
    processing_daily_rainfall().run()
    pass

if __name__ == '__main__':
    main()



    pass