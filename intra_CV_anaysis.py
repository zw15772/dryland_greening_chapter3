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

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

def global_get_gs(pix):
    global_northern_hemi_gs = (5, 6, 7, 8, 9, 10)
    global_southern_hemi_gs = (11, 12, 1, 2, 3, 4)
    tropical_gs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    r, c = pix
    if r < 240:
        return global_northern_hemi_gs
    elif 240 <= r < 480:
        return tropical_gs
    elif r >= 480:
        return global_southern_hemi_gs
    else:
        raise ValueError('r not in range')

class Intra_CV_preprocessing():
    def __init__(self):
        pass

    def run(self):
        # self.extract_GS_return_biweekly()
        # self.calculate_intra_CV()
        # self.calculate_intra_stats()
        # self.calculate_total_amount_of_precip()  ## here we calculate the total amount of precip of CRU/GPCP and ERA5
        # self.trend_analysis()
        self.detrend()
        pass

    def extract_GS_return_biweekly(self):  ## extract growing season but return biweekly data

        fdir_all = data_root + rf'\biweekly\Precip\\'
        # print(fdir_all);exit()
        # r'D:\Project3\Data\biweekly\LAI4g'
        outdir = result_root + f'\extract_GS\extract_GS_return_biweekly\\Precip\\'
        # print(outdir);exit()
        Tools().mk_dir(outdir, force=True)
        date_list = []

        for year in range(1982, 2020):
            for mon in range(1, 13):
                for day in [1, 15]:
                    date_list.append(datetime.datetime(year, mon, day))

        for f in tqdm(os.listdir(fdir_all)):
            # if not '015' in f:
            #     continue

            outf = outdir + f.split('.')[0] + '.npy'

            fpath = join(fdir_all, f)

            spatial_dict = T.load_npy(fpath)
            # pprint(spatial_dict);exit()
            annual_spatial_dict = {}
            annual_spatial_dict_index = {}
            for pix in spatial_dict:
                gs_mon = global_get_gs(pix)

                vals = spatial_dict[pix]
                vals = np.array(vals)
                if T.is_all_nan(vals):
                    continue
                if np.nanstd(vals) == 0:
                    continue

                vals[vals < -999] = np.nan

                vals_dict = T.dict_zip(date_list, vals)
                # pprint(vals_dict);exit()
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)
                        date_list_index.append(i)

                consecutive_ranges = self.group_consecutive_vals(date_list_index)
                # date_dict = dict(zip(list(range(len(date_list))), date_list))
                date_dict = T.dict_zip(list(range(len(date_list))), date_list)
                # pprint(date_dict);exit()

                # annual_vals_dict = {}
                annual_gs_list = []
                annual_gs_list_idx = []
                # print(consecutive_ranges);exit()
                # print(len(consecutive_ranges[0]))
                # print(len(consecutive_ranges))

                if len(consecutive_ranges[0]) > 12:  # tropical
                    consecutive_ranges = np.reshape(consecutive_ranges, (-1, 24))
                # try:
                #     print(np.shape(consecutive_ranges))
                # except:
                #     print(consecutive_ranges)
                #     exit()
                # print(consecutive_ranges);exit()

                for idx in consecutive_ranges:
                    # print(len(idx))
                    date_gs = [date_dict[i] for i in idx]
                    # print(date_gs);exit()
                    # print(len(gs_mon)*2)
                    if not len(date_gs) == len(gs_mon) * 2:
                        continue
                    year = date_gs[0].year

                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    # print(len(vals_gs))
                    ## return monthly data
                    annual_gs_list.append(vals_gs)
                    ## return month_index
                    annual_gs_list_idx.append(idx)

                annual_gs_list = np.array(annual_gs_list)
                # pprint(annual_gs_list)
                # print(annual_gs_list.shape)
                # exit()

                # if T.is_all_nan(annual_gs_list):
                #     continue
                annual_spatial_dict[pix] = annual_gs_list
                # print(len(annual_gs_list[0]))
                # #
                # plt.imshow(annual_gs_list)
                # plt.show()
                annual_spatial_dict_index[pix] = annual_gs_list_idx

            np.save(outf, annual_spatial_dict)
            np.save(outf + '_index', annual_spatial_dict_index)

        pass

    def group_consecutive_vals(self, in_list):
        # 连续值分组
        ranges = []
        #### groupby 用法
        ### when in_list=468, how to groupby
        for _, group in groupby(enumerate(in_list), lambda index_item: index_item[0] - index_item[1]):

            group = list(map(itemgetter(1), group))
            if len(group) > 1:
                ranges.append(list(range(group[0], group[-1] + 1)))
            else:
                ranges.append([group[0]])
        return ranges

    def calculate_intra_CV(self):
        fdir_all = data_root + rf'biweekly\LAI4g\\'
        outdir = result_root + f'\intra_CV_annual\\'
        Tools().mk_dir(outdir, force=True)
        spatial_dict = {}
        for f in os.listdir(fdir_all):

            fpath = join(fdir_all, f)
            dic = T.load_npy(fpath)
            spatial_dict.update(dic)
        # pprint(spatial_dict);exit()
        CV_spatial_dict = {}
        for pix in tqdm(spatial_dict):
            CV_list=[]
            vals= spatial_dict[pix]
            vals = np.array(vals)
            vals_reshape = np.reshape(vals, (-1, 24))
            if T.is_all_nan(vals):
                continue


            for val in vals_reshape:
                if T.is_all_nan(val):
                    print('nan')
                    # CV_list.append(np.nan)
                    exit()

                CV= np.nanstd(val)/np.nanmean(val)*100
                CV_list.append(CV)
            # print(CV_list)

            CV_spatial_dict[pix] = CV_list


        np.save(outdir + 'CV_LAI4g', CV_spatial_dict)


        pass

    def calculate_intra_stats(self):
        fdir_all = data_root + rf'biweekly\\'
        outdir = result_root + f'\intra_stats_annual\\'
        Tools().mk_dir(outdir, force=True)

        for fdir in os.listdir(fdir_all):

            spatial_dict=T.load_npy_dir(fdir_all+fdir)

            # pprint(spatial_dict);exit()
            std_spatial_dict = {}
            mean_spatial_dict = {}
            max_spatial_dict = {}
            min_spatial_dict = {}
            for pix in tqdm(spatial_dict):
                std_list = []
                mean_list = []
                max_list = []
                min_list = []
                vals = spatial_dict[pix]
                vals = np.array(vals)
                vals_reshape = np.reshape(vals, (-1, 24))
                if T.is_all_nan(vals):
                    continue

                for val in vals_reshape:
                    if T.is_all_nan(val):
                        print('nan')
                        exit()
                    std = np.nanstd(val)
                    std_list.append(std)
                    mean = np.nanmean(val)
                    mean_list.append(mean)
                    max = np.nanmax(val)
                    max_list.append(max)
                    min = np.nanmin(val)
                    min_list.append(min)


                std_spatial_dict[pix] = std_list
                mean_spatial_dict[pix] = mean_list
                max_spatial_dict[pix] = max_list
                min_spatial_dict[pix] = min_list



            np.save(outdir + f'std_{fdir}', std_spatial_dict)
            np.save(outdir + f'mean_{fdir}', mean_spatial_dict)
            np.save(outdir + f'max_{fdir}', max_spatial_dict)
            np.save(outdir + f'min_{fdir}', min_spatial_dict)


        pass
    def calculate_total_amount_of_precip(self):
        fdir= data_root + rf'biweekly\Precip\\'
        spatial_dic= T.load_npy_dir(fdir)
        outdir = result_root + rf'total_precip\ERA5\\'
        Tools().mk_dir(outdir, force=True)
        total_precip_dic = {}
        for pix in tqdm(spatial_dic):
            vals = spatial_dic[pix]
            vals = np.array(vals)
            # vals_reshape = np.reshape(vals, (-1, 12))
            vals_reshape = np.reshape(vals, (-1, 24))
            if T.is_all_nan(vals):
                continue
            total_precip_list = []
            for val in vals_reshape:

                total_precip = np.nansum(val)
                total_precip_list.append(total_precip)
            total_precip_dic[pix] = total_precip_list
        np.save(outdir + 'total_precip', total_precip_dic)

        pass

    def trend_analysis(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root + rf'total_precip\\'
        outdir = result_root + rf'total_precip\\trend_analysis\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):



            outf = outdir + f.split('.')[0]
            if os.path.isfile(outf + '_trend.tif'):
                continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r, c = pix
                if r<120:
                    continue
                landcover_value=crop_mask[pix]
                if landcover_value==16 or landcover_value==17 or landcover_value==18:
                    continue
                if dic_modis_mask[pix]==12:
                    continue

                    ## ignore the last one year

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
                # print(time_series)

                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                if len(set(time_series)) == 1:
                    continue
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

            # plt.imshow(arr_trend_dryland, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

    def detrend(self):


        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = rf'D:\Project3\Result\total_precip\\raw\\'
        outdir = result_root + rf'total_precip\\Detrend\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            print(f)

            outf = outdir + f.split('.')[0] + '.npy'

            dic = dict(np.load(fdir + f, allow_pickle=True, ).item())

            detrend_zscore_dic = {}

            for pix in tqdm(dic):
                dryland_values = dic_dryland_mask[pix]
                if np.isnan(dryland_values):
                    continue
                r, c = pix

                # print(len(dic[pix]))
                time_series = dic[pix][0:38]
                # print(len(time_series))
                # print(time_series)

                time_series = np.array(time_series)
                # plt.plot(time_series)
                # plt.show()

                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.std(time_series) == 0:
                    continue
                ##### if count of nan is more than 50%, then skip
                if np.sum(np.isnan(time_series)) / len(time_series) > 0.5:
                    continue

                # mean = np.nanmean(time_series)
                # std=np.nanstd(time_series)
                # if std == 0:
                #     continue
                # delta_time_series = (time_series - mean) / std
                # if np.isnan(time_series).any():
                #     continue

                time_series = T.interp_nan(time_series)
                    # print(np.nanmean(time_series))
                    # plt.plot(time_series)

                detrend_delta_time_series = signal.detrend(time_series) + np.nanmean(time_series)
                    # plt.plot(detrend_delta_time_series)
                    # plt.show()

                detrend_zscore_dic[pix] = detrend_delta_time_series


            np.save(outf, detrend_zscore_dic)

        pass

class moving_window():
    def __init__(self):
        pass
    def run(self):
        # self.moving_window_extraction()
        # self.moving_window_extraction_for_LAI()
        # self.moving_window_trend_anaysis()
        # self.moving_window_CV_extraction_anaysis()
        self.moving_window_CV_trends()
        # self.moving_window_average_anaysis()
        # self.produce_trend_for_each_slides()
        # self.calculate_trend_spatial()
        # self.calculate_trend_trend()
        # self.convert_trend_trend_to_tif()

        # self.plot_moving_window_time_series_area()
        # self.calculate_browning_greening_average_trend()
        # self.plot_moving_window_time_series()
        pass
    def moving_window_extraction(self):

        fdir = result_root + rf'\total_precip\Detrend\\'
        outdir = result_root + rf'total_precip\\extract_window\\15\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):


            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)
            # if os.path.isfile(outf):
            #     continue

            dic = T.load_npy(fdir + f)
            window = 15

            new_x_extraction_by_window = {}
            for pix in tqdm(dic):

                time_series = dic[pix]
                time_series = np.array(time_series)

                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    print('error')
                    continue
                # print((len(time_series)))
                ### if all values are identical, then continue
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
        for i in range(len(x)):
            if i + window >= len(x):
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

    def moving_window_trend_anaysis(self):
        window_size=15
        fdir = result_root + rf'extract_window\extract_relative_change_window_CV\\'
        outdir = result_root + rf'\\extract_window\\extract_relative_change_window_CV_trend\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):

            dic = T.load_npy(fdir + f)
            slides = 39-window_size
            outf = outdir + f.split('.')[0] + f'.npy'
            print(outf)
            if os.path.isfile(outf):
                continue

            new_x_extraction_by_window = {}
            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):
                trend_list = []
                p_value_list = []

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
                    print(len(time_series))
                    slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series)), time_series)
                    trend_list.append(slope)
                    p_value_list.append(p_value)
                trend_dic[pix]=trend_list
                p_value_dic[pix]=p_value_list
                ## save
            np.save(outf, trend_dic)
            np.save(outf+'_p_value', p_value_dic)
            ##tiff
            # arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            #
            # p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')

    def moving_window_CV_extraction_anaysis(self):
        window_size=15
        fdir = result_root + rf'total_precip\\extract_window\\15\\'
        outdir = result_root + rf'\\total_precip\\extract_window\\extract_window_CV\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):

            dic = T.load_npy(fdir + f)
            slides = 38-window_size
            outf = outdir + f.split('.')[0] + f'.npy'
            print(outf)
            # if os.path.isfile(outf):
            #     continue

            new_x_extraction_by_window = {}
            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):
                trend_list = []


                time_series_all = dic[pix]
                if len(time_series_all)<23:
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
                    print(len(time_series))

                    if np.nanmean(time_series)==0:
                        continue
                    cv=np.nanstd(time_series)/np.nanmean(time_series)*100
                    trend_list.append(cv)

                trend_dic[pix]=trend_list

            np.save(outf, trend_dic)

            ##tiff
            # arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            #
            # p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')

    def moving_window_CV_trends(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)
        variable='GPCC'

        f = result_root + rf'total_precip\\extract_window\\extract_window_CV\\total_precip_{variable}.npy'
        outdir = result_root + rf'total_precip\extract_window\\CV_trend\\'
        T.mk_dir(outdir, force=True)
        dic=T.load_npy(f)
        result_dic_trend={}
        result_dic_p_value={}
        for pix in tqdm(dic):
            r,c=pix
            if r<120:
                continue
            vals=dic[pix]
            land_cover_val=crop_mask[pix]
            if land_cover_val==16 or land_cover_val==17 or land_cover_val==18:
                continue
            modis_val=dic_modis_mask[pix]
            if modis_val==12:
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            slope, b, r, p_value = T.nan_line_fit(np.arange(len(vals)), vals)
            result_dic_trend[pix]=slope
            result_dic_p_value[pix]=p_value
        array_slope=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic_trend)
        array_slope_mask=array_slope*array_mask
        array_p_value=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic_p_value)
        array_p_value_mask=array_p_value*array_mask

        DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_slope_mask,outdir+f'{variable}_CV_trend.tif')
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_p_value_mask,outdir+f'{variable}_CV_p_value.tif')

        outf=outdir+f'{variable}_CV_trend.npy'
        np.save(outf,result_dic_trend)







        pass

    def moving_window_average_anaysis(self):
        window_size = 15
        fdir = result_root + rf'extract_window\\extract_original_window\\{window_size}\\'
        outdir = result_root + rf'\\extract_window\\extract_original_window_average\\{window_size}\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):

            dic = T.load_npy(fdir + f)
            slides = 39 - window_size
            outf = outdir + f.split('.')[0] + f'.npy'
            print(outf)
            if os.path.isfile(outf):
                continue

            new_x_extraction_by_window = {}
            trend_dic = {}


            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                time_series_all = np.array(time_series_all)
                for ss in range(slides):
                    if np.isnan(np.nanmean(time_series_all)):
                        print('error')
                        continue

                    ### if all values are identical, then continue
                    if len(time_series_all)<24:
                        continue
                    time_series = time_series_all[ss]
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    print(len(time_series))
                    ##average
                    average=np.nanmean(time_series)

                    trend_list.append(average)

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)

            ##tiff
            # arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            #
            # p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')


    def produce_trend_for_each_slides(self):  ## 从上一个函数生成的一个像素24个trend, 变成 一个trend 一张图
        fdir=rf'D:\Project3\Result\extract_window\extract_relative_change_window_trend\\'
        dryland_mask=join(data_root,'Base_data','dryland_mask.tif')
        dic_dryland=DIC_and_TIF().spatial_tif_to_dic(dryland_mask)


        for f in os.listdir(fdir):
            if not 'LAI4g_p_value' in f:
                continue


            dic=T.load_npy(fdir+f)
            result_dic={}


            for slide in range(1,25):
                slide_f=f'{slide:02d}'

                for pix in dic:
                    dryland_val=dic_dryland[pix]

                    vals=dic[pix]
                    vals=np.array(vals)
                    vals=vals*dryland_val
                    vals=np.array(vals)
                    if len(vals)!=24:
                        continue
                    result_dic[pix]=vals[slide-1]
                DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(result_dic,fdir+f.split('.')[0]+f'_{slide_f}.tif')


        pass
    def calculate_trend_spatial(self):
        fdir = result_root + rf'multi_regression_moving_window\window15_relative_change\TIFF\\'
        outdir = result_root + rf'multi_regression_moving_window\window15_relative_change_trend\\'
        T.mk_dir(outdir, force=True)
        val_list=['GLEAM_SMroot_LAI4g','VPD_LAI4g']

        for val in val_list:
            array_list=[]
            for f in os.listdir(fdir):
                if not f.endswith('.tif'):
                    continue

                fname=f.split('.')[0]
                if not val in fname:
                    continue


                print(f)
                array=ToRaster().raster2array(fdir+f)[0]
                array=np.array(array)
                array[array<-999]=np.nan
                array_list.append(array)
            array_list=np.array(array_list)

            trend_list=[]


            ### calculate trend for each pixel across all slides
            for i in range(array_list.shape[1]):
                for j in range(array_list.shape[2]):
                    vals=array_list[:,i,j]
                    if np.isnan(np.nanmean(vals)):
                        trend_list.append(np.nan)
                        continue
                    slope, b, r, p_value = T.nan_line_fit(np.arange(len(vals)), vals)
                    trend_list.append(slope)
            trend_list=np.array(trend_list)
            trend_list=trend_list.reshape(array_list.shape[1],array_list.shape[2])
            outf=outdir+val+'.tif'
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(trend_list,outf)












            ##save








        T.mk_dir(outdir, force=True)
    def calculate_trend_trend(self):  ## calculate the trend of trend
        fdir = result_root + rf'extract_window\\extract_original_window_trend\\15\\GPCC\\'
        outdir = result_root + rf'\\extract_window\\extract_original_window_trend_trend\\15\\'
        T.mk_dir(outdir, force=True)
        dryland_mask=join(data_root,'Base_data','dryland_mask.tif')
        dic_dryland=DIC_and_TIF().spatial_tif_to_dic(dryland_mask)

        for f in os.listdir(fdir):
            if not 'npy' in f:
                continue
            if 'p_value' in f:
                continue


            dic = T.load_npy(fdir + f)

            outf = outdir + f.split('.')[0] + f'.npy'
            print(outf)



            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):

                time_series_all = dic[pix]
                dryland_value=dic_dryland[pix]
                if np.isnan(dryland_value):
                    continue
                time_series_all = np.array(time_series_all)

                if len(time_series_all) < 24:
                    continue

                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series_all)), time_series_all)

                trend_dic[pix]=slope
                p_value_dic[pix]=p_value
                ## save
            np.save(outf, trend_dic)
            np.save(outf+'_p_value', p_value_dic)

            ##tiff

    def convert_trend_trend_to_tif(self):
        fdir = result_root + rf'extract_window\\extract_original_window_trend_trend\\15\\'
        outdir = result_root + rf'\\extract_window\\extract_original_window_trend_trend\\15\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):

            if 'tif' in f:
                continue

            dic=T.load_npy(fdir+f)

            arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(dic)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outdir + f.split('.')[0] + '.tif')









    def plot_moving_window_time_series_area(self): ## plot the time series of moving window and calculate the area of greening and browning

        f = data_root + rf'\Base_data\\glc_025\\glc2000_025.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)


        fdir = result_root + rf'extract_window\\extract_original_window_trend\\15\\'

        dic_trend=T.load_npy(fdir+'LAI4g.npy')
        dic_p_value=T.load_npy(fdir+'LAI4g.npy_p_value.npy')

        area_dic={}
        for ss in range(39-15):
            print(ss)

            greening_area=0
            browning_area=0
            no_change_area=0

            for pix in tqdm(dic_trend):
                landcover=val_dic[pix]
                if landcover==16:

                    continue

                # print(len(dic_trend[pix]))
                if len(dic_trend[pix])<24:
                    continue
                trend=dic_trend[pix][ss]
                p_value=dic_p_value[pix][ss]
                if trend>0 and p_value<0.1:
                    greening_area+=1
                elif trend<0 and p_value<0.1:
                    browning_area+=1
                else:
                    no_change_area+=1
                greening_area_percent=greening_area/(greening_area+browning_area+no_change_area)
                browning_area_percent=browning_area/(greening_area+browning_area+no_change_area)
                no_change_area_percent=no_change_area/(greening_area+browning_area+no_change_area)


            area_dic[ss]=[greening_area_percent,browning_area_percent,no_change_area_percent]
        df=pd.DataFrame(area_dic)


        df=df.T
        ##plot
        color_list=['green','red','grey']
        df.plot(kind='bar',stacked=True,color=color_list,legend=False)
        plt.legend(['Greening','Browning','No change'],loc='upper left',bbox_to_anchor=(1.0, 1.0))
        plt.ylabel('percentage')
        plt.xlabel('moving window')
        plt.xticks(np.arange(0, 24, 1))
        plt.title('Area of greening and browning')
        plt.show()
        exit()

    def calculate_browning_greening_average_trend(self): ## each winwow, greening or browning pixels average trend
        f = data_root + rf'\Base_data\\glc_025\\glc2000_025.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        fdir = result_root + rf'extract_window\\extract_original_window_trend\\15\\'
        outdir = result_root + rf'\\extract_window\\extract_original_window_trend\\15\\'
        T.mk_dir(outdir, force=True)
        dic_trend=T.load_npy(fdir+'LAI4g.npy')
        dic_p_value=T.load_npy(fdir+'LAI4g.npy_p_value.npy')

        area_dic = {}
        for ss in range(39 - 15):
            print(ss)

            greening_value = []
            browning_value = []
            no_change_value = []
            all_value=[]
            non_sig_greening_value=[]
            non_sig_browning_value=[]
            for pix in tqdm(dic_trend):
                landcover = val_dic[pix]
                if landcover == 16:
                    continue
                # print(len(dic_trend[pix]))
                if len(dic_trend[pix]) < 24:
                    continue
                trend = dic_trend[pix][ss]
                p_value = dic_p_value[pix][ss]

                if p_value<0.1:
                    if trend>0:
                        value=trend
                        greening_value.append(value)
                    elif trend<0:
                        value=trend
                        browning_value.append(value)
                    else:
                        raise
                else:
                    if trend>0:
                        value=trend
                        non_sig_greening_value.append(value)
                    elif trend<0:
                        value=trend
                        non_sig_browning_value.append(value)
                    else:
                        continue

                    value=trend
                    no_change_value.append(value)
                all_value.append(value)


            greening_value_average = np.nanmean(greening_value)
            browning_value_average = np.nanmean(browning_value)
            no_change_value_average = np.nanmean(no_change_value)
            non_sig_greening_value_average=np.nanmean(non_sig_greening_value)
            non_sig_browning_value_average=np.nanmean(non_sig_browning_value)
            all_value_average=np.nanmean(all_value)
            area_dic[ss] = [greening_value_average, browning_value_average, no_change_value_average,all_value_average,non_sig_greening_value_average,non_sig_browning_value_average]

        df = pd.DataFrame(area_dic)
        df = df.T
        ##plot
        color_list = ['green', 'red', 'grey','black','cyan','orange']
        df.plot(kind='bar', stacked=False, color=color_list, legend=False)
        plt.legend(['Greening', 'Browning', 'No change','all_value','non-sig-greening','non-sig-browning'], loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.ylabel('LAI(m3/m3/year/year)')
        plt.xlabel('moving window')
        plt.xticks(np.arange(0, 24, 1))
        #### set line
        plt.axhline(y=-0.02, color='black', linestyle='--', linewidth=0.5)
        plt.axhline(y=0.02, color='black', linestyle='--', linewidth=0.5)
        plt.title('')
        plt.show()
        exit()

    def plot_moving_window_time_series(self): ### each winwow, greening or browning pixels average original
        fdir_trend = result_root + rf'extract_window\\extract_original_window_trend\\15\\GPCC\\'


        dic_trend = T.load_npy(fdir_trend + 'GPCC.npy')


        area_dic = {}
        for ss in range(39 - 15):
            print(ss)

            trend_value_list = []

            for pix in tqdm(dic_trend):
                # print(len(dic_trend[pix]))
                if len(dic_trend[pix]) < 24:
                    continue
                trend_value = dic_trend[pix][ss]

                trend_value_list.append(trend_value)
            trend_value_average = np.nanmean(trend_value_list)
            area_dic[ss] = [trend_value_average]
        df_new = pd.DataFrame(area_dic)
        df_new = df_new.T
        ##plot
        color_list = ['black']
        df_new.plot( color=color_list, legend=False)
        plt.legend(['trend'], loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.ylabel('precipitaton(mm/year/year)')
        plt.xlabel('moving window')
        plt.xticks(np.arange(0, 24, 1))
        plt.show()






        ##


        ####


        df_new = pd.DataFrame(area_dic)
        df_new = df_new.T
        ##plot
        color_list = ['black']
        df_new.plot( color=color_list, legend=False)
        plt.legend(['trend'], loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.ylabel('precipitaton(mm/year/year)')
        plt.xlabel('moving window')
        plt.xticks(np.arange(0, 24, 1))
        plt.show()
class extract_rainfall:
    ## 1) extract rainfall CV
    ## 2) extract rainfall total
    ## 3) extract rainfall frequency
    ## extract dry frequency
    ## 4) extract rainfall intensity
    ## 5) extract rainfall wet spell
    ## 6) extract rainfall dry spell
    def run(self):
        # self.define_quantile_threshold()
        # self.extract_rainfall_CV_total()
        # self.extract_rainfall_CV()
        # self.rainfall_frequency()
        # self.peak_timing_of_rainfall()
        # self.rainfall_intensity()
        # self.dry_spell()
        self.trend_analysis()

        # self.check_spatial_map()
        pass

    def define_quantile_threshold(self):
        # 1) extract extreme wet event based on 90th percentile and calculate frequency and total duration
        # 2) extract extreme dry event based on 10th percentile and calculate frequency and total duration
        # 3) extract wet event intensity
        ## 4) extract dry event intensity
        ## extract VPD and calculate the frequency of VPD>2kpa
        fdir=rf'E:\Data\ERA5_precip\\ERA5_daily\dict\\precip_transform\\'
        outdir = rf'E:\Data\\ERA5_precip\\ERA5_daily\dict\\define_quantile_threshold\\'
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            spatial_dic = np.load(fdir+f,allow_pickle=True).item()
            result_dic={}
            for pix in tqdm(spatial_dic):

                vals=spatial_dic[pix]
                vals_flatten=[item for sublist in vals for item in sublist]
                vals_flatten = np.array(vals_flatten)

                if T.is_all_nan(vals_flatten):
                    continue
                # plt.bar(range(len(vals_flatten)),vals_flatten)
                # plt.show()


                val_90th= np.percentile(vals_flatten,90)
                val_10th = np.percentile(vals_flatten, 10)
                val_95th = np.percentile(vals_flatten, 95)
                val_5th = np.percentile(vals_flatten, 5)
                val_99th = np.percentile(vals_flatten, 99)
                val_1st = np.percentile(vals_flatten, 1)
                dic_i={
                    '90th':val_90th,
                    '10th':val_10th,
                    '95th':val_95th,
                    '5th':val_5th,
                    '99th':val_99th,
                    '1st':val_1st
                }
                result_dic[pix]=dic_i
            outf=outdir+f
            np.save(outf,result_dic)

    def extract_extreme_rainfall_event(self):
        ENSO_type = 'La_nina'
        fdir_threshold = data_root+rf'ERA5\ERA5_daily\dict\define_quantile_threshold\\'
        fdir_yearly_all=rf'D:\Project3\Data\ERA5\ERA5_daily\dict\ENSO_year_extraction\\{ENSO_type}\\'
        outdir = data_root+rf'\ERA5\ERA5_daily\dict\\extreme_event_extraction\\{ENSO_type}\\'
        T.mk_dir(outdir,force=True)
        spatial_threshold_dic=T.load_npy_dir(fdir_threshold)
        result_dic = {}
        for f in T.listdir(fdir_yearly_all):
            spatial_dic = T.load_npy(fdir_yearly_all+f)
            for pix in tqdm(spatial_dic):
                if not pix in spatial_threshold_dic:
                    continue
                threshold_dic=spatial_threshold_dic[pix]

                val_90th = threshold_dic['90th']
                print(val_90th)
                val_10th = threshold_dic['10th']
                print(val_10th)
                EI_nino_dic= spatial_dic[pix]
                result_dic_i = {}
                for year_range in EI_nino_dic:

                    extreme_wet_event = []
                    extreme_dry_event = []
                    for val in EI_nino_dic[year_range]:
                        if val > val_90th:
                            extreme_wet_event.append(val)

                    ## calculate the frequency and average intensity of extreme wet event and extreme dry event
                    ## intensity
                    average_intensity_extreme_wet_event = np.nanmean(extreme_wet_event)

                    ## frequency
                    frequency_extreme_wet_event = len(extreme_wet_event)




                    result_dic_i[year_range] = {
                        f'{ENSO_type}_average_intensity_extreme_wet_event':average_intensity_extreme_wet_event,

                        f'{ENSO_type}_frequency_extreme_wet_event':frequency_extreme_wet_event,



                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def extract_rainfall_CV_total(self):  ## extract total and CV of rainfall
        fdir = data_root+rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir_CV = data_root+rf'\ERA5\ERA5_daily\dict\\rainfall_CV_total\\'

        T.mk_dir(outdir_CV,force=True)

        for f in T.listdir(fdir):

            spatial_dic = np.load(fdir+f,allow_pickle=True).item()
            result_dic = {}

            for pix in tqdm(spatial_dic):
                ### ui==if northern hemisphere
                r,c = pix
                vals = spatial_dic[pix]
                vals_flatten = np.array(vals).flatten()


                result_dic_i = {}

                for i in range(38):

                    if 120<r<=240:  # Northern hemisphere
                        ### April to October is growing season

                        vals_growing_season = vals_flatten[i*365+120:(i+1)*365+304]

                    elif 240<r<480:### whole year is growing season

                        vals_growing_season = vals_flatten[i*365:(i+1)*365]


                    else: ## october to April is growing season  Southern hemisphere
                        if i >=37:
                            break


                        vals_growing_season = vals_flatten[i*365+304:(i+1)*365+120]

                    vals_growing_season = np.array(vals_growing_season)
                    if T.is_all_nan(vals_growing_season):
                        continue
                    CV = np.std(vals_growing_season)/np.mean(vals_growing_season)
                    total = np.nansum(vals_growing_season)
                    result_dic_i[i] = {f'CV_rainfall':CV,
                                       }
                result_dic[pix] = result_dic_i

            outf = outdir_CV+f

            np.save(outf,result_dic)

    def extract_rainfall_CV(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'E:\Data\ERA5\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\\ERA5\ERA5_daily\dict\\rainfall_CV\\'

        T.mk_dir(outdir_CV,force=True)



        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r,c = pix
            vals = spatial_dic[pix]
            vals_flatten = np.array(vals).flatten()


            CV_list = []

            for i in range(38):

                if 120<r<=240:  # Northern hemisphere
                    ### April to October is growing season

                    vals_growing_season = vals_flatten[i*365+120:(i+1)*365+304]

                elif 240<r<480:### whole year is growing season

                    vals_growing_season = vals_flatten[i*365:(i+1)*365]


                else: ## october to April is growing season  Southern hemisphere
                    if i >=37:
                        break


                    vals_growing_season = vals_flatten[i*365+304:(i+1)*365+120]

                vals_growing_season = np.array(vals_growing_season)
                if T.is_all_nan(vals_growing_season):
                    continue
                CV = np.std(vals_growing_season)/np.mean(vals_growing_season)
                CV_list.append(CV)
            result_dic[pix] = CV_list

        outf = outdir_CV+'CV_rainfall.npy'

        np.save(outf,result_dic)


    def rainfall_frequency(self):

        fdir =rf'E:\Data\\ERA5_precip\\ERA5_daily\dict\\precip_transform\\'
        outdir= rf'E:\Data\\ERA5_precip\\ERA5_daily\dict\\rainfall_frequency\\'
        threshold_f=rf'E:\Data\\ERA5_precip\\ERA5_daily\dict\\define_quantile_threshold\\'
        dic_threshold = T.load_npy_dir(threshold_f)
        T.mk_dir(outdir,force=True)

        spatial_dic = T.load_npy_dir(fdir)

        result_dic_wet_frequency = {}
        result_dic_wet_extreme_frequency = {}

        for pix in tqdm(spatial_dic):

            if not pix in dic_threshold:
                continue
            vals = spatial_dic[pix]

            threshold = dic_threshold[pix]
            threshold_wet = threshold['90th']
            threhold_wet_extreme = threshold['95th']

            frequency_wet_list = []
            frequency_wet_extreme_list = []

            for val in vals:

                if T.is_all_nan(val):
                    continue
                ## wet event>90th percentile and <95th percentile

                frequency_wet = len(np.where((val > threshold_wet) & (val < threhold_wet_extreme))[0])
                frequency_wet_list.append(frequency_wet)
                frequency_wet_extreme = len(np.where(val > threhold_wet_extreme)[0])
                frequency_wet_extreme_list.append(frequency_wet_extreme)
            # print(frequency_wet_list)
            # print(frequency_wet_extreme_list)
            # exit()

            result_dic_wet_frequency[pix] = frequency_wet_list
            result_dic_wet_extreme_frequency[pix] = frequency_wet_extreme_list

        np.save(outdir + 'wet_frequency_90th.npy', result_dic_wet_frequency)
        np.save(outdir + 'wet_frequency_95th.npy', result_dic_wet_extreme_frequency)

    def peak_timing_of_rainfall(self):
        ## using moving window to calculate the peak timing of rainfall
        fdir = rf'E:\Data\\ERA5_precip\ERA5_daily\dict\\precip_transform\\'
        outdir = rf'E:\Data\\ERA5_precip\ERA5_daily\dict\\peak_timing_of_rainfall\\'
        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)

        result_dic = {}
        for pix in tqdm(spatial_dic):
            peak_timing_list = []

            vals = spatial_dic[pix]
            for val in vals:

                if T.is_all_nan(val):
                    continue



    def rainfall_intensity(self):
        fdir = rf'E:\Data\\ERA5_precip\ERA5_daily\dict\\precip_transform\\'
        outdir = rf'E:\Data\\ERA5_precip\ERA5_daily\dict\\rainfall_intensity\\'
        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)

        result_dic = {}
        for pix in tqdm(spatial_dic):
            intensity_list = []

            vals = spatial_dic[pix]
            for val in vals:
                ## calculate the average intensity of rainfall events

                if T.is_all_nan(val):
                    continue
                intensity = np.nanmean(val)
                intensity_list.append(intensity)
            result_dic[pix] = intensity_list
        np.save(outdir + 'rainfall_intensity.npy', result_dic)


    def dry_spell(self):

        fdir = rf'E:\Data\\ERA5_precip\ERA5_daily\dict\\precip_transform\\'
        outdir = rf'E:\Data\\ERA5_precip\ERA5_daily\dict\\dry_spell\\'
        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)

        average_dry_spell_annual_dic = {}
        maxmum_dry_spell_annual_dic = {}


        for pix in tqdm(spatial_dic):
            average_dry_spell_annual_list = []
            maxmum_dry_spell_annual_list = []

            vals = spatial_dic[pix]
            for val in vals:
                ## calculate the average intensity of rainfall events

                if T.is_all_nan(val):
                    continue
                vals_wet = val.copy()

                vals_wet[vals_wet >= 1] = np.nan

                dry_index = np.where(~np.isnan(vals_wet))
                if len(dry_index[0]) == 0:
                    continue
                dry_index = np.array(dry_index)
                dry_index = dry_index.flatten()
                dry_index_groups = T.group_consecutive_vals(dry_index)

                # plt.bar(range(len(val)), val)
                # plt.bar(range(len(val)), vals_wet)
                # print(dry_index_groups)
                # plt.show()
                ## calcuate average wet spell
                dry_spell = []
                for group in dry_index_groups:
                    dry_spell.append(len(group))
                dry_spell = np.array(dry_spell)

                average_dry_spell = np.nanmean(dry_spell)
                average_dry_spell_annual_list.append(average_dry_spell)

                maxmum_wet_spell = np.nanmax(dry_spell)
                maxmum_dry_spell_annual_list.append(maxmum_wet_spell)

            average_dry_spell_annual_dic[pix] = average_dry_spell_annual_list
            maxmum_dry_spell_annual_dic[pix] = maxmum_dry_spell_annual_list
        np.save(outdir + 'average_dry_spell.npy', average_dry_spell_annual_dic)
        np.save(outdir + 'maxmum_dry_spell.npy', maxmum_dry_spell_annual_dic)


    pass
    def trend_analysis(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=rf'E:\Data\ERA5_precip\ERA5_daily\dict\dry_spell\\'
        outdir = rf'E:\Data\ERA5_precip\ERA5_daily\dict\dry_spell\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            outf = outdir + f.split('.')[0]
            if os.path.isfile(outf + '_trend.tif'):
                continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r, c = pix
                if r < 120:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                    ## ignore the last one year

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
                # print(time_series)

                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                if len(set(time_series)) == 1:
                    continue
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

            # plt.imshow(arr_trend_dryland, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)
    pass

    def check_spatial_map(self):
        fdir = data_root + rf'\ERA5\ERA5_daily\dict\\dry_spell\\'
        spatial_dic= T.load_npy_dir(fdir)
        key_list = ['average_dry_spell','maximum_dry_spell']

        for key in key_list:
            spatial_dict_num = {}
            spatial_dict_mean = {}

            for pix in spatial_dic:

                annual_dict = spatial_dic[pix]
                if len(annual_dict)==0:
                    continue

                valid_year = 0
                vals_list = []
                for year in annual_dict:
                    dict_i = annual_dict[year]
                    if not key in dict_i:
                        continue
                    val = dict_i[key]
                    vals_list.append(val)

                    valid_year+=1
                vals_mean = np.nanmean(vals_list)
                spatial_dict_num[pix] = valid_year
                spatial_dict_mean[pix] = vals_mean

            arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_mean)
            plt.figure()
            plt.imshow(arr,interpolation='nearest')
            plt.title(key)
            plt.colorbar()
        plt.show()



        #     spatial_dict_test[pix] = np.nanmean(vals['average_dry_spell'])
        # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_test)
        # plt.imshow(arr,interpolation='nearest')
        # # plt.title(key)
        # plt.show()




        pass



class Check_data():
    def __init__(self):
        pass
    def run(self):
        # self.check_data()
        self.testrobinson()
        pass
    def check_data(self):
        # fdir_all = data_root + rf'\biweekly\LAI4g\\'
        fdir_all = result_root + f'\extract_GS\extract_GS_return_biweekly\\'
        spatial_dict=   {}
        for f in os.listdir(fdir_all):
            if 'index' in f:
                continue
            fpath = join(fdir_all, f)
            dic = T.load_npy(fpath)
            spatial_dict.update(dic)

            for pix in spatial_dict:
                vals = spatial_dict[pix]
                vals = np.array(vals)
                print(vals)
                exit()
                if T.is_all_nan(vals):
                    continue
                if np.nanstd(vals) == 0:
                    continue
                vals[vals < -999] = np.nan
                plt.plot(vals)
                plt.show()
    pass




    def testrobinson(self):

        fdir_trend = rf'E:\Data\ERA5_precip\ERA5_daily\dict\dry_spell\\'
        temp_root = rf'E:\Data\ERA5_precip\ERA5_daily\dict\dry_spell\\'
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):

            if not f.endswith('.tif'):
                continue
            if not 'trend' in f:
                continue

            fname = f.split('.')[0]
            fname_p_value = fname.replace('trend', 'p_value')
            print(fname_p_value)
            fpath = fdir_trend + f
            p_value_f = fdir_trend + fname_p_value+'.tif'
            print(p_value_f)
            # exit()
            m, ret = Plot().plot_Robinson(fpath, vmin=-0.5, vmax=0.5, is_discrete=True, colormap_n=5,)

            Plot().plot_Robinson_significance_scatter(m,p_value_f,temp_root,0.05, s=5, marker='x')
            plt.title(f'{fname}')
            plt.show()


class PLOT():
    def __init__(self):
        pass
    def run(self):
        # self.intra_inter_CV_boxplot()
        self.intra_inter_CV_scatter()
    def intra_inter_CV_boxplot(self):
        f_LAI_CV = result_root + rf'\extract_window\extract_detrend_original_window_CV\\LAI4g_CV_trend.tif'
        f_precip_CV = result_root + rf'intra_stats_annual\trend_analysis\\CV_precip_trend.tif'

        arr_LAI_CV = ToRaster().raster2array(f_LAI_CV)[0]
        arr_precip_CV = ToRaster().raster2array(f_precip_CV)[0]
        arr_LAI_CV = np.array(arr_LAI_CV)
        arr_precip_CV = np.array(arr_precip_CV)
        arr_LAI_CV[arr_LAI_CV < -999] = np.nan
        arr_precip_CV[arr_precip_CV < -999] = np.nan
        arr_LAI_CV = arr_LAI_CV.flatten()
        arr_precip_CV = arr_precip_CV.flatten()
        df=pd.DataFrame({'LAI_CV':arr_LAI_CV,'precip_CV':arr_precip_CV})
        df=df.dropna()
        # bins=np.arange(-2,2,0.2)
        bins=np.linspace(-2,2,20)

        df_group, bins_list_str=T.df_bin(df,'precip_CV',bins)
        x_list=[]
        y_list=[]
        err_list=[]
        box_list=[]

        for name,df_group_i in df_group:
            left = name[0].left
            vals = df_group_i['LAI_CV'].tolist()
            mean = np.nanmean(vals)
            # err=np.nanstd(vals)
            err,_,_=T.uncertainty_err(vals)
            box_list.append(vals)

            x_list.append(left)
            y_list.append(mean)
            err_list.append(err)
        plt.plot(x_list,y_list)
        #
        plt.fill_between(x_list, np.array(y_list) - np.array(err_list), np.array(y_list) + np.array(err_list), alpha=0.5)
        # plt.boxplot(box_list,positions=x_list,showfliers=False,widths=0.08)
        plt.xticks(x_list,bins_list_str,rotation=45)
        plt.ylabel('inter_LAI_CV')
        plt.xlabel('intra_precip_CV')
        plt.show()
        pass
    def intra_inter_CV_scatter(self):

        f_LAI_CV = result_root + rf'\extract_window\extract_detrend_original_window_CV\\LAI4g_CV_trend.tif'
        f_precip_CV = result_root + rf'intra_stats_annual\trend_analysis\\CV_precip_trend.tif'
        arr_LAI_CV = ToRaster().raster2array(f_LAI_CV)[0]
        arr_precip_CV = ToRaster().raster2array(f_precip_CV)[0]
        arr_LAI_CV = np.array(arr_LAI_CV)
        arr_precip_CV = np.array(arr_precip_CV)
        arr_LAI_CV[arr_LAI_CV < -2] = np.nan
        arr_precip_CV[arr_precip_CV < -2] = np.nan
        arr_precip_CV[arr_precip_CV>2]=np.nan
        arr_LAI_CV[arr_LAI_CV>2]=np.nan

        arr_LAI_CV = arr_LAI_CV.flatten()
        arr_precip_CV = arr_precip_CV.flatten()
        KDE_plot().plot_scatter(arr_precip_CV,arr_LAI_CV,cmap='Spectral',s=5)
        plt.xlim(-0.5,1.5)
        # plt.ylim(-2,2)

        plt.xlabel('intra_precip_CV')
        plt.ylabel('inter_LAI_CV')
        plt.show()

        pass




        pass



def main():

    # Intra_CV_preprocessing().run()
    # moving_window().run()
    # extract_rainfall().run()
    # PLOT().run()
    Check_data().run()

    pass

if __name__ == '__main__':
    main()