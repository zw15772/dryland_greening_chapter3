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

class extract_rainfall_annual_based_on_weekly():
    ## 1) extract rainfall CV
    ## 2) extract rainfall total
    ## 3) extract rainfall frequency
    ## extract dry frequency
    ## 4) extract rainfall intensity
    ## 5) extract rainfall wet spell
    ## 6) extract rainfall dry spell

    ## 1) extract rainfall CV
    ## 2) extract rainfall total
    ## 3) extract rainfall frequency
    ## extract dry frequency
    ## 4) extract rainfall intensity
    ## 5) extract rainfall wet spell
    ## 6) extract rainfall dry spell
    def run(self):
        # self.define_quantile_threshold()

        # self.extract_rainfall_CV()
        # self.extract_rainfall_std()
        # self.extract_rainfall_mean()
        # self.dry_spell()
        # # #
        # self.rainfall_extreme_wet_event()
        # self.rainfall_intensity()

        # self.peak_rainfall_timing()
        # self.extract_LAI4g_CV()
        self.trend_analysis()

        # self.check_spatial_map()
        pass

    def define_quantile_threshold(self):
        # 1) extract extreme wet event based on 90th percentile and calculate frequency and total duration
        # 2) extract extreme dry event based on 10th percentile and calculate frequency and total duration
        # 3) extract wet event intensity
        ## 4) extract dry event intensity
        ## extract VPD and calculate the frequency of VPD>2kpa
        fdir = rf'E:\Data\ERA5_precip\\ERA5_daily\dict\\precip_transform\\'
        outdir = rf'E:\Data\\ERA5_precip\\ERA5_daily\dict\\define_quantile_threshold\\'
        T.mk_dir(outdir, force=True)

        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            spatial_dic = np.load(fdir + f, allow_pickle=True).item()
            result_dic = {}
            for pix in tqdm(spatial_dic):

                vals = spatial_dic[pix]
                vals_flatten = [item for sublist in vals for item in sublist]
                vals_flatten = np.array(vals_flatten)

                if T.is_all_nan(vals_flatten):
                    continue
                # plt.bar(range(len(vals_flatten)),vals_flatten)
                # plt.show()

                val_90th = np.percentile(vals_flatten, 90)
                val_10th = np.percentile(vals_flatten, 10)
                val_95th = np.percentile(vals_flatten, 95)
                val_5th = np.percentile(vals_flatten, 5)
                val_99th = np.percentile(vals_flatten, 99)
                val_1st = np.percentile(vals_flatten, 1)
                dic_i = {
                    '90th': val_90th,
                    '10th': val_10th,
                    '95th': val_95th,
                    '5th': val_5th,
                    '99th': val_99th,
                    '1st': val_1st
                }
                result_dic[pix] = dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def extract_rainfall_CV(self):  ## extract CV of rainfall based on biweekly
        fdir = rf'D:\Project3\Data\biweekly\Precip\\'
        outdir_CV = rf'E:\Data\\ERA5_biweekly\\extract_rainfall_annual\\CV_rainfall\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            vals_reshape=np.reshape(vals, (-1, 24))
            CV_list = []
            for val in vals_reshape:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)

                CV = np.std(val) / np.mean(val) * 100
                # print(CV)
                CV_list.append(CV)
            result_dic[pix] = CV_list

        outf = outdir_CV + 'CV_rainfall.npy'

        np.save(outf, result_dic)

    def extract_rainfall_std(self):  ## extract std of rainfall ready for multiregression
        fdir = rf'D:\Project3\Data\biweekly\Precip\\'
        outdir_CV = rf'E:\Data\\ERA5_biweekly\\extract_rainfall_annual\\std_rainfall\\'
        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            vals_reshape = np.reshape(vals, (-1, 24))
            std_list = []
            for val in vals_reshape:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)

                std = np.std(val)

                std_list.append(std)
            result_dic[pix] = std_list

        outf = outdir_CV + 'std_rainfall.npy'

        np.save(outf, result_dic)

    def extract_rainfall_mean(self):  ## extract std of rainfall ready for multiregression
        fdir = rf'D:\Project3\Data\biweekly\Precip\\'
        outdir_CV = rf'E:\Data\\ERA5_biweekly\\extract_rainfall_annual\\mean_rainfall\\'
        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            vals_reshape = np.reshape(vals, (-1, 24))
            mean_list = []
            for val in vals_reshape:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)

                mean = np.mean(val)

                mean_list.append(mean)
            result_dic[pix] = mean_list

        outf = outdir_CV + 'mean_rainfall.npy'

        np.save(outf, result_dic)

    def rainfall_extreme_wet_event(self):

        fdir = rf'D:\Project3\Data\biweekly\Precip\\'
        outdir = rf'E:\Data\\ERA5_biweekly\\extract_rainfall_annual\\extreme_frequency_wet\\'
        T.mk_dir(outdir, force=True)
        threshold_f = rf'E:\Data\\\ERA5_daily\dict\\define_quantile_threshold\\'
        dic_threshold = T.load_npy_dir(threshold_f)
        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)

        result_dic_wet_frequency = {}
        result_dic_wet_extreme_frequency = {}

        for pix in tqdm(spatial_dic):

            if not pix in dic_threshold:
                continue
            vals = spatial_dic[pix]
            vals_reshape = np.reshape(vals, (-1, 24))

            threshold = dic_threshold[pix]
            threshold_wet = threshold['90th']
            threhold_wet_extreme = threshold['95th']

            frequency_wet_list = []
            frequency_wet_extreme_list = []

            for val in vals_reshape:

                if T.is_all_nan(val):
                    continue
                ## wet event>90th percentile and <95th percentile

                frequency_wet = len(np.where((val > threshold_wet) & (val < threhold_wet_extreme))[0])
                frequency_wet_list.append(frequency_wet)
                frequency_wet_extreme = len(np.where(val > threhold_wet_extreme)[0]) / len(val) * 100
                frequency_wet_extreme_list.append(frequency_wet_extreme)
            # print(frequency_wet_list)
            # print(frequency_wet_extreme_list)
            # exit()

            result_dic_wet_frequency[pix] = frequency_wet_list
            result_dic_wet_extreme_frequency[pix] = frequency_wet_extreme_list

        np.save(outdir + 'wet_frequency_90th.npy', result_dic_wet_frequency)
        np.save(outdir + 'wet_frequency_95th.npy', result_dic_wet_extreme_frequency)

    def rainfall_intensity(self):
        fdir = rf'D:\Project3\Data\biweekly\Precip\\'
        outdir = rf'E:\Data\\ERA5_biweekly\\extract_rainfall_annual\\rainfall_intensity\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)

        result_dic = {}
        for pix in tqdm(spatial_dic):
            intensity_list = []

            vals = spatial_dic[pix]
            vals_reshape = np.reshape(vals, (-1, 24))
            for val in vals_reshape:
                ## calculate the average intensity of rainfall events

                if T.is_all_nan(val):
                    continue
                intensity = np.nanmean(val)
                intensity_list.append(intensity)
            result_dic[pix] = intensity_list
        np.save(outdir + 'rainfall_intensity.npy', result_dic)

    def dry_spell(self):

        fdir = rf'D:\Project3\Data\biweekly\Precip\\'
        outdir = rf'E:\Data\\ERA5_biweekly\\extract_rainfall_annual\\dry_spell\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)

        average_dry_spell_annual_dic = {}
        maxmum_dry_spell_annual_dic = {}

        for pix in tqdm(spatial_dic):
            average_dry_spell_annual_list = []
            maxmum_dry_spell_annual_list = []

            vals = spatial_dic[pix]
            vals_reshape = np.reshape(vals, (-1, 24))
            for val in vals_reshape:
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

    def peak_rainfall_timing(self):  ## Weighted Mean of the Peak Rainfall Timing
        from scipy.ndimage import gaussian_filter1d
        fdir = rf'D:\Project3\Data\biweekly\Precip\\'
        outdir = rf'E:\Data\\ERA5_biweekly\\extract_rainfall_annual\\peak_rainfall_timing\\'
        T.mk_dir(outdir, force=True)

        T.mk_dir(outdir, force=True)
        spatial_dic = T.load_npy_dir(fdir)

        result_dic = {}
        for pix in tqdm(spatial_dic):
            r, c = pix

            vals = spatial_dic[pix]
            rainfall_peak_list = []
            for val in vals:

                if T.is_all_nan(val):
                    continue
                ## smooth rainfall
                smoothed_rainfall = SMOOTH().mid_window_smooth(val, 5)
                # plt.plot(time, smoothed_rainfall, label='Smoothed Rainfall')
                # plt.show()

                ## find peaks
                # max_index = T.pick_max_indx_from_1darray(smoothed_rainfall, 0, 365)
                max_indexs, max_values = T.pick_max_n_index(smoothed_rainfall, 1)
                print(max_indexs[0])

                rainfall_peak_list.append(max_indexs[0])

            result_dic[pix] = rainfall_peak_list

        np.save(outdir + 'peak_rainfall_timing.npy', result_dic)


    def extract_LAI4g_CV(self):  ##
        fdir = rf'D:\Project3\Data\biweekly\LAI4g\\'
        outdir_CV = rf'E:\Data\\ERA5_biweekly\\extract_rainfall_annual\\LAI4g_CV\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            vals_reshape=np.reshape(vals, (-1, 24))
            CV_list = []
            for val in vals_reshape:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)

                CV = np.std(val) / np.mean(val) * 100
                # print(CV)
                CV_list.append(CV)
            result_dic[pix] = CV_list

        outf = outdir_CV + 'LAI4g_CV.npy'

        np.save(outf, result_dic)

    def trend_analysis(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'E:\Data\ERA5_biweekly\extract_rainfall_annual\dry_spell\\'
        outdir = rf'E:\Data\ERA5_biweekly\extract_rainfall_annual\trend_analysis\\'
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

            # plt.imshow(arr_trend, cmap='jet', vmin=-0.01, vmax=0.01)
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
        spatial_dic = T.load_npy_dir(fdir)
        key_list = ['average_dry_spell', 'maximum_dry_spell']

        for key in key_list:
            spatial_dict_num = {}
            spatial_dict_mean = {}

            for pix in spatial_dic:

                annual_dict = spatial_dic[pix]
                if len(annual_dict) == 0:
                    continue

                valid_year = 0
                vals_list = []
                for year in annual_dict:
                    dict_i = annual_dict[year]
                    if not key in dict_i:
                        continue
                    val = dict_i[key]
                    vals_list.append(val)

                    valid_year += 1
                vals_mean = np.nanmean(vals_list)
                spatial_dict_num[pix] = valid_year
                spatial_dict_mean[pix] = vals_mean

            arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_mean)
            plt.figure()
            plt.imshow(arr, interpolation='nearest')
            plt.title(key)
            plt.colorbar()
        plt.show()

        #     spatial_dict_test[pix] = np.nanmean(vals['average_dry_spell'])
        # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_test)
        # plt.imshow(arr,interpolation='nearest')
        # # plt.title(key)
        # plt.show()

        pass







class extract_rainfall_annual_based_on_daily():
    ## 1) extract rainfall CV
    ## 2) extract rainfall total
    ## 3) extract rainfall frequency
    ## extract dry frequency
    ## 4) extract rainfall intensity
    ## 5) extract rainfall wet spell
    ## 6) extract rainfall dry spell
    def run(self):
        # self.define_quantile_threshold()

        # self.extract_rainfall_CV()
        # self.extract_rainfall_std()
        # self.extract_rainfall_mean()
        # self.extract_rainfall_sum()
        # self.dry_spell()
        # self.extract_rainfall_event_size()
        # self.extract_heavy_rainfall_days()
        # self.extract_rainfall_frequency()
        # self.extract_rainfall_seasonal_distribution()
        # #
        # self.rainfall_extreme_wet_event()
        # self.rainfall_intensity()
        # self.extract_seasonal_rainfall_event_size()
        # self.extract_seasonal_rainfall_intervals()


        # self.peak_rainfall_timing()
        # self.aggreate_AVHRR_LAI()
        # self.tif_to_dic()
        # self.extract_annual_LAI()
        self.relative_change()

        # self.detrend()
        # self.trend_analysis()

        # self.check_spatial_map()
        pass

    def define_quantile_threshold(self):
        # 1) extract extreme wet event based on 90th percentile and calculate frequency and total duration
        # 2) extract extreme dry event based on 10th percentile and calculate frequency and total duration
        # 3) extract wet event intensity
        ## 4) extract dry event intensity
        ## extract VPD and calculate the frequency of VPD>2kpa
        fdir = rf'E:\Data\ERA5_precip\\ERA5_daily\dict\\precip_transform\\'
        outdir = rf'E:\Data\\ERA5_precip\\ERA5_daily\dict\\define_quantile_threshold\\'
        T.mk_dir(outdir, force=True)

        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            spatial_dic = np.load(fdir + f, allow_pickle=True).item()
            result_dic = {}
            for pix in tqdm(spatial_dic):

                vals = spatial_dic[pix]
                vals_flatten = [item for sublist in vals for item in sublist]
                vals_flatten = np.array(vals_flatten)

                if T.is_all_nan(vals_flatten):
                    continue
                # plt.bar(range(len(vals_flatten)),vals_flatten)
                # plt.show()

                val_90th = np.percentile(vals_flatten, 90)
                val_10th = np.percentile(vals_flatten, 10)
                val_95th = np.percentile(vals_flatten, 95)
                val_5th = np.percentile(vals_flatten, 5)
                val_99th = np.percentile(vals_flatten, 99)
                val_1st = np.percentile(vals_flatten, 1)
                dic_i = {
                    '90th': val_90th,
                    '10th': val_10th,
                    '95th': val_95th,
                    '5th': val_5th,
                    '99th': val_99th,
                    '1st': val_1st
                }
                result_dic[pix] = dic_i
            outf = outdir + f
            np.save(outf, result_dic)



    def extract_rainfall_CV(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'E:\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\ERA5_daily\dict\\extract_rainfall_annual\\CV_rainfall\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            CV_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)

                CV = np.std(val) / np.mean(val) *100
                # print(CV)
                CV_list.append(CV)
            result_dic[pix] = CV_list

        outf = outdir_CV + 'CV_rainfall.npy'

        np.save(outf, result_dic)

    def extract_rainfall_event_size(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'E:\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\ERA5_daily\dict\\extract_rainfall_annual\\rainfall_event_size\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            CV_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue
                ### This is calculated as the total amount of precipitation during
                # the period of interest divided by the number of rainy days.

                val = np.array(val)

                ## number of rainy days is when precip>1

                val_rainy = val[val>1]
                total_precip = np.nansum(val_rainy)

                stats = total_precip / len(val_rainy)

                # print(CV)
                CV_list.append(stats)
            result_dic[pix] = CV_list

        outf = outdir_CV + 'rainfall_event_size.npy'

        np.save(outf, result_dic)

    def extract_rainfall_frequency(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'E:\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\ERA5_daily\dict\\extract_rainfall_annual\\rainfall_frequency\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            CV_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue
                ### This is the total number of precipitation events during the period of interest.

                val = np.array(val)

                ## number of rainy days is when precip>1

                val_rainy = val[val > 1]

                stats = len(val_rainy)

                # print(CV)
                CV_list.append(stats)
            result_dic[pix] = CV_list

        outf = outdir_CV + 'rainfall_frequency.npy'

        np.save(outf, result_dic)


    def extract_seasonal_rainfall_event_size(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'E:\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\ERA5_daily\dict\\extract_rainfall_annual\\seasonal_rainfall_event_size\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            CV_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue
                ### This is the total number of precipitation events during the period of interest.

                val = np.array(val)

                ## number of rainy days is when precip>1

                val_rainy = val[val > 1]
                if len(val_rainy) == 0:
                    continue
                SI=np.sum(val_rainy)
                SImax=np.max(val_rainy)

                n=len(val_rainy)
                temp=0
                for i in val_rainy:
                    pi=i/SI
                    q=1/n
                    temp+=pi*np.log2(pi/q)
                Dsize=temp*SI/SImax

                CV_list.append(Dsize)

            result_dic[pix] = CV_list

        outf = outdir_CV + 'seasonal_rainfall_event_size.npy'

        np.save(outf, result_dic)

    def extract_seasonal_rainfall_intervals(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'E:\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\ERA5_daily\dict\\extract_rainfall_annual\\seasonal_rainfall_intervals\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}
        # for f in os.listdir(fdir):
        #     if not '050.npy' in f:
        #         continue
        #     spatial_dic=np.load(fdir+f, allow_pickle=True, encoding='latin1').item()

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            CV_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue
                ### This is the total number of precipitation events during the period of interest.

                val = np.array(val)


                val[val >= 1] = np.nan

                dry_index = np.where(~np.isnan(val))
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


                tI=np.sum(dry_spell)
                tmax=np.max(dry_spell)

                ## n is number of dry spells

                n=len(dry_spell)
                temp=0
                for i in dry_spell:
                    lnti=i
                    pi=lnti/tI
                    q=1/n
                    temp+=pi*np.log2(pi/q)

                Dinterval=temp*tI/tmax


                CV_list.append(Dinterval)

            result_dic[pix] = CV_list

        outf = outdir_CV + 'seasonal_rainfall_intervals.npy'

        np.save(outf, result_dic)

    def extract_heavy_rainfall_days(self):  ##
        fdir = rf'E:\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\ERA5_daily\dict\\extract_rainfall_annual\\heavy_rainfall_days\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            CV_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue
                ### This counts the number of days where the precipitation exceeds 10 mm.

                val = np.array(val)

                ## number of rainy days is when precip>1

                val_heavy_rainfall = val[val > 10]

                stats = len(val_heavy_rainfall)

                # print(CV)
                CV_list.append(stats)
            result_dic[pix] = CV_list

        outf = outdir_CV + 'heavy_rainfall_days.npy'

        np.save(outf, result_dic)

    def extract_rainfall_std(self):  ## extract std of rainfall ready for multiregression
        fdir = rf'E:\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\ERA5_daily\dict\\extract_rainfall_annual\\std_rainfall\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            std_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)

                std=np.std(val)

                std_list.append(std)
            result_dic[pix] = std_list

        outf = outdir_CV + 'std_rainfall.npy'

        np.save(outf, result_dic)

    def extract_rainfall_mean(self):  ## extract std of rainfall ready for multiregression
        fdir = rf'E:\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\ERA5_daily\dict\\extract_rainfall_annual\\mean_rainfall\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            mean_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)

                mean=np.mean(val)

                mean_list.append(mean)
            result_dic[pix] = mean_list

        outf = outdir_CV + 'mean_rainfall.npy'

        np.save(outf, result_dic)

    def extract_rainfall_sum(self):  ## extract std of rainfall ready for multiregression
        fdir = rf'E:\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\ERA5_daily\dict\\extract_rainfall_annual\\sum_rainfall\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            mean_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue


                val = np.array(val)
                val_rainy=val[val>1]

                sum=np.sum(val_rainy)

                mean_list.append(sum)
            result_dic[pix] = mean_list

        outf = outdir_CV + 'sum_rainfall.npy'

        np.save(outf, result_dic)



    def rainfall_extreme_wet_event(self):

        fdir = rf'E:\Data\\\ERA5_daily\dict\\precip_transform\\'
        outdir = rf'E:\Data\\\ERA5_daily\dict\\extract_rainfall_annual\\rainfall_extreme_wet_event\\'
        threshold_f = rf'E:\Data\\\ERA5_daily\dict\\define_quantile_threshold\\'
        dic_threshold = T.load_npy_dir(threshold_f)
        T.mk_dir(outdir, force=True)

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
                frequency_wet_extreme = len(np.where(val > threhold_wet_extreme)[0])/len(val) * 100
                frequency_wet_extreme_list.append(frequency_wet_extreme)
            # print(frequency_wet_list)
            # print(frequency_wet_extreme_list)
            # exit()

            result_dic_wet_frequency[pix] = frequency_wet_list
            result_dic_wet_extreme_frequency[pix] = frequency_wet_extreme_list

        np.save(outdir + 'wet_frequency_90th.npy', result_dic_wet_frequency)
        np.save(outdir + 'wet_frequency_95th.npy', result_dic_wet_extreme_frequency)


    def rainfall_intensity(self):
        fdir = rf'E:\Data\\\ERA5_daily\dict\\precip_transform\\'
        outdir = rf'E:\Data\\\ERA5_daily\dict\\extract_rainfall_annual\\rainfall_intensity\\'
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
                val_rainy = val[val > 1]
                intensity = np.nanmean(val_rainy)
                intensity_list.append(intensity)
            result_dic[pix] = intensity_list
        np.save(outdir + 'rainfall_intensity.npy', result_dic)

    def dry_spell(self):

        fdir = rf'E:\Data\\ERA5_daily\dict\\precip_transform\\'
        outdir = rf'E:\Data\ERA5_daily\dict\\extract_rainfall_annual\\dry_spell\\'
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

    def peak_rainfall_timing(self):  ## Weighted Mean of the Peak Rainfall Timing
        from scipy.ndimage import gaussian_filter1d
        time = np.arange(0, 365)
        fdir = rf'E:\Data\ERA5_daily\dict\\precip_transform\\'
        outdir = rf'E:\Data\ERA5_daily\dict\\extract_rainfall_annual\\peak_rainfall_timing\\'
        T.mk_dir(outdir, force=True)
        spatial_dic = T.load_npy_dir(fdir)

        result_dic = {}
        for pix in tqdm(spatial_dic):
            r, c = pix


            vals = spatial_dic[pix]
            rainfall_peak_list = []
            for val in vals:


                if T.is_all_nan(val):
                    continue
                ## smooth rainfall
                smoothed_rainfall = SMOOTH().mid_window_smooth(val, 5)
                # plt.plot(time, smoothed_rainfall, label='Smoothed Rainfall')
                # plt.show()

                ## find peaks
                # max_index = T.pick_max_indx_from_1darray(smoothed_rainfall, 0, 365)
                max_indexs, max_values = T.pick_max_n_index(smoothed_rainfall,1)
                print(max_indexs[0])

                rainfall_peak_list.append(max_indexs[0])

            result_dic[pix] = rainfall_peak_list

        np.save(outdir + 'peak_rainfall_timing.npy', result_dic)

    def aggreate_AVHRR_LAI(self):  # aggregate biweekly data to monthly
        fdir_all = rf'D:\Project3\Data\\LAI4g\\\scales_LAI4g_weekly\\'
        outdir = rf'D:\Project3\Data\\LAI4g\\scales_LAI4g_weekly_monthly\\'
        Tools().mk_dir(outdir, force=True)

        year_list = list(range(1982, 2021))
        month_list = list(range(1, 13))

        for year in tqdm(year_list):
            for month in tqdm(month_list):
                month=rf'{month:02d}'

                data_list = []
                for f in tqdm(os.listdir(fdir_all)):

                    if not f.endswith('.tif'):
                        continue

                    data_year = f.split('.')[0][0:4]
                    data_month = f.split('.')[0][4:6]

                    if not int(data_year) == year:
                        continue
                    if not int(data_month) == int(month):
                        continue
                    arr = ToRaster().raster2array(fdir_all + f)[0]
                    # arr=arr/1000 ###
                    arr_unify = arr[:720][:720,
                                :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
                    arr_unify = np.array(arr_unify)
                    arr_unify[arr_unify == 65535] = np.nan
                    arr_unify[arr_unify < 0] = np.nan
                    arr_unify[arr_unify > 7] = np.nan
                    # 当变量是LAI 的时候，<0!!
                    data_list.append(arr_unify)
                data_list = np.array(data_list)
                print(data_list.shape)
                # print(len(data_list))
                # exit()

                ##define arr_average and calculate arr_average

                arr_average = np.nanmax(data_list, axis=0)
                arr_average = np.array(arr_average)
                arr_average[arr_average < 0] = np.nan
                arr_average[arr_average > 7] = np.nan
                if np.isnan(np.nanmean(arr_average)):
                    continue
                if np.nanmean(arr_average) < 0.:
                    continue
                # plt.imshow(arr_average)
                # plt.title(f'{year}{month}')
                # plt.show()

                # save

                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_average, outdir + '{year}{month}.tif'.format(year=year, month=month))


    def tif_to_dic(self):  ## monthly data

        fdir_all = rf'D:\Project3\Data\\LAI4g\\'

        NDVI_mask_f = rf'D:\Project3\Data/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan

        year_list = list(range(1982, 2021))


        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            if not 'scales_LAI4g_monthly' in fdir:
                continue

            outdir =rf'D:\Project3\Data\monthly_data\\LAI4g\\'
            # if os.path.isdir(outdir):
            #     pass

            T.mk_dir(outdir, force=True)
            all_array = []  #### so important  it should be go with T.mk_dic

            for f in os.listdir(fdir_all+ fdir):
                if not f.endswith('.tif'):
                    continue
                if int(f.split('.')[0][0:4]) not in year_list:
                    continue


                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir_all + fdir + '\\' + f)
                array = np.array(array, dtype=float)


                array_unify = array[:720][:720,
                              :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

                array_unify[array_unify < -999] = np.nan
                # array_unify[array_unify > 7] = np.nan
                # array[array ==0] = np.nan

                array_unify[array_unify < 0] = np.nan



                # plt.imshow(array)
                # plt.show()
                array_mask = np.array(array_mask, dtype=float)
                # plt.imshow(array_mask)
                # plt.show()
                array_dryland = array_unify * array_mask
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
                    np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                    temp_dic = {}
            np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)




    def extract_annual_LAI(self):  ## extract annaul LAI

        fdir = rf'D:\Project3\Data\monthly_data\\LAI4g\\'
        outdir = rf'E:\Data\ERA5_daily\dict\\extract_rainfall_annual\\annual_LAI\\'
        outf=outdir + 'annual_LAI.npy'
        Tools().mk_dir(outdir, force=True)
        annual_spatial_dict = {}
        dict=T.load_npy_dir(fdir)
        for pix in tqdm(dict):
            time_series = dict[pix]
            time_series=np.array(time_series)
            time_series[time_series==65535]=np.nan
            time_series[time_series<0]=np.nan
            time_series [time_series>7000]=np.nan

            if T.is_all_nan(time_series):
                continue

            annual_time_series_reshape = np.reshape(time_series, (-1, 12))

            annual_time_series = np.nanmean(annual_time_series_reshape, axis=1)


            annual_spatial_dict[pix] = annual_time_series


        np.save(outf, annual_spatial_dict)

        pass


    def relative_change(self, ):  ## calculate annual relative change of LAI
        fdir=rf'E:\Data\ERA5_daily\dict\extract_rainfall_annual\annual_LAI4g\\'

        outdir = rf'E:\Data\ERA5_daily\dict\\extract_rainfall_annual\\annual_LAI4g\\'
        Tools().mk_dir(outdir, force=True)
        annual_spatial_dict = {}
        dict = T.load_npy_dir(fdir)
        for pix in tqdm(dict):
            time_series = dict[pix]
            time_series[time_series == 65535] = np.nan
            if T.is_all_nan(time_series):
                continue

            # plt.plot(time_series)

            plt.plot(time_series)
            average=np.nanmean(time_series)
            relative_change = (time_series - average) / average * 100

            annual_spatial_dict[pix] = relative_change


            # print((detrended_annual_time_series))
            # plt.plot(relative_change, color='r')
            # plt.show()

            annual_spatial_dict[pix] = relative_change
        np.save(outdir + 'relative_change_annual_LAI4g.npy', annual_spatial_dict)

        pass

        pass

    def detrend(self): ## detrend LAI4g

        fdir = rf'E:\Data\ERA5_daily\dict\\extract_rainfall_annual\\annual_LAI4g\\'
        outdir = rf'E:\Data\ERA5_daily\dict\\extract_rainfall_annual\\annual_LAI4g\\'
        Tools().mk_dir(outdir, force=True)
        annual_spatial_dict = {}
        dict = T.load_npy_dir(fdir)
        for pix in tqdm(dict):
            time_series = dict[pix]
            time_series[time_series==65535]=np.nan
            if T.is_all_nan(time_series):
                continue

            plt.plot(time_series)


            detrended_annual_time_series = signal.detrend(time_series)+np.mean(time_series)
            # print((detrended_annual_time_series))
            # plt.plot(detrended_annual_time_series)
            # plt.show()

            annual_spatial_dict[pix] = detrended_annual_time_series


        np.save(outdir + 'detrended_annual_LAI.npy', annual_spatial_dict)

        pass


    def trend_analysis(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'E:\Data\ERA5_daily\dict\extract_rainfall_annual\annual_LAI\\'
        outdir = rf'E:\Data\ERA5_daily\dict\extract_rainfall_annual\\trend_analysis_moving_window\\'
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

            plt.imshow(arr_trend, cmap='jet', vmin=-0.01, vmax=0.01)

            plt.colorbar()
            plt.title(f)
            plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

    pass

    def check_spatial_map(self):
        fdir = rf'D:\Project3\Data\monthly_data\LAI4g\\'
        spatial_dic = T.load_npy_dir(fdir)
        key_list = ['average_dry_spell', 'maximum_dry_spell']

        for key in key_list:
            spatial_dict_num = {}
            spatial_dict_mean = {}

            for pix in spatial_dic:

                annual_dict = spatial_dic[pix]
                if len(annual_dict) == 0:
                    continue

                valid_year = 0
                vals_list = []
                for year in annual_dict:
                    dict_i = annual_dict[year]
                    if not key in dict_i:
                        continue
                    val = dict_i[key]
                    vals_list.append(val)

                    valid_year += 1
                vals_mean = np.nanmean(vals_list)
                spatial_dict_num[pix] = valid_year
                spatial_dict_mean[pix] = vals_mean

            arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_mean)
            plt.figure()
            plt.imshow(arr, interpolation='nearest')
            plt.title(key)
            plt.colorbar()
        plt.show()

        #     spatial_dict_test[pix] = np.nanmean(vals['average_dry_spell'])
        # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_test)
        # plt.imshow(arr,interpolation='nearest')
        # # plt.title(key)
        # plt.show()

        pass

class moving_window():
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass
    def run(self):
        # self.moving_window_extraction()
        # self.moving_window_extraction_for_LAI()

        # self.moving_window_CV_extraction_anaysis()

        # self.moving_window_average_anaysis()
        # self.moving_window_trend_anaysis()
        self.trend_analysis()
        # self.robinson()

        pass
    def moving_window_extraction(self):

        fdir_all =  rf'E:\Data\ERA5_daily\dict\extract_rainfall_annual\\\\'
        outdir = rf'E:\Data\ERA5_daily\\dict\\extract_window\\'
        T.mk_dir(outdir, force=True)
        for fdir in os.listdir(fdir_all):
            if fdir not in ['seasonal_rainfall_intervals', 'seasonal_rainfall_event_size',
                            'rainfall_frequency','heavy_rainfall_days','rainfall_event_size','sum_rainfall','annual_LAI4g']:
                continue


            for f in os.listdir(fdir_all + fdir):
                if not 'relative_change_annual_LAI4g' in f:
                    continue


                outf = outdir + f.split('.')[0] + '.npy'
                print(outf)
                if os.path.isfile(outf):
                    continue

                dic = T.load_npy(fdir_all + fdir + '\\' + f)
                window = 15

                new_x_extraction_by_window = {}
                for pix in tqdm(dic):

                    time_series = dic[pix]
                    time_series = np.array(time_series)


                    # time_series[time_series < -999] = np.nan
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


    def moving_window_CV_extraction_anaysis(self):
        window_size=15
        fdir = rf'E:\Data\ERA5_daily\dict\\extract_window\\'
        outdir =  rf'E:\Data\ERA5_daily\dict\\moving_window_average_anaysis\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'detrend' in f:
                continue

            dic = T.load_npy(fdir + f)
            slides = 39-window_size
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
                    # print(len(time_series))

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


    def moving_window_average_anaysis(self): ## each window calculating the average
        window_size = 15

        fdir=rf'E:\Data\ERA5_daily\dict\\extract_window\\'
        outdir = rf'E:\Data\ERA5_daily\dict\\moving_window_average_anaysis\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not f.split('.')[0] in ['seasonal_rainfall_intervals', 'seasonal_rainfall_event_size',
        'rainfall_frequency','heavy_rainfall_days','rainfall_event_size','sum_rainfall']:
                continue

            dic = T.load_npy(fdir + f)

            slides = 38 - window_size   ## revise!!
            outf = outdir + f.split('.')[0] + f'_average.npy'
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
                    if len(time_series_all)<23:
                        continue


                    time_series = time_series_all[ss]
                    # print(time_series)
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    # print(len(time_series))
                    ##average
                    average=np.nanmean(time_series)
                    # print(average)

                    trend_list.append(average)

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)

    def moving_window_std_extraction_anaysis(self):
        window_size=15
        fdir = rf'E:\Data\ERA5_daily\dict\\extract_window\\'
        outdir =  rf'E:\Data\ERA5_daily\dict\\moving_window_average_anaysis\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'detrend' in f:
                continue

            dic = T.load_npy(fdir + f)
            slides = 39-window_size
            outf = outdir + f.split('.')[0] + f'_std.npy'
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
                    # print(len(time_series))

                    if np.nanmean(time_series)==0:
                        continue
                    cv=np.nanstd(time_series)
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
        window_size = 15

        fdir=rf'E:\Data\ERA5_daily\dict\\extract_window\\'
        outdir = rf'E:\Data\ERA5_daily\dict\\moving_window_average_anaysis\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'relative_change_annual_LAI4g' in f:
                continue


            dic = T.load_npy(fdir + f)

            slides = 39 - window_size
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
                    if len(time_series_all)<24:
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
    def trend_analysis(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'E:\Data\ERA5_daily\dict\moving_window_average_anaysis\\'
        outdir = rf'E:\Data\ERA5_daily\dict\trend_analysis_moving_window\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.split('.')[0] in ['seasonal_rainfall_intervals', 'seasonal_rainfall_event_size',
                                       'rainfall_frequency', 'heavy_rainfall_days', 'rainfall_event_size',
                                       'sum_rainfall']:
                continue

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

    pass

    def robinson(self):
        fdir=rf'E:\Data\ERA5_daily\dict\extract_rainfall_annual\trend_analysis_moving_window\\'
        temp_root=result_root+r'Result_new\trend_anaysis\\robinson\\'
        out_pdf_fdir=result_root+r'Result_new\trend_anaysis\\robinson\\pdf\\'

        T.mk_dir(out_pdf_fdir,force=True)


        variable='peak_rainfall_timing'
        f_trend=fdir+variable+'_trend.tif'

        f_p_value=fdir+variable+'_p_value.tif'


        m,ret=Plot().plot_Robinson(f_trend, vmin=-2,vmax=2,is_discrete=True,colormap_n=7,)
        self.plot_Robinson_significance_scatter(m, f_p_value,temp_root,0.05,s=5)


        # plt.title(f'{variable}_(%/yr)')
        # plt.title(f'{variable}_(mm/day/yr)')
        plt.title(f'{variable}_(day/yr)')
        # plt.title('r')
        # plt.show()
        ## save fig pdf
        #save pdf
        plt.savefig(out_pdf_fdir+variable+'.pdf', dpi=300, bbox_inches='tight')
        plt.close()

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

    # extract_rainfall_annual_based_on_weekly().run()
    # extract_rainfall_annual_based_on_daily().run()

    moving_window().run()
    # PLOT().run()
    # Check_data().run()

    pass

if __name__ == '__main__':
    main()