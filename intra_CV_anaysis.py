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

class extract_heatevent():
    def run (self):
        # self.extract_climatology()
        # self.extract_extreme_heat_frequency()
        # self.extract_interannual_temperature()
        self.detrend()



    def extract_climatology(self):
        fdir=rf'C:\Users\wenzhang1\Desktop\max_temp\\transform\\'
        outdir=rf'E:\Data\\ERA5_daily\\extract_heatevent\\extract_climatology\\'

        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            spatial_dic = T.load_npy(fdir+f)
            anomaly_dic = {}
            for pix in tqdm(spatial_dic):
                r, c = pix
                vals = spatial_dic[pix]
                vals = np.array(vals)
                if np.isnan(np.nanmean(vals)):
                    continue
                vals_flatten = vals.flatten()
                plt.plot(vals_flatten, 'k')
                self.daily_climatology_anomaly(vals_flatten)
                anomaly_dic[pix] = self.daily_climatology_anomaly(vals_flatten)

            np.save(outdir+f, anomaly_dic)
    def daily_climatology_anomaly(self, vals):
        '''
        juping
        :param vals: 40 * 365
        :return:
        '''
        pix_anomaly = []
        climatology_means = []
        for day in range(1, 366):
            one_day = []
            for i in range(len(vals)):
                d = i % 365 + 1
                if day == d:
                    one_day.append(vals[i])
            mean = np.nanmean(one_day)
            std = np.nanstd(one_day)
            climatology_means.append(mean)
        for i in range(len(vals)):
            d_ind = i % 365
            mean_ = climatology_means[d_ind]
            anomaly = vals[i] - mean_
            pix_anomaly.append(anomaly)
        pix_anomaly = np.array(pix_anomaly)
        return pix_anomaly
    def extract_extreme_heat_frequency(self):

        # fdir = rf'E:\Data\\ERA5\\max_temp\\climatology_anomaly\\'
        fdir=rf'C:\Users\wenzhang1\Desktop\max_temp\deseasonal\\'
        outdir = rf'E:\Data\\ERA5_daily\\extract_heatevent\\\\'

        T.mk_dir(outdir, force=True)
        average_heat_spell_annual_dic = {}
        maxmum_heat_spell_annual_dic = {}
        heat_event_count_annual_dic = {}

        spatial_dic = T.load_npy_dir(fdir)
        for pix in tqdm(spatial_dic):
            r, c = pix

            vals = spatial_dic[pix]
            vals=np.array(vals)
            ##resha 38 year
            average_heat_spell_annual_list = []
            maxmum_heat_spell_annual_list = []
            heat_event_count_annual_list = []

            vals_reshape = vals.reshape(38, 365)
            for val in vals_reshape:
                if T.is_all_nan(val):
                    continue
                vals_heat = val.copy()
                # print(vals_heat);exit()

                vals_heat[vals_heat <= 5] = np.nan

                heat_index = np.where(~np.isnan(vals_heat))
                heat_index = heat_index[0]
                heat_index = np.array(heat_index)

                heat_index_groups = T.group_consecutive_vals(heat_index)
                # print(heat_index_groups)

                # plt.bar(range(len(val)), val)
                # plt.bar(range(len(val)), vals_heat, alpha=0.5)
                # # print(dry_index_groups)
                # plt.show()
                ## calcuate average wet spell
                heat_spell = []
                for group in heat_index_groups:
                    if len(group) < 5:
                        continue
                    heat_days=np.array(group)

                    heat_spell.append(len(heat_days))
                    # print(heat_spell)
                heat_spell = np.array(heat_spell)
                if len(heat_spell) == 0:

                    heat_event_count_annual_list.append(0)
                    average_heat_spell_annual_list.append(0)
                    maxmum_heat_spell_annual_list.append(0)

                    continue

                frequency = len(heat_spell)
                heat_event_count_annual_list.append(frequency)

                average_heat = np.nanmean(heat_spell)
                average_heat_spell_annual_list.append(average_heat)

                maxmum_wet_spell = np.nanmax(heat_spell)
                maxmum_heat_spell_annual_list.append(maxmum_wet_spell)

            average_heat_spell_annual_dic[pix] = average_heat_spell_annual_list
            maxmum_heat_spell_annual_dic[pix] = maxmum_heat_spell_annual_list
            heat_event_count_annual_dic[pix] = heat_event_count_annual_list

        np.save(outdir + 'average_heat_spell.npy', average_heat_spell_annual_dic)
        np.save(outdir + 'maxmum_heat_spell.npy', maxmum_heat_spell_annual_dic)
        np.save(outdir + 'heat_event_frequency.npy', heat_event_count_annual_dic)

    ###
    def extract_tmax_interannual(self):  ## extract std of rainfall ready for multiregression
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
                val_rainy = val[val > 1]

                sum = np.sum(val_rainy)

                mean_list.append(sum)
            result_dic[pix] = mean_list

        outf = outdir_CV + 'sum_rainfall.npy'

        np.save(outf, result_dic)

    def extract_interannual_temperature(self):

        ## here use annual temperature to extract
        fdir=rf'D:\Project3\Data\monthly_data\\tmax\\'
        outdir = rf'E:Project3\Data\ERA5_daily\dict\extract_heatevent_annual\heat_event_extraction\\'

        T.mk_dir(outdir, force=True)
        average_heat_event_temp_annual_dic = {}


        spatial_dic = T.load_npy_dir(fdir)
        for pix in tqdm(spatial_dic):
            r, c = pix

            vals = spatial_dic[pix]
            vals=np.array(vals)
            ##resha 38 year
            average_temp_annual_list = []


            vals_reshape = vals.reshape(39, 12)
            for val in vals_reshape:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)
                average_val=np.nanmean(val)
                average_temp_annual_list.append(average_val)

            average_heat_event_temp_annual_dic[pix] = average_temp_annual_list
        np.save(outdir + 'average_annual_tmax.npy', average_heat_event_temp_annual_dic)

    ###
    def detrend(self): ## detrend LAI4g

        f = rf'E:\Project3\Data\ERA5_daily\dict\extract_heatevent_annual\heat_event_extraction\\average_annual_tmax.npy'
        outdir = rf'E:\Project3\Data\ERA5_daily\dict\extract_heatevent_annual\heat_event_extraction\\'
        Tools().mk_dir(outdir, force=True)
        annual_spatial_dict = {}
        dict = T.load_npy(f)
        for pix in tqdm(dict):
            time_series = dict[pix]
            # time_series[time_series==65535]=np.nan
            if T.is_all_nan(time_series):
                continue
            if np.sum(time_series) == 0:
                continue

            plt.plot(time_series)
            # plt.show()


            detrended_annual_time_series = signal.detrend(time_series)+np.mean(time_series)
            # print((detrended_annual_time_series))
            # plt.plot(detrended_annual_time_series)
            # plt.show()

            annual_spatial_dict[pix] = detrended_annual_time_series


        np.save(outdir + 'detrended_average_annual_tmax.npy', annual_spatial_dict)

        pass
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
        # self.extract_rainfall_intensity()
        # self.extract_heavy_rainfall_days()
        # self.extract_rainfall_frequency()
        # self.extract_rainfall_seasonal_distribution()
        # #
        # self.rainfall_extreme_wet_event()

        # self.extract_rainfall_seasonality()
        self.extract_rainfall_seasonality_all_year()
        # self.extract_seasonal_rainfall_intervals()
        # self.extract_seasonal_rainfall_seasonality()


        # self.peak_rainfall_timing()
        # self.aggreate_AVHRR_LAI()
        # self.tif_to_dic()
        # self.extract_annual_LAI()
        # self.relative_change()

        # self.detrend()
        # self.trend_analysis()
        # self.mask_spatial_map()

        # self.check_spatial_map()
        # self.check_spatial_map2()
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

    def extract_rainfall_intensity(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'E:\Project3\Data\CRU-JRA\precip_transform\\'
        outdir = rf'E:\Project3\Data\CRU-JRA\\extract_rainfall_annual\\rainfall_intensity\\'
        T.mk_dir(outdir, force=True)
        # f='E:\Project3\Data\CRU-JRA\Precip_transform\\per_pix_dic_028.npy'
        # spatial_dic = T.load_npy(f)

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
            # print(CV_list)
            # print(len(CV_list))
            result_dic[pix] = CV_list

        outf = outdir + 'rainfall_intensity.npy'

        np.save(outf, result_dic)

    def extract_rainfall_frequency(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'E:\Project3\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Project3\Data\ERA5_daily\dict\\extract_rainfall_annual\\rainfall_frequency\\'

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


    def extract_rainfall_seasonality(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'E:Project3\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:Project3\Data\\ERA5_daily\dict\\extract_rainfall_annual\\rainfall_seasonality\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]

            SImax = np.max(vals)
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


                n=len(val_rainy)
                temp=0
                for i in val_rainy:
                    pi=i/SI
                    q=1/n
                    temp+=pi*np.log2(pi/q)
                Dsize=temp*SI/SImax

                CV_list.append(Dsize)

            result_dic[pix] = CV_list

        outf = outdir_CV + 'rainfall_seasonality.npy'

        np.save(outf, result_dic)

    def extract_rainfall_seasonality_all_year(self):  ## extract CV of rainfall ready for multiregression
        from scipy.stats import entropy
        fdir = rf'E:\Project3\Data\CRU-JRA\precip_transform\\'
        outdir_CV = rf'E:\Project3\Data\CRU-JRA\\extract_rainfall_annual\\rainfall_seasonality_all_year\\'

        T.mk_dir(outdir_CV, force=True)


        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]

            # print(len(vals))
            # exit()

            SImax = np.max(vals)
            CV_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue
                vals = np.array(val)
                ## set vals <1 to 0

                val[vals < 1] = 0



                ### This is the total number of precipitation events during the period of interest.

                ## count the number of 0s

                ## number of rainy days is when precip>1
                if len(val) == 0:
                    continue
                SI=np.sum(val)


                epsilon = 1e-10
                uniform_distribution = np.full(365, 1 / 365) + epsilon
                observed=val/SI

                uniform_distribution = uniform_distribution / uniform_distribution.sum()
                # exit()
                RE=entropy(observed, uniform_distribution, base=2)*SI/SImax

                Dsize=RE
                # print(Dsize)

                CV_list.append(Dsize)


            # print(CV_list)

            result_dic[pix] = CV_list

        outf = outdir_CV + 'rainfall_seasonality_all_year_new.npy'

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
        fdir = rf'E:\Project3\Data\CRU-JRA\Precip_transform\\'
        outdir_CV = rf'E:\Project3\Data\CRU-JRA\\extract_rainfall_annual\\sum_rainfall\\'

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
                # plt.plot(val)
                # plt.show()
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




    def dry_spell(self):

        fdir = rf'E:\Project3\Data\CRU-JRA\precip_transform\\'
        outdir = rf'E:\Project3\Data\CRU-JRA\\extract_rainfall_annual\\dry_spell\\'
        T.mk_dir(outdir, force=True)
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan
        dic_dryland=DIC_and_TIF().spatial_arr_to_dic(array_mask)


        spatial_dic = T.load_npy_dir(fdir)

        average_dry_spell_annual_dic = {}
        maxmum_dry_spell_annual_dic = {}

        for pix in tqdm(spatial_dic):
            val_dryland = dic_dryland[pix]
            if np.isnan(val_dryland):
                continue

            average_dry_spell_annual_list = []
            maxmum_dry_spell_annual_list = []

            vals = spatial_dic[pix]
            vals_flatten = np.array(vals).flatten()
            if T.is_all_nan(vals_flatten):
                continue
            print(len(vals_flatten))

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
            # print(average_dry_spell_annual_list)
            # print(maxmum_dry_spell_annual_list)

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
        outdir = rf'D:\Project3\Data\\LAI4g\\scales_LAI4g_monthly\\'
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

        fdir = rf'E:\Project3\Data\ERA5_daily\dict\extract_rainfall_annual\sum_rainfall\\'
        outdir = rf'E:\Project3\Data\ERA5_daily\dict\extract_rainfall_annual\sum_rainfall\\'
        Tools().mk_dir(outdir, force=True)
        annual_spatial_dict = {}
        dict = T.load_npy_dir(fdir)
        for pix in tqdm(dict):
            time_series = dict[pix]
            # time_series[time_series==65535]=np.nan
            if T.is_all_nan(time_series):
                continue
            if np.sum(time_series) == 0:
                continue

            plt.plot(time_series)


            detrended_annual_time_series = signal.detrend(time_series)+np.mean(time_series)
            # print((detrended_annual_time_series))
            # plt.plot(detrended_annual_time_series)
            # plt.show()

            annual_spatial_dict[pix] = detrended_annual_time_series


        np.save(outdir + 'detrended_sum_rainfall.npy', annual_spatial_dict)

        pass


    def trend_analysis(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'E:\Project3\Data\ERA5_daily\dict\extract_rainfall_annual\\rainfall_frequency\\'
        outdir = rf'E:\Project3\Data\ERA5_daily\dict\extract_rainfall_annual\rainfall_frequency\\\trend\\'
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
        fdir = rf'E:\Project3\Data\ERA5_daily\dict\extract_rainfall_annual\seasonal_rainfall_seasonality\\'
        spatial_dic = T.load_npy_dir(fdir)
        key_list = ['seasonal_rainfall_seasonality']

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
    def mask_spatial_map(self):

        fdir = rf'E:\Project3\Data\CRU-JRA\precip_transform\\'
        outdir = rf'E:\Project3\Data\CRU-JRA\\precip_transform_dryland\\'
        T.mk_dir(outdir, force=True)
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan
        dic_dryland = DIC_and_TIF().spatial_arr_to_dic(array_mask)


        result_dic = {}
        for f in T.listdir(fdir):
            dic_data = T.load_npy(fdir + f)


            for pix in tqdm(dic_data):
                val_dryland = dic_dryland[pix]
                data=dic_data[pix]
                if np.isnan(val_dryland):
                    result_dic[pix] = np.nan

                else:
                    result_dic[pix] = data

            T.save_npy(result_dic, outdir+f)

    def check_spatial_map2(self): ## plot spatial map
        fdir=rf'E:\Project3\Data\CRU-JRA\precip_transform_dryland\\'

        spatial_dic = T.load_npy_dir(fdir)

        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr,interpolation='nearest')
        # plt.title(key)
        plt.show()


        pass




    pass

    def check_spatial_map2(self):
        fdir=rf'E:\Project3\Data\ERA5_daily\dict\extract_rainfall_annual\seasonal_rainfall_seasonality\\'
        spatial_dic = T.load_npy_dir(fdir)

        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr,interpolation='nearest')
        # plt.title(key)
        plt.show()

        pass

class check_correlation(): ## check CRU and ERA5 correlation
    def __init__(self):
        pass
    def run(self):
        self.check_spatial_map()
        # self.check_daily_trend()
        # self.foo()
        pass
    def check_spatial_map(self):
        fdir_ERA5=rf'E:\Project3\Data\ERA5_daily\dict\precip_transform\\'
        fdir_CRU=rf'E:\Project3\Data\\CRU-JRA\\Precip_transform\\'
        result_dict = {}

        spatial_dic_CRU = T.load_npy(fdir_CRU+f)
        spatial_dic_ERA5 = T.load_npy(fdir_ERA5+f)

        for pix in tqdm(spatial_dic_ERA5):
            r, c=pix
            # if not r==444:
            #     continue
            # if not c==817:
            #     continue

            if pix not in spatial_dic_CRU:
                continue

            vals_ERA5 = spatial_dic_ERA5[pix]
            if len(vals_ERA5) == 0:
                continue

            vals_ERA5_flatten = np.array(vals_ERA5).flatten()
            if T.is_all_nan(vals_ERA5_flatten):

                continue
            vals_CRU = spatial_dic_CRU[pix][0:-1]
            vals_CRU_flatten = np.array(vals_CRU).flatten()
            plt.plot(vals_ERA5_flatten,'b')
            plt.plot(vals_CRU_flatten,'r')
            # print(len(vals_ERA5))
            plt.show()
            # print(len(vals_CRU))
            #
            # if len(vals_ERA5) == 0 or len(vals_CRU) == 0:
            #     continue
            corr, p = T.nan_correlation(vals_ERA5_flatten, vals_CRU_flatten)
            result_dict[pix] = corr

        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(result_dict)
        plt.imshow(arr,interpolation='nearest')
        # plt.title(key)
        plt.show()
        outdir = rf'E:\Project3\Data\ERA5_daily\dict\correlation_CRU_ERA5\\'
        T.mk_dir(outdir, force=True)
        outf = outdir+'_correlation.npy'
        DIC_and_TIF(pixelsize=.25).arr_to_tif(arr,outf.replace('.npy','.tif'))
        np.save(outf, result_dict)
        pass

    def check_daily_trend(self):## a,b,r,p=
        # fdir_ERA5=rf'E:\Project3\Data\ERA5_daily\dict\precip_transform\\'
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir_CRU=rf'E:\Project3\Data\\CRU-JRA\\Precip_transform\\'
        result_dict = {}
        for f in os.listdir(fdir_CRU):
            print(f)

            spatial_dic_ERA5 = T.load_npy(fdir_CRU+f)

            for pix in tqdm(spatial_dic_ERA5):

                r, c = pix
                if r < 120:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                vals_ERA5 = spatial_dic_ERA5[pix][0:-1]
                if len(vals_ERA5) == 0:
                    continue

                vals_ERA5_flatten = np.array(vals_ERA5).flatten()
                if T.is_all_nan(vals_ERA5_flatten):

                    continue

                a,b,r,p = T.nan_line_fit(np.array(range(len(vals_ERA5_flatten))), vals_ERA5_flatten)
                result_dict[pix] = a

        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(result_dict)
        # plt.imshow(arr,interpolation='nearest')
        # # plt.title(key)
        # plt.show()
        outdir = rf'E:\Project3\Data\ERA5_daily\dict\correlation_CRU_ERA5\\trend_CRU\\'
        T.mk_dir(outdir,force=True)
        outf = outdir + 'trend.npy'
        DIC_and_TIF(pixelsize=.25).arr_to_tif(arr,outf.replace('.npy', '.tif'))

        np.save(outf, result_dict)
        pass

    def foo(self):
        f=rf'E:\Project3\Data\ERA5_daily\dict\correlation_CRU_ERA5\trend_ERA5\\per_pix_dic_103.npy'

        dic = np.load(f, allow_pickle=True, encoding='latin1').item()
        spatial_dic = {}
        for pix in dic:
            spatial_dic[pix] = dic[pix]
        arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f.replace('.npy', '.tif'))
        plt.imshow(arr, interpolation='nearest', cmap='jet')
        plt.colorbar()
        plt.title(f)
        plt.show()

        pass









class extract_rainfall_annual_based_on_monthly():  ## here process monthly GPCC and CRU data and calculate seasonality
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass

    def run(self):
        self.aggregate_monthly()
        # self.extract_rainfall_CV()
        # self.extract_seasonality()
        # self.trend_analysis()
        # self.plot_spatial()
    def aggregate_monthly(self):
        fdir = rf'E:Project3\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:Project3\Data\\ERA5_monthly\dict\\aggregate_monthly\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}
        dic_month={'Jan':31,'Feb':28,'Mar':31,'Apr':30,'May':31,'Jun':30,'Jul':31,'Aug':31,'Sep':30,'Oct':31,'Nov':30,'Dec':31}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]


            sum_list = []
            for val in vals:
                if T.is_all_nan(val):
                    continue


                val = np.array(val)
                ## aggregate monthly
                print(len(val))
                ## based on dic_month above to first split 365 into 12 parts


                for i in range(12):
                    if i == 11:
                        break
                    val_i = val[i * dic_month['Jan']:(i + 1) * dic_month['Jan']]
                    sum_list.append(np.nansum(val_i) )


            result_dic[pix] = sum_list

        outf = outdir_CV + 'EAR5_precip_monthly.npy'

        np.save(outf, result_dic)



    def extract_seasonality(self):
        from scipy.stats import entropy
        fdir=rf'E:\Project3\Data\ERA5_monthly\dict\aggregate_monthly\\'
        outdir='E:\Project3\Data\ERA5_monthly\dict\\extract_rainfall\\'

        T.mkdir(outdir,force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            SI_max = np.nanmax(vals)
            if T.is_all_nan(vals):
                continue

            CV_list = []
            vals_reshape = np.array(vals).reshape(-1, 12)

            for val in vals_reshape:

                ### This is  total number of precipitation events during the period of interest.

                val = np.array(val)
                if len(val) == 0:
                    continue
                n = len(val)  ## 12 months
                temp = 0
                for i in val:
                    SI = np.sum(val)
                    pi = i / SI
                    q = 1 / n
                    temp += pi * np.log2(pi / q)
                Dsize = temp * SI / SI_max

                CV_list.append(Dsize)


            result_dic[pix] = CV_list

        outf = outdir + 'seasonal_rainfall_CRU.npy'

        np.save(outf, result_dic)


    def extract_rainfall_CV(self):  ## extract CV of rainfall ready for multiregression
        fdir = 'D:\Project3\Data\monthly_data\CRU\DIC\\'
        outdir = 'E:\Project3\Data\ERA5_monthly\dict\\extract_rainfall\\'

        T.mk_dir(outdir, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            vals = spatial_dic[pix]
            vals_reshape = vals.reshape(-1, 12)
            CV_list = []
            for val in vals_reshape:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)

                CV = np.std(val) / np.mean(val) *100
                # print(CV)
                CV_list.append(CV)
            result_dic[pix] = CV_list

        outf = outdir + 'CV_rainfall_CRU.npy'

        np.save(outf, result_dic)



    pass

    def trend_analysis(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'E:\Project3\Data\ERA5_daily\dict\moving_window_average_anaysis\\'
        outdir = rf'E:\Project3\Data\ERA5_monthly\dict\\trend_analysis_moving_window\\\\'
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

            plt.imshow(arr_trend, cmap='jet', vmin=1, vmax=1)

            plt.colorbar()
            plt.title(f)
            plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

    pass

    def plot_spatial(self):

        fdir = rf'seasonal_rainfall_seasonality\\'

        # dic = T.load_npy(fdir)
        dic=T.load_npy_dir(fdir)

        result_dic={}


        for pix in dic:
            r, c = pix
            # if r<480:
            #     continue
            vals = dic[pix]
            vals_reshape = vals.reshape(-1, 12)
            vals_sum = []

            for val in vals_reshape:
                if T.is_all_nan(val):
                    continue

                val = np.array(val)
                val[val >9999] = np.nan
                val[val < 0] = np.nan
                SI = np.nansum(val)
                vals_sum.append(SI)
            multiyear_mean = np.mean(vals_sum)

            result_dic[pix]=multiyear_mean
        ###
        arr=DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(result_dic)
        plt.imshow(arr,interpolation='nearest',vmin=0,vmax=500)
        plt.show()






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
        # self.moving_window_std_anaysis()
        # self.moving_window_trend_anaysis()
        self.trend_analysis()
        # self.robinson()

        pass
    def moving_window_extraction(self):

        fdir_all =  rf'E:\Project3\Data\ERA5_daily\dict\extract_heatevent_annual\\'
        outdir = rf'E:\Project3\Data\ERA5_daily\dict\\extract_window\\'
        T.mk_dir(outdir, force=True)
        for fdir in os.listdir(fdir_all):
            if fdir not in ['heat_event_extraction' ]:
                continue
            # if fdir not in ['seasonal_rainfall_intervals', 'seasonal_rainfall_event_size',
            #                 'rainfall_frequency','heavy_rainfall_days','rainfall_event_size','sum_rainfall','annual_LAI4g']:
            #     continue


            for f in os.listdir(fdir_all + fdir):
                if not 'detrended_average_annual_tmax' in f:
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
        fdir = rf'E:Project3\Data\ERA5_daily\dict\\extract_window\\'
        outdir =  rf'E:Project3\Data\ERA5_daily\dict\\moving_window_average_anaysis\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'detrended_average_annual_tmax' in f:
                continue

            dic = T.load_npy(fdir + f)
            slides = 39-window_size
            outf = outdir + f.split('.')[0] + f'_interannual_CV.npy'
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

        fdir=rf'E:\Project3\Data\ERA5_daily\dict\\extract_window\\'
        outdir = rf'E:\Project3\Data\ERA5_daily\dict\\moving_window_average_anaysis\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not f.split('.')[0] in ['detrended_average_annual_tmax']:
                continue

            dic = T.load_npy(fdir + f)

            slides = 38 - window_size   ## revise!!
            outf = outdir + f.split('.')[0] + f'.npy'
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
                    # print(len(time_series))
                    ##average
                    average=np.nanmean(time_series)
                    # print(average)

                    trend_list.append(average)

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)

    def moving_window_std_anaysis(self):
        window_size=15
        fdir = rf'E:Project3\Data\ERA5_daily\dict\\extract_window\\'
        outdir =  rf'E:Project3\Data\ERA5_daily\dict\\moving_window_average_anaysis\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'detrended_sum_rainfall' in f:
                continue

            dic = T.load_npy(fdir + f)
            slides = 38-window_size
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
            if not f.split('.')[0] in ['average_heat_spell', 'heat_event_frequency',
               'maxmum_heat_spell']:
                continue


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
                    if len(time_series_all)<23:
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

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'E:\Project3\Data\ERA5_daily\dict\moving_window_average_anaysis\\'
        outdir = rf'E:\Project3\Data\ERA5_daily\dict\\\trend_analysis_moving_window\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            # if not f.split('.')[0] in ['seasonal_rainfall_intervals', 'seasonal_rainfall_event_size',
            #                            'rainfall_frequency', 'heavy_rainfall_days', 'rainfall_event_size',

            if not f.split('.')[0] in ['detrended_sum_rainfall_std', 'detrended_sum_rainfall',
                                       'rainfall_seasonality_all_year','heat_event_frequency',
                'heavy_rainfall_days','maxmum_dry_spell']:
                continue
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
                    trend_dic[pix] = slope/average*100
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
        fdir=rf'E:Project3\Data\ERA5_daily\dict\\trend_analysis_moving_window\\'
        temp_root=result_root+r'Result_new\trend_anaysis\\robinson\\'
        out_pdf_fdir=rf'E:Project3\Data\ERA5_daily\dict\\trend_analysis_moving_window\\pdf\\'

        T.mk_dir(out_pdf_fdir,force=True)


        variable='rainfall_seasonality_all_year'
        f_trend=fdir+variable+'_trend.tif'

        f_p_value=fdir+variable+'_p_value.tif'


        m,ret=Plot().plot_Robinson(f_trend, vmin=-2,vmax=2,is_discrete=True,colormap_n=11,)
        self.plot_Robinson_significance_scatter(m, f_p_value,temp_root,0.05,s=3)


        # plt.title(f'{variable}_(%/yr2)')
        plt.title(f'rainfall_seasonality(%/yr)')
        # plt.title(f'{variable}_(day/yr)')
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
class partial_correlation_CV():
    def __init__(self):
        pass
        self.result_root = rf'E:\Project3\Data\ERA5_daily\dict\\'

        self.fdir_X = self.result_root+rf'\moving_window_average_anaysis\\'
        self.fdir_Y = self.result_root+rf'moving_window_average_anaysis\\'
        self.fy_list = ['detrended_annual_LAI4g_CV']
        self.fx_list=['detrended_sum_rainfall_interannual_CV','CV_rainfall_interseasonal',
                      'heat_event_frequency']


        self.outdir = self.result_root+rf'\partial_correlation_CV\\CV3\\'
        T.mk_dir(self.outdir, force=True)

        self.outpartial = self.result_root + rf'\partial_correlation_CV\\CV3\\partial_corr3.npy'
        self.outpartial_pvalue = self.result_root + rf'\partial_correlation_CV\\CV3\\\\partial_corr_pvalue3.npy'

    def run(self):
        df=self.build_df(self.fdir_X,self.fdir_Y,self.fx_list,self.fy_list)
        #
        self.cal_partial_corr(df,self.fx_list)
        # self.cal_single_correlation()
        # self.cal_single_correlation_ly()
        self.plot_partial_correlation()
        # self.partial_correlation_sign()
        # self.statistical_analysis()
        # self.robinson()


    def build_df(self,fdir_X,fdir_Y,fx_list,fy):
        df = pd.DataFrame()

        filey = fdir_Y + fy[0] + '.npy'
        print(filey)

        dic_y = T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue
            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
            yvals=yvals[0:23]
            if yvals[0] == None:
                continue

            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        for xvar in fx_list:

            # print(var_name)
            x_val_list = []
            filex = fdir_X + xvar + '.npy'
            # filex = fdir_X + xvar + f'_{period}.npy'

            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
            dic_x = T.load_npy(filex)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                xvals = dic_x[pix]
                xvals = np.array(xvals)
                xvals = xvals[0:23]
                if len(xvals) == 0:
                    x_val_list.append([])
                    continue

                xvals = T.interp_nan(xvals)
                if xvals[0] == None:
                    x_val_list.append([])
                    continue

                x_val_list.append(xvals)

            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list
        T.print_head_n(df)

        # exit()

        return df

    def cal_partial_corr(self,df,x_var_list, ):


        outf_corr = self.outpartial
        outf_pvalue = self.outpartial_pvalue

        partial_correlation_dic= {}
        partial_p_value_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']
            # y_vals = T.remove_np_nan(y_vals)
            # y_vals = T.interp_nan(y_vals)
            if len(y_vals) == 0:
                continue

            # y_vals_detrend = signal.detrend(y_vals)
            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:

                x_vals = row[x]

                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                # x_vals = T.interp_nan(x_vals)
                # if len(y_vals) == 18:
                #     x_vals = x_vals[:-1]

                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue
                # x_vals_detrend = signal.detrend(x_vals) #detrend
                df_new[x] = x_vals
                # df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals  # nodetrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1, how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()

            if len(df_new) <= 3:
                continue
            partial_correlation = {}
            partial_correlation_p_value = {}
            for x in x_var_list_valid_new:
                x_var_list_valid_new_cov = copy.copy(x_var_list_valid_new)
                x_var_list_valid_new_cov.remove(x)
                r, p = self.partial_corr(df_new, x, 'y', x_var_list_valid_new_cov)
                partial_correlation[x] = r
                partial_correlation_p_value[x] = p

            partial_correlation_dic[pix] = partial_correlation
            partial_p_value_dic[pix] = partial_correlation_p_value
        T.save_npy(partial_correlation_dic, outf_corr)
        T.save_npy(partial_p_value_dic, outf_pvalue)





            # print(df_new)


    def cal_single_correlation(self):
        f_x= result_root + rf'\anomaly\OBS_extend\\wet_frequency_90th.npy'
        f_y = result_root + rf'\anomaly\OBS_extend\\LAI4g.npy'
        dic_x = T.load_npy(f_x)
        dic_y = T.load_npy(f_y)

        spatial_r_dic = {}

        for pix in tqdm(dic_x):
            if not pix in dic_y:
                continue
            x_val = dic_x[pix]

            y_val = dic_y[pix]

            x_val = T.interp_nan(x_val)
            y_val = T.interp_nan(y_val)
            if x_val[0] == None:
                continue
            y_val = y_val[0:38]

            if len(y_val) == 0:
                continue

            if np.isnan(np.nanmean(x_val)):
                continue
            if len(x_val) != len(y_val):
                continue
            ## remove nan

       ####
            r, p = stats.pearsonr(x_val, y_val)
            # print(r)
            # print(p)
            spatial_r_dic[pix] = r
        arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_r_dic)
        plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='jet',interpolation='nearest')
        plt.colorbar()
        plt.show()


    def cal_single_correlation_ly(self):
        f_x= result_root + rf'\anomaly\OBS_extend\\CV_rainfall.npy'
        f_y = result_root + rf'\anomaly\OBS_extend\\LAI4g.npy'
        outdir = join(result_root, 'anomaly', 'cal_single_correlation_ly')
        T.mk_dir(outdir, force=True)
        dic_x = T.load_npy(f_x)
        dic_y = T.load_npy(f_y)

        spatial_r_dic_cv = {}
        spatial_r_dic_lai = {}
        correlation_dict = {}

        for pix in tqdm(dic_x):
            if not pix in dic_y:
                continue
            x_val = dic_x[pix]

            y_val = dic_y[pix]

            x_val = T.interp_nan(x_val)
            y_val = T.interp_nan(y_val)
            if x_val[0] == None:
                continue
            y_val = y_val[0:38]

            if len(y_val) == 0:
                continue

            if np.isnan(np.nanmean(x_val)):
                continue
            if len(x_val) != len(y_val):
                continue
            ## remove nan

       ####
            # r, p = stats.pearsonr(x_val, y_val)
            # print(r)
            r_lai,_ = stats.pearsonr(list(range(len(y_val))), y_val)
            r_cv,_ = stats.pearsonr(list(range(len(x_val))), x_val)
            r,p = stats.pearsonr(x_val, y_val)
            # print(p)
            # spatial_r_dic[pix] = r
            spatial_r_dic_cv[pix] = r_cv
            spatial_r_dic_lai[pix] = r_lai
            correlation_dict[pix] = r
        outf_cv = join(outdir, 'CV_trend.tif')
        outf_lai = join(outdir, 'LAI_trend.tif')
        outf_corr = join(outdir, 'correlation.tif')
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(spatial_r_dic_cv, outf_cv)
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(spatial_r_dic_lai, outf_lai)
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(correlation_dict, outf_corr)



    def plot_partial_correlation(self):

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        fdir = self.result_root + rf'\partial_correlation_CV\\CV3\\'
        f_partial = fdir + 'partial_corr3.npy'
        # f_pvalue = fdir + 'partial_corr_pvalue3.npy'


        # partial_correlation_dic = np.load(f_partial, allow_pickle=True, encoding='latin1').item()
        partial_correlation_p_value_dic = np.load(f_partial, allow_pickle=True, encoding='latin1').item()


        var_list = []
        for pix in partial_correlation_p_value_dic:

            r,c=pix
            if r < 120:
                continue
            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue
            if dic_modis_mask[pix] == 12:
                continue

            vals = partial_correlation_p_value_dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in partial_correlation_p_value_dic:
                r, c = pix
                if r < 120:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                dic_i = partial_correlation_p_value_dic[pix]


                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            arr = arr * array_mask
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, fdir + f'partial_corr_{var_i}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr, vmin=-1, vmax=1)

            plt.title(var_i)
            plt.colorbar()

        plt.show()

    def partial_correlation_sign(self):
        fdir= rf'E:Project3\Data\ERA5_daily\dict\\partial_correlation_CV\\CV3\\'
        for f in os.listdir(fdir):
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue
            if 'pvalue' in f:
                continue
            f_correlation = fdir + f
            f_pvalue = fdir + f.split('.')[0] + '_pvalue.tif'

            arr = ToRaster().raster2array(f_correlation)[0]
            arr_pvalue = ToRaster().raster2array(f_pvalue)[0]

            ### pvalue >=0.05 ==1 and pvalue < 0.05 ==0
            arr_pvalue[arr_pvalue <-999] = np.nan

            mask = arr_pvalue < 0.05

            array_result = arr * mask
            array_result[array_result == 0] = np.nan

            outdir = fdir+'partial_correlation_sign_CV\\'

            T.mk_dir(outdir)
            outf= outdir + f
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_result, outf)


        pass
    def statistical_analysis(self):
        fdir= rf'E:\Project3\Data\ERA5_daily\dict\partial_correlation_CV\CV3\partial_correlation_sign_CV\\'

        for f in os.listdir(fdir):
            if not 'tif' in f:
                continue
            if 'xml' in f:
                continue
            f_temp = rf'E:\Project3\Data\ERA5_daily\dict\partial_correlation_CV\CV3\\partial_corr_rainfall_seasonality_all_year.tif'
            arr_temp = ToRaster().raster2array(f_temp)[0]
            arr_temp[arr_temp < -999] = np.nan
            arr_temp_flattened = arr_temp.flatten()
            arr_temp_flattened = arr_temp_flattened[~np.isnan(arr_temp_flattened)]


            arr=ToRaster().raster2array(fdir+f)[0]
            arr[arr < -999] = np.nan
            arr_flatten = arr.flatten()


            ## calculate areas when correlation is >0 and correlation is < 0

            arr_flatten_value = arr_flatten[~np.isnan(arr_flatten)]
            positive = arr_flatten_value[arr_flatten_value > 0]
            negative = arr_flatten_value[arr_flatten_value < 0]
            print(f, len(positive)/len(arr_temp_flattened), len(negative)/len(arr_temp_flattened))




    def partial_corr(self, df, x, y, cov):
        df = pd.DataFrame(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        # print(df)
        df = df.dropna()
        # try:
        # print(x)
        # print(y)
        stats_result = pg.partial_corr(data=df, x=x, y=y, covar=cov, method='pearson').round(3)
        r = float(stats_result['r'])
        p = float(stats_result['p-val'])
        return r, p


    pass




























    def maximum_trend(self):
        fdir = result_root + rf'extract_GS_return_monthly_data\OBS_LAI\\'
        outdir = result_root + rf'extract_GS_return_monthly_data\OBS_LAI_maximum_trend\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
            dic_maximum_trend = {}
            for pix in tqdm(dic, desc=f):
                time_series = dic[pix]
                if len(time_series) == 0:
                    pass

    def robinson(self):
        fdir=rf'E:\Project3\Data\ERA5_daily\dict\partial_correlation_CV\CV3\\partial_correlation_sign_CV\\'
        # temp_root=result_root+r'Result_new\trend_analysis_moving_window\\robinson\\'
        out_pdf_fdir=rf'E:\Project3\Data\ERA5_daily\dict\partial_correlation_CV\CV3\\partial_correlation_sign_CV\pdf\\'

        T.mk_dir(out_pdf_fdir,force=True)
        for f in os.listdir(fdir):
            variable=f.split('.')[0]
            if not f.endswith('.tif'):
                continue

            fpath=fdir+f


            Plot().plot_Robinson(fpath, vmin=-1,vmax=1,is_discrete=True,colormap_n=11,)
        # self.plot_Robinson_significance_scatter(m, f_p_value,temp_root,0.05,s=2)

            # plt.title(f'{variable}_(%/yr)')
            # plt.title(f'{variable}')
            # plt.title(f'{variable}_(day/yr)')
            # plt.title(f'(%/100mm/yr)')
            # plt.title('r')
            # plt.show()
            ## save fig pdf
            #save pdf
            plt.savefig(out_pdf_fdir+variable+'.pdf', dpi=300, bbox_inches='tight')
            plt.close()





class TRENDY_model:
    ## 1)

    def __init__(self):
        pass

    def run(self):
        # self.TIFF_to_dic()

        # self.extract_annual_LAI()
        # self.detrend()
        # self.moving_window_extraction()
        # self.moving_window_CV_anaysis()
        # self.trend_analysis()
        self.plt_basemap()

        pass
    def TIFF_to_dic(self):
        fdir_all=rf'E:\Project3\Data\TRENDY_LAI\unify_tiff\\'

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan

        year_list = list(range(1982, 2021))

        # 作为筛选条件
        for fdir in os.listdir(fdir_all):


            outdir  = rf'E:\Project3\Data\TRENDY_LAI_DIC\\{fdir}\\'


            T.mk_dir(outdir, force=True)
            all_array = []  #### so important  it should be go with T.mk_dic

            for f in os.listdir(fdir_all + fdir):
                if not f.endswith('.tif'):
                    continue
                if int(f.split('.')[0][0:4]) not in year_list:
                    continue

                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
                    fdir_all + fdir + '\\' + f)
                array = np.array(array, dtype=float)

                array_unify = array[:720][:720,
                              :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

                array_unify[array_unify < -999] = np.nan


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

        fdir_all = rf'E:\Project3\Data\TRENDY_LAI\TRENDY_LAI_DIC\\'
        outdir = rf'E:Project3\Data\TRENDY_LAI\\extract_annual_LAI\\'
        for fdir in os.listdir(fdir_all):


            outf=outdir + f'{fdir}_annual.npy'
            print(outf)
            if os.path.exists(outf):
                continue
            Tools().mk_dir(outdir, force=True)
            annual_spatial_dict = {}
            dict=T.load_npy_dir(fdir_all+fdir)
            for pix in tqdm(dict):
                time_series = dict[pix]
                time_series=np.array(time_series)
                print(time_series.shape)

                if T.is_all_nan(time_series):
                    continue

                annual_time_series_reshape = np.reshape(time_series, (-1, 12))

                annual_time_series = np.nanmean(annual_time_series_reshape, axis=1)


                annual_spatial_dict[pix] = annual_time_series


            np.save(outf, annual_spatial_dict)

        pass

    def detrend(self): ## detrend LAI4g

        fdir = rf'E:\Project3\Data\TRENDY_LAI\extract_annual_LAI\\'
        outdir = rf'E:Project3\Data\TRENDY_LAI\\detrend\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            dict = T.load_npy(fdir + f)
            annual_spatial_dict = {}
            for pix in tqdm(dict):
                time_series = dict[pix]

                if T.is_all_nan(time_series):
                    continue

                plt.plot(time_series)


                detrended_annual_time_series = signal.detrend(time_series)+np.mean(time_series)
                # print((detrended_annual_time_series))
                # plt.plot(detrended_annual_time_series)
                # plt.show()

                annual_spatial_dict[pix] = detrended_annual_time_series
            outf=outdir +f'{f.split(".")[0]}_detrend.npy'
            np.save(outf, annual_spatial_dict)



        pass

    def moving_window_extraction(self):

        fdir = rf'E:\Project3\Data\TRENDY_LAI\detrend\\'
        outdir = rf'E:\Project3\Data\TRENDY_LAI\moving_window_extraction\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):


            outf = outdir + f
            print(outf)
            if os.path.isfile(outf):
                continue

            dic = T.load_npy(fdir + f)
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
    def moving_window_CV_anaysis(self):
        window_size=15
        fdir = rf'E:\Project3\Data\TRENDY_LAI\moving_window_extraction\\'
        outdir =  rf'E:\Project3\Data\TRENDY_LAI\moving_window_CV\\'
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

                time_series_all = dic[pix]
                if len(time_series_all)<23:
                    continue
                time_series_all = np.array(time_series_all)
                # print(time_series_all)
                for ss in range(slides):
                    if np.isnan(np.nanmean(time_series_all)):
                        print('error')
                        continue
                    # print((len(time_series)))
                    ### if all values are identical, then continue
                    time_series=time_series_all[ss]
                    # print(time_series)
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

    def trend_analysis(self):  ##each window average trend

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'E:\Project3\Data\TRENDY_LAI\moving_window_CV\\'
        outdir = rf'E:\Project3\Data\TRENDY_LAI\trend_analysis\moving_window_CV\\'
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
    def plt_basemap(self):
        fdir = rf'E:\Project3\Data\TRENDY_LAI\trend_analysis\moving_window_CV\\'

        count = 1
        fig = plt.figure(figsize=(10, 15))
        for f in os.listdir(fdir):
            # if not 'CABLE' in f:
            #     continue
            if not f.endswith('.tif'):
                continue
            if 'p_value' in f:
                continue


            print(f)

            outf = fdir + f.split('.')[0]
            if os.path.isfile(outf + '_trend.tif'):
                continue
            print(outf)

            fpath = join(fdir, f )


            ax = plt.subplot(5, 3, count)
            count = count + 1


            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr = Tools().mask_999999_arr(arr, warning=False)
            # plt.imshow(arr, cmap='PiYG', interpolation='nearest', vmin=-50, vmax=50)
            # plt.show()
            # arr = arr[:120]
            # arr_m = ma.masked_where(np.isnan(arr), arr)
            lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
            # plt.imshow(arr,interpolation='nearest', cmap='PiYG', vmin=-.1, vmax=.1)
            # plt.show()
            # print(lat_list);exit()
            # plt.show()
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180, resolution='i',
                        ax=ax)

            ret = m.pcolormesh(lon_list, lat_list, arr, cmap='PiYG', zorder=1, vmin=-1, vmax=1)
            coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
            ## set basemap size


            plt.title(f.replace('_trend.tif', '').replace('_S2', '').replace('_lai', '').replace('_LAI', '').replace('_detrend', '').replace('_annual', ''))
            # plt.imshow(arr, cmap='PiYG', interpolation='nearest', vmin=-50, vmax=50)
            # m.pcolormesh(lon_list, lat_list, arr_m, cmap='PiYG', zorder=-1, vmin=-1, vmax=1)

            # plt.tight_layout()


            # cax = plt.axes([0.5 - 0.3 / 2, 0.1, 0.3, 0.02])
            # plt.colorbar(mappable=ret, ax=ax, orientation='horizontal')
            ## set colorbar (CV %/year)
            # colorbar = plt.colorbar(mappable=ret, ax=ax, orientation='horizontal')

            ## set name of colorbar
            # colorbar.set_label('CV %/year', fontsize=12)




            # plt.colorbar(ax=ax, cax=cax, orientation='horizontal', extend='both')
        outdir=rf'E:\Project3\Data\TRENDY_LAI\trend_analysis\moving_window_CV\\Figure\\'
        T.mk_dir(outdir, force=True)

        # outf = join(outdir, 'trend_analysis.pdf')
        #
        # plt.savefig(outf, dpi=300, bbox_inches='tight')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        plt.show()

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
    # extract_rainfall_annual_based_on_monthly().run()

    # extract_heatevent().run()
    extract_rainfall_annual_based_on_daily().run()
    # TRENDY_model().run()
    # check_correlation().run()

    # moving_window().run()
    # partial_correlation_CV().run()

    # PLOT().run()
    # Check_data().run()

    pass

if __name__ == '__main__':
    main()