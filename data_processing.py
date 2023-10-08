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

class data_processing():
    def __init__(self):
        pass
    def run(self):
        # self.tif_to_dic()
        self.split_data()
    def tif_to_dic(self):

        fdir = data_root+'monthly_data\SPEI3\\TIFF\\'
        outdir=data_root+'monthly_data\SPEI3\\DIC\\'
        T.mk_dir(outdir, force=True)

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan

        year_list = list(range(1982, 2021))
        all_array=  []# 作为筛选条件

        for f in tqdm(sorted(os.listdir(fdir)), desc='loading...'):
            if f.startswith('.'):
                continue
            if not f.endswith('.tif'):
                continue
            if isfile(outdir):
                continue
            print(f)

            if int(f.split('.')[0][0:4]) not in year_list:  #
                continue

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
                fdir + f)
            array = np.array(array, dtype=float)
            # extract 360 and 720
            array_unify = array[:720][:720,
                          :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

            array_unify[array_unify < -999] = np.nan
            # array[array ==0] = np.nan
            # array_unify[array_unify < 0] = np.nan  # 当变量是LAI 的时候，<0!!
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
    def split_data(self):

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask= DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir_all=data_root+'monthly_data\\'
        for fdir in os.listdir(fdir_all):
            if not 'LAI4g' in fdir:
                continue

            outdir=data_root+rf'split\{fdir}\\'
            T.mk_dir(outdir,force=True)

            for fdir_dic in os.listdir(fdir_all+fdir):
                if not 'DIC' in fdir_dic:
                    continue
                dic = {}
                dic_i = {}
                dic_ii = {}
                for f in os.listdir(fdir_all+fdir+'\\'+fdir_dic):
                    if not f.endswith('.npy'):
                        continue

                    dic_temp=np.load(fdir_all+fdir+'\\'+fdir_dic+'\\'+f,allow_pickle=True).item()
                    dic.update(dic_temp)

                for pix in tqdm(dic):
                    if pix not in dic_dryland_mask:
                        continue
                    time_series=dic[pix]
                    time_series[time_series > 10000] = np.nan
                    time_series=np.array(time_series)/100.
                    time_series[time_series >7] = np.nan


                    if len(time_series) < 12*39:
                        continue
                    time_series_i=time_series[:12*19]
                    time_series_ii=time_series[12*19:]

                    dic_i[pix]=time_series_i
                    dic_ii[pix]=time_series_ii
                np.save(outdir+'1982_2000.npy',dic_i)
                np.save(outdir+'2001_2020.npy',dic_ii)



class statistic_analysis():
    def __init__(self):
        pass
    def run(self):
        # self.trend_analysis()
        # self.detrend_zscore()
        # self.detrend_zscore_monthly()
        self.zscore()
        self.detrend()


    def trend_analysis(self):


        fdir = data_root + rf'Extraction\\'
        outdir = result_root + rf'trend_analysis\\original\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            outf=outdir+f.split('.')[0]
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
            trend_dic = {}
            p_value_dic = {}
            for pix in dic_mask_lc:
                if pix not in dic:
                    continue
                time_series = dic[pix]

                if dic_mask_lc[pix] == 'Crop':
                    continue
                val_lc_change = array_mask_landcover_change[pix]
                if val_lc_change == np.nan:
                    continue
                time_series = np.array(time_series)

                time_series[time_series < -99.] = np.nan
                slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                trend_dic[pix] = slope
                p_value_dic[pix] = p_value

            arr_trend = DIC_and_TIF().pix_dic_to_spatial_arr(trend_dic)
            p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(p_value_dic)

            DIC_and_TIF().arr_to_tif(arr_trend, outf + '_trend.tif')
            DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

            # plt.imshow(arr_trend, cmap='jet', vmin=-0.03, vmax=0.03)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()



    def detrend_zscore(self):

        dic_mask_lc_file = data_root+'Base_data/LC_reclass2.npy'
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = data_root+'//Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)

        product_list=  ['MCD','CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'DLEM_S2_lai' 'CLM5', 'IBIS_S2_lai',
                          'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',''
                           'LPX-Bern_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai','Trendy_ensemble']


        for period in ['early', 'peak', 'late', 'early_peak', 'early_peak_late']:

            for product in product_list:
                outdir = result_root + rf'detrend_zscore\{period}\\'
                outf=outdir+product+'.npy'
                # print(outf)
                # exit()
                f = result_root + rf'extraction_original_val\LAI\{product}\\during_{period}_{product}.npy'

                Tools().mk_dir(outdir,force=True)
                dic = {}


                dic = dict(np.load( f, allow_pickle=True, ).item())


                detrend_zscore_dic={}

                for pix in tqdm(dic):

                    val_lc_change = array_mask_landcover_change[pix]
                    if val_lc_change < -9999:
                        continue
                    if pix not in dic_mask_lc:
                        continue
                    if dic_mask_lc[pix] == 'Crop':
                        continue
                    if array_mask_landcover_change[pix] == np.nan:
                        continue
                    print(len(dic[pix]))
                    time_series = dic[pix]
                    print(len(time_series))

                    time_series=np.array(time_series)
                    # plt.plot(time_series)
                    # plt.show()

                    time_series[time_series < -999] = np.nan

                    if np.isnan(np.nanmean(time_series)):
                        continue
                    if np.nanmean(time_series) <= 0.:
                        continue

                    delta_time_series = []
                    mean = np.nanmean(time_series)
                    std=np.nanstd(time_series)
                    if std == 0:
                        continue
                    delta_time_series = (time_series - mean) / std

                    detrend_delta_time_series = signal.detrend(delta_time_series)

                    # plt.plot(detrend_delta_time_series)
                    # plt.show()

                    detrend_zscore_dic[pix] = detrend_delta_time_series

                np.save(outf, detrend_zscore_dic)

    def detrend_zscore_monthly(self): #  detrend based on each month


        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir_all = data_root + 'split\\'
        for fdir in os.listdir(fdir_all):


            outdir = result_root + rf'detrend_zscore\\{fdir}\\'
            # if os.path.isdir(outdir):
            #     continue
            Tools().mk_dir(outdir, force=True)

            for f in os.listdir(fdir_all+fdir):

                outf=outdir+f.split('.')[0]
                print(outf)


                dic = dict(np.load(fdir_all+fdir+'\\'+f, allow_pickle=True, ).item())

                detrend_zscore_dic={}


                for pix in tqdm(dic):
                    detrend_delta_time_series_list = []
                    if pix not in dic_dryland_mask:
                        continue

                    print(len(dic[pix]))
                    time_series = dic[pix]
                    print(len(time_series))

                    time_series=np.array(time_series)
                    time_series[time_series < -999] = np.nan


                    if np.isnan(np.nanmean(time_series)):
                        continue
                    if np.nanmean(time_series) <= 0.:
                        continue
                    time_series_reshape=time_series.reshape(-1,12)
                    time_series_reshape_T=time_series_reshape.T
                    for i in range(len(time_series_reshape_T)):
                        time_series_i=time_series_reshape_T[i]

                        mean = np.nanmean(time_series_i)
                        std=np.nanstd(time_series_i)
                        if std == 0:
                            continue

                        delta_time_series = (time_series_i - mean) / std
                        if np.isnan(delta_time_series).any():
                            continue

                        detrend_delta_time_series = signal.detrend(delta_time_series)
                        detrend_delta_time_series_list.append(detrend_delta_time_series)
                    detrend_delta_time_series_array=np.array(detrend_delta_time_series_list)
                    detrend_delta_time_series_array=detrend_delta_time_series_array.T
                    detrend_delta_time_series_result=detrend_delta_time_series_array.flatten()

                    # detrend_delta_time_series_result2=detrend_delta_time_series_array.reshape(-1)   ##flatten and reshape 是一个东西
                    ##plot
                    # plt.plot(detrend_delta_time_series_result1,'r' ,linewidth=0.5, marker='*', markerfacecolor='blue', markersize=1 )
                    # plt.plot(detrend_delta_time_series_result,'b' ,linewidth=1,linestyle='--')
                    # plt.show()

                    detrend_zscore_dic[pix] = detrend_delta_time_series_result

                np.save(outf, detrend_zscore_dic)


    def zscore(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)


        fdir_all=data_root+'split\\'


        for fdir in os.listdir(fdir_all):


            outdir = result_root + rf'zscore\\{fdir}\\'
            # if os.path.isdir(outdir):
            #     continue
            Tools().mk_dir(outdir, force=True)

            for f in os.listdir(fdir_all + fdir):

                outf = outdir + f.split('.')[0]
                print(outf)

                dic = dict(np.load(fdir_all + fdir + '\\' + f, allow_pickle=True, ).item())

                zscore_dic = {}

                for pix in tqdm(dic):
                    delta_time_series_list = []
                    if pix not in dic_dryland_mask:
                        continue

                    # print(len(dic[pix]))
                    time_series = dic[pix]
                    # print(len(time_series))

                    time_series = np.array(time_series)
                    time_series[time_series < -999] = np.nan

                    if np.isnan(np.nanmean(time_series)):
                        continue
                    if np.nanmean(time_series) <= 0.:
                        continue
                    time_series_reshape = time_series.reshape(-1, 12)
                    time_series_reshape_T = time_series_reshape.T
                    for i in range(len(time_series_reshape_T)):
                        time_series_i = time_series_reshape_T[i]

                        mean = np.nanmean(time_series_i)
                        std = np.nanstd(time_series_i)

                        delta_time_series = (time_series_i - mean) / std

                        delta_time_series_list.append(delta_time_series)
                    delta_time_series_array = np.array(delta_time_series_list)
                    delta_time_series_array = delta_time_series_array.T
                    delta_time_series_result = delta_time_series_array.flatten()

                    # detrend_delta_time_series_result2=detrend_delta_time_series_array.reshape(-1)   ##flatten and reshape 是一个东西
                    ##plot
                    # plt.plot(detrend_delta_time_series_result1,'r' ,linewidth=0.5, marker='*', markerfacecolor='blue', markersize=1 )
                    # plt.plot(detrend_delta_time_series_result,'b' ,linewidth=1,linestyle='--')
                    # plt.show()

                    zscore_dic[pix] = delta_time_series_result

                np.save(outf, zscore_dic)

        pass

    def detrend(self):  # detrend based on two period 1982-2000 and 2001-2020

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir_all = result_root + 'zscore\\'
        for fdir in os.listdir(fdir_all):


            outdir = result_root + rf'detrend_zscore_Yang\\{fdir}\\'
            # if os.path.isdir(outdir):
            #     continue
            Tools().mk_dir(outdir, force=True)

            for f in os.listdir(fdir_all + fdir):

                outf = outdir + f.split('.')[0]
                print(outf)

                dic = dict(np.load(fdir_all + fdir + '\\' + f, allow_pickle=True, ).item())

                detrend_zscore_dic = {}

                for pix in tqdm(dic):

                    if pix not in dic_dryland_mask:
                        continue

                    print(len(dic[pix]))
                    time_series = dic[pix]
                    print(len(time_series))

                    time_series = np.array(time_series)
                    time_series[time_series < -999] = np.nan

                    if np.isnan(np.nanmean(time_series)):
                        continue

                    # if np.isnan(time_series).any():
                    #     continue
                    detrend_time_series=T.detrend_vals(time_series)
                    detrend_zscore_dic[pix] = detrend_time_series
                    # plt.plot(detrend_time_series)
                    # plt.show()

                np.save(outf, detrend_zscore_dic)
class moving_window():
    def __init__(self):
        pass
    def run(self):
        self.moving_window_extraction()
        pass
    def moving_window_extraction(self):
        variables=['LAI4g','NDVI4g','GPP_CFE','GPP_baseline']
        fdir = data_root + rf'Extraction\\'
        outdir = result_root + rf'\\extract_window\\extract_detrend_original_window\\'
        T.mk_dir(outdir, force=True)
        for variable in variables:

            f=fdir+variable+'.npy'
            outf=outdir+variable+'.npy'
            outf_i = join(outdir, fdir)
            if os.path.isfile(outf_i):
                continue
            dic = T.load_npy(f)
            window = 15

            new_x_extraction_by_window = {}
            for pix in tqdm(dic):

                time_series = dic[pix]
                time_series = np.array(time_series)

                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    print('error')
                    continue
                print((len(time_series)))
                ### if all values are identical, then continue
                if np.nanmax(time_series) == np.nanmin(time_series):
                    continue

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
                #     x_anomaly=x_vals[i]-x_mean
                #
                #     anomaly.append(x_anomaly)
                new_x_extraction_by_window.append(x_vals)
        return new_x_extraction_by_window
class multi_regression_window():
    def __init__(self):
        self.fdirX=result_root+rf'extract_window\extract_detrend_original_window\15_year_window_1982_2020\X\\'
        self.y_f=result_root+rf'extract_window\extract_detrend_original_window\15_year_window_1982_2020\Y\\LAI4g_clean.npy'

        self.xvar_list = ['Tmax', 'GLEAM_SMroot']
        self.y_var = ['LAI4g_clean']
        pass

    def run(self):

        window = 39-15

        # step 1 build dataframe
        for i in range(window):
            outdir = result_root + rf'multi_regression_moving_window\\window{window}\\'
            df_i = self.build_df(self.fdirX, self.y_f, self.xvar_list, i)

            T.mk_dir(outdir,force=True)
            outf= result_root + rf'multi_regression_moving_window\\window15\\LAI_SMroot_window{i:02d}.npy'
            # if os.path.isfile(outf):
            #     continue
            print(outf)

            self.cal_multi_regression_beta(df_i,self.xvar_list, outf)  # 修改参数
            # self.plt_multi_regression_result(outdir,self.y_var,i)

    def build_df(self, fdir_X, y_f, xvar_list, w):


        df = pd.DataFrame()
        dic_y=T.load_npy(y_f)
        pix_list = []
        y_val_list=[]

        for pix in dic_y:
            if len(dic_y[pix]) == 0:
                continue
            vals = dic_y[pix][w]
            # print(vals)
            # exit()
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            vals = vals
            pix_list.append(pix)
            y_val_list.append(vals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        for xvar in xvar_list:


            x_val_list = []
            x_arr = T.load_npy(fdir_X+xvar+'.npy')
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in x_arr:
                    x_val_list.append([])
                    continue
                vals = x_arr[pix][w]
                vals = np.array(vals)
                if len(vals) == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(vals)

            df[xvar] = x_val_list

        return df



    def __linearfit(self, x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
        for i in range(0, int(N)):
            sx += x[i]
            sy += y[i]
            sxx += x[i] * x[i]
            syy += y[i] * y[i]
            sxy += x[i] * y[i]
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
        b = (sy - a * sx) / N
        r = -(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
        return a, b, r


    def cal_multi_regression_beta(self, df, x_var_list, outf):

        multi_derivative = {}
        multi_pvalue = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']
            y_vals = T.remove_np_nan(y_vals)
            if len(y_vals) == 0:
                continue
            y_vals = np.array(y_vals)
            # y_vals_detrend=signal.detrend(y_vals)


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:
                x_vals = row[x]

                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals = T.interp_nan(x_vals)
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
            linear_model = LinearRegression()

            linear_model.fit(df_new[x_var_list_valid_new], df_new['y'])
            # coef_ = np.array(linear_model.coef_) / y_mean
            coef_ = np.array(linear_model.coef_)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))
            ## pvalue
            X=df_new[x_var_list_valid_new]
            Y=df_new['y']
            try:
                sse = np.sum((linear_model.predict(X) -Y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])

                se = np.array([
                    np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                    for i in range(sse.shape[0])
                ])

                t = coef_ / se
                p = 2 * (1 - stats.t.cdf(np.abs(t), Y.shape[0] - X.shape[1]))
            except:
                p=np.nan

            multi_derivative[pix] = coef_dic
            multi_pvalue[pix] = p

        T.save_npy(multi_derivative, outf)
        T.save_npy(multi_pvalue, outf.replace('.npy', '_pvalue.npy'))

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir,y_var,w):

        f='D:\Project3\Result\multi_regression_moving_window\window15\\LAI_SMroot_window00.npy'

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:
            # print(pix)
            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in dic:
                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f'{multi_regression_result_dir}\\{var_i}_{y_var}_{w:02d}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr,vmin=-5,vmax=5)

            plt.title(var_i)
            plt.colorbar()

        plt.show()

class multi_regression():
    def __init__(self):
        self.fdirX=data_root+rf'detrend_zscore\\'
        self.fdirY=data_root+rf'detrend_zscore\\'
        self.xvar=['Tmax','GLEAM_SMroot']
        self.y_var=['LAI4g']
        self.multi_regression_result_dir=result_root+rf'multi_regression\\'
        T.mk_dir(self.multi_regression_result_dir,force=True)

        self.multi_regression_result_f = result_root + rf'multi_regression\\LAI_SMroot.npy'
        pass

    def run(self):

        #step 1 build dataframe
        df = self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var)

        # # # step 2 cal correlation
        self.cal_multi_regression_beta(df, self.xvar)  # 修改参数

        # step 3 plot
        # self.plt_multi_regression_result(self.multi_regression_result_dir,self.y_var[0])

    def build_df(self,fdir_X,fdir_Y,fx_list,fy):

        window=15

        df = pd.DataFrame()
        dic_y=T.load_npy(fdir_Y+fy[0]+'.npy')
        pix_list = []
        y_val_list=[]


        for pix in dic_y:
            vals = dic_y[pix]
            # print(vals)
            # exit()
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            vals = vals
            pix_list.append(pix)
            y_val_list.append(vals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        x_var_list = []
        for xvar in fx_list:

            x_var_list.append(xvar)
            # print(var_name)
            x_val_list = []
            x_arr = T.load_npy(fdir_X+xvar+'.npy')
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in x_arr:
                    x_val_list.append([])
                    continue
                vals = x_arr[pix]
                vals = np.array(vals)
                if len(vals) == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(vals)
            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list

        return df

    def __linearfit(self, x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
        for i in range(0, int(N)):
            sx += x[i]
            sy += y[i]
            sxx += x[i] * x[i]
            syy += y[i] * y[i]
            sxy += x[i] * y[i]
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
        b = (sy - a * sx) / N
        r = -(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
        return a, b, r


    def cal_multi_regression_beta(self, df, x_var_list):


        outf = self.multi_regression_result_f

        multi_derivative = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']
            y_vals = T.remove_np_nan(y_vals)
            if len(y_vals) == 0:
                continue

            y_vals_detrend = signal.detrend(y_vals)
            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:
                x_vals = row[x]

                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals = T.interp_nan(x_vals)
                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue
                x_vals_detrend = signal.detrend(x_vals) #detrend
                # df_new[x] = x_vals
                df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals_detrend

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
            linear_model = LinearRegression()

            linear_model.fit(df_new[x_var_list_valid_new], df_new['y'])
            # coef_ = np.array(linear_model.coef_) / y_mean
            coef_ = np.array(linear_model.coef_)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))
            # print(df_new['y'])
            # exit()
            multi_derivative[pix] = coef_dic
        T.save_npy(multi_derivative, outf)

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir,y_var):

        f=self.multi_regression_result_f

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:
            # print(pix)
            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in dic:
                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f'{multi_regression_result_dir}\\{var_i}_{y_var}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr,vmin=-5,vmax=5)

            plt.title(var_i)
            plt.colorbar()

        plt.show()

class selection():
    def __init__(self):
        pass
    def run(self):
        self.select_drying_wetting_trend()
        self.select_drought_event()
        pass
    def select_drying_wetting_trend(self):

        f_sm=data_root+'Extraction\\GLEAM_SMroot.npy'


        dic=T.load_npy(f_sm)
        result_dic={}
        result_tif_dic={}

        for pix in tqdm(dic):
            time_series=dic[pix]
            time_series=np.array(time_series)
            time_series[time_series<-999]=np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            a,b,r,p=T.nan_line_fit(np.arange(len(time_series)),time_series)
            if a>0 and p<0.05:
                result_dic[pix]='sig_wetting'
                result_tif_dic[pix]=2
            elif a<0 and p<0.05:
                result_dic[pix]='sig_drying'
                result_tif_dic[pix]=-2
            elif a>0 and p>0.05:
                result_dic[pix]='non_sig_wetting'
                result_tif_dic[pix]=1
            else:
                result_dic[pix]='non_sig_drying'
                result_tif_dic[pix]=-1
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(result_tif_dic,result_root+'\\SM_trend_label.tif')
        T.save_npy(result_dic,data_root+'Base_data\\GLEAM_SMroot_trend_label_mark.npy')

        pass

class pick_event():
    def __init__(self):
        pass

    def run(self):

        # self.pick_drought_event()
        # self.extract_variables_during_droughts_GS()
        # self.extract_variables_after_droughts_GS() ##### extract variables after droughts mean
        # self.extract_variables_after_droughts_GS_in_nth_year()  ### extract variables after droughts in nth year
        # self.multiregression_based_on_during_droughts()  ###
        # self.statistic_variables()
        # self.rename_variables()


        # self.plot_df()
        self.plt_spatial_df()

    def pick_drought_event(self):

        fdir = result_root+rf'detrend_zscore_Yang\\SPEI3\\'
        outdir = result_root + rf'pick_event_scheme2\\SPEI3\\'
        T.mk_dir(outdir, force=True)

        spatial_dic={}
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            dic_i=T.load_npy(fdir+f)
            spatial_dic.update(dic_i)

            threshold_upper=-2
            threshold_bottom=-3
            threshold_start=-1
            outf=outdir+f.split('.')[0]+f'_({threshold_bottom},{threshold_upper}).df'


            print(outf)
            event_dic={}
            for pix in spatial_dic:
                vals=spatial_dic[pix]

                drought_events_list_extreme, _=self.kernel_find_drought_period(vals,threshold_upper,threshold_bottom,threshold_start)
                if len(drought_events_list_extreme)==0:
                    continue
                event_dic[pix]=drought_events_list_extreme
            df=pd.DataFrame()
            pix_list=[]
            drought_range_list=[]
            for pix in event_dic:
                events_list=event_dic[pix]
                for event in events_list:
                    pix_list.append(pix)
                    drought_range_list.append(event)
            df['pix']=pix_list
            df['drought_range']=drought_range_list
            T.print_head_n(df)
            T.save_df(df,outf)
            self.__df_to_excel(df, outf)


        pass



    def kernel_find_drought_period(self, vals, threshold_upper, threshold_bottom, threshold_start):

        vals = np.array(vals)

        start_of_drought_list = []
        end_of_drought_list = []
        for i in range(len(vals)):
            if i + 1 == len(vals):
                break
            val_left = vals[i]
            vals_right = vals[i + 1]
            if val_left < threshold_start and vals_right > threshold_start:
                end_of_drought_list.append(i + 1)
            if val_left > threshold_start and vals_right < threshold_start:
                start_of_drought_list.append(i)

        drought_events_list = []
        for s in start_of_drought_list:
            for e in end_of_drought_list:
                if e > s:
                    drought_events_list.append((s, e))
                    break

        drought_events_list_extreme = []
        drought_timing_list = []
        for event in drought_events_list:
            s = event[0]
            e = event[1]
            min_index = T.pick_min_indx_from_1darray(vals, list(range(s, e)))
            drought_timing_month = min_index % 12 + 1
            min_val = vals[min_index]
            if min_val <  threshold_upper and min_val > threshold_bottom:
                drought_events_list_extreme.append(event)
                drought_timing_list.append(drought_timing_month)
        return drought_events_list_extreme, drought_timing_list

        pass
    def extract_variables_during_droughts_GS(self):
        fdir_drought = result_root + rf'pick_event_scheme2\\SPEI3\\'
        fdir_variables_all = result_root + rf'detrend_zscore_Yang\\'
        for f in os.listdir(fdir_drought):
            if not f.endswith('.df'):
                continue
            time_range= f.split('_')[0]+'_'+f.split('_')[1]
            threshold=f.split('_')[-1].split('.')[0]
            print(threshold)
            df= T.load_df(fdir_drought + f)
            df_new=pd.DataFrame()
            outdir = result_root + rf'pick_event_scheme2\\extract_variables_during_droughts_GS\\'
            T.mk_dir(outdir, force=True)
            # print(outdir)
            fvariable_list = ['Tmax', 'GLEAM_SMroot', 'LAI4g', 'NDVI4g']

            for fvariable in os.listdir(fdir_variables_all):
                if not fvariable in fvariable_list:
                    continue

                fvariable_path=fdir_variables_all+fvariable+'\\'+f'{time_range}.npy'
                print(fvariable_path)
                var_name=fvariable.split('.')[0]+'_'+threshold+'_'+time_range
                print(var_name)
                # exit()

                data_dict=T.load_npy(fvariable_path)
                # pix_list = T.get_df_unique_val_list(df, 'pix')

                mean_list = []
                for i, row in tqdm(df.iterrows(), total=len(df), ):
                    pix = row['pix']
                    GS = global_get_gs(pix)

                    drought_range = row['drought_range']
                    e,s = drought_range[1],drought_range[0]
                    picked_index = []
                    for idx in range(s,e+1):
                        mon = idx % 12 + 1
                        if not mon in GS:
                            continue

                        picked_index.append(idx)

                    if len(picked_index) == 0:
                        mean_list.append(np.nan)
                        continue
                    if not pix in data_dict:
                        mean_list.append(np.nan)
                        continue
                    vals = data_dict[pix]
                    # print(len(vals))
                    if picked_index[-1] >= len(vals):
                        mean_list.append(np.nan)
                        continue
                    picked_vals = T.pick_vals_from_1darray(vals,picked_index)
                    mean = np.nanmean(picked_vals)
                    if mean == 0:
                        mean_list.append(np.nan)
                        continue
                    mean_list.append(mean)

                df_new['pix'] = df['pix']
                df_new[f'{var_name}'] = mean_list
            T.print_head_n(df_new)
            return df_new

            # T.save_df(df_new, outdir + f'{time_range}_{threshold}.df')
            # self.__df_to_excel(df_new, outdir + f'{time_range}_{threshold}.df')

    def extract_variables_after_droughts_GS(self):
        n_list = [ 1, 2, 3, 4]
        fdir_drought = result_root + rf'pick_event_scheme2\\SPEI3\\'
        fdir_variables_all = result_root + rf'detrend_zscore_Yang\\'
        for f in os.listdir(fdir_drought):
            if not f.endswith('.df'):
                continue
            time_range = f.split('_')[0] + '_' + f.split('_')[1]
            threshold = f.split('_')[-1].split('.')[0]
            print(threshold)
            df = T.load_df(fdir_drought + f)
            df_new = pd.DataFrame()

            outdir = result_root + rf'pick_event_scheme2\\extract_variables_during_droughts_GS\\'
            T.mk_dir(outdir, force=True)
            # print(outdir)
            fvariable_list=['Tmax','GLEAM_SMroot','LAI4g','NDVI4g']

            for fvariable in os.listdir(fdir_variables_all):
                if not fvariable in fvariable_list:
                    continue
                fvariable_path = fdir_variables_all + fvariable + '\\' + f'{time_range}.npy'
                print(fvariable_path)
                var_name = fvariable.split('.')[0] + '_' + threshold + '_' + time_range
                print(var_name)

                data_dict = T.load_npy(fvariable_path)

            ################ add during drought year
                mean_list = []
                for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_0_{var_name}'):
                    pix = row['pix']
                    GS = global_get_gs(pix)
                    if not pix in data_dict:
                        mean_list.append(np.nan)
                        continue
                    vals = data_dict[pix]
                    drought_range = row['drought_range']

                    drought_range_index = list(range(drought_range[0], drought_range[-1] + 1))

                    picked_index = []
                    for idx in drought_range_index:
                        mon = idx % 12 + 1
                        if not mon in GS:
                            continue
                        if idx >= len(vals):
                            picked_index = []
                            break
                        picked_index.append(idx)
                    if len(picked_index) == 0:
                        mean_list.append(np.nan)
                        continue
                    picked_vals = T.pick_vals_from_1darray(vals, picked_index)
                    mean = np.nanmean(picked_vals)
                    if mean == 0:
                        mean_list.append(np.nan)
                        continue
                    mean_list.append(mean)
                df_new[f'{var_name}_post_0_GS'] = mean_list

                ################ add post drought year

                for n in n_list:

                    delta_mon = n*12
                    mean_list = []
                    for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_{n}_{var_name}'):
                        pix = row['pix']
                        GS = global_get_gs(pix)
                        if not pix in data_dict:
                            mean_list.append(np.nan)
                            continue
                        vals = data_dict[pix]
                        drought_range = row['drought_range']
                        end_mon = drought_range[-1]
                        post_drought_range = []
                        for m in range(delta_mon):
                            post_drought_range.append(end_mon + m + 1)
                        picked_index = []
                        for idx in post_drought_range:
                            mon = idx % 12 + 1
                            if not mon in GS:
                                continue
                            if idx >= len(vals):
                                picked_index = []
                                break
                            picked_index.append(idx)
                        if len(picked_index) == 0:
                            mean_list.append(np.nan)
                            continue
                        picked_vals = T.pick_vals_from_1darray(vals, picked_index)
                        mean = np.nanmean(picked_vals)
                        if mean == 0:
                            mean_list.append(np.nan)
                            continue
                        mean_list.append(mean)
                    df_new[f'{var_name}_post_{n}_GS'] = mean_list



            df_new['pix'] = df['pix']

            T.print_head_n(df_new)


            T.save_df(df_new, outdir + f'{time_range}_{threshold}.df')
            self.__df_to_excel(df_new, outdir + f'{time_range}_{threshold}.df')

    def extract_variables_after_droughts_GS_in_nth_year(self):
        n_list = [1, 2, 3, 4]
        fdir_drought = result_root + rf'pick_event_scheme2\\SPEI3\\'
        fdir_variables_all = result_root + rf'detrend_zscore_Yang\\'
        for f in os.listdir(fdir_drought):
            if not f.endswith('.df'):
                continue
            time_range = f.split('_')[0] + '_' + f.split('_')[1]
            threshold = f.split('_')[-1].split('.')[0]
            print(threshold)
            df = T.load_df(fdir_drought + f)
            df_new = pd.DataFrame()

            outdir = result_root + rf'pick_event_scheme2\\extract_variables_after_droughts_GS_in_nth_year\\'
            outf = outdir + f'{time_range}_{threshold}.df'
            if os.path.exists(outf):
                continue
            T.mk_dir(outdir, force=True)
            # print(outdir)
            fvariable_list=['Tmax','GLEAM_SMroot','LAI4g','NDVI4g']

            for fvariable in os.listdir(fdir_variables_all):
                if not fvariable in fvariable_list:
                    continue
                fvariable_path = fdir_variables_all + fvariable + '\\' + f'{time_range}.npy'
                print(fvariable_path)
                var_name = fvariable.split('.')[0] + '_' + threshold + '_' + time_range
                print(var_name)

                data_dict = T.load_npy(fvariable_path)

                ################ add during drought year
                mean_list = []
                for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_0_{var_name}'):
                    pix = row['pix']
                    GS = global_get_gs(pix)
                    if not pix in data_dict:
                        mean_list.append(np.nan)
                        continue
                    vals = data_dict[pix]
                    drought_range = row['drought_range']

                    drought_range_index = list(range(drought_range[0], drought_range[-1] + 1))

                    picked_index = []
                    for idx in drought_range_index:
                        mon = idx % 12 + 1
                        if not mon in GS:
                            continue
                        if idx >= len(vals):
                            picked_index = []
                            break
                        picked_index.append(idx)
                    if len(picked_index) == 0:
                        mean_list.append(np.nan)
                        continue
                    picked_vals = T.pick_vals_from_1darray(vals, picked_index)
                    mean = np.nanmean(picked_vals)
                    if mean == 0:
                        mean_list.append(np.nan)
                        continue
                    mean_list.append(mean)
                df_new[f'{var_name}_post_0_GS'] = mean_list

                ################ add post nth year

                for n in n_list:

                    mean_list = []
                    for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_{n}_{var_name}'):
                        pix = row['pix']
                        GS = global_get_gs(pix)
                        if not pix in data_dict:
                            mean_list.append(np.nan)
                            continue
                        vals = data_dict[pix]
                        drought_range = row['drought_range']
                        end_mon = drought_range[-1]
                        post_drought_range = []
                        assert n>0
                        for m in range((n-1)*12,n*12):

                            post_drought_range.append(end_mon + m + 1)
                        picked_index = []
                        for idx in post_drought_range:
                            mon = idx % 12 + 1
                            if not mon in GS:
                                continue
                            if idx >= len(vals):
                                picked_index = []
                                break
                            picked_index.append(idx)
                        if len(picked_index) == 0:
                            mean_list.append(np.nan)
                            continue
                        picked_vals = T.pick_vals_from_1darray(vals, picked_index)
                        mean = np.nanmean(picked_vals)
                        if mean == 0:
                            mean_list.append(np.nan)
                            continue
                        mean_list.append(mean)
                    df_new[f'{var_name}_post_{n}_GS'] = mean_list


            df_new['pix'] = df['pix']

            T.print_head_n(df_new)


            T.save_df(df_new, outdir + f'{time_range}_{threshold}.df')
            self.__df_to_excel(df_new, outdir + f'{time_range}_{threshold}.df')
    def statistic_variables(self):
        time_range = ['1982_2000', '2001_2020']
        # threshold = '(-4,-3)'
        threshold = '(-3,-2)'
        threshold = '(-2,-1)'


        fdir = result_root + rf'pick_event_scheme2\\extract_variables_after_droughts_GS\\'
        variable_list=['GLEAM_SMroot','LAI4g','NDVI4g']
        for time_range in time_range:
            plt.figure(figsize=(10, 5))
            flag = 1


            f_path = fdir + f'{time_range}_{threshold}.df'

            df = T.load_df(f_path)

            n_list = [0, 1, 2, 3, 4]

            for variable in variable_list:
                plt.subplot(1, 3, flag)
                average_list = []
                for n in n_list:
                    vals=df[f'{variable}_{threshold}_{time_range}_post_{n}_GS'].tolist()
                    average=np.nanmean(vals)
                    average_list.append(average)
                plt.bar(n_list,average_list,label=variable)
                flag = flag + 1
                plt.legend()
                plt.ylim(-0.6, 0.3)
                plt.title(f'{variable}_{threshold}_{time_range}')
                plt.xticks(n_list, [f'post_{n}' for n in n_list])
                plt.xticks(rotation=45)
            plt.show()



    def multiregression_based_on_during_droughts(self):   ## LAI/SM
        time_range=['1982_2000','2001_2020']
        threshold='(-4,-3)'
        plt.figure(figsize=(10,5))
        flag=1

        fdir=result_root+rf'pick_event_scheme2\\extract_variables_after_droughts_GS\\'
        for time_range in time_range:
            plt.subplot(1,2,flag)

            f_path=fdir+f'{time_range}_{threshold}.df'

            df=T.load_df(f_path)

            n_list=[0,1,2,3,4]

            bar_list={}

            for n in n_list:
                outf = result_root + rf'multi_regression\\LAI_SMroot_post_{n}_GS.npy'
                df_new = pd.DataFrame()

                # x_var_list=[f'Tmax_{threshold}_{time_range}_post_{n}_GS',f'GLEAM_SMroot_{threshold}_{time_range}_post_{n}_GS']
                # y_vals = df[f'LAI4g_{threshold}_{time_range}_post_{n}_GS']

                x_var_list=[f'LAI4g_{threshold}_{time_range}_post_{n}_GS',f'Tmax_{threshold}_{time_range}_post_{n}_GS']
                y_vals = df[f'GLEAM_SMroot_{threshold}_{time_range}_post_{n}_GS']



                df_new['y'] = y_vals
                for x_var in x_var_list:
                    x_vals=df[x_var]
                    df_new[x_var]=x_vals


                ## remove nan
                df_new = df_new.dropna()

                # T.print_head_n(df_new)

                linear_model = LinearRegression()

                linear_model.fit(df_new[x_var_list], df_new['y'])
                coef_ = np.array(linear_model.coef_)
                coef_dic = dict(zip(x_var_list, coef_))
                bar_list[n]=coef_dic

            SM_corr_list=[]
            for n in bar_list:
                SM_corr=bar_list[n][f'GLEAM_SMroot_{threshold}_{time_range}_post_{n}_GS']
                SM_corr = bar_list[n][f'LAI4g_{threshold}_{time_range}_post_{n}_GS']
                SM_corr_list.append(SM_corr)
            plt.bar(range(len(SM_corr_list)),SM_corr_list)
            plt.xticks(range(len(SM_corr_list)), [f'post_{n}' for n in n_list])
            plt.ylim(0,0.6)
            plt.title(f'delta LAI/delta GLEAM_SMroot_{threshold}_{time_range}')

            flag=flag+1

        plt.show()




            # plt.bar(bar_list[n].keys(),bar_list[n].values(),label=f'post_{n}')
        # plt.legend()
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # plt.show()

















        pass


    def concat_df(self):
        fdir = result_root + rf'pick_event\\extract_variables_during_droughts_GS\\'

        df_list=[]
        for f in os.listdir(fdir):
            if not f.endswith('.df'):
                continue
            df=T.load_df(fdir+f)
            df_list.append(df)
        df=pd.concat(df_list,axis=0)

        T.print_head_n(df)
        T.save_df(df, result_root + rf'pick_event\\extract_variables_during_droughts_GS\\concat_df.df')
        self.__df_to_excel(df, result_root + rf'pick_event\\extract_variables_during_droughts_GS\\concat_df.df')

    # pd.concat([df,df1],axis=1)

    def __df_to_excel(self, df, dff, n=1000, random=False):
        dff = dff.split('.')[0]
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass
    def plot_df(self):
        f=result_root+'pick_event\extract_variables_during_droughts_GS\\concat_df.df'
        df=T.load_df(f)
        time_range_list=['1982_2000','2001_2020']
        varname='LAI4g'

        threshold=[-4,-3,-2,-1]

        val_mean_dic = {}
        label_list = []


        for th in threshold:

            for time_range in time_range_list:

                th1=th
                th2=th+1
                if th2>threshold[-1]:
                    continue

                threshold_str=f'({th1},{th2})'

                column_name=f'{varname}_{threshold_str}_{time_range}'

                print(column_name)
                vals=df[column_name]
                vals=T.remove_np_nan(vals)
                vals=np.array(vals)
                val_mean=np.nanmean(vals)
                val_mean_dic[column_name]=val_mean
                label_list.append(join(column_name.split('_')[1],column_name.split('_')[2]))
        plt.bar(range(len(val_mean_dic)),val_mean_dic.values(),tick_label=label_list)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plt_spatial_df(self):


        flag=1
        plt.figure(figsize=(10, 5))

        for n in [0,1,2,3]:
            arr_list = []
            plt.subplot(2, 2, flag)

            for period in ['1982_2000','2001_2020']:

                f=result_root+f'pick_event_scheme2\extract_variables_after_droughts_GS\\{period}_(-2,-1).df'
                f = result_root + f'pick_event_scheme2\extract_variables_after_droughts_GS_in_nth_year\\{period}_(-2,-1).df'
                df=T.load_df(f)
                print(df)
                # col_name=rf'NDVI4g_(-2,-1)_{period}_post_{n}_GS'
                col_name = rf'NDVI4g_(-2,-1)_{period}_post_{n}_GS'

                dic={}
                dic_group=T.df_groupby(df,'pix')
                for pix in dic_group:
                    df_i=dic_group[pix]
                    val=df_i[col_name].tolist()
                    mean=np.nanmean(val)
                    dic[pix]=mean
                arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(dic)
                arr_list.append(arr)
            arr_difference=(arr_list[1]-arr_list[0])
                # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr,result_root+'pick_event\extract_variables_during_droughts_GS\\1982_2000_(-3,-2).tif')
            plt.imshow(arr_difference,vmin=-0.5,vmax=0.5,cmap='RdBu',interpolation='nearest')
            plt.colorbar()
            title=f'NDVI4g_(-2,-1)_post_{n}_GS'
            plt.title(title)
            flag=flag+1
        plt.show()

    def rename_variables(self):
        fdir = result_root + rf'pick_event_scheme2\\extract_variables_after_droughts_GS_in_nth_year_rename\\'
        outdir = result_root + rf'pick_event_scheme2\\extract_variables_after_droughts_GS_in_nth_year_rename2\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not f.endswith('.df'):
                continue
            df=T.load_df(fdir+f)
            rename_dic={}
            for col in df.columns:
                if '_nth' in col:
                    rename_dic[col]=col.replace('_nth','')
            df=df.rename(columns=rename_dic)
            T.print_head_n(df)
            T.save_df(df,outdir+f)
            self.__df_to_excel(df,outdir+f)


def global_get_gs(pix):
    r,c = pix
    if r > 360:
        return [11, 12, 1, 2, 3, 4]
    else:
        return [5,6, 7, 8, 9, 10]



class build_dataframe():


    def __init__(self):

        self.this_class_arr = result_root + 'Dataframe\zscore\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'zscore.df'

        pass

    def run(self):

        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        # df = self.add_detrend_zscore_to_df(df)
        # df=self.add_AI_classfication(df)
        df=self.add_SM_trend_label(df)

        #
        # df = self.add_row(df)


        # df = self.add_GLC_landcover_data_to_df(df)

        # df=self.__rename_dataframe_columns(df)
        # df=self.show_field(df)
        # df = self.drop_field_df(df)

        T.save_df(df, self.dff)

        self.__df_to_excel(df, self.dff)

    def __gen_df_init(self, file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df = self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df(self, file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __df_to_excel(self, df, dff, n=1000, random=False):
        dff = dff.split('.')[0]
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass

    def foo1(self, df):

        f = result_root + rf'zscore\LAI4g.npy'
        dic = {}
        outf = self.dff
        result_dic = {}
        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []
        f_name = f.split('.')[0]
        print(f_name)

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 1982
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y + 1)
                y = y + 1

        df['pix'] = pix_list
        df[f_name] = change_rate_list
        df['year'] = year
        # T.save_df(df, outf)
        # df = df.head(1000)
        # df.to_excel(outf+'.xlsx')
        return df

    def foo2(self, df):  # 新建trend

        f = 'zscore\daily_Y\peak/during_peak_LAI3g_zscore.npy'
        val_array = np.load(f)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)

        # exit()

        pix_list = []
        for pix in tqdm(val_dic):
            pix_list.append(pix)
        df['pix'] = pix_list

        return df

    def add_detrend_zscore_to_df(self, df):
        fdir = result_root + rf'zscore\\'

        for variable in ['LAI4g','NDVI4g','GPP_CFE','GPP_baseline']:


            f=fdir+variable+'.npy'

            NDVI_dic = T.load_npy(f)


            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in NDVI_dic:
                    NDVI_list.append(np.nan)
                    continue

                vals = NDVI_dic[pix]
                print(len(vals))
                # if len(vals) != 20:
                #     NDVI_list.append(np.nan)
                #     continue
                try:
                    v1 = vals[year - 1981]
                    NDVI_list.append(v1)
                except:
                    NDVI_list.append(np.nan)

            df[variable] = NDVI_list
        return df

    def add_row(self, df):
        r_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix
            r_list.append(r)
        df['row'] = r_list
        return df

    def add_max_trend_to_df(self, df):

        fdir = data_root + rf'/Base_data/lc_trend/'
        for f in (os.listdir(fdir)):
            # print()
            if not 'max_trend' in f:
                continue
            if not f.endswith('.npy'):
                continue
            if 'p_value' in f:
                continue

            val_array = np.load(fdir + f)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                val = val * 20
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name] = val_list

        return df

    def add_NDVI_mask(self, df):
        f = data_root + rf'/Base_data/NDVI_mask.tif'

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def add_GLC_landcover_data_to_df(self, df):

        f = data_root + rf'\Base_data\LC_reclass2.npy'

        val_dic = T.load_npy(f)

        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)

        df['landcover_GLC'] = val_list
        return df
    def add_AI_classfication(self, df):

        f = data_root + rf'\\Base_data\dryland_AI.tif\\dryland_classfication.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)


        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            if val==0:
                label='Arid'
            elif val==1:
                label='Semi-Arid'
            else:
                label='Sub-Humid'

            val_list.append(label)

        df['AI_classfication'] = val_list
        return df
    def add_SM_trend_label(self, df):

        f = data_root + rf'\\Base_data\GLEAM_SMroot_trend_label_mark.npy'


        val_dic = T.load_npy(f)


        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]

            val_list.append(val)

        df['wetting_drying_trend'] = val_list
        return df


class plot_dataframe():
    def __init__(self):
        pass
    def run(self):

        # self.plot_annual_zscore_based_region()
        self.plot_annual_zscore_based_trend()
        pass

    def plot_annual_zscore_based_region(self):   #based on semi-arid, arid and sub-humid
        df= T.load_df(result_root + 'Dataframe\zscore\zscore.df')

        product_list = ['LAI4g','NDVI4g','GPP_CFE','GPP_baseline']

        fig = plt.figure()
        i = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:

            ax = fig.add_subplot(2, 2, i)

            flag = 0
            color_list=['blue','green','red','orange']

            for variable in product_list:

                colunm_name = variable
                df_region = df[df['AI_classfication'] == region]
                mean_value_yearly, up_list, bottom_list, fit_value_yearly, k_value, p_value = self.calculation_annual_average(df_region, colunm_name)
                xaxis = range(len(mean_value_yearly))
                xaxis = list(xaxis)

                ax.plot(xaxis, mean_value_yearly, label=variable, color=color_list[flag])
                # ax.plot(xaxis, fit_value_yearly, label='k={:0.2f},p={:0.4f}'.format(k_value, p_value), linestyle='--')

                # print(f'{region}_{variable}', 'k={:0.2f},p={:0.4f}'.format(k_value, p_value))
                flag = flag + 1


            plt.legend()
            plt.xlabel('year')
            plt.title(f'{region}')
            # create xticks

            yearlist = list(range(1982, 2021))
            yearlist_str = [int(i) for i in yearlist]
            ax.set_xticks(xaxis[::5])
            ax.set_xticklabels(yearlist_str[::5], rotation=45)


            major_yticks = np.arange(-1.1, 1)
            ax.set_yticks(major_yticks)

            plt.grid(which='major', alpha=0.5)
            plt.tight_layout()
            i = i + 1

        plt.show()


    def plot_annual_zscore_based_trend(self):  ##based on wetting, drying, significant wetting and drylng trend
        df= T.load_df(result_root + 'Dataframe\zscore\zscore.df')

        product_list = ['LAI4g','NDVI4g','GPP_CFE','GPP_baseline']

        fig = plt.figure()
        i = 1

        for region in ['sig_wetting', 'sig_drying', 'non_sig_wetting', 'non_sig_drying']:

            ax = fig.add_subplot(2, 2, i)

            flag = 0
            color_list=['blue','green','red','orange']

            for variable in product_list:

                colunm_name = variable
                df_region = df[df['wetting_drying_trend'] == region]
                mean_value_yearly, up_list, bottom_list, fit_value_yearly, k_value, p_value = self.calculation_annual_average(df_region, colunm_name)
                xaxis = range(len(mean_value_yearly))
                xaxis = list(xaxis)

                ax.plot(xaxis, mean_value_yearly, label=variable, color=color_list[flag])
                # ax.plot(xaxis, fit_value_yearly, label='k={:0.2f},p={:0.4f}'.format(k_value, p_value), linestyle='--')

                # print(f'{region}_{variable}', 'k={:0.2f},p={:0.4f}'.format(k_value, p_value))
                flag = flag + 1


            plt.legend()
            plt.xlabel('year')
            plt.title(f'{region}')
            # create xticks

            yearlist = list(range(1982, 2021))
            yearlist_str = [int(i) for i in yearlist]
            ax.set_xticks(xaxis[::5])
            ax.set_xticklabels(yearlist_str[::5], rotation=45)


            major_yticks = np.arange(-1.1, 1)
            ax.set_yticks(major_yticks)

            plt.grid(which='major', alpha=0.5)
            plt.tight_layout()
            i = i + 1

        plt.show()

    def calculation_annual_average(self,df,column_name):
        dic = {}
        mean_val = {}
        confidence_value = {}
        std_val = {}
        # year_list = df['year'].to_list()
        # year_list = set(year_list)  # 取唯一
        # year_list = list(year_list)
        # year_list.sort()

        year_list = []
        for i in range(1982, 2021):
            year_list.append(i)
        print(year_list)

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []
            confidence_value[year] = []

        for year in year_list:
            df_pick = df[df['year'] == year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row[column_name]
                dic[year].append(val)
            val_list = np.array(dic[year])
            # val_list[val_list>1000]=np.nan

            n = len(val_list)
            mean_val_i = np.nanmean(val_list)
            std_val_i = np.nanstd(val_list)
            se = stats.sem(val_list)
            h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
            confidence_value[year] = h
            mean_val[year] = mean_val_i
            std_val[year] = std_val_i

        # a, b, r = KDE_plot().linefit(xaxis, val)
        mean_val_list = []  # mean_val_list=下面的mean_value_yearly

        for year in year_list:
            mean_val_list.append(mean_val[year])
        xaxis = range(len(mean_val_list))
        xaxis = list(xaxis)
        print(len(mean_val_list))
        # r, p_value = stats.pearsonr(xaxis, mean_val_list)
        # k_value, b_value = np.polyfit(xaxis, mean_val_list, 1)
        k_value, b_value, r, p_value = T.nan_line_fit(xaxis, mean_val_list)
        print(k_value)

        mean_value_yearly = []
        up_list = []
        bottom_list = []
        fit_value_yearly = []
        p_value_yearly = []

        for year in year_list:
            mean_value_yearly.append(mean_val[year])
            # up_list.append(mean_val[year] + confidence_value[year])
            # bottom_list.append(mean_val[year] - confidence_value[year])
            up_list.append(mean_val[year] + 0.125 * std_val[year])
            bottom_list.append(mean_val[year] - 0.125 * std_val[year])

            fit_value_yearly.append(k_value * (year - year_list[0]) + b_value)



        return mean_value_yearly, up_list, bottom_list, fit_value_yearly, k_value, p_value
        # exit()

class check_data():
    def run (self):
        self.plot_sptial()
        # self.plot_time_series()
        pass
    def plot_sptial(self):

        f= result_root+ rf'detrend_zscore_Yang\GLEAM_SMroot\\1982_2000.npy'
        # f = data_root + rf'split\NDVI4g\2001_2020.npy'
        dic=T.load_npy(f)

        len_dic={}
        for pix in dic:
            vals=dic[pix]

            # len_dic[pix]=np.nanmean(vals)
            len_dic[pix] = len(vals)
        arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(len_dic)
        plt.imshow(arr)
        plt.colorbar()
        plt.title('SPEI')
        plt.show()
    def plot_time_series(self):
        f=data_root+rf'split\LAI4g\1982_2000.npy'
        # f= result_root+ rf'detrend_zscore_Yang\LAI4g\\1982_2000.npy'
        dic=T.load_npy(f)
        for pix in dic:
            vals=dic[pix]
            vals=np.array(vals)
            # if not len(vals)==19*12:
            #     continue
            # if True in np.isnan(vals):
            #     continue
            # print(len(vals))
            if np.isnan(np.nanmean(vals)):
                continue
            plt.plot(vals)
            plt.show()


class Dataframe_func:

    def __init__(self,df,is_clean_df=True):
        print('add lon lat')
        df = self.add_lon_lat(df)

        print('add NDVI mask')
        # df = self.add_NDVI_mask(df)

        # if is_clean_df == True:
        #     df = self.clean_df(df)

        # print('add landcover')
        # df = self.add_GLC_landcover_data_to_df(df)

        print('add Aridity Index')
        df = self.add_AI_to_df(df)


        print('add AI_reclass')
        df = self.AI_reclass(df)


        self.df = df

    def clean_df(self,df):

        df = df[df['lat']>=30]
        # df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['NDVI_MASK'] == 1]
        # df = df[df['ELI_significance'] == 1]
        return df

    def add_GLC_landcover_data_to_df(self, df):
        f = join(data_root,'GLC2000/reclass_lc_dic.npy')
        val_dic=T.load_npy(f)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)
        df['landcover_GLC'] = val_list
        return df

    def add_NDVI_mask(self,df):
        # f =rf'C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        f = join(data_root, 'Base_data', 'NDVI_mask.tif')
        print(f)

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def add_AI_to_df(self, df):
        f = join(data_root, 'Base_data/Aridity_Index/aridity_index.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'HI_class')
        return df

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        return df


    def AI_reclass(self,df):
        AI_class = []
        for i,row in df.iterrows():
            AI = row['HI_class']
            if AI < 0.65:
                AI_class.append('Dryland')
            elif AI >= 0.65:
                AI_class.append('Humid')
            elif np.isnan(AI):
                AI_class.append(np.nan)
            else:
                print(AI)
                raise ValueError('AI error')
        df['HI_class'] = AI_class
        return df





def main():
    # data_processing().run()
    # statistic_analysis().run()
    pick_event().run()
    # selection().run()
    # multi_regression().run()
    # moving_window().run()
    # multi_regression_window().run()
    # build_dataframe().run()
    # plot_dataframe().run()
    # check_data().run()



    pass

if __name__ == '__main__':
    main()