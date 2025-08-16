# coding='utf-8'
import sys


import lytools
import pingouin
import pingouin as pg
import xymap
from matplotlib.pyplot import xticks
from numba.core.compiler_machinery import pass_info
from numba.cuda.libdevice import fdiv_rd
from openpyxl.styles.builtins import percent, total
# from green_driver_trend_contribution import *
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import t
from statsmodels.sandbox.regression.gmm import results_class_dict
from sympy.abc import alpha

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
result_root = 'D:/Project3/Result/'




class multi_regression_beta():
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'

        self.fdirX=self.result_root+rf'3mm\moving_window_multi_regression\moving_window\window_detrend_growing_season\\'
        self.fdir_Y=self.result_root+rf'\3mm\moving_window_multi_regression\moving_window\window_detrend_growing_season\\'

        self.xvar_list = ['sum_rainfall_detrend','Tmax_detrend','VPD_detrend']
        self.y_var = ['GLOBMAP_LAI_relative_change_detrend']


    def run(self):
        # self.anomaly()
        # self.detrend()
        # self.moving_window_extraction()

        self.window = 38-15+1
        outdir = self.result_root + rf'3mm\moving_window_multi_regression\multiresult_relative_change_detrend\multi_regression_result_detrend_growing_season_GLOBMAP_LAI\\'
        T.mk_dir(outdir, force=True)

        # # ####step 1 build dataframe
        for i in range(self.window):

            df_i = self.build_df(self.fdirX, self.fdir_Y, self.xvar_list, self.y_var,i)
            outf= outdir+rf'\\window{i:02d}.npy'
            if os.path.isfile(outf):
                continue
            print(outf)
        # #
            self.cal_multi_regression_beta(df_i,self.xvar_list, outf)  # 修改参数
        ##step 2 crate individial files
        self.plt_multi_regression_result(outdir,self.y_var)
# #
        # ##step 3 covert to time series

        self.convert_files_to_time_series(outdir,self.y_var) ## 这里乘以100
        ### step 4 build dataframe using build Dataframe function and then plot here
        # self.plot_moving_window_time_series() not use
        ## spatial trends of sensitivity
        self.calculate_trend_trend(outdir)
        # self.composite_beta()
        # plot robinson
        # self.plot_robinson()
        # self.plot_sensitivity_preicipation_trend()

    def anomaly(self):  ### anomaly GS

        fdir = rf'E:\Project3\Result\3mm\moving_window_multi_regression\original\\'

        outdir = rf'E:\Project3\Result\3mm\moving_window_multi_regression\\anomaly_ecosystem_year\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)
            # if os.path.isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')

            dic = np.load(fdir + f, allow_pickle=True, ).item()

            anomaly_dic = {}

            for pix in tqdm(dic):

                r, c = pix

                time_series = dic[pix]['ecosystem_year']
                print(len(time_series))

                time_series = np.array(time_series)

                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                # plt.plot(time_series)
                # plt.show()

                mean = np.nanmean(time_series)

                delta_time_series = (time_series - mean)

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

        fdir=result_root + rf'\3mm\moving_window_multi_regression\anomaly_ecosystem_year\\selected_variables\\'
        outdir=result_root + rf'\3mm\moving_window_multi_regression\anomaly_ecosystem_year\\selected_variables\\detrend\\'
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
                detrend_delta_time_series = signal.detrend(time_series)+np.nanmean(time_series)
                # plt.plot(time_series)
                # plt.plot(detrend_delta_time_series)
                # plt.show()

                detrend_zscore_dic[pix] = detrend_delta_time_series

            np.save(outf, detrend_zscore_dic)

    def moving_window_extraction(self):

        fdir_all = self.result_root + rf'3mm\moving_window_multi_regression\anomaly_ecosystem_year\selected_variables\detrend\\'

        outdir = self.result_root  + rf'\3mm\moving_window_multi_regression\moving_window\window_detrend_ecosystem_year\\'
        T.mk_dir(outdir, force=True)
        # outdir = self.result_root + rf'\3mm\extract_LAI4g_phenology_year\moving_window_extraction\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir_all):

            if not f.endswith('.npy'):
                continue

            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)

            # if os.path.isfile(outf):
            #     continue
            # if os.path.isfile(outf):
            #     continue

            dic = T.load_npy(fdir_all + f)
            window = 15

            new_x_extraction_by_window = {}
            for pix in tqdm(dic):

                time_series = dic[pix]
                # time_series = dic[pix]

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
            if i + window >= len(x)+1:  ####revise  here!!
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




    def build_df(self, fdir_X, fdir_Y, xvar_list,y_var,w):

        df = pd.DataFrame()
        dic_y=T.load_npy(fdir_Y+y_var[0]+'.npy')
        pix_list = []
        y_val_list=[]

        for pix in dic_y:
            r,c= pix


            if len(dic_y[pix]) == 0:
                continue
            vals = dic_y[pix][w]
            # print(vals)
            # exit()
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            vals = np.array(vals,dtype=float)


            vals[vals>999.0] = np.nan
            vals[vals<-999.0] = np.nan

            pix_list.append(pix)
            y_val_list.append(vals)

        df['pix'] = pix_list
        df['y'] = y_val_list

        ##df histogram



        # build x

        for xvar in xvar_list:


            x_val_list = []
            x_arr = T.load_npy(fdir_X+xvar+'.npy')
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in x_arr:
                    x_val_list.append([])
                    continue
                # print(len(x_arr[pix]))
                if len(x_arr[pix]) < self.window:
                    x_val_list.append([])
                    continue
                vals = x_arr[pix][w]
                vals = np.array(vals)
                vals = np.array(vals, dtype=float)
                vals[vals > 999] = np.nan
                vals[vals < -999] = np.nan
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
            r,c=pix

            y_vals = row['y']
            y_vals[y_vals<-999]=np.nan
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
            # print(df_new['y'])

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

    def plt_multi_regression_result(self, multi_regression_result_dir,y_var):
        fdir = multi_regression_result_dir
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if 'pvalue' in f:
                continue
            print(f)

            w=f.split('\\')[-1].split('.')[0][-2:]


            w=int(w)

            dic = T.load_npy(fdir+f)
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
                arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
                outdir=fdir+'TIFF\\'
                T.mk_dir(outdir)
                outf=outdir+f.replace('.npy','')

                DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, outf + f'_{var_i}.tif')
                std = np.nanstd(arr)
                mean = np.nanmean(arr)
                vmin = mean - std
                vmax = mean + std
                # plt.figure()
                # arr[arr > 0.1] = 1
                # plt.imshow(arr,vmin=-0.5,vmax=0.5)
                #
                # plt.title(var_i)
                # plt.colorbar()

            # plt.show()
    def convert_files_to_time_series(self, multi_regression_result_dir,y_var):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        # average_LAI_f = self.result_root + rf'state_variables\LAI4g_1982_2020.npy'
        # average_LAI_dic = T.load_npy(average_LAI_f)  ### normalized Co2 effect


        fdir = multi_regression_result_dir+'\\'+'TIFF\\'



        variable_list = ['sum_rainfall_detrend']



        for variable in variable_list:
            array_list = []

            for f in os.listdir(fdir):

                if not variable in f:
                    continue
                if not f.endswith('.tif'):
                    continue
                if 'pvalue' in f:
                    continue
                print(f)

                array= ToRaster().raster2array(fdir+f)[0]
                array = np.array(array)


                array_list.append(array)
            array_list=np.array(array_list)

            ## array_list to dic
            dic=DIC_and_TIF(pixelsize=0.5).void_spatial_dic()
            result_dic = {}
            for pix in dic:
                r, c = pix

                if r < 60:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue


                dic[pix]=array_list[:,r,c] ## extract time series




                time_series=dic[pix]
                time_series = np.array(time_series)
                time_series = time_series*100  ###currently no multiply %/100mm
                result_dic[pix]=time_series
                if np.nanmean(dic[pix])<=5:
                    continue
                # print(len(dic[pix]))
                # exit()
            outdir=multi_regression_result_dir+'\\'+'npy_time_series\\'
            print(outdir)
            # exit()
            T.mk_dir(outdir,force=True)
            outf=outdir+rf'\\{variable}.npy'
            np.save(outf,result_dic)

        pass

    def plot_moving_window_time_series(self):
        df= T.load_df(result_root + rf'\3mm\Dataframe\moving_window_multi_regression\\phenology_LAI_mean_trend.df')

        # variable_list = ['precip_detrend','rainfall_frenquency_detrend']
        variable_list = ['precip', 'rainfall_frenquency','rainfall_seasonality_all_year','rainfall_intensity']

        df=df.dropna()
        df=self.df_clean(df)

        fig = plt.figure()
        i = 1

        for variable in variable_list:

            ax = fig.add_subplot(2, 2, i)

            vals = df[f'{variable}'].tolist()

            vals_nonnan = []

            for val in vals:
                if type(val) == float:  ## only screening
                    continue
                if np.isnan(np.nanmean(val)):
                    continue
                if np.nanmean(val) <=-999:
                    continue

                vals_nonnan.append(val)
            ###### calculate mean
            vals_mean = np.array(vals_nonnan)  ## axis=0, mean of each row  竖着加
            vals_mean = np.nanmean(vals_mean, axis=0)
            vals_mean = vals_mean.tolist()
            plt.plot(vals_mean, label=variable)

            i = i + 1

        plt.xlabel('year')

        plt.ylabel(f'{variable}_LAI4g')
        # plt.legend()

        plt.show()
    def calculate_trend_trend(self,outdir):  ## calculate the trend of trend

    ## here input is the npy file
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=outdir+'\\npy_time_series\\'
        outdir_trend=outdir+'\\trend\\'
        T.mk_dir(outdir_trend,force=True)




        for f in os.listdir(fdir):
            if not f.endswith('npy'):
                continue

            if 'p_value' in f:
                continue


            dic = T.load_npy(fdir + f)

            outf = outdir + f.split('.')[0] + f'_trend.npy'
            print(outf)



            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):

                time_series_all = dic[pix]
                dryland_value=dic_dryland_mask[pix]
                if np.isnan(dryland_value):
                    continue
                time_series_all = np.array(time_series_all)

                if len(time_series_all) < 24:
                    continue
                time_series_all[time_series_all < -999] = np.nan

                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series_all)), time_series_all)

                trend_dic[pix]=slope
                p_value_dic[pix]=p_value

            arr_trend=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(trend_dic)
            arr_p_value = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(p_value_dic)
            # plt.imshow(arr_trend)
            # plt.colorbar()
            # plt.show()
            outf = outdir_trend + f.split('.')[0] + f'_trend.tif'
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_trend,outf)
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_p_value, outf + '_p_value.tif')
                ## save
            # np.save(outf, trend_dic)
            # np.save(outf+'_p_value', p_value_dic)

            ##tiff

    def composite_beta(self):
        f_1=rf'D:\Project3\Result\3mm\moving_window_multi_regression\multiresult_zscore_detrend\SNU_LAI\npy_time_series\\beta.npy'
        f_2=rf'D:\Project3\Result\3mm\moving_window_multi_regression\multiresult_zscore_detrend\LAI4g\npy_time_series\beta.npy'
        f_3=rf'D:\Project3\Result\3mm\moving_window_multi_regression\multiresult_zscore_detrend\\GLOBALMAP\npy_time_series\\beta.npy'
        dic1=np.load(f_1,allow_pickle=True).item()
        dic2=np.load(f_2,allow_pickle=True).item()
        dic3=np.load(f_3,allow_pickle=True).item()
        average_dic= {}

        for pix in dic1:
            if not pix in dic2:
                continue
            if not pix in dic3:
                continue
            value1=dic1[pix]
            value2=dic2[pix]
            value3=dic3[pix]
            value1[value1<-999]=np.nan
            value2[value2<-999]=np.nan
            value3[value3<-999]=np.nan



            value1=np.array(value1)
            value2=np.array(value2)
            value3=np.array(value3)
            if np.isnan(np.nanmean(value1)) or np.isnan(np.nanmean(value2)) or np.isnan(np.nanmean(value3)):
                continue
            if len(value1)!=24 or len(value2)!=24 or len(value3)!=24:
                print(pix,len(value1),len(value2),len(value3))
                continue


            average_val=np.nanmean([value1,value2,value3],axis=0)
            # print(average_val)
            average_dic[pix]=average_val
            # plt.plot(value1,color='blue')
            # plt.plot(value2,color='green')
            # plt.plot(value3,color='orange')
            # plt.plot(average_val,color='red')
            # plt.legend(['GlOBMAP','SNU','LAI4g','average'])
            # plt.show()

        np.save(rf'D:\Project3\Result\3mm\moving_window_multi_regression\\multiresult_zscore_detrend\composite_LAI\\composite_LAI_beta_mean.npy',average_dic)



    def plot_robinson(self):

        fdir_trend = result_root+rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year\\\npy_time_series\trend\\'
        temp_root = result_root+rf'\3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year\\\npy_time_series\trend\\'
        outdir = result_root+rf'\3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year\\npy_time_series\\\trend_plot\\'
        T.mk_dir(outdir, force=True)
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
            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=-2, vmax=2, is_discrete=True, colormap_n=7, cmap='RdBu')

            Plot_Robinson().plot_Robinson_significance_scatter(m,p_value_f,temp_root,0.05, s=0.2, marker='.')
            plt.title(f'{fname}')
            # plt.show()
            outf = outdir + f+'1.pdf'
            plt.savefig(outf)
            plt.close()

    pass


    def df_clean(self,df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]
        df=df[df['LC_max']<10]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df
    def plot_significant_percentage_area(self):  ### insert bar plot for all spatial map to calculate percentage

        dff = result_root + rf'3mm\Dataframe\Trend\\Trend.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        ##plt histogram of LAI
        df=df[df['LAI4g_1983_2020_trend']<30]
        df=df[df['LAI4g_1983_2020_trend']>-30]

        vals_p_value = df['LAI4g_1983_2020_p_value'].values
        significant_browning_count = 0
        non_significant_browning_count = 0
        significant_greening_count = 0
        non_significant_greening_count = 0

        for i in range(len(vals_p_value)):
            if vals_p_value[i] < 0.05:
                if df['LAI4g_1983_2020_trend'].values[i] > 0:
                    significant_greening_count = significant_greening_count + 1
                else:
                    significant_browning_count = significant_browning_count + 1
            else:
                if df['LAI4g_1983_2020_trend'].values[i] > 0:
                    non_significant_browning_count = non_significant_browning_count + 1
                else:
                    non_significant_greening_count = non_significant_greening_count + 1
            ## plot bar
        ##calculate percentage
        significant_greening_percentage = significant_greening_count / len(vals_p_value)*100
        non_significant_greening_percentage = non_significant_greening_count / len(vals_p_value)*100
        significant_browning_percentage = significant_browning_count / len(vals_p_value)*100
        non_significant_browning_percentage = non_significant_browning_count / len(vals_p_value)*100

        count = [non_significant_browning_percentage,significant_browning_percentage, significant_greening_percentage,

                 non_significant_greening_percentage]
        print(count)
        labels = ['non_significant_browning','significant_browning', 'significant_greening',
                  'non_significant_greening']
        color_list=['navajowhite','chocolate','navy','lightblue',]
        ##gap = 0.1
        df_new=pd.DataFrame({'count':count})
        df_new_T=df_new.T


        df_new_T.plot.barh( stacked=True, color=color_list,legend=False,width=0.1,)
        ## add legend
        plt.legend(labels)

        plt.ylabel('Percentage (%)')
        plt.tight_layout()

        plt.show()

class Figure1():
    def __init__(self):
        self.map_width = 8.2 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run (self):
        # self.bivariate_map()
        # self.Aridity_CV()
        # self.Aridity_beta()
        # self.figure1f()

        # self.Figure1c_robinson()
        # self.heatmap_LAImin_max_CV_Figure1d()
        self.statistical_analysis()



    def bivariate_map(self):  ## figure 1  ## LAImin and LAImax bivariate
        import xymap


        fdir =result_root + rf'\3mm\extract_composite_phenology_year\trend\\'

        outdir =result_root + rf'\3mm\extract_composite_phenology_year\bivariate\\'

        T.mkdir(outdir)

        outtif = join(outdir,'CV_trend2.tif')
        # outtif = join(outdir, 'LAImin_LAImax.tif')

        # fpath1 = join(fdir,'composite_LAI_detrend_relative_change_min_trend.tif')
        fpath1 = join(fdir,'composite_LAI_CV_trend.tif')
        # fpath2 = join(fdir,'composite_LAI_detrend_relative_change_max_trend.tif')
        fpath2 = join(fdir,'composite_LAI_relative_change_mean_trend.tif')

        #1
        # tif1_label, tif2_label = 'LAImin_trend','LAImax_trend'
        #2
        tif1_label, tif2_label = 'LAI_CV_trend','LAI_relative_change_mean_trend'

        #1
        # min1, max1 = -1, 1
        # min2, max2 = -1, 1

        #2
        min1, max1 = -.3, .3
        min2, max2 = -.5, .5

        arr1 = ToRaster().raster2array(fpath1)[0]
        arr2 = ToRaster().raster2array(fpath2)[0]

        arr1[arr1<-9999] = np.nan
        arr2[arr2<-9999] = np.nan

        arr1_flattened = arr1.flatten()
        arr2_flattened = arr2.flatten()


        # plt.hist(arr1_flattened,bins=100)
        # plt.title('arr1')
        # plt.figure()
        # plt.hist(arr2_flattened,bins=100)
        # plt.title('arr2')
        # plt.show()

        # choice 1
        # upper_left_color = (193,92,156)
        # upper_right_color =(112, 196, 181)
        # lower_left_color = (237, 125, 49)
        # lower_right_color = (0, 0, 110)
        # center_color = (240, 240, 240)

        ## CV greening option

        upper_left_color = (194, 0, 120)
        upper_right_color = (0,170,237)
        lower_left_color = (233, 55, 43)
        # lower_right_color = (160, 108, 168)
        lower_right_color = (234, 233, 46)
        center_color = (240, 240, 240)


        xymap.Bivariate_plot_1(res = 7,
                         alpha = 255,
                         upper_left_color = upper_left_color, #
                         upper_right_color = upper_right_color, #
                         lower_left_color = lower_left_color, #
                         lower_right_color = lower_right_color, #
                         center_color = center_color).plot_bivariate(
                                                                    fpath1, fpath2,
                                                                    tif1_label, tif2_label,
                                                                    min1, max1,
                                                                    min2, max2,
                                                                    outtif,
                                                                    n_x = 5, n_y = 5
                                                                    )

        T.open_path_and_file(outdir)




    def RGBA_to_tif(self,blend_arr,outf,originX, originY, pixelWidth, pixelHeight):
        import PIL.Image as Image
        img = Image.fromarray(blend_arr.astype('uint8'), 'RGBA')
        img.save(outf)
        # define a projection and extent
        raster = gdal.Open(outf)
        geotransform = raster.GetGeoTransform()
        raster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outRasterSRS = osr.SpatialReference()
        projection = self.wkt_84()
        # outRasterSRS.ImportFromEPSG(4326)
        # outRasterSRS.ImportFromEPSG(projection)
        # raster.SetProjection(outRasterSRS.ExportToWkt())
        raster.SetProjection(projection)
        pass











    def Figure1_robinson(self):  # convert figure to robinson

        fdir_trend = result_root + rf'3mm\extract_composite_phenology_year\bivariate\\'
        temp_root = result_root + rf'\3mm\extract_composite_phenology_year\bivariate\\'
        outdir = result_root + rf'\3mm\extract_composite_phenology_year\\bivariate\\ROBINSON\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):
            if not 'CV_trend_test' in f:
                continue

            if not f.endswith('.tif'):
                continue

            fname = f.split('.')[0]

            fpath = fdir_trend + f
            outf=outdir + fname + '.tif'
            srcSRS=self.wkt_84()
            dstSRS=self.wkt_robinson()

            ToRaster().resample_reproj(fpath,outf, 50000, srcSRS=srcSRS, dstSRS=dstSRS)

            T.open_path_and_file(outdir)


    def wkt_robinson(self):
        wkt='''PROJCRS["World_Robinson",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["World_Robinson",
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
    ID["ESRI",54030]]
        '''
        return wkt


    def wkt_84(self):
        wkt = '''GEOGCRS["WGS 84",
    ENSEMBLE["World Geodetic System 1984 ensemble",
        MEMBER["World Geodetic System 1984 (Transit)"],
        MEMBER["World Geodetic System 1984 (G730)"],
        MEMBER["World Geodetic System 1984 (G873)"],
        MEMBER["World Geodetic System 1984 (G1150)"],
        MEMBER["World Geodetic System 1984 (G1674)"],
        MEMBER["World Geodetic System 1984 (G1762)"],
        MEMBER["World Geodetic System 1984 (G2139)"],
        ELLIPSOID["WGS 84",6378137,298.257223563,
            LENGTHUNIT["metre",1]],
        ENSEMBLEACCURACY[2.0]],
    PRIMEM["Greenwich",0,
        ANGLEUNIT["degree",0.0174532925199433]],
    CS[ellipsoidal,2],
        AXIS["geodetic latitude (Lat)",north,
            ORDER[1],
            ANGLEUNIT["degree",0.0174532925199433]],
        AXIS["geodetic longitude (Lon)",east,
            ORDER[2],
            ANGLEUNIT["degree",0.0174532925199433]],
    USAGE[
        SCOPE["Horizontal component of 3D system."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["EPSG",4326]]'''
        return wkt

    def statistical_analysis(self):  # ## calculating percentage

        dff=rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\\Trend_all.df'
        df=T.load_df(dff)
        df=self.df_clean(df)


        T.print_head_n(df)
        x_var = 'composite_LAI_detrend_relative_change_min_trend'
        y_var = 'composite_LAI_detrend_relative_change_max_trend'
        ## x_var >0 and y_var >0==1;  2 x_var>0 and y_var<0 3 x_var<0 and y_var<0 4 x_var<0 and y_var>0
        result_list=[]
        label_list=[]


        df_pos_pos=df[(df[x_var]>0)&(df[y_var]>0)]

        df_pos_neg=df[(df[x_var]>0)&(df[y_var]<0)]
        df_neg_pos=df[(df[x_var]<0)&(df[y_var]>0)]
        df_neg_neg=df[(df[x_var]<0)&(df[y_var]<0)]
        percentage_pos_pos=len(df_pos_pos)/len(df)*100
        result_list.append(percentage_pos_pos)
        label_list.append('++')
        percentage_pos_neg=len(df_pos_neg)/len(df)*100
        result_list.append(percentage_pos_neg)
        label_list.append('+-')
        percentage_neg_pos=len(df_neg_pos)/len(df)*100
        result_list.append(percentage_neg_pos)
        label_list.append('-+')
        percentage_neg_neg=len(df_neg_neg)/len(df)*100
        result_list.append(percentage_neg_neg)
        label_list.append('--')

        # upper_left_color = (193,92,156)
        # upper_right_color =(112, 196, 181)
        # lower_left_color = (237, 125, 49)
        # lower_right_color = (0, 0, 110)
        # center_color = (240, 240, 240)
        color_list=[   (112, 196, 181),
            (0, 0, 110),

            (193, 92, 156),

                    (237, 125, 49)]
        ## rgb_to_hex
        print(result_list);exit()
        color_list = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in color_list]
        fig = plt.figure(figsize=(3, 3))


        for i in range(len(result_list)):
            plt.bar(label_list[i],result_list[i],color=color_list[i],width=0.7,alpha=0.8)
        plt.ylabel('Percentage (%)')
        plt.tight_layout()
        # plt.show()
        ## save figure
        plt.savefig(rf'D:\Project3\Result\3mm\extract_composite_phenology_year\bivariate\Figure1d.pdf',dpi=600,bbox_inches='tight')








    def heatmap_LAImin_max_CV_Figure1d(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        dff=rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\\Trend_all.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        print(len(df))
        # df = df[df['detrended_SNU_LAI_CV_p_value'] < 0.05]
        # df = df[df['LAI4g_detrend_CV_p_value'] < 0.05]
        # df = df[df['GlOBMAP_detrend_CV_p_value'] < 0.05]
        df=df[df['composite_LAI_CV_p_value'] < 0.05]
        # # print(len(df));exit()


        # plt.show();exit()

        T.print_head_n(df)
        x_var = 'composite_LAI_detrend_relative_change_min_trend'
        y_var = 'composite_LAI_detrend_relative_change_max_trend'
        z_var = 'composite_LAI_CV_trend'

        # x_var = 'GLOBMAP_LAI_relative_change_detrend_min_trend'
        # y_var = 'GLOBMAP_LAI_relative_change_detrend_max_trend'
        # z_var = 'GlOBMAP_detrend_CV_trend'
        plt.hist(df[x_var])
        plt.show()
        plt.hist(df[y_var])
        plt.show()

        bin_x = np.linspace(-1.5, 1.5,11, )

        bin_y = np.linspace(-1.5, 1.5, 11)
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure(figsize=(self.map_width, self.map_height))

        matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y,round_x=4,round_y=4)
        # pprint(matrix_dict);exit()

        my_cmap = T.cmap_blend(color_list = ['#000000','r', 'b'])
        my_cmap = 'RdBu'
        self.plot_df_bin_2d_matrix(matrix_dict,-1,1,x_ticks_list,y_ticks_list,cmap=my_cmap,
                              is_only_return_matrix=False)
        plt.colorbar()
        plt.xticks(rotation=45)
        plt.tight_layout()
        pprint(matrix_dict)
        # plt.show()


        matrix_dict_count, x_ticks_list, y_ticks_list = self.df_bin_2d_count(df, val_col_name=z_var,
                                                              col_name_x=x_var,
                                                              col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        pprint(matrix_dict_count)
        scatter_size_dict = {
            (1,20): 5,
            (20,50): 20,
            (50,100): 50,
            (100,200): 75,
            (200,400): 100,
            (400,800): 200,
            (800,np.inf): 250
        }
        matrix_dict_count_normalized = {}
        # Normalize counts for circle size
        for key in matrix_dict_count:
            num = matrix_dict_count[key]
            for key2 in scatter_size_dict:
                if num >= key2[0] and num < key2[1]:
                    matrix_dict_count_normalized[key] = scatter_size_dict[key2]
                    break
        pprint(matrix_dict_count_normalized)
        reverse_x = list(range(len(bin_y)-1))[::-1]
        reverse_x_dict = {}
        for i in range(len(bin_y)-1):
            reverse_x_dict[i] = reverse_x[i]
        # print(reverse_x_dict);exit()
        for x,y in matrix_dict_count_normalized:
            plt.scatter(y,reverse_x_dict[x],s=matrix_dict_count_normalized[(x,y)],c='gray',edgecolors='none',alpha=.5)
        for x,y in matrix_dict_count_normalized:
            plt.scatter(y,reverse_x_dict[x],s=matrix_dict_count_normalized[(x,y)],c='none',edgecolors='gray',alpha=1)

        plt.xlabel('Trend in LAImin (%)')
        plt.ylabel('Trend in LAImax (%)')

        plt.show()
        # plt.savefig(outf)
        # plt.close()

    def figure1f(self):

        dff = rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\\Trend_all.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        print(len(df))

        df = df[df['composite_LAI_CV_p_value'] < 0.05]
        # # print(len(df));exit()

        # plt.show();exit()

        T.print_head_n(df)
        x_var = 'composite_LAI_detrend_relative_change_min_trend'
        y_var = 'composite_LAI_detrend_relative_change_max_trend'

        df_LAImax_pos=df[df['composite_LAI_detrend_relative_change_max_trend']>0]
        df_LAImax_neg=df[df['composite_LAI_detrend_relative_change_max_trend']<0]
        values_pos_LAImin=df_LAImax_pos['composite_LAI_detrend_relative_change_min_trend'].tolist()
        values_neg_LAImin=df_LAImax_neg['composite_LAI_detrend_relative_change_min_trend'].tolist()
        values_pos_LAImin_array=np.array(values_pos_LAImin)
        values_neg_LAImin_array=np.array(values_neg_LAImin)

        sns.kdeplot(values_pos_LAImin_array, fill=True,color='#785187',label='LAImax>0')
        sns.kdeplot(values_neg_LAImin_array, fill=True,color='#e66d50',label='LAImax<0')
        plt.axvline(np.nanmean(values_pos_LAImin_array), color='purple', linestyle='--', linewidth=1)
        plt.axvline(np.nanmean(values_neg_LAImin_array), color='orange', linestyle='--', linewidth=1)
        plt.xlabel('Trends in LAImin')
        plt.xlim(-2.5,2.5)
        plt.legend()


        plt.show()

        pass

    def df_bin_2d_count(self,df,val_col_name,col_name_x,col_name_y,bin_x,bin_y,round_x=2,round_y=2):
        df_group_y, _ = self.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = self.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                rt_mean = len(vals)
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict,x_ticks_list,y_ticks_list

    def df_bin(self, df, col, bins):
        df_copy = df.copy()
        df_copy[f'{col}_bins'] = pd.cut(df[col], bins=bins)
        df_group = df_copy.groupby([f'{col}_bins'],observed=True)
        bins_name = df_group.groups.keys()
        bins_name_list = list(bins_name)
        bins_list_str = [str(i) for i in bins_name_list]
        # for name,df_group_i in df_group:
        #     vals = df_group_i[col].tolist()
        #     mean = np.nanmean(vals)
        #     err,_,_ = self.uncertainty_err(SM)
        #     # x_list.append(name)
        #     y_list.append(mean)
        #     err_list.append(err)
        return df_group, bins_list_str
    def plot_df_bin_2d_matrix(self,matrix_dict,vmin,vmax,x_ticks_list,y_ticks_list,cmap='RdBu',
                              is_only_return_matrix=False):
        print(x_ticks_list)
        keys = list(matrix_dict.keys())
        r_list = []
        c_list = []
        for r, c in keys:
            r_list.append(r)
            c_list.append(c)
        r_list = set(r_list)
        c_list = set(c_list)

        row = len(r_list)
        col = len(c_list)
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in matrix_dict:
                    val_pix = matrix_dict[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        matrix = np.array(spatial, dtype=float)
        matrix = matrix[::-1]
        if is_only_return_matrix:
            return matrix
        plt.imshow(matrix,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.xticks(range(len(c_list)), x_ticks_list)
        plt.yticks(range(len(r_list)), y_ticks_list[::-1])
        # plt.colorbar()
        # plt.show()
        #
    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df







class Figure2():
    def __init__(self):
        self.map_width = 8.2 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        ## step1
        # self.generate_bivarite_map()
    ## Step2
        # self.generate_df()

        ## step3
        build_dataframe().run()


        ## step4
        self.generate_three_dimension()  ## generate three_dimension_growing_season.tif [8 class]

        # step 4 add field 8 class again to df reuse step 3

        # step 5

        self.plot_figure2b_test()  ## 1-8 map +CV LAI trends + CV interannual rainfall trend
        # self.plot_figure2a_Robinson()





    def generate_bivarite_map(self):  ##

        import xymap
        tif_rainfall = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\growing_season\\trend\\\detrended_sum_rainfall_CV_trend.tif'
        # tif_CV=  result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'
        tif_sensitivity = result_root + rf'3mm\moving_window_multi_regression\multiresult_relative_change_detrend\multi_regression_result_detrend_growing_season_composite\trend\\composite_LAI_beta_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outdir = result_root + rf'3mm\\\bivariate_analysis\\composite_LAI\\'
        T.mk_dir(outdir, force=True)
        outtif = outdir + rf'\\interannual_CVrainfall_beta_growing_season.tif'

        tif1 = tif_rainfall
        tif2 = tif_sensitivity

        dic1 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif1)
        dic2 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif2)
        dics = {'interannual_CVrainfall': dic1,
                'beta': dic2}
        df = T.spatial_dics_to_df(dics)
        # print(df)
        df['interannual_CVrainfall_increase'] = df['interannual_CVrainfall'] > 0
        df['beta_increase'] = df['beta'] > 0

        print(df)
        label_list = []
        for i, row in df.iterrows():
            if row['interannual_CVrainfall_increase'] and row['beta_increase']:
                label_list.append(1)
            elif row['interannual_CVrainfall_increase'] and not row['beta_increase']:
                label_list.append(2)
            elif not row['interannual_CVrainfall_increase'] and row['beta_increase']:
                label_list.append(3)
            else:
                label_list.append(4)

        df['label'] = label_list
        result_dic = T.df_to_spatial_dic(df, 'label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(result_dic, outtif)

    pass


    def generate_df(self):
        ##rainfall_trend +sensitivity+ greening
        variable='composite_LAI'
        ftiff=result_root + rf'3mm\bivariate_analysis\\{variable}\\interannual_CVrainfall_beta_growing_season.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(ftiff)

        dic_beta_CVrainfall=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(array)

        f_CVLAItiff=result_root+rf'3mm\extract_composite_phenology_year\trend\\composite_LAI_CV_trend.tif'
        array_CV_LAI,originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_CVLAItiff)
        dic_CV_LAI=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(array_CV_LAI)


        df=T.spatial_dics_to_df({'CVrainfall_beta':dic_beta_CVrainfall,f'{variable}_CV':dic_CV_LAI})

        T.save_df(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\three_dimension_growing_season_growing_season.df')
        T.df_to_excel(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\three_dimension_growing_season_growing_season.xlsx')
        exit()





    def generate_three_dimension(self):
        variable='composite_LAI'
        dff=result_root + rf'\3mm\bivariate_analysis\Dataframe\\three_dimension_growing_season.df'
        df=T.load_df(dff)
        self.df_clean(df)
        df=df[df['CVrainfall_beta']>=0]



        df[f"{variable}_CV_trends"] = df[f"{variable}_CV"].apply(lambda x: "increaseCV" if x >= 0 else "decreaseCV")

        category_mapping = {
            ("increaseCV", 1): 1,
            ("increaseCV", 2): 2,
            ("increaseCV", 3): 3,
            ("increaseCV", 4): 4,
            ("decreaseCV", 1): 5,
            ("decreaseCV", 2): 6,
            ("decreaseCV", 3): 7,
            ("decreaseCV", 4): 8,
        }

        # Apply the mapping to create a new column 'new_category'
        df["CV_rainfall_beta_LAI"] = df.apply(
            lambda row: category_mapping[(row[f"{variable}_CV_trends"""], row["CVrainfall_beta"])],
            axis=1)
        # T.save_df(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend.df')
        # T.df_to_excel(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend.xlsx')

        # Display the result
        print(df)
        outdir = result_root + rf'\3mm\bivariate_analysis\\{variable}\\'
        T.mk_dir(outdir, force=True)
        outf = outdir + rf'CV_rainfall_beta_LAI_growing_season.tif'

        spatial_dic = T.df_to_spatial_dic(df, 'CV_rainfall_beta_LAI')
        DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(spatial_dic, outf)
        ##save pdf

        # plt.savefig(outf)
        # plt.close()





    def plot_figure2b(self):

        variable='composite'
        dff = rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\Three_dimension.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df=df.dropna()
        # df_sig=df[df['detrended_SNU_LAI_CV_p_value']<0.05]
        dic_label = {1: 'CVLAI+_CVrainfall+_posbeta', 2: 'CVLAI+_CVrainfall+_negbeta', 3: 'CVLAI+_CVrainfall-posbeta',
                     4: 'CVLAI+_CVrainfall-negbeta',
                     5: 'CVLAI-_CVrainfall+_posbeta', 6: 'CVLAI-_CVrainfall+_negbeta', 7: 'CVLAI-_CVrainfall-posbeta',
                     8: 'CVLAI-_CVrainfall-negbeta'}
        dic = {}



        df_greening = df[df[f'CV_rainfall_beta_LAI_{variable}'] < 5]
        count_green = len(df_greening)


        df_browning = df[df[f'CV_rainfall_beta_LAI_{variable}'] >= 5]
        count_brown = len(df_browning)



        greening_percentage = count_green / len(df)
        browning_percentage = count_brown / len(df)
        # print(greening_percentage,browning_percentage);exit()
        # print(greening_sum,browning_sum)


        ## count the number of pixels
        for i in range(1, 9):

            if i < 5:
                df_i = df[df[f'CV_rainfall_beta_LAI_{variable}'] == i]
                count = len(df_i)
                dic[i] = count / count_green * 100

            else:
                df_i = df[df[f'CV_rainfall_beta_LAI_{variable}'] == i]
                count = len(df_i)
                dic[i] = count / count_brown * 100
        # pprint(dic);exit()
        ## I want to index 1234 to 5678 and 5678 to 1234


        # Colors from your color scheme
        colors = ['','#33a02c','#1f78b4', '#fb9a99',
                  '#a6cee3', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']






        # Assign segments to two bars: Wetting and Drying
        CVrainfall_pos_indices = [ 6,5, 1, 2,]  # Class 1,2,5,6
        CVrainfall_neg_indices = [8,7, 3, 4]

        # Prepare stacked bar data
        wetting_values = [dic[i] for i in CVrainfall_pos_indices]
        drying_values = [dic[i] for i in CVrainfall_neg_indices]
        wetting_colors = [colors[i] for i in CVrainfall_pos_indices]
        drying_colors = [colors[i] for i in CVrainfall_neg_indices]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot wetting (upper bar)
        left = 0
        for val, color in zip(wetting_values, wetting_colors):
            ax.barh(y=1, width=val, left=left, color=color, edgecolor='black', height=0.2)
            left += val

        # Plot drying (lower bar)
        left = 0
        for val, color in zip(drying_values, drying_colors):
            ## barheight
            ax.barh(y=0, width=val, left=left, color=color, edgecolor='black', height=0.2)
            left += val

        # Aesthetics
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['CVrainfall+', 'CVrainfall-'])
        ax.set_xlabel('%')
        ax.set_xlim(0, 150)

        # Optional: Add vertical line at 50% or labels

        plt.tight_layout()
        plt.show()
        # outdir=result_root + rf'\3mm\bivariate_analysis\Barplot\\'
        # T.mk_dir(outdir, force=True)
        # ## save the figure
        # plt.savefig(outdir + rf'{variable}.pdf')


        pass


    def plot_figure2b_test(self):

        variable='composite'
        dff = rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\Three_dimension_growing_season.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df=df.dropna()
        # df_sig=df[df['detrended_SNU_LAI_CV_p_value']<0.05]
        dic_label = {1: 'CVLAI+_CVrainfall+_posbeta', 2: 'CVLAI+_CVrainfall+_negbeta', 3: 'CVLAI+_CVrainfall-posbeta',
                     4: 'CVLAI+_CVrainfall-negbeta',
                     5: 'CVLAI-_CVrainfall+_posbeta', 6: 'CVLAI-_CVrainfall+_negbeta', 7: 'CVLAI-_CVrainfall-posbeta',
                     8: 'CVLAI-_CVrainfall-negbeta'}
        dic = {}
        ## 加字段 wet dry , sum rainfall trend >0 ==wet and sum rainfall trend <0 == dry significant pvalue<0.05

        df['wet_dry'] = 'unknown'
        df.loc[df['sum_rainfall_trend'] > 0, 'wet_dry'] = 'wetting'
        df.loc[df['sum_rainfall_trend'] < 0, 'wet_dry'] = 'drying'
        # df['wet_dry'] = 'unknown'

        # df.loc[(df['sum_rainfall_trend'] > 0) & (df['sum_rainfall_p_value'] < 0.05), 'wet_dry'] = 'wetting'
        # df.loc[(df['sum_rainfall_trend'] < 0) & (df['sum_rainfall_p_value'] < 0.05), 'wet_dry'] = 'drying'

        df_greening = df[df[f'CV_rainfall_beta_LAI_{variable}_growing_season'] < 5]
        count_green = len(df_greening)


        df_browning = df[df[f'CV_rainfall_beta_LAI_{variable}_growing_season'] >= 5]
        count_brown = len(df_browning)


        wet_dry_ratio={}

        ## count the number of pixels
        for i in range(1, 9):

            if i < 5:
                df_i = df[df[f'CV_rainfall_beta_LAI_{variable}_growing_season'] == i]
                count = len(df_i)
                dic[i] = count / len(df) * 100
                wet_count=len(df_i[df_i['wet_dry']=='wetting'])
                dry_count=len(df_i[df_i['wet_dry']=='drying'])
                wet_ratio = wet_count / len(df_i) * 100
                dry_ratio = dry_count / len(df_i) * 100
                wet_dry_ratio[i]=(wet_ratio,dry_ratio)

            else:
                df_i = df[df[f'CV_rainfall_beta_LAI_{variable}_growing_season'] == i]
                count = len(df_i)
                dic[i] = count / len(df) * 100
                wet_count = len(df_i[df_i['wet_dry'] == 'wetting'])
                dry_count = len(df_i[df_i['wet_dry'] == 'drying'])
                wet_ratio = wet_count / len(df_i) * 100
                dry_ratio = dry_count / len(df_i) * 100
                wet_dry_ratio[i] = (wet_ratio, dry_ratio)
        pprint(dic);exit()
        # pprint(wet_dry_ratio);exit()
        ## I want to add new column when df[df[f'CV_rainfall_beta_LAI_{variable}'] == 3] and df[df[f'wet_dry'] == 'drying']
        ## give this column a name is extraction
        df['extraction'] = -999

        df.loc[(df[f'CV_rainfall_beta_LAI_{variable}_growing_season'] == 3) & (df['wet_dry'] == 'drying'), 'extraction'] = 0
        df.loc[(df[f'CV_rainfall_beta_LAI_{variable}_growing_season'] == 3) & (df['wet_dry'] == 'wetting'), 'extraction'] = 1
        ##
        # spatial_mask_dic=T.df_to_spatial_dic(df,'extraction')

        # outf = rf'D:\Project3\Result\3mm\bivariate_analysis\extraction_mask.tif'
        # array=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_mask_dic)
        # array[array==-999]=np.nan
        # DIC_and_TIF().arr_to_tif(array,outf);exit()





        ##

        # Colors from your color scheme
        colors = ['','#33a02c','#1f78b4', '#fb9a99',
                  '#a6cee3', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']



        # Assign segments to two bars: CVIAV+ and CVIAV-
        CVrainfall_pos_indices = [ 6,5, 1, 2,]  # Class 1,2,5,6
        ##
        CVrainfall_neg_indices = [8,7, 3, 4]


        # Prepare stacked bar data
        CV_IAV_positive_values = [dic[i] for i in CVrainfall_pos_indices]
        CVIAV_negative_values = [dic[i] for i in CVrainfall_neg_indices]

        CV_IAV_positive_colors = [colors[i] for i in CVrainfall_pos_indices]
        CVIAV_negative_colors = [colors[i] for i in CVrainfall_neg_indices]


        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot CVIAV+ (upper bar)
        left=0
        for i, (val, color) in enumerate(zip(CV_IAV_positive_values, CV_IAV_positive_colors)):
            idx = CVrainfall_pos_indices[i]
            wet_ratio, dry_ratio = wet_dry_ratio[idx]
            wet_width = val * (wet_ratio / 100)
            dry_width = val * (dry_ratio / 100)

            # drying portion ( hatch)
            ax.barh(y=1, width=dry_width, left=left, color=color, edgecolor='black', height=0.2,)
            # wetting portion (no hatch)
            ax.barh(y=1, width=wet_width, left=left + dry_width, color=color, edgecolor='black', height=0.2,
                   )

            left += val

        # plot CVIAV- (lower bar)
        left = 0
        for i, (val, color) in enumerate(zip(CVIAV_negative_values, CVIAV_negative_colors)):
            idx = CVrainfall_neg_indices[i]
            wet_ratio, dry_ratio = wet_dry_ratio[idx]
            wet_width = val * (wet_ratio / 100)
            dry_width = val * (dry_ratio / 100)

            # drying portion ( hatch)
            # ax.barh(y=0, width=dry_width, left=left, color=color, edgecolor='black', height=0.2, hatch='///')
            ax.barh(y=0, width=dry_width, left=left, color=color, edgecolor='black', height=0.2,)
            # wetting portion (no hatch)
            ax.barh(y=0, width=wet_width, left=left + dry_width, color=color, edgecolor='black', height=0.2,
                   )

            left += val


        # Aesthetics
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['CVrainfall+', 'CVrainfall-'])
        ax.set_xlabel('%')
        ax.set_xlim(0, 62)

        # Optional: Add vertical line at 50% or labels

        # plt.tight_layout()
        # plt.show()
        outdir=result_root + rf'\3mm\bivariate_analysis\Barplot\\'
        T.mk_dir(outdir, force=True)
        ## save the figure
        plt.savefig(outdir + rf'{variable}.pdf')


        pass





    def plot_figure2a_Robinson(self):

        fdir_trend = result_root + rf'\3mm\bivariate_analysis\composite_LAI\\'
        temp_root = result_root + rf'\3mm\bivariate_analysis\\composite_LAI\\temp\\'
        outdir = result_root + rf'\3mm\bivariate_analysis\\ROBINSON\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):

            if not f.endswith('.tif'):
                continue

            fname = f.split('.')[0]
            if not 'CV_rainfall_beta_LAI_composite_growing_season' in fname:
                continue

            fpath = fdir_trend + f
            ## use this  color_list = [ '#33a02c','#1f78b4',
         #               '#fb9a99',  '#a6cee3', '#fdbf6f',
         # '#ff7f00', '#6a3d9a', '#b15928']


            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=8, is_discrete=True, colormap_n=9, )




            # arr = ToRaster().raster2array(fpath)[0]
            # arr[arr<-999]=np.nan
            # plt.imshow(arr,cmap=my_cmap,vmin=1,vmax=8,interpolation='nearest')
            # plt.colorbar()
            # plt.show()


            # plt.title(f'{fname}')
            # plt.show()
            outf = outdir + 'CV_rainfall_beta_LAI_composite_growing_season.pdf'
            plt.savefig(outf)
            plt.close()
            # exit()



    def LAImin_LAImax_index_ratio_group(self,):

        import matplotlib.cm as cm

        fdir_max=result_root+rf'3mm\relative_change_growing_season\moving_window_min_max_anaysis\max\trend_analysis\\'
        fdir_min=result_root+rf'3mm\relative_change_growing_season\moving_window_min_max_anaysis\min\trend_analysis\\'
        outdir=result_root+rf'\3mm\relative_change_growing_season\\moving_window_min_max_anaysis\\ratio\\'
        T.mk_dir(outdir,force=True)

        variables_list = ['composite_LAI', 'TRENDY_ensemble',
                            'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',

                          'YIBs_S2_Monthly_lai']

        dic_label_name = {'composite_LAI': 'Composite LAI',
                          'TRENDY_ensemble': 'TRENDY ensemble',
            'CABLE-POP_S2_lai': 'CABLE-POP',
                          'CLASSIC_S2_lai': 'CLASSIC',
                          'CLM5': 'CLM5',
                          'DLEM_S2_lai': 'DLEM',
                          'IBIS_S2_lai': 'IBIS',
                          'ISAM_S2_lai': 'ISAM',
                          'ISBA-CTRIP_S2_lai': 'ISBA-CTRIP',
                          'JSBACH_S2_lai': 'JSBACH',
                          'JULES_S2_lai': 'JULES',
                          'LPJ-GUESS_S2_lai': 'LPJ-GUESS',
                          'LPX-Bern_S2_lai': 'LPX-Bern',
                          'ORCHIDEE_S2_lai': 'ORCHIDEE',

                          'YIBs_S2_Monthly_lai': 'YIBs',

                          }
        result_dic={}

        for variable in variables_list:
            percentage_dic={}


            lai_max_trend_path = fdir_max+f'{variable}_detrend_max_trend.tif'
            lai_min_trend_path = fdir_min+f'{variable}_detrend_min_trend.tif'

            output_classification_path = outdir + f'{variable}_detrend_relative_change_ratio_classification.tif'

            LAImax_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(lai_max_trend_path)
            LAImax_arr[LAImax_arr < -99] = np.nan
            LAImax_arr[LAImax_arr > 99] = np.nan

            LAImin_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(lai_min_trend_path)
            LAImin_arr[LAImin_arr < -99] = np.nan
            LAImin_arr[LAImin_arr > 99] = np.nan



            # === Initialize classification map ===
            class_map = np.full_like(LAImax_arr, np.nan, dtype=np.uint8)

            # === Define classification logic ===
            ratio = np.full_like(LAImax_arr, np.nan, dtype=np.float32)
            valid_mask = (~np.isnan(LAImax_arr)) & (~np.isnan(LAImin_arr)) & (np.abs(LAImin_arr) > 0.001)
            ratio[valid_mask] = LAImax_arr[valid_mask] / LAImin_arr[valid_mask]

            # Case 1: both +, ratio > 1
            class_map[(LAImax_arr > 0) & (LAImin_arr > 0) & (ratio > 1)] = 1

            # Case 2: both +, ratio ≈ 1
            class_map[(LAImax_arr > 0) & (LAImin_arr > 0) & (np.isclose(ratio, 1, atol=0.1))] = 2

            # Case 3: both +, ratio < 1
            class_map[(LAImax_arr > 0) & (LAImin_arr > 0) & (ratio < 1)] = 3

            # Case 4: max +, min -
            class_map[(LAImax_arr > 0) & (LAImin_arr < 0)] = 4

            # Case 5: max -, min +
            class_map[(LAImax_arr < 0) & (LAImin_arr > 0)] = 5

            # Case 6: both -, ratio > 1
            class_map[(LAImax_arr < 0) & (LAImin_arr < 0) & (ratio > 1)] = 6


            # Case 7: both -, ratio ≈ 1
            class_map[(LAImax_arr < 0) & (LAImin_arr < 0) & (np.isclose(ratio, 1, atol=0.1))] = 7

            # Case 8: both -, ratio < 1
            class_map[(LAImax_arr < 0) & (LAImin_arr < 0) & (ratio < 1)] = 8

            # Case 9: denominator ≈ 0 (unstable)
            class_map[np.abs(LAImin_arr) < 0.001] = 9

            class_map = class_map.astype(np.float32)
            class_map[class_map == 0] = np.nan

            # === Save the classification map ===
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(class_map, output_classification_path)
            class_map[class_map<-99]=np.nan

            total_valid = np.count_nonzero(~np.isnan(class_map))
            for k in range(1, 10):
                percentage = np.count_nonzero(class_map == k) / total_valid * 100
                percentage_dic[k] = percentage


            result_dic[variable] = percentage_dic


        alpha_list = [1] + [1] + [0.7] * 14


        fig, ax = plt.subplots(figsize=(6, 4))

        df_new=pd.DataFrame(result_dic)
        df_new=df_new.T
        df_new=df_new.reset_index()
        df_new.columns = ['variable'] + [str(i) for i in range(1, 10)]

        df_melted = df_new.melt(
            id_vars='variable',
            value_vars=[str(i) for i in range(1, 10)],

            var_name='class',
            value_name='percentage'
        )


        variables = df_melted['variable'].unique()
        classes = sorted(df_melted['class'].unique())

        cmap = cm.get_cmap('Set3', len(classes))  # 也可以用 'viridis', 'cool', 'turbo' 等
        color_list = [cmap(i) for i in range(len(classes))]

        # 初始化底部为 0
        bottom = np.zeros(len(variables))

        for i, cls in enumerate(classes):
            df_class = df_melted[df_melted['class'] == cls].set_index('variable').reindex(variables)
            values = df_class['percentage'].values

            ax.bar(
                variables,
                values,
                bottom=bottom,
                width=0.6,
                label=f'Class {cls}',
                alpha=0.8,
                color=color_list[i],
                edgecolor='black'

            )

            # 更新底部
            bottom += values


        # 设置图例和格式
        ax.set_ylabel('Percentage (%)', fontsize=10,font='Arial')
        ## set xticks dicname
        ax.set_xticks(range(len(variables)))
        ax.set_xticklabels(dic_label_name.values(), rotation=90, fontsize=10,font='Arial')

        ax.legend(title="Class")
        plt.tight_layout()
        plt.show()




    def classfication_LAImin_LAImax_index(self):
        fmax=result_root+rf'\3mm\extract_composite_phenology_year\trend\\composite_LAI_detrend_relative_change_max_trend.tif'
        fmin=result_root+rf'\3mm\extract_composite_phenology_year\trend\\composite_LAI_detrend_relative_change_min_trend.tif'
        array_max, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fmax)
        array_min, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fmin)
        array_max[array_max<-99]=np.nan
        array_max[array_max>99]=np.nan
        array_min[array_min<-99]=np.nan
        array_min[array_min>99]=np.nan


        trend_max=np.array(array_max).flatten()
        trend_min=np.array(array_min).flatten()
        delta=trend_max-trend_min


        # 设置一个微小阈值来判断是否为 "接近于0"
        eps_trend = 0.0001
        eps_delta = 0.0001

        # 初始化类别图
        category = np.full(trend_max.shape, np.nan)

        # 类别 1：max↑ min↑ 幅度相近
        category[(trend_max > eps_trend) & (trend_min > eps_trend) & (np.abs(delta) < eps_delta)] = 1

        # 类别 2：max↑ min↓ 且差值较大
        category[(trend_max > eps_trend) & (trend_min < -eps_trend) & (delta > eps_delta)] = 2

        # 类别 3：max↓ min↑ 且差值较大（负）
        category[(trend_max < -eps_trend) & (trend_min > eps_trend) & (delta < -eps_delta)] = 3

        # 类别 4：max↓ min↓ 幅度相近
        category[(trend_max < -eps_trend) & (trend_min < -eps_trend) & (np.abs(delta) < eps_delta)] = 4

        # 类别 5：max↑ min↑ 但 max 多
        category[(trend_max > eps_trend) & (trend_min > eps_trend) & (delta > eps_delta)] = 5

        # 类别 6：max↑ min↑ 但 min 多
        category[(trend_max > eps_trend) & (trend_min > eps_trend) & (delta < -eps_delta)] = 6  # min≈0, max变
        ## calculate the percentage of each category
        category_temp=category
        category_temp=category_temp[~np.isnan(category_temp)]
        category_count = np.unique(category_temp, return_counts=True)

        category_percentage = category_count[1] / len(category_temp)*100

        plt.bar(category_count[0], category_percentage)
        plt.show()

        ## reshape
        category=category.reshape(array_max.shape)
        # plt.imshow(category,interpolation='nearest',cmap='jet_r',vmin=1,vmax=6)
        # plt.show()
        outdir=result_root+rf'\3mm\extract_composite_phenology_year\trend\\'
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(category,outdir+'delta_category.tif')



        pass


    def statistic_trend_bar(self):
        fdir = result_root + rf'\3mm\extract_composite_phenology_year\trend\\'
        variable='LAImin_LAImax_index2'

        f_trend_path=fdir+f'{variable}_trend.tif'
        f_pvalue_path=fdir+f'{variable}_pvalue.tif'


        arr_corr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_trend_path)
        arr_pvalue, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_pvalue_path)
        arr_corr[arr_corr<-99]=np.nan
        arr_corr[arr_corr>99]=np.nan
        arr_corr=arr_corr[~np.isnan(arr_corr)]

        arr_pvalue[arr_pvalue<-99]=np.nan
        arr_pvalue[arr_pvalue>99]=np.nan
        arr_pvalue=arr_pvalue[~np.isnan(arr_pvalue)]
        ## corr negative and positive
        arr_corr = arr_corr.flatten()
        arr_pvalue = arr_pvalue.flatten()
        arr_pos=len(arr_corr[arr_corr>0])/len(arr_corr)*100
        arr_neg=len(arr_corr[arr_corr<0])/len(arr_corr)*100


        ## significant positive and negative
        ## 1 is significant and 2 positive or negative

        mask_pos = (arr_corr > 0) & (arr_pvalue < 0.05)
        mask_neg = (arr_corr < 0) & (arr_pvalue < 0.05)


        # 满足条件的像元数
        count_positive_sig = np.sum(mask_pos)
        count_negative_sig = np.sum(mask_neg)

        # 百分比
        significant_positive = (count_positive_sig / len(arr_corr)) * 100
        significant_negative = (count_negative_sig / len(arr_corr)) * 100
        result_dic = {

            'sig neg': significant_negative,
            'non sig neg': arr_neg,
            'non sig pos': arr_pos,
            'sig pos': significant_positive



        }
        # df_new=pd.DataFrame(result_dic,index=[variable])
        # ## plot
        # df_new=df_new.T
        # df_new=df_new.reset_index()
        # df_new.columns=['Variable','Percentage']
        # df_new.plot.bar(x='Variable',y='Percentage',rot=45,color='green')
        # plt.show()
        color_list = [
            '#008837',
            '#a6dba0',

            '#c2a5cf',
            '#7b3294',
        ]
        width = 0.4
        alpha_list = [1, 0.5, 0.5, 1]

        # 逐个画 bar
        for i, (key, val) in enumerate(result_dic.items()):
            plt.bar(i , val, color=color_list[i], alpha=alpha_list[i], width=width)
            plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
            plt.ylabel('Percentage')
            plt.title(variable)

        plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
        plt.show()




    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

class Figure3_beta_2():
    def __init__(self):
        self.map_width = 8.2 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run(self):
        ## step1
        # self.generate_bivarite_map()
    ## Step2
        # self.generate_df()
        #
        # ## step3
        # build_dataframe().run()


        ## step4
        # self.generate_three_dimension()  ## generate three_dimension_growing_season.tif [8 class]

        # step 4 add field 8 class again to df reuse step 3

        # step 5

        self.plot_figure2b_test()  ## 1-8 map +CV LAI trends + CV interannual rainfall trend
        # self.plot_figure2a_Robinson()





    def generate_bivarite_map(self):  ##

        import xymap
        tif_rainfall = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\growing_season\\trend\\\detrended_sum_rainfall_CV_trend.tif'
        # tif_CV=  result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'
        tif_sensitivity = result_root + rf'3mm\moving_window_multi_regression\multiresult_relative_change_detrend\multi_regression_result_detrend_growing_season_composite\trend\\composite_LAI_beta_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outdir = result_root + rf'3mm\\\bivariate_analysis\\composite_LAI\\'
        T.mk_dir(outdir, force=True)
        outtif = outdir + rf'\\interannual_CVrainfall_beta_growing_season.tif'

        tif1 = tif_rainfall
        tif2 = tif_sensitivity

        dic1 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif1)
        dic2 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif2)
        dics = {'interannual_CVrainfall': dic1,
                'beta': dic2}
        df = T.spatial_dics_to_df(dics)
        # print(df)
        df['interannual_CVrainfall_increase'] = df['interannual_CVrainfall'] > 0
        df['beta_increase'] = df['beta'] > 0

        print(df)
        label_list = []
        for i, row in df.iterrows():
            if row['interannual_CVrainfall_increase'] and row['beta_increase']:
                label_list.append(1)
            elif row['interannual_CVrainfall_increase'] and not row['beta_increase']:
                label_list.append(2)
            elif not row['interannual_CVrainfall_increase'] and row['beta_increase']:
                label_list.append(3)
            else:
                label_list.append(4)

        df['label'] = label_list
        result_dic = T.df_to_spatial_dic(df, 'label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(result_dic, outtif)

    pass


    def generate_df(self):
        ##rainfall_trend +sensitivity+ greening
        variable='composite_LAI'
        ftiff=result_root + rf'3mm\bivariate_analysis\\{variable}\\interannual_CVrainfall_beta.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(ftiff)

        dic_beta_CVrainfall=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(array)

        f_CVLAItiff=result_root+rf'\3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\trend\\CV_intraannual_rainfall_trend.tif'
        array_CV_LAI,originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_CVLAItiff)
        dic_CV_LAI=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(array_CV_LAI)


        df=T.spatial_dics_to_df({'CVrainfall_beta':dic_beta_CVrainfall,f'CV_intraannual_rainfall':dic_CV_LAI})

        T.save_df(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\three_dimension.df')
        T.df_to_excel(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\three_dimension.xlsx')
        # exit()





    def generate_three_dimension(self):
        variable='composite_LAI'
        dff=result_root + rf'\3mm\bivariate_analysis\Dataframe\\three_dimension.df'
        df=T.load_df(dff)
        self.df_clean(df)
        df=df[df['composite_LAI_CV_trend']>=0]



        df[f"CV_intraannual_rainfall"] = df[f"CV_intraannual_rainfall"].apply(lambda x: "increaseCV" if x >= 0 else "decreaseCV")

        category_mapping = {
            ("increaseCV", 1): 1,
            ("increaseCV", 2): 2,
            ("increaseCV", 3): 3,
            ("increaseCV", 4): 4,
            ("decreaseCV", 1): 5,
            ("decreaseCV", 2): 6,
            ("decreaseCV", 3): 7,
            ("decreaseCV", 4): 8,
        }

        # Apply the mapping to create a new column 'new_category'
        df["CV_inter_intra_rainfall_beta"] = df.apply(
            lambda row: category_mapping[(row[f"CV_intraannual_rainfall"], row["CVrainfall_beta"])],
            axis=1)
        # T.save_df(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend.df')
        # T.df_to_excel(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend.xlsx')

        # Display the result
        print(df)
        outdir = result_root + rf'\3mm\bivariate_analysis\\{variable}\\'
        T.mk_dir(outdir, force=True)
        outf = outdir + rf'CV_inter_intra_rainfall_beta.tif'

        spatial_dic = T.df_to_spatial_dic(df, 'CV_inter_intra_rainfall_beta')
        DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(spatial_dic, outf)
        ##save pdf

        # plt.savefig(outf)
        # plt.close()





    def plot_figure2b(self):

        variable='composite'
        dff = rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\Three_dimension.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df=df.dropna()
        # df_sig=df[df['detrended_SNU_LAI_CV_p_value']<0.05]
        dic_label = {1: 'CVLAI+_CVrainfall+_posbeta', 2: 'CVLAI+_CVrainfall+_negbeta', 3: 'CVLAI+_CVrainfall-posbeta',
                     4: 'CVLAI+_CVrainfall-negbeta',
                     5: 'CVLAI-_CVrainfall+_posbeta', 6: 'CVLAI-_CVrainfall+_negbeta', 7: 'CVLAI-_CVrainfall-posbeta',
                     8: 'CVLAI-_CVrainfall-negbeta'}
        dic = {}



        df_greening = df[df[f'CV_rainfall_beta_LAI_{variable}'] < 5]
        count_green = len(df_greening)


        df_browning = df[df[f'CV_rainfall_beta_LAI_{variable}'] >= 5]
        count_brown = len(df_browning)



        greening_percentage = count_green / len(df)
        browning_percentage = count_brown / len(df)
        # print(greening_percentage,browning_percentage);exit()
        # print(greening_sum,browning_sum)


        ## count the number of pixels
        for i in range(1, 9):

            if i < 5:
                df_i = df[df[f'CV_rainfall_beta_LAI_{variable}'] == i]
                count = len(df_i)
                dic[i] = count / count_green * 100

            else:
                df_i = df[df[f'CV_rainfall_beta_LAI_{variable}'] == i]
                count = len(df_i)
                dic[i] = count / count_brown * 100
        # pprint(dic);exit()
        ## I want to index 1234 to 5678 and 5678 to 1234


        # Colors from your color scheme
        colors = ['','#33a02c','#1f78b4', '#fb9a99',
                  '#a6cee3', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']






        # Assign segments to two bars: Wetting and Drying
        CVrainfall_pos_indices = [ 6,5, 1, 2,]  # Class 1,2,5,6
        CVrainfall_neg_indices = [8,7, 3, 4]

        # Prepare stacked bar data
        wetting_values = [dic[i] for i in CVrainfall_pos_indices]
        drying_values = [dic[i] for i in CVrainfall_neg_indices]
        wetting_colors = [colors[i] for i in CVrainfall_pos_indices]
        drying_colors = [colors[i] for i in CVrainfall_neg_indices]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot wetting (upper bar)
        left = 0
        for val, color in zip(wetting_values, wetting_colors):
            ax.barh(y=1, width=val, left=left, color=color, edgecolor='black', height=0.2)
            left += val

        # Plot drying (lower bar)
        left = 0
        for val, color in zip(drying_values, drying_colors):
            ## barheight
            ax.barh(y=0, width=val, left=left, color=color, edgecolor='black', height=0.2)
            left += val

        # Aesthetics
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['CVrainfall+', 'CVrainfall-'])
        ax.set_xlabel('%')
        ax.set_xlim(0, 150)

        # Optional: Add vertical line at 50% or labels

        plt.tight_layout()
        plt.show()
        # outdir=result_root + rf'\3mm\bivariate_analysis\Barplot\\'
        # T.mk_dir(outdir, force=True)
        # ## save the figure
        # plt.savefig(outdir + rf'{variable}.pdf')


        pass


    def plot_figure2b_test(self):

        variable='composite'
        dff = rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\Three_dimension.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df=df.dropna()
        # df_sig=df[df['detrended_SNU_LAI_CV_p_value']<0.05]
        dic_label = {1: 'CVLAI+_CVrainfall+_posbeta', 2: 'CVLAI+_CVrainfall+_negbeta', 3: 'CVLAI+_CVrainfall-posbeta',
                     4: 'CVLAI+_CVrainfall-negbeta',
                     5: 'CVLAI-_CVrainfall+_posbeta', 6: 'CVLAI-_CVrainfall+_negbeta', 7: 'CVLAI-_CVrainfall-posbeta',
                     8: 'CVLAI-_CVrainfall-negbeta'}
        dic = {}
        ## 加字段 wet dry , sum rainfall trend >0 ==wet and sum rainfall trend <0 == dry significant pvalue<0.05





        beta_sum=0
        CV_intra_sum=0
        CV_inter_sum=0

        ## count the number of pixels
        for i in [1,3,5,7]:


            df_i = df[df[f'CV_inter_intra_rainfall_beta'] == i]
            count = len(df_i)
            beta_sum+=count

        beta_percent=beta_sum/len(df)*100
        for i in [1,2,5,6]:
            df_i = df[df[f'CV_inter_intra_rainfall_beta'] == i]
            count = len(df_i)
            CV_inter_sum+=count

        CV_inter_percent=CV_inter_sum/len(df)*100
        for i in [1,2,3,4]:
            df_i = df[df[f'CV_inter_intra_rainfall_beta'] == i]
            count = len(df_i)
            CV_intra_sum+=count

        CV_intra_percent=CV_intra_sum/len(df)*100

        ## plt the figure bar
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.bar([1,2,3], [beta_percent,CV_inter_percent,CV_intra_percent,], color=['#1f78b4','#33a02c','#ff7f00', ])
        plt.show()





        #
        #
        #
        #
        # # Colors from your color scheme
        # colors = ['','#33a02c','#1f78b4', '#fb9a99',
        #           '#a6cee3', '#fdbf6f', '#ff7f00', '#6a3d9a', '#b15928']







    def plot_figure2a_Robinson(self):

        fdir_trend = result_root + rf'\3mm\bivariate_analysis\composite_LAI\\'
        temp_root = result_root + rf'\3mm\bivariate_analysis\\composite_LAI\\temp\\'
        outdir = result_root + rf'\3mm\bivariate_analysis\\ROBINSON\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):

            if not f.endswith('.tif'):
                continue

            fname = f.split('.')[0]
            if not 'CV_rainfall_beta_LAI_composite_growing_season' in fname:
                continue

            fpath = fdir_trend + f
            ## use this  color_list = [ '#33a02c','#1f78b4',
         #               '#fb9a99',  '#a6cee3', '#fdbf6f',
         # '#ff7f00', '#6a3d9a', '#b15928']


            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=8, is_discrete=True, colormap_n=9, )




            # arr = ToRaster().raster2array(fpath)[0]
            # arr[arr<-999]=np.nan
            # plt.imshow(arr,cmap=my_cmap,vmin=1,vmax=8,interpolation='nearest')
            # plt.colorbar()
            # plt.show()


            # plt.title(f'{fname}')
            # plt.show()
            outf = outdir + 'CV_rainfall_beta_LAI_composite_growing_season.pdf'
            plt.savefig(outf)
            plt.close()
            # exit()



    def LAImin_LAImax_index_ratio_group(self,):

        import matplotlib.cm as cm

        fdir_max=result_root+rf'3mm\relative_change_growing_season\moving_window_min_max_anaysis\max\trend_analysis\\'
        fdir_min=result_root+rf'3mm\relative_change_growing_season\moving_window_min_max_anaysis\min\trend_analysis\\'
        outdir=result_root+rf'\3mm\relative_change_growing_season\\moving_window_min_max_anaysis\\ratio\\'
        T.mk_dir(outdir,force=True)

        variables_list = ['composite_LAI', 'TRENDY_ensemble',
                            'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',

                          'YIBs_S2_Monthly_lai']

        dic_label_name = {'composite_LAI': 'Composite LAI',
                          'TRENDY_ensemble': 'TRENDY ensemble',
            'CABLE-POP_S2_lai': 'CABLE-POP',
                          'CLASSIC_S2_lai': 'CLASSIC',
                          'CLM5': 'CLM5',
                          'DLEM_S2_lai': 'DLEM',
                          'IBIS_S2_lai': 'IBIS',
                          'ISAM_S2_lai': 'ISAM',
                          'ISBA-CTRIP_S2_lai': 'ISBA-CTRIP',
                          'JSBACH_S2_lai': 'JSBACH',
                          'JULES_S2_lai': 'JULES',
                          'LPJ-GUESS_S2_lai': 'LPJ-GUESS',
                          'LPX-Bern_S2_lai': 'LPX-Bern',
                          'ORCHIDEE_S2_lai': 'ORCHIDEE',

                          'YIBs_S2_Monthly_lai': 'YIBs',

                          }
        result_dic={}

        for variable in variables_list:
            percentage_dic={}


            lai_max_trend_path = fdir_max+f'{variable}_detrend_max_trend.tif'
            lai_min_trend_path = fdir_min+f'{variable}_detrend_min_trend.tif'

            output_classification_path = outdir + f'{variable}_detrend_relative_change_ratio_classification.tif'

            LAImax_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(lai_max_trend_path)
            LAImax_arr[LAImax_arr < -99] = np.nan
            LAImax_arr[LAImax_arr > 99] = np.nan

            LAImin_arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(lai_min_trend_path)
            LAImin_arr[LAImin_arr < -99] = np.nan
            LAImin_arr[LAImin_arr > 99] = np.nan



            # === Initialize classification map ===
            class_map = np.full_like(LAImax_arr, np.nan, dtype=np.uint8)

            # === Define classification logic ===
            ratio = np.full_like(LAImax_arr, np.nan, dtype=np.float32)
            valid_mask = (~np.isnan(LAImax_arr)) & (~np.isnan(LAImin_arr)) & (np.abs(LAImin_arr) > 0.001)
            ratio[valid_mask] = LAImax_arr[valid_mask] / LAImin_arr[valid_mask]

            # Case 1: both +, ratio > 1
            class_map[(LAImax_arr > 0) & (LAImin_arr > 0) & (ratio > 1)] = 1

            # Case 2: both +, ratio ≈ 1
            class_map[(LAImax_arr > 0) & (LAImin_arr > 0) & (np.isclose(ratio, 1, atol=0.1))] = 2

            # Case 3: both +, ratio < 1
            class_map[(LAImax_arr > 0) & (LAImin_arr > 0) & (ratio < 1)] = 3

            # Case 4: max +, min -
            class_map[(LAImax_arr > 0) & (LAImin_arr < 0)] = 4

            # Case 5: max -, min +
            class_map[(LAImax_arr < 0) & (LAImin_arr > 0)] = 5

            # Case 6: both -, ratio > 1
            class_map[(LAImax_arr < 0) & (LAImin_arr < 0) & (ratio > 1)] = 6


            # Case 7: both -, ratio ≈ 1
            class_map[(LAImax_arr < 0) & (LAImin_arr < 0) & (np.isclose(ratio, 1, atol=0.1))] = 7

            # Case 8: both -, ratio < 1
            class_map[(LAImax_arr < 0) & (LAImin_arr < 0) & (ratio < 1)] = 8

            # Case 9: denominator ≈ 0 (unstable)
            class_map[np.abs(LAImin_arr) < 0.001] = 9

            class_map = class_map.astype(np.float32)
            class_map[class_map == 0] = np.nan

            # === Save the classification map ===
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(class_map, output_classification_path)
            class_map[class_map<-99]=np.nan

            total_valid = np.count_nonzero(~np.isnan(class_map))
            for k in range(1, 10):
                percentage = np.count_nonzero(class_map == k) / total_valid * 100
                percentage_dic[k] = percentage


            result_dic[variable] = percentage_dic


        alpha_list = [1] + [1] + [0.7] * 14


        fig, ax = plt.subplots(figsize=(6, 4))

        df_new=pd.DataFrame(result_dic)
        df_new=df_new.T
        df_new=df_new.reset_index()
        df_new.columns = ['variable'] + [str(i) for i in range(1, 10)]

        df_melted = df_new.melt(
            id_vars='variable',
            value_vars=[str(i) for i in range(1, 10)],

            var_name='class',
            value_name='percentage'
        )


        variables = df_melted['variable'].unique()
        classes = sorted(df_melted['class'].unique())

        cmap = cm.get_cmap('Set3', len(classes))  # 也可以用 'viridis', 'cool', 'turbo' 等
        color_list = [cmap(i) for i in range(len(classes))]

        # 初始化底部为 0
        bottom = np.zeros(len(variables))

        for i, cls in enumerate(classes):
            df_class = df_melted[df_melted['class'] == cls].set_index('variable').reindex(variables)
            values = df_class['percentage'].values

            ax.bar(
                variables,
                values,
                bottom=bottom,
                width=0.6,
                label=f'Class {cls}',
                alpha=0.8,
                color=color_list[i],
                edgecolor='black'

            )

            # 更新底部
            bottom += values


        # 设置图例和格式
        ax.set_ylabel('Percentage (%)', fontsize=10,font='Arial')
        ## set xticks dicname
        ax.set_xticks(range(len(variables)))
        ax.set_xticklabels(dic_label_name.values(), rotation=90, fontsize=10,font='Arial')

        ax.legend(title="Class")
        plt.tight_layout()
        plt.show()




    def classfication_LAImin_LAImax_index(self):
        fmax=result_root+rf'\3mm\extract_composite_phenology_year\trend\\composite_LAI_detrend_relative_change_max_trend.tif'
        fmin=result_root+rf'\3mm\extract_composite_phenology_year\trend\\composite_LAI_detrend_relative_change_min_trend.tif'
        array_max, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fmax)
        array_min, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fmin)
        array_max[array_max<-99]=np.nan
        array_max[array_max>99]=np.nan
        array_min[array_min<-99]=np.nan
        array_min[array_min>99]=np.nan


        trend_max=np.array(array_max).flatten()
        trend_min=np.array(array_min).flatten()
        delta=trend_max-trend_min


        # 设置一个微小阈值来判断是否为 "接近于0"
        eps_trend = 0.0001
        eps_delta = 0.0001

        # 初始化类别图
        category = np.full(trend_max.shape, np.nan)

        # 类别 1：max↑ min↑ 幅度相近
        category[(trend_max > eps_trend) & (trend_min > eps_trend) & (np.abs(delta) < eps_delta)] = 1

        # 类别 2：max↑ min↓ 且差值较大
        category[(trend_max > eps_trend) & (trend_min < -eps_trend) & (delta > eps_delta)] = 2

        # 类别 3：max↓ min↑ 且差值较大（负）
        category[(trend_max < -eps_trend) & (trend_min > eps_trend) & (delta < -eps_delta)] = 3

        # 类别 4：max↓ min↓ 幅度相近
        category[(trend_max < -eps_trend) & (trend_min < -eps_trend) & (np.abs(delta) < eps_delta)] = 4

        # 类别 5：max↑ min↑ 但 max 多
        category[(trend_max > eps_trend) & (trend_min > eps_trend) & (delta > eps_delta)] = 5

        # 类别 6：max↑ min↑ 但 min 多
        category[(trend_max > eps_trend) & (trend_min > eps_trend) & (delta < -eps_delta)] = 6  # min≈0, max变
        ## calculate the percentage of each category
        category_temp=category
        category_temp=category_temp[~np.isnan(category_temp)]
        category_count = np.unique(category_temp, return_counts=True)

        category_percentage = category_count[1] / len(category_temp)*100

        plt.bar(category_count[0], category_percentage)
        plt.show()

        ## reshape
        category=category.reshape(array_max.shape)
        # plt.imshow(category,interpolation='nearest',cmap='jet_r',vmin=1,vmax=6)
        # plt.show()
        outdir=result_root+rf'\3mm\extract_composite_phenology_year\trend\\'
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(category,outdir+'delta_category.tif')



        pass


    def statistic_trend_bar(self):
        fdir = result_root + rf'\3mm\extract_composite_phenology_year\trend\\'
        variable='LAImin_LAImax_index2'

        f_trend_path=fdir+f'{variable}_trend.tif'
        f_pvalue_path=fdir+f'{variable}_pvalue.tif'


        arr_corr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_trend_path)
        arr_pvalue, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_pvalue_path)
        arr_corr[arr_corr<-99]=np.nan
        arr_corr[arr_corr>99]=np.nan
        arr_corr=arr_corr[~np.isnan(arr_corr)]

        arr_pvalue[arr_pvalue<-99]=np.nan
        arr_pvalue[arr_pvalue>99]=np.nan
        arr_pvalue=arr_pvalue[~np.isnan(arr_pvalue)]
        ## corr negative and positive
        arr_corr = arr_corr.flatten()
        arr_pvalue = arr_pvalue.flatten()
        arr_pos=len(arr_corr[arr_corr>0])/len(arr_corr)*100
        arr_neg=len(arr_corr[arr_corr<0])/len(arr_corr)*100


        ## significant positive and negative
        ## 1 is significant and 2 positive or negative

        mask_pos = (arr_corr > 0) & (arr_pvalue < 0.05)
        mask_neg = (arr_corr < 0) & (arr_pvalue < 0.05)


        # 满足条件的像元数
        count_positive_sig = np.sum(mask_pos)
        count_negative_sig = np.sum(mask_neg)

        # 百分比
        significant_positive = (count_positive_sig / len(arr_corr)) * 100
        significant_negative = (count_negative_sig / len(arr_corr)) * 100
        result_dic = {

            'sig neg': significant_negative,
            'non sig neg': arr_neg,
            'non sig pos': arr_pos,
            'sig pos': significant_positive



        }
        # df_new=pd.DataFrame(result_dic,index=[variable])
        # ## plot
        # df_new=df_new.T
        # df_new=df_new.reset_index()
        # df_new.columns=['Variable','Percentage']
        # df_new.plot.bar(x='Variable',y='Percentage',rot=45,color='green')
        # plt.show()
        color_list = [
            '#008837',
            '#a6dba0',

            '#c2a5cf',
            '#7b3294',
        ]
        width = 0.4
        alpha_list = [1, 0.5, 0.5, 1]

        # 逐个画 bar
        for i, (key, val) in enumerate(result_dic.items()):
            plt.bar(i , val, color=color_list[i], alpha=alpha_list[i], width=width)
            plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
            plt.ylabel('Percentage')
            plt.title(variable)

        plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
        plt.show()




    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

class Figure4:

    def __init__(self):
        self.dff=result_root+rf'\3mm\SHAP_beta\Dataframe\\moving_window.df'

    def run(self):
        # self.check_df_attributes()
        # self.plot_X_Y()
        self.bivariate_map()
        pass


    def check_df_attributes(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print(df.columns.tolist())
        print(len(df))
        # exit()
        pass
    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def bivariate_map(self): ## this plot for VPD and heavy rainfall days vs beta
        import xymap


        fdir =result_root + rf'3mm\SHAP_beta\png\RF_composite_LAI_beta\pdp_shap_beta_ALL_sig2\bivariate\\'

        outdir =result_root + rf'3mm\SHAP_beta\png\RF_composite_LAI_beta\pdp_shap_beta_ALL_sig2\bivariate\\'

        T.mkdir(outdir)

        outtif = join(outdir,'beta_VPD.tif')
        # outtif = join(outdir, 'LAImin_LAImax.tif')

        # fpath1 = join(fdir,'composite_LAI_detrend_relative_change_min_trend.tif')
        fpath1 = join(fdir,'composite_LAI_beta_trend.tif')
        # fpath2 = join(fdir,'composite_LAI_detrend_relative_change_max_trend.tif')
        fpath2 = join(fdir,'heavy_rainfall_days_trend.tif')

        #1
        # tif1_label, tif2_label = 'LAImin_trend','LAImax_trend'
        #2
        tif1_label, tif2_label = 'composite_LAI_beta_trend','VPD_trend'

        #1
        # min1, max1 = -1, 1
        # min2, max2 = -1, 1

        #2
        min1, max1 = -.5, .5
        min2, max2 = -.01, .01

        arr1 = ToRaster().raster2array(fpath1)[0]
        arr2 = ToRaster().raster2array(fpath2)[0]

        arr1[arr1<-9999] = np.nan
        arr2[arr2<-9999] = np.nan

        arr1_flattened = arr1.flatten()
        arr2_flattened = arr2.flatten()


        # plt.hist(arr1_flattened,bins=100)
        # plt.title('arr1')
        # plt.figure()
        # plt.hist(arr2_flattened,bins=100)
        # plt.title('arr2')
        # plt.show()

        # choice 1
        # upper_left_color = (193,92,156)
        # upper_right_color =(112, 196, 181)
        # lower_left_color = (237, 125, 49)
        # lower_right_color = (0, 0, 110)
        # center_color = (240, 240, 240)

        ## CV greening option

        upper_left_color = (194, 0, 120)
        upper_right_color = (0,170,237)
        lower_left_color = (233, 55, 43)
        # lower_right_color = (160, 108, 168)
        lower_right_color = (234, 233, 46)
        center_color = (240, 240, 240)


        xymap.Bivariate_plot_1(res = 2,
                         alpha = 255,
                         upper_left_color = upper_left_color, #
                         upper_right_color = upper_right_color, #
                         lower_left_color = lower_left_color, #
                         lower_right_color = lower_right_color, #
                         center_color = center_color).plot_bivariate(
                                                                    fpath1, fpath2,
                                                                    tif1_label, tif2_label,
                                                                    min1, max1,
                                                                    min2, max2,
                                                                    outtif,
                                                                    n_x = 5, n_y = 5
                                                                    )

        T.open_path_and_file(outdir)

    def plot_X_Y(self):
        dff=result_root+rf'3mm\SHAP_beta\Dataframe\\Trend.df'
        # dff=result_root+rf'\3mm\SHAP_beta\Dataframe\\moving_window.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        # df=df[df['continent']=='Africa']
        df=df[df['extraction_mask']==1]
        # df=df[df['composite_LAI_beta']>-5]
        # df=df[df['composite_LAI_beta']<5]

        Y=df['composite_LAI_beta_mean_trend'].tolist()
        X=df['sand'].tolist()
        Y=np.array(Y)
        X=np.array(X)
        mask = (~np.isnan(X)) & (~np.isnan(Y))
        Y=Y[mask]
        X=X[mask]
        ### add fiting line
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
        plt.plot(X, intercept + slope * X, color='r')
        plt.text(0.5, 0.5, f'r={r_value:.2f}, p={p_value:.2e}', transform=plt.gca().transAxes)

        KDE_plot().plot_scatter(X,Y)
        plt.ylim(-.5,.5)
        plt.xlabel('')
        plt.ylabel('composite_LAI_beta')
        plt.show()

        pass
class Plot_Robinson:
    def __init__(self):
        # plt.figure(figsize=(15.3 * centimeter_factor, 8.2 * centimeter_factor))
        self.map_width = 15.3 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass

    def robinson_template(self):
        '''
                :param fpath: tif file
                :param is_reproj: if True, reproject file from 4326 to Robinson
                :param res: resolution, meter
                '''
        color_list = [
            ['', '#fb9a99','#1f78b4',
             '#33a02c', '#a6cee3', '#fdbf6f',
             '#ff7f00', '#6a3d9a', '#b15928']
        ]
        # Blue represents high values, and red represents low values.
        plt.figure(figsize=(self.map_width, self.map_height))
        m = Basemap(projection='robin', lon_0=0, lat_0=90., resolution='c')

        m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        # for obj in meridict:
        #     line = meridict[obj][0][0]
        coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        polys = m.fillcontinents(color='#FFFFFF', lake_color='#EFEFEF', zorder=90)
    def plot_Robinson_significance_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5, s=20,
                                           c='k', marker='x',
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


    def plot_Robinson(self, fpath, ax=None, cmap=None, vmin=None, vmax=None, is_plot_colorbar=True, is_reproj=True,
                      res=25000, is_discrete=False, colormap_n=11):
        '''
        :param fpath: tif file
        :param is_reproj: if True, reproject file from 4326 to Robinson
        :param res: resolution, meter
        ## trend color list
        '''
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        color_list = [ '#33a02c','#1f78b4',
                       '#fb9a99',  '#a6cee3', '#fdbf6f',
         '#ff7f00', '#6a3d9a', '#b15928']


        # Blue represents high values, and red represents low values.
        if ax == None:
            # plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list)
        elif type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        if not is_reproj:
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        else:
            fpath_robinson = self.Robinson_reproj(fpath, fpath + '_robinson-reproj.tif', res=res)
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
        m = Basemap(projection='robin', lon_0=0, lat_0=90., ax=ax, resolution='c')
        ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax, )

        # m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        # m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        # meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        # meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        # for obj in meridict:
        #     line = meridict[obj][0][0]
        # coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        # polys = m.fillcontinents(color='#FFFFFF', lake_color='#EFEFEF', zorder=90)
        if is_plot_colorbar:
            if is_discrete:
                bounds = np.linspace(vmin, vmax, colormap_n)
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cax, kw = mpl.colorbar.make_axes(ax, location='bottom', pad=0.05, shrink=0.5)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds,
                                                 orientation='horizontal')
            else:
                cbar = plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom', pad=0.05)
        return m, ret

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







    def heatmap(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        f_sensitivity_trend = result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year_SNU_LAI\npy_time_series\\sum_rainfall_detrend_trend.tif'
        f_rainfall_trend=result_root+rf'\3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\trend\\\detrended_sum_rainfall_CV_trend.tif'
        f_CVLAI=result_root + rf'3mm\extract_SNU_LAI_phenology_year\moving_window_extraction\trend\\detrended_SNU_LAI_CV_trend.tif'
        outf = result_root + rf'\3mm\heatmap\\heatmap_CVLAI.pdf'
        arr_LAI_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_CVLAI)

        arr_LAI_trend[arr_LAI_trend < -999] = np.nan

        arr_LAI_sensitivity_precip, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_sensitivity_trend)
        arr_precip_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_rainfall_trend)
        arr_LAI_sensitivity_precip[arr_LAI_sensitivity_precip < -999] = np.nan
        arr_precip_trend[arr_precip_trend < -999] = np.nan
        arr_LAI_trend=np.array(arr_LAI_trend)
        arr_LAI_sensitivity_precip=np.array(arr_LAI_sensitivity_precip)
        arr_precip_trend=np.array(arr_precip_trend)

        dic_LAI_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_trend)
        dic_arr_LAI_sensitivity_precip=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_sensitivity_precip)
        dic_precip_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_precip_trend)

        result_dic={
            'LAI_CV_trend':dic_LAI_trend,
            'LAI_sensitivity_precip_trend':dic_arr_LAI_sensitivity_precip,
            'interannnual_rainallCV_trend':dic_precip_trend
        }
        # plt.hist(result_dic['LAI_CV_trend'].values())
        # plt.show()
        # plt.hist(result_dic['LAI_sensitivity_precip_trend'].values())
        # plt.show()
        # plt.hist(result_dic['interannnual_rainallCV_trend'].values())
        # plt.show();exit()
        df=T.spatial_dics_to_df(result_dic)
        T.print_head_n(df)
        x_var = 'LAI_sensitivity_precip_trend'
        y_var = 'interannnual_rainallCV_trend'
        z_var = 'LAI_CV_trend'
        # bin_x = [ -0.6,-0.4,-0.2,0,0.2,0.4,0.6,]
        bin_x = np.linspace(-2.5, 2.5, 11)
        # bin_y = [ -4, -3, -2, -1, 0, 1, 2, 3, 4, ]
        bin_y = np.linspace(-1.5, 1.5, 11)
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure(figsize=(self.map_width, self.map_height))

        matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)

        my_cmap = T.cmap_blend(color_list = ['#000000','r', 'b'])
        my_cmap = 'GnBu'
        self.plot_df_bin_2d_matrix(matrix_dict,-0.8,0.8,x_ticks_list,y_ticks_list,cmap=my_cmap,
                              is_only_return_matrix=False)
        plt.colorbar()
        pprint(matrix_dict)


        matrix_dict_count, x_ticks_list, y_ticks_list = self.df_bin_2d_count(df, val_col_name=z_var,
                                                              col_name_x=x_var,
                                                              col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        pprint(matrix_dict_count)
        scatter_size_dict = {
            (1,20): 5,
            (20,50): 20,
            (50,100): 50,
            (100,np.inf): 100
        }
        matrix_dict_count_normalized = {}
        # Normalize counts for circle size
        for key in matrix_dict_count:
            num = matrix_dict_count[key]
            for key2 in scatter_size_dict:
                if num >= key2[0] and num < key2[1]:
                    matrix_dict_count_normalized[key] = scatter_size_dict[key2]
                    break
        pprint(matrix_dict_count_normalized)
        reverse_x = list(range(len(bin_y)-1))[::-1]
        reverse_x_dict = {}
        for i in range(len(bin_y)-1):
            reverse_x_dict[i] = reverse_x[i]
        # print(reverse_x_dict);exit()
        for x,y in matrix_dict_count_normalized:
            plt.scatter(y,reverse_x_dict[x],s=matrix_dict_count_normalized[(x,y)],c='gray',edgecolors='none',alpha=.5)
        for x,y in matrix_dict_count_normalized:
            plt.scatter(y,reverse_x_dict[x],s=matrix_dict_count_normalized[(x,y)],c='none',edgecolors='gray',alpha=1)

        plt.xlabel('beta')
        plt.ylabel('Trend in Rainfall (mm/yr)')

        plt.show()
        # plt.savefig(outf)
        # plt.close()




    #     plt.savefig(result_root + rf'Data_frame\\Frequency\\Trendy_{region}.pdf', dpi=300, )
    #     plt.close()


    def heatmap_count(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        f_sensitivity_trend = result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year\npy_time_series\trend\\sum_rainfall_sensitivity_trend.tif'
        f_rainfall_trend=result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\ecosystem_year\trend\\\sum_rainfall_trend.tif'
        f_CVLAI=result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'

        arr_LAI_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_CVLAI)

        arr_LAI_trend[arr_LAI_trend < -999] = np.nan

        arr_LAI_sensitivity_precip, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_sensitivity_trend)
        arr_precip_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_rainfall_trend)
        arr_LAI_sensitivity_precip[arr_LAI_sensitivity_precip < -999] = np.nan
        arr_precip_trend[arr_precip_trend < -999] = np.nan
        arr_LAI_trend=np.array(arr_LAI_trend)
        arr_LAI_sensitivity_precip=np.array(arr_LAI_sensitivity_precip)
        arr_precip_trend=np.array(arr_precip_trend)

        dic_LAI_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_trend)
        dic_arr_LAI_sensitivity_precip=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_sensitivity_precip)
        dic_precip_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_precip_trend)

        result_dic={
            'LAI_CV':dic_LAI_trend,
            'LAI_sensitivity_precip_trend':dic_arr_LAI_sensitivity_precip,
            'Preci_trend':dic_precip_trend
        }
        # plt.hist(result_dic['LAI_CV'].values())
        # plt.show()
        # plt.hist(result_dic['LAI_sensitivity_precip_trend'].values())
        # plt.show()
        # plt.hist(result_dic['Preci_trend'].values())
        # plt.show();exit()
        df=T.spatial_dics_to_df(result_dic)
        T.print_head_n(df)
        x_var = 'LAI_sensitivity_precip_trend'
        y_var = 'Preci_trend'
        z_var = 'LAI_CV'
        # bin_x = [ -0.6,-0.4,-0.2,0,0.2,0.4,0.6,]
        bin_x = np.linspace(-0.7, 1.5, 13)
        # bin_y = [ -4, -3, -2, -1, 0, 1, 2, 3, 4, ]
        bin_y = np.linspace(-3, 3, 13)
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure()

        matrix_dict,x_ticks_list,y_ticks_list = self.df_bin_2d_count(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)

        self.plot_df_bin_2d_matrix(matrix_dict,0,200,x_ticks_list,y_ticks_list,
                              is_only_return_matrix=False)

        plt.xlabel('beta')
        plt.ylabel('Trend in Rainfall (mm/yr)')

        plt.colorbar()
        plt.show()


    def df_bin_2d_count(self,df,val_col_name,col_name_x,col_name_y,bin_x,bin_y,round_x=2,round_y=2):
        df_group_y, _ = self.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = self.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                rt_mean = len(vals)
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict,x_ticks_list,y_ticks_list

    def df_bin(self, df, col, bins):
        df_copy = df.copy()
        df_copy[f'{col}_bins'] = pd.cut(df[col], bins=bins)
        df_group = df_copy.groupby([f'{col}_bins'],observed=True)
        bins_name = df_group.groups.keys()
        bins_name_list = list(bins_name)
        bins_list_str = [str(i) for i in bins_name_list]
        # for name,df_group_i in df_group:
        #     vals = df_group_i[col].tolist()
        #     mean = np.nanmean(vals)
        #     err,_,_ = self.uncertainty_err(SM)
        #     # x_list.append(name)
        #     y_list.append(mean)
        #     err_list.append(err)
        return df_group, bins_list_str

    def plot_df_bin_2d_matrix(self,matrix_dict,vmin,vmax,x_ticks_list,y_ticks_list,cmap='RdBu',
                              is_only_return_matrix=False):
        keys = list(matrix_dict.keys())
        r_list = []
        c_list = []
        for r, c in keys:
            r_list.append(r)
            c_list.append(c)
        r_list = set(r_list)
        c_list = set(c_list)

        row = len(r_list)
        col = len(c_list)
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in matrix_dict:
                    val_pix = matrix_dict[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        matrix = np.array(spatial, dtype=float)
        matrix = matrix[::-1]
        if is_only_return_matrix:
            return matrix
        plt.imshow(matrix,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.xticks(range(len(c_list)), x_ticks_list)
        plt.yticks(range(len(r_list)), y_ticks_list[::-1])

class Figure2():
    def __init__(self):
        self.map_width = 8.2 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        self.__color_idx_array()
        pass

    def run(self):


        # self.statistic_bar_CV_greening()
        # self.bivariate_scheme3()
        # self.Figure2a_robinson()


        # self.Figure2b_test()

        # self.CV_greening_heatmap2()
        # self.statistic_bar()


        pass

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def statistic_bar_CV_greening(self):
        f_trend=rf'D:\Project3\Result\3mm\relative_change_growing_season\\TRENDY\trend_analysis\\composite_LAI_mean_trend.tif'
        f_cv=rf'D:\Project3\Result\3mm\extract_composite_phenology_year\trend\composite_LAI_CV_trend.tif'

        array_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_trend)
        array_cv, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_cv)

        array_trend[array_trend < -999] = np.nan
        array_cv[array_cv < -999] = np.nan
        plt.hist(array_trend.flatten())
        plt.show()
        plt.hist(array_cv.flatten())
        plt.show()
        # exit()
        mask = ~np.isnan(array_trend) & ~np.isnan(array_cv)
        trend_map_valid = array_trend[mask]
        cv_map_valid = array_cv[mask]



        n_bins =5

        cv_bins=np.linspace(-1,1,n_bins)
        trend_bins=np.linspace(-0.5,0.5,n_bins)
        # 5. Combine into a DataFrame
        df = pd.DataFrame({'CV': cv_map_valid, 'Trend': trend_map_valid})
        df=df.dropna()

        result_dic={}
        ###
        x_tick_list=[]
        percent_matrix = np.zeros((n_bins - 1, n_bins - 1))


        for i in range(n_bins):
            if i == n_bins - 1:
                continue
            df_group=df[(df['CV'] > cv_bins[i]) & (df['CV'] <= cv_bins[i + 1])]
            print(len(df_group))

            x_tick_list.append([i, 0])
            for j in range(n_bins):
                if j == n_bins - 1:
                    continue
                df_groupii=df_group[(df_group['Trend'] > trend_bins[j]) & (df_group['Trend'] <= trend_bins[j + 1])]['Trend']

                count=len(df_groupii)/len(df_group) *100

                percent_matrix[i,j]=count



        # print(result_dic);exit()
        x_label_list=['strongly CV-','moderately CV-','slight CV-','slight CV+','moderately CV+','strongly CV+']


        ## plt result
        # x轴标签 = CV组（主分组）
        x_labels = [f"{cv_bins[i]:.2f}~{cv_bins[i + 1]:.2f}" for i in range(n_bins - 1)]
        trend_labels = [f"{trend_bins[i]:.2f}~{trend_bins[i + 1]:.2f}" for i in range(n_bins - 1)]

        x = np.arange(len(x_labels))  # 每个主组位置
        width = 0.2  # 每个子柱宽度

        plt.figure(figsize=(self.map_width, self.map_height))
        color_list=['brown','red',   'green', 'blue', 'purple']
        flag=0

        for j in range(n_bins - 1):  # 对于每个 trend 组

            plt.bar(x + j * width, percent_matrix[:, j], width, label=f"Trend {trend_labels[j]}", color=color_list[flag])
            flag+=1

        plt.xticks(x + width * ((n_bins - 2) / 2), x_labels, rotation=45)
        plt.ylabel('Percentage (%)')

        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.show()

    def CV_greening_heatmap2(self):
        dff=rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\\Trend_all.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        df=df[df['composite_LAI_CV_p_value']<0.05]
        df=df[df['composite_LAI_relative_change_mean_p_value']<0.05]
        df=df.dropna()

        df['Group'] = df.apply(self.classify, axis=1)

            # Group and compute average LAImin trend
        result_dic_value={}
        result_pixel_percentage={}
        for group in df['Group'].unique():
            if group==2:
                print(df[df['Group']==group]['composite_LAI_detrend_relative_change_min_trend'])
                print(np.nanmean(df[df['Group']==group]['composite_LAI_detrend_relative_change_max_trend']))

            df_group = df[df['Group'] == group]
            LAImax = df_group['composite_LAI_detrend_relative_change_max_trend'].tolist()
            LAImin = df_group['composite_LAI_detrend_relative_change_min_trend'].tolist()
            LAImax_mean=np.nanmean(LAImax)
            LAImin_mean=np.nanmean(LAImin)
            df_group_percentage=len(df_group)/len(df)*100


            result_dic_value[group]=[LAImax_mean,LAImin_mean]
            result_pixel_percentage[group]=df_group_percentage

        ## plot LAImin and LAImax at the same time for all gropus
        dic_name={1:'CV+ & Greening',2:'CV+ & Browning',3:'CV- & Greening',4:'CV- & Browning'}

        plt.figure(figsize=(self.map_width, self.map_height))
        for i in range(1,5):
            plt.bar(i,result_dic_value[i][0],color='blue',width=0.2)
            plt.bar(i,result_dic_value[i][1],color='red',width=0.2)
        plt.xticks([1,2,3,4],list(dic_name.values()),rotation=0)
        ## add y=0 line
        plt.axhline(y=0, color='grey', linestyle='-')
        ## add text of percentage
        for i in range(1,5):
            plt.text(i,result_dic_value[i][0],f'{result_pixel_percentage[i]:.2f}',ha='center',va='top')

        plt.ylabel('Trend (%/yr)')
        plt.show()











        # Count how many pixels per group (optional)





    def classify(self,row):
        if row['composite_LAI_CV_trend'] > 0 and row['composite_LAI_relative_change_mean_trend'] > 0:
            return 1
        elif row['composite_LAI_CV_trend'] > 0 and row['composite_LAI_relative_change_mean_trend'] < 0:
            return 2
        elif row['composite_LAI_CV_trend'] < 0 and row['composite_LAI_relative_change_mean_trend'] > 0:
            return 3
        elif row['composite_LAI_CV_trend'] < 0 and row['composite_LAI_relative_change_mean_trend'] < 0:
            return 4
        else:
            return 'Other'

        # Apply classification

    def bivariate_scheme3(self):

        fdir = result_root + rf'\3mm\extract_composite_phenology_year\trend\\'

        outdir = result_root + rf'\3mm\extract_composite_phenology_year\bivariate\\'

        T.mkdir(outdir)

        outtif = join(outdir, 'CV_trend123.tif')
        # outtif = join(outdir, 'LAImin_LAImax.tif')

        # fpath1 = join(fdir,'composite_LAI_detrend_relative_change_min_trend.tif')
        fpath1 = join(fdir, 'composite_LAI_CV_trend.tif')
        # fpath2 = join(fdir,'composite_LAI_detrend_relative_change_max_trend.tif')
        fpath2 = join(fdir, 'composite_LAI_relative_change_mean_trend.tif')

        # 1
        # tif1_label, tif2_label = 'LAImin_trend','LAImax_trend'
        # 2
        tif1_label, tif2_label = 'LAI_CV_trend', 'LAI_trend'

        # 2
        bins_x = np.array([-np.inf, -0.5, 0, 0.5, np.inf])
        bins_y = np.array([-np.inf, -0.3, 0, 0.3, np.inf])

        arr1 = ToRaster().raster2array(fpath1)[0]
        arr2 = ToRaster().raster2array(fpath2)[0]

        arr1[arr1 < -9999] = np.nan
        arr2[arr2 < -9999] = np.nan

        dict1 = DIC_and_TIF().spatial_arr_to_dic(arr1)
        dict2 = DIC_and_TIF().spatial_arr_to_dic(arr2)
        dict_list = {'LAI_CV_trend': dict1, 'LAI_trend': dict2}
        df_new = T.spatial_dics_to_df(dict_list)
        df_new = df_new.dropna(how='any')
        ##
        T.print_head_n(df_new)

        arr_count = np.zeros((len(bins_x) - 1, len(bins_y) - 1)).flatten()
        arr = np.ones((360, 720, 4)) * 0
        for i, row in tqdm(df_new.iterrows(), total=len(df_new)):
            pix = row['pix']
            x = row[tif1_label]
            y = row[tif2_label]
            color_idx, color = self.get_color(x, y, bins_x, bins_y)
            arr_count[color_idx - 1] = arr_count[color_idx - 1] + 1
            # print('binsx',bins_x)
            # print('binsy',bins_y)
            # print(x,y,color_idx,color);exit()

            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            a = 255
            arr[pix] = np.array([r, g, b, a])
        self.RGBA_to_tif(arr, outtif, -180, 90, 0.5, -0.5)
        arr_count = arr_count.reshape((len(bins_x) - 1, len(bins_y) - 1))

        self.plot_legend(outdir,arr_count)

        T.open_path_and_file(outdir)


    def get_color(self, x, y, bins_x, bins_y):
        # color_array = [
        #     ['#f26d21 ', '#f598a1', '#ec1d8f', '#ec1d0f'],
        #     ['#c7e86e', '#7BC8F6', '#d3a3cb', '#f26d21'],
        #     ['#0e6b3f ', '#98cdd2', '#5d4a8d', '#f598a1'],
        #     ['#0e6b3f ', '#98cdd2', '#5d4a8d', '#7BC8F6'],
        # ][::-1]
        color_idx_array = self.color_idx_array
        color_dict = self.color_dict
        color_idx_array = np.array(color_idx_array)
        idx_x = np.digitize(x, bins_x) - 1
        idx_y = np.digitize(y, bins_y) - 1
        color_idx = color_idx_array[idx_y][idx_x]
        color = color_dict[color_idx]
        return color_idx, color

    def __color_idx_array(self):
        self.color_idx_array = [
                                   [1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 11, 12],
                                   [13, 14, 15, 16]
                               ][::-1]
        self.color_dict = {
            4: '#3182bd',
            3: '#6baed6',
            2: '#bd9e39',
            1: '#8c6d31',
            12: '#e6550d',
            11: '#fd8d3c',
            10: '#ad494a',
            9: '#843c39',
            8: '#31a354',
            7: '#74c476',
            6: '#e7969c',
            5: '#d6616b',
            16: '#756bb1',
            15: '#9e9ac8',
            14: '#cedb9c',
            13: '#b5cf6b'
        }

    def plot_legend(self, outdir,count_arr):

        color_array = []
        for row in self.color_idx_array:
            color_array_i = []
            for col in row:
                color = self.color_dict[col]
                rgb_color = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)  ### convert hex to rgb
                color_array_i.append(rgb_color)
            color_array.append(color_array_i)
        color_array = np.array(color_array)[::-1]
        sns.heatmap(self.color_idx_array,annot=count_arr, fmt='g',alpha=0,annot_kws={'alpha':1,'color':'k'})
        plt.imshow(color_array)

        plt.show()
        # plt.savefig(join(outdir, 'legend.pdf'))
        # plt.close()
        pass

    def Figure2a_robinson(self):

        fdir_trend = result_root + rf'3mm\extract_composite_phenology_year\bivariate\\'
        temp_root = result_root + rf'\3mm\extract_composite_phenology_year\bivariate\\'
        outdir = result_root + rf'\3mm\extract_composite_phenology_year\\bivariate\\ROBINSON\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):

            if not 'CV_trend123' in f:
                continue

            if not f.endswith('.tif'):
                continue
            print(f)

            fname = f.split('.')[0]

            fpath = fdir_trend + f
            outf = outdir + fname + '.tif'
            srcSRS = self.wkt_84()
            dstSRS = self.wkt_robinson()

            ToRaster().resample_reproj(fpath, outf, 50000, srcSRS=srcSRS, dstSRS=dstSRS)

            T.open_path_and_file(outdir)

    def wkt_robinson(self):
        wkt = '''PROJCRS["World_Robinson",
       BASEGEOGCRS["WGS 84",
           DATUM["World Geodetic System 1984",
               ELLIPSOID["WGS 84",6378137,298.257223563,
                   LENGTHUNIT["metre",1]]],
           PRIMEM["Greenwich",0,
               ANGLEUNIT["Degree",0.0174532925199433]]],
       CONVERSION["World_Robinson",
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
       ID["ESRI",54030]]
           '''
        return wkt

    def wkt_84(self):
        wkt = '''GEOGCRS["WGS 84",
       ENSEMBLE["World Geodetic System 1984 ensemble",
           MEMBER["World Geodetic System 1984 (Transit)"],
           MEMBER["World Geodetic System 1984 (G730)"],
           MEMBER["World Geodetic System 1984 (G873)"],
           MEMBER["World Geodetic System 1984 (G1150)"],
           MEMBER["World Geodetic System 1984 (G1674)"],
           MEMBER["World Geodetic System 1984 (G1762)"],
           MEMBER["World Geodetic System 1984 (G2139)"],
           ELLIPSOID["WGS 84",6378137,298.257223563,
               LENGTHUNIT["metre",1]],
           ENSEMBLEACCURACY[2.0]],
       PRIMEM["Greenwich",0,
           ANGLEUNIT["degree",0.0174532925199433]],
       CS[ellipsoidal,2],
           AXIS["geodetic latitude (Lat)",north,
               ORDER[1],
               ANGLEUNIT["degree",0.0174532925199433]],
           AXIS["geodetic longitude (Lon)",east,
               ORDER[2],
               ANGLEUNIT["degree",0.0174532925199433]],
       USAGE[
           SCOPE["Horizontal component of 3D system."],
           AREA["World."],
           BBOX[-90,-180,90,180]],
       ID["EPSG",4326]]'''
        return wkt

    def RGBA_to_tif(self,blend_arr,outf,originX, originY, pixelWidth, pixelHeight):
        import PIL.Image as Image
        img = Image.fromarray(blend_arr.astype('uint8'), 'RGBA')
        img.save(outf)
        # define a projection and extent
        raster = gdal.Open(outf)
        geotransform = raster.GetGeoTransform()
        raster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outRasterSRS = osr.SpatialReference()
        projection = self.wkt_84()
        # outRasterSRS.ImportFromEPSG(4326)
        # outRasterSRS.ImportFromEPSG(projection)
        # raster.SetProjection(outRasterSRS.ExportToWkt())
        raster.SetProjection(projection)
        pass



    def Figure2b_test(self):
        fdir = result_root + rf'\3mm\extract_composite_phenology_year\trend\\'

        outdir = result_root + rf'\3mm\extract_composite_phenology_year\bivariate\\'
        self.plot_legend(outdir)

        T.mkdir(outdir)

        outtif = join(outdir, 'CV_trend_test.tif')
        # outtif = join(outdir, 'LAImin_LAImax.tif')

        # fpath1 = join(fdir,'composite_LAI_detrend_relative_change_min_trend.tif')
        fpath1 = join(fdir, 'composite_LAI_CV_trend.tif')
        # fpath2 = join(fdir,'composite_LAI_detrend_relative_change_max_trend.tif')
        fpath2 = join(fdir, 'composite_LAI_relative_change_mean_trend.tif')

        # 1
        # tif1_label, tif2_label = 'LAImin_trend','LAImax_trend'
        # 2
        tif1_label, tif2_label = 'LAI_CV_trend', 'LAI_trend'

        # 2
        bins_x = np.array([-np.inf, -0.5, 0, 0.5, np.inf])
        bins_y = np.array([-np.inf, -0.3, 0, 0.3, np.inf])

        arr1 = ToRaster().raster2array(fpath1)[0]
        arr2 = ToRaster().raster2array(fpath2)[0]

        arr1[arr1 < -9999] = np.nan
        arr2[arr2 < -9999] = np.nan

        dict1 = DIC_and_TIF().spatial_arr_to_dic(arr1)
        dict2 = DIC_and_TIF().spatial_arr_to_dic(arr2)
        dict_list = {'LAI_CV_trend': dict1, 'LAI_trend': dict2}
        df_new = T.spatial_dics_to_df(dict_list)
        df_new = df_new.dropna(how='any')
        ##
        T.print_head_n(df_new)

        arr_count=np.zeros((len(bins_x)-1,len(bins_y)-1)).flatten()
        for i, row in tqdm(df_new.iterrows(), total=len(df_new)):
            pix = row['pix']
            x = row[tif1_label]
            y = row[tif2_label]
            color_idx, color = self.get_color(x, y, bins_x, bins_y)
            arr_count[color_idx-1]=arr_count[color_idx-1]+1

        arr_count=arr_count.reshape((len(bins_x)-1,len(bins_y)-1))
        arr_percentage=arr_count/np.nansum(arr_count)*100
        arr_log=np.log10(arr_count)
        fig,ax=plt.subplots(1,1,figsize=(5,5))

        plt.imshow(arr_percentage, cmap='GnBu', vmin=5, vmax=40)
        ## add line y=0 and x=0
        ax.axhline(y=1.5, color='k', linewidth=1)
        ax.axvline(x=1.5, color='k', linewidth=1)
        ## add label
        ax.set_xlabel('Trend in CVLAI (%/yr)', fontsize=10)
        ax.set_ylabel('Trend in LAI (%/yr)',fontsize=10)
        ## xtick is empty
        ax.set_xticks([])
        ax.set_yticks([])
        ## set colorbar name 'percentage
        # cbar = plt.colorbar()
        # cbar.ax.set_title('Percentage', fontsize=10)

        # plt.tight_layout()
        plt.colorbar()

        plt.show()

        # plt.savefig(outtif, dpi=300, bbox_inches='tight')
        # plt.close()

    pass






    def statistic_bar(self):
        dff=rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\Trend.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        print(len(df))
        # df=df[df['detrended_SNU_LAI_CV_p_value']<0.05]
        # df=df[df['SNU_LAI_relative_change_p_value']<0.05]

        df = df[df['LAI4g_detrend_CV_p_value'] < 0.05]
        df = df[df['LAI4g_p_value'] < 0.05]

        # print(len(df));exit()
        SNU_LAI_trend_values = df['LAI4g_trend'].tolist()

        SNU_LAI_CV_values = df['LAI4g_detrend_CV_trend'].tolist()


        SNU_LAI_trend_values = np.array(SNU_LAI_trend_values)

        SNU_LAI_CV_values = np.array(SNU_LAI_CV_values)
        ## CV>0 and trend >0 , CV>0 and trend <0 CV < 0 and trend > 0 CV < 0 and trend < 0
        class1=np.logical_and(SNU_LAI_CV_values>0,SNU_LAI_trend_values>0)
        class2=np.logical_and(SNU_LAI_CV_values>0,SNU_LAI_trend_values<0)
        class3=np.logical_and(SNU_LAI_CV_values<0,SNU_LAI_trend_values>0)
        class4=np.logical_and(SNU_LAI_CV_values<0,SNU_LAI_trend_values<0)
        ## calculate the percentage of each class
        class1_percentage = np.sum(class1)/len(df)*100
        class2_percentage = np.sum(class2)/len(df)*100
        class3_percentage = np.sum(class3)/len(df)*100
        class4_percentage = np.sum(class4)/len(df)*100
        print(class1_percentage,class2_percentage,class3_percentage,class4_percentage)
        plt.bar([1,2,3,4],[class1_percentage,class2_percentage,class3_percentage,class4_percentage])
        plt.xticks([1,2,3,4],['CV>0 and trend >0','CV>0 and trend <0','CV < 0 and trend > 0','CV < 0 and trend < 0'])
        plt.ylabel('Percentage')
        plt.show()


    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']
        return df





    def mode(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        dff=rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\\Trend.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        print(len(df))
        df = df[df['detrended_SNU_LAI_CV_p_value'] < 0.05]
        df=df[df['SNU_LAI_relative_change_p_value']<0.05]
        df=df[df['SNU_LAI_relative_change_trend']>0]
        df = df[df['SNU_LAI_CV'] < 0]
        LAI_max_list=[]
        LAI_min_list=[]


        for i,row in df.iterrows():
            time_series_LAImin=row['SNU_LAI_relative_change_detrend_min']

            time_series_LAImin=np.array(time_series_LAImin)
            # print(time_series_LAImin)
            if np.isnan(time_series_LAImin).all():
                continue

            if time_series_LAImin.shape[0]!=24:
                continue
            time_series_LAImax=row['SNU_LAI_relative_change_detrend_max']
            time_series_LAImax=np.array(time_series_LAImax)
            if time_series_LAImax.shape[0]!=24:
                continue

            LAI_max_list.append(time_series_LAImax)
            LAI_min_list.append(time_series_LAImin)
        ## average
        LAI_max_list=np.array(LAI_max_list)
        LAI_min_list=np.array(LAI_min_list)
        LAI_max_list_avg=np.mean(LAI_max_list,axis=0)
        LAI_min_list_avg=np.mean(LAI_min_list,axis=0)
        slope_laimin, intercept_laimin, r_value_laimin, p_value_laimin, std_err_laimin = stats.linregress(np.arange(len(LAI_min_list_avg)), LAI_min_list_avg)
        slope_laimax, intercept_laimax, r_value_laimax, p_value_laimax, std_err_laimax = stats.linregress(np.arange(len(LAI_max_list_avg)), LAI_max_list_avg)
        plt.figure(figsize=(self.map_width, self.map_height))



        plt.plot(LAI_min_list_avg,'b')
        ## plot regression line
        plt.plot(np.arange(len(LAI_min_list_avg)), [slope_laimin * x + intercept_laimin for x in np.arange(len(LAI_min_list_avg))],
                 linestyle='--', color='b', alpha=0.5)
        plt.text(10, -15, f'{slope_laimin:.2f}*x+{intercept_laimin:.2f} p={p_value_laimin:.2f}',
                 fontsize=12)
        print(f'{slope_laimin:.2f}*LAImin+{intercept_laimin:.2f}')
        print(p_value_laimin)

        plt.plot(LAI_max_list_avg,'r')
        plt.plot(np.arange(len(LAI_max_list_avg)),
                 [slope_laimax * x + intercept_laimax for x in np.arange(len(LAI_max_list_avg))],
                 linestyle='--', color='r', alpha=0.5)
        ## text regression model y=ax+b
        plt.text(10, 20, f'{slope_laimax:.2f}*x+{intercept_laimax:.2f} p={p_value_laimax:.2f}',
                 fontsize=12)
        print(f'{slope_laimax:.2f}*LAImax+{intercept_laimax:.2f}')
        print(p_value_laimax)

        plt.xticks(range(0, 24, 3))
        window_size = 15

        # set xticks with 1982-1997, 1998-2013,.. 2014-2020
        year_range = range(1983, 2021)
        year_range_str = []
        for year in year_range:

            start_year = year
            end_year = year + window_size - 1
            if end_year > 2020:
                break
            year_range_str.append(f'{start_year}-{end_year}')
        plt.xticks(range(len(year_range_str))[::3], year_range_str[::3], rotation=45, ha='right')


        plt.ylabel ('Relative change (%)')
        plt.tight_layout()
        plt.legend(['LAImin','',  'LAImax',''], loc='upper left')

        plt.show()



        # plt.show();exit()




        # plt.close()

class build_dataframe():


    def __init__(self):

        self.this_class_arr = (result_root+rf'\3mm\Multiregression\partial_correlation\Obs\obs_climate_fire\Dataframe\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + rf'Dataframe.df'

        pass

    def run(self):



        df = self.__gen_df_init(self.dff)
        # df=self.foo2(df)
        # df=self.add_detrend_zscore_to_df(df)
        # df=self.append_attributes(df)
        df=self.add_trend_to_df(df)
        # df=self.ensemble_to_df(df)
        # df=self.add_wet_dry_to_df(df)
        # df=self.add_8class_to_df(df)


        # df=self.add_aridity_to_df(df)
        # df=self.add_dryland_nondryland_to_df(df)
        # df=self.add_MODIS_LUCC_to_df(df)
        # df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        # df=self.add_landcover_classfication_to_df(df)
        # df=self.add_maxmium_LC_change(df)
        # df=self.add_row(df)
        #
        # df=self.add_lat_lon_to_df(df)



        # df=self.rename_columns(df)
        # df = self.drop_field_df(df)
        df=self.show_field(df)


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
    def append_attributes(self, df):  ## add attributes
        fdir = result_root+ rf'3mm\extract_SNU_LAI_phenology_year\moving_window_min_max_anaysis\\'
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
                continue
            if not 'relative_change' in f:
                continue

            # array=np.load(fdir+f)
            # dic = DIC_and_TIF().spatial_arr_to_dic(array)
            dic=T.load_npy(fdir+f)
            key_name = f.split('.')[0]
            print(key_name)

            # df[key_name] = df['pix'].map(dic)
            # T.print_head_n(df)
            df=T.add_spatial_dic_to_df(df,dic,key_name)
        return df

    def add_detrend_zscore_to_df(self, df):

        fdir=rf'D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\extraction_LAI4g\\'


        for f in os.listdir(fdir):
            variable=f.split('.')[0]
            print(variable)
            if not 'LAI4g_detrend_CV_p_value' in variable:
                continue




            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year
                # pix = row.pix
                pix = row['pix']
                r, c = pix

                if not pix in val_dic:
                    NDVI_list.append(np.nan)
                    continue

                vals = val_dic[pix]
                print(len(vals))

                if len(vals)==0:
                    NDVI_list.append(np.nan)
                    continue


                if len(vals)==33 :
                    nan_list=np.array([np.nan]*5)
                    vals=np.append(vals,nan_list)



                v1= vals[year - 0]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)


            df[f'{variable}'] = NDVI_list
        # exit()
        return df
    def add_wet_dry_to_df(self,df):
        # T.print_head_n(df);exit()

        # df['CO2_rainfall']=df['CO2']*df['rainfall_intensity_average_zscore']
        df['wet_dry'] = 'unknown'
        df.loc[df['sum_rainfall_trend'] > 0, 'wet_dry'] = 'wetting'
        df.loc[df['sum_rainfall_trend'] < 0, 'wet_dry'] = 'drying'
        return df

    def add_row(self, df):
        r_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix
            r_list.append(r)
        df['row'] = r_list
        return df

    def foo2(self, df):  # 新建trend

        f = result_root + rf'3mm\relative_change_growing_season\moving_window_min_max_anaysis\max\trend_analysis\\TRENDY_ensemble_detrend_max_trend.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        # val_array = np.load(f)
        # val_array[val_array<-99]=np.nan
        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        # plt.imshow(val_array)
        # plt.colorbar()
        # plt.show()

        # exit()

        pix_list = []
        for pix in tqdm(val_dic):
            val = val_dic[pix]
            if np.isnan(val):
                continue
            pix_list.append(pix)
        df['pix'] = pix_list
        T.print_head_n(df)

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
    def add_continent_to_df(self, df):
        tiff = rf'E:\Project3\Data\Base_data\\continent_05.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'continent'

        dic_convert={1:'Africa',2:'Asia',3:'Australia',4: np.nan, 5:'South_America', 6: np.nan, 7:'Europe',8:'North_America',255: np.nan}

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            # print(val)

            val_convert=dic_convert[val]

            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val_convert)
        df[f_name] = val_list
        return df

    pass
    def add_lat_lon_to_df(self, df):
        D=DIC_and_TIF(pixelsize=0.5)
        df=T.add_lon_lat_to_df(df,D)
        return df


    def add_soil_texture_to_df(self, df):
        fdir = data_root + rf'\Base_data\\SoilGrid\SOIL_Grid_05_unify\\weighted_average\\'
        for f in (os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue

            tiff = fdir + f

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
            array = np.array(array, dtype=float)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
            fname=f.split('.')[0]
            print(fname)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[fname] = val_list
        return df
        pass

    def add_rooting_depth_to_df(self, df):
        tiff=rf'D:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth_05.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'rooting_depth'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f_name] = val_list
        return df

        pass
    def add_Ndepostion_to_df(self, df):
        fdir='D:\Project3\Result\extract_GS\OBS_LAI_extend\\noy\\'
        val_dic = T.load_npy_dir(fdir)
        f_name = 'Noy'
        print(f_name)
        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            year = row.year
            # pix = row.pix
            pix = row['pix']
            r, c = pix

            if not pix in val_dic:
                NDVI_list.append(np.nan)
                continue

            vals = val_dic[pix]

            # print(len(vals))
            ##### if len vals is 38, the end of list add np.nan

            # if len(vals) == 38:
            #     vals=np.append(vals,np.nan)
            #     v1 = vals[year - 1982]
            #     NDVI_list.append(v1)
            # if len(vals)==39:
            # v1 = vals[year - 1982]
            # v1 = vals[year - 1982]
            v1 = vals[year - 0]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)

        df[f_name] = NDVI_list
        # exit()

        return df

    def add_area_to_df(self, df):
        area_dic=DIC_and_TIF(pixelsize=0.25).calculate_pixel_area()
        f_name = 'pixel_area'
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in area_dic:
                val_list.append(np.nan)
                continue
            val = area_dic[pix]
            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f_name] = val_list
        return df


    def add_ozone_to_df(self, df):
            f=rf'D:\Project3\Result\extract_GS\OBS_LAI_extend\\ozone.npy'
            val_dic = T.load_npy(f)
            f_name = 'ozone'
            print(f_name)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year_range
                # pix = row.pix
                pix = row['pix']
                r, c = pix

                if not pix in val_dic:
                    NDVI_list.append(np.nan)
                    continue

                vals = val_dic[pix]

                # print(len(vals))
                ##### if len vals is 38, the end of list add np.nan

                if len(vals) == 37:
                    ## append 2 nan
                    vals=np.append(vals,np.nan)
                    vals=np.append(vals,np.nan)

                    v1 = vals[year - 0]
                    NDVI_list.append(v1)


            df[f_name] = NDVI_list
            # exit()

            return df

            pass
    def add_root_depth_to_df(self, df):
        tiff=rf'D:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'rooting_depth'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val = val_dic[pix]
            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f_name] = val_list
        return df

        pass
    def add_precipitation_CV_to_df(self, df):
        fdir='D:\Project3\Result\state_variables\\CV_monthly\\'
        for f in os.listdir(fdir):

            val_dic = T.load_npy(fdir + f)
            f_name = f.split('.')[0]+'_CV'
            print(f_name)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix'][0]
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name] = val_list
        return df

        pass

    def add_events(self, df):
        fdir = result_root + rf'relative_change\events_extraction\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dict_ = T.load_npy(fdir + f)
            key_name = f.split('.')[0]+'_event_level'
            print(key_name)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                year= row['year']
                if not pix in dict_:
                    val_list.append(np.nan)
                    continue
                dic_i = dict_[pix]
                mode=np.nan

                for key in dic_i:
                    idx_list=dic_i[key]
                    for idx in idx_list:
                        yeari=idx+1982
                        if yeari==year:
                            mode=key
                            break
                val_list.append(mode)
            df[key_name] = val_list

        T.print_head_n(df)
        return df


    def add_trend_to_df_scenarios(self,df):
        mode_list=['wet','dry']
        for mode in mode_list:
            period_list=['1982_2000','2001_2020','1982_2020']
            for period in period_list:

                fdir=result_root+rf'\monte_carlo\{mode}\\{period}\\'

                for f in os.listdir(fdir):
                    # print(f)
                    # exit()
                    if not 'trend' in f:
                        continue


                    if not f.endswith('.tif'):
                        continue



                    variable=(f.split('.')[0])



                    array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
                    array = np.array(array, dtype=float)

                    val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

                    # val_array = np.load(fdir + f)
                    # val_dic=T.load_npy(fdir+f)

                    # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                    f_name=f.split('.')[0]+'_'+period
                    print(f_name)

                    val_list=[]
                    for i,row in tqdm(df.iterrows(),total=len(df)):
                        pix=row['pix']
                        if not pix in val_dic:
                            val_list.append(np.nan)
                            continue
                        val=val_dic[pix]
                        if val<-99:
                            val_list.append(np.nan)
                            continue
                        if val>99:
                            val_list.append(np.nan)
                            continue
                        val_list.append(val)
                    df[f'{f_name}']=val_list

        return df

    def add_trend_to_df(self, df):
        # for col in df.columns:
        #     print(col)
        # exit()
        model_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                           'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                           'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                           'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                           'ORCHIDEE_S2_lai',
                           'YIBs_S2_Monthly_lai',

                           ]
        model_list = ['composite_LAI']
        for model in model_list:
            fdir=rf'D:\Project3\Result\3mm\Multiregression\partial_correlation\Obs\obs_climate_fire\result\\{model}\\'
            # fdir=rf'D:\Project3\Result\3mm\Multiregression\partial_correlation\Obs\obs_climate_fire\result\\{model}\\'

            for f in os.listdir(fdir):



                if not f.endswith('.tif'):
                    continue
                if not 'zscore' in f:
                    continue



                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
                array = np.array(array, dtype=float)

                val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

                # val_array = np.load(fdir + f)
                # val_dic=T.load_npy(fdir+f)

                # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                f_name = f.split('.')[0]

                if 'beta' in f_name:
                    fname_new=f_name
                else:
                    fname_new=f'{model}_{f_name}'
                print(fname_new)
                # fname_new=f_name

                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    val = val_dic[pix]
                    if val < -99:
                        val_list.append(np.nan)
                        continue
                    # if val > 99:
                    #     val_list.append(np.nan)
                    #     continue
                    val_list.append(val)


                df[f'{fname_new}'] = val_list


        return df
    def ensemble_to_df(self,df):
        for col in df.columns:
            print(col)


        model_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',
                      'YIBs_S2_Monthly_lai',

                      ]
        # model_list = ['GLOBMAP_LAI', 'LAI4g', 'SNU_LAI']
        var_list = ['CV_intraannual_rainfall_ecosystem_year', 'detrended_sum_rainfall_CV',
                    'sensitivity','Fire_sum_average']

        for var in var_list:

                ## pick all model field and avergae
            df[f'TRENDY_mean_{var}_zscore_norm']=df[[f'{model}_{var}_zscore_norm' for model in model_list]].mean(axis=1)
            ## df to dic
        for var in var_list:
            # outdir=result_root+rf'\3mm\Multiregression\partial_correlation\TRENDY\obs_climate_fire\result\\TRENDY_mean\\'
            outdir=result_root+rf'\3mm\Multiregression\partial_correlation\TRENDY\Result\climate_fire_sensitivity\\'
            T.mk_dir(outdir,force=True)
            outf=outdir+f'TRENDY_mean_{var}_zscore_norm.tif'
            dic=T.df_to_spatial_dic(df,f'TRENDY_mean_{var}_zscore_norm')

            tiff=DIC_and_TIF().pix_dic_to_tif(dic,outf)
        return df


    def add_8class_to_df(self, df):
        fdir=rf'D:\Project3\Result\3mm\bivariate_analysis\composite_LAI\\'
        for f in os.listdir(fdir):
            if not 'CV_rainfall_beta_LAI_composite_growing_season' in f:
                continue



            if not f.endswith('.tif'):
                continue

            variable = (f.split('.')[0])

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

            # val_array = np.load(fdir + f)
            # val_dic=T.load_npy(fdir+f)

            # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)

            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -99:
                    val_list.append(np.nan)
                    continue
                # if val > 99:
                #     val_list.append(np.nan)
                #     continue
                val_list.append(val)


            df[f'{f_name}'] = val_list


        return df


    def add_mean_to_df(self, df):
        fdir=rf'D:\Project3\Result\state_variables\mean\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            variable = (f.split('.')[0])
            if not 'GPCC' in variable:
                continue


            val_dic=T.load_npy(fdir+f)

            # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)

            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < 0:
                    val_list.append(np.nan)
                    continue
                if val > 9999:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            # df[f'{f_name}'] = val_list
            df['MAP'] = val_list


        return df





    def rename_columns(self, df):
        ## print columns
        for col in df.columns:
            print(col)
        # exit()
        df = df.rename(columns={'TRENDY_mean_sensitivity_zscore_norm_norm_2': 'TRENDY_median_detrended_sum_rainfall_CV_zscore_norm',
                                'TRENDY_median_TRENDY_median_CV_intraannual_rainfall_ecosystem_year_zscore_norm': 'TRENDY_median_CV_intraannual_rainfall_ecosystem_year_zscore_norm',
                                'TRENDY_average_TRENDY_average_detrended_sum_rainfall_CV_zscore_norm': 'TRENDY_average_detrended_sum_rainfall_CV_zscore_norm',
                                'TRENDY_average_TRENDY_average_CV_intraannual_rainfall_ecosystem_year_zscore_norm': 'TRENDY_average_CV_intraannual_rainfall_ecosystem_year_zscore_norm',





                            }

                               )



        return df
    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=[


                              'TRENDY_mean_sensitivity_zscore_norm_norm_2',





                              ])
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
    def add_MODIS_LUCC_to_df(self, df):
        f = data_root + rf'\Base_data\MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
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
            vals = val_dic[pix]
            val_list.append(vals)
        df['MODIS_LUCC'] = val_list
        return df



    def add_landcover_data_to_df(self, df):

        f = data_root + rf'\Base_data\\glc_025\\glc2000_05.tif'

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
            vals = val_dic[pix]
            val_list.append(vals)

        df['landcover_GLC'] = val_list
        return df
    def add_landcover_classfication_to_df(self, df):

        val_list=[]
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix=row['pix']
            landcover=row['landcover_GLC']
            if landcover==0 or landcover==4:
                val_list.append('Evergreen')
            elif landcover==2 or landcover==4 or landcover==5:
                val_list.append('Deciduous')
            elif landcover==6:
                val_list.append('Mixed')
            elif landcover==11 or landcover==12:
                val_list.append('Shrub')
            elif landcover==13 or landcover==14 or landcover==15:
                val_list.append('Grass')
            elif landcover==16 or landcover==17 or landcover==18:
                val_list.append('Cropland')
            else:
                val_list.append(np.nan)
        df['landcover_classfication']=val_list

        return df


        pass
    def add_maxmium_LC_change(self, df): ##

        f = data_root+rf'\Base_data\lc_trend\\max_trend.tif'

        array, origin, pixelWidth, pixelHeight, extent = ToRaster().raster2array(f)
        array[array <-99] = np.nan

        LC_dic =DIC_and_TIF().spatial_arr_to_dic(array)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix

            val= LC_dic[pix]
            df.loc[i,'LC_max'] = val
        return df

    def add_aridity_to_df(self,df):  ## here is original aridity index not classification

        f=data_root+rf'Base_data\\aridity_index_05\\aridity_index.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)

        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        # val_array = np.load(fdir + f)

        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        f_name='Aridity'
        print(f_name)
        val_list=[]
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix=row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val=val_dic[pix]
            if val<-99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f'{f_name}']=val_list

        return df




    def add_AI_classfication(self, df):

        f = data_root + rf'\Base_data\aridity_index_05\\aridity_index.tif'

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
            elif val==2:
                label='Sub-Humid'
            elif val<-99:
                label=np.nan
            else:
                raise




            val_list.append(label)

        df['AI_classfication'] = val_list
        return df
    def add_dryland_nondryland_to_df(self, df):
        fpath=data_root+rf'\\Base_data\\aridity_index_05\\aridity_index.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue

            val = val_dic[pix]
            if val <= 0.65:
                label = 'Dryland'
            elif 0.65 < val <= 0.8:
                label = 'Sub-humid'
            elif 0.8 < val <= 1.5:
                label = 'Humid'
            elif val > 1.5:
                label = 'Very Humid'
            elif val < -99:
                label = np.nan
            else:
                raise
            val_list.append(label)
        df['Dryland_Humid'] = val_list

        return df




    def show_field(self, df):
        for col in df.columns:
            print(col)
        return df
        pass

class partial_correlation():
    def __init__(self):
        pass

        self.fdirX = result_root + rf'3mm\Multiregression\partial_correlation\Obs\obs_climate\\input\\X\\'
        self.fdirY = result_root + rf'\3mm\Multiregression\partial_correlation\Obs\obs_climate\\input\\Y\\'

    def run(self):

        self.xvar_list = ['rainfall_frenquency_zscore',
                          'detrended_sum_rainfall_growing_season_zscore',
                          ]
        self.model_list = ['GLOBMAP_LAI','SNU_LAI','LAI4g']



        for model in self.model_list:
            self.outdir = result_root + rf'\3mm\Multiregression\partial_correlation\Obs\obs_climate\\result\\\\{model}\\'
            T.mk_dir(self.outdir, force=True)
            self.outpartial = self.outdir + rf'\partial_corr_{model}.npy'
            self.outpartial_pvalue = self.outdir + rf'\partial_pvalue_{model}.npy'

            y_var = f'{model}_detrend_CV_zscore.npy'
            x_var_list = self.xvar_list + [f'{model}_sensitivity_zscore']

            #
            df=self.build_df(self.fdirX,self.fdirY,x_var_list,y_var)
            #
            self.cal_partial_corr(df,x_var_list, )
            # #
            # # # # # # self.check_data()
            self.plot_partial_correlation()
            self.plot_partial_correlation_p_value()

            # self.maximum_partial_corr()
            # self.normalized_partial_corr(model)
            # self.normalized_partial_corr_unpacked(model)
            # self.normalized_partial_corr_ensemble(model)
            # self.plot_pdf()
            # self.statistic_corr()
            # self.statistic_trend()

            # self.pft_test2()
            # self.pft_max_label()
            # self.pft_corr()
            # self.aridity_bin()


    def check_data(self):
        f=result_root+rf'\3mm\Multiregression\zscore\\Fire_sum_max_zscore.npy'
        dic=T.load_npy(f)

        val_list=[]
        for pix in dic:
            val=dic[pix]
            val_list.append(val)
        val_list=np.array(val_list)
        val_list=val_list[~np.isnan(val_list)]
        val_list=val_list.flatten()
        plt.hist(val_list)
        plt.show()



        pass
    def build_df(self,fdir_X,fdir_Y,fx_list,fy):
        df = pd.DataFrame()

        filey = fdir_Y + fy
        print(filey)

        dic_y = T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix][0:22]

            if len(yvals) == 0:
                continue
            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
            yvals=yvals
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
            filex = fdir_X + xvar+'.npy'


            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
            dic_x = T.load_npy(filex)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                xvals = dic_x[pix][0:22]
                xvals = np.array(xvals)
                xvals = xvals
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
                # print(x_var_list_valid_new_cov)
                x_var_list_valid_new_cov.remove(x)
                # print(x_var_list_valid_new_cov)
                r, p = self.partial_corr(df_new, x, 'y', x_var_list_valid_new_cov)
                partial_correlation[x] = r
                partial_correlation_p_value[x] = p

            partial_correlation_dic[pix] = partial_correlation
            partial_p_value_dic[pix] = partial_correlation_p_value
        T.save_npy(partial_correlation_dic, outf_corr)
        T.save_npy(partial_p_value_dic, outf_pvalue)





            # print(df_new)


    def cal_single_correlation(self):
        f_x= result_root + rf'\3mm\Multiregression\input\\sum_rainfall.npy'
        f_y = result_root + rf'\3mm\Multiregression\input\\composite_LAI_beta_mean.npy'
        outdir=join(result_root, 'Multiregression', 'correlation')
        T.mk_dir(outdir, force=True)
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
            y_val = y_val[0:24]

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
        arr=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_r_dic)

        outf=outdir+'\\sum_rainfall.tif'
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr,outf)
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

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)



        f_partial = self.outpartial
        # f_pvalue = self.outpartial_pvalue
        outdir= self.outdir


        partial_correlation_dic = np.load(f_partial, allow_pickle=True, encoding='latin1').item()
        # partial_correlation_p_value_dic = np.load(f_pvalue, allow_pickle=True, encoding='latin1').item()


        var_list = []
        for pix in partial_correlation_dic:



            vals = partial_correlation_dic[pix]
            # vals = partial_correlation_p_value_dic[pix]


            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in partial_correlation_dic:
                r, c = pix
                if r < 60:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                dic_i = partial_correlation_dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, self.outdir + f'{var_i}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            # plt.figure()
            # arr[arr > 0.1] = 1
            # plt.imshow(arr, vmin=-1, vmax=1)
            #
            # plt.title(var_i)
            # plt.colorbar()

        # plt.show()

    def plot_partial_correlation_p_value(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)




        f_pvalue = self.outpartial_pvalue
        outdir= self.outdir



        partial_correlation_p_value_dic = np.load(f_pvalue, allow_pickle=True, encoding='latin1').item()


        var_list = []
        for pix in partial_correlation_p_value_dic:




            vals = partial_correlation_p_value_dic[pix]


            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in partial_correlation_p_value_dic:
                r, c = pix
                if r < 60:
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
            arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, self.outdir + f'{var_i}_p_value.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            # plt.figure()
            # arr[arr > 0.1] = 1
            # plt.imshow(arr, vmin=-1, vmax=1)
            #
            # plt.title(var_i)
            # plt.colorbar()

        # plt.show()


    def maximum_partial_corr(self):
        fdir=self.outdir
        array_dic_all={}
        array_arg={}

        var_name_list = []
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if 'p_value' in f:
                continue
            if 'max_label' in f:
                continue
            if 'maximum_partial_corr' in f:
                continue
            var_name=f.split('.')[0]
            var_name_list.append(var_name)
            print(f)
            fpath = join(fdir, f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            array_dic_all[var_name]=spatial_dict

        spatial_df = T.spatial_dics_to_df(array_dic_all)
        max_key_list = []
        max_val_list = []
        for i,row in spatial_df.iterrows():
            vals = row[var_name_list].tolist()
            vals = np.array(vals)
            var_name_list_array = np.array(var_name_list)
            vals_no_nan = vals[~np.isnan(vals)]
            var_name_list_array_no_nan = var_name_list_array[~np.isnan(vals)]
            vals_dict = T.dict_zip(var_name_list_array_no_nan, vals_no_nan)
            # if True in np.isnan(vals):
                # max_key_list.append(np.nan)
                # max_val_list.append(np.nan)
                # continue
            max_key = T.get_max_key_from_dict(vals_dict)
            max_val = vals_dict[max_key]
            max_key_list.append(max_key)
            max_val_list.append(max_val)
            # print(vals_dict)
            # print(max_key)
            # print(max_val)
            # exit()
        spatial_df['max_key'] = max_key_list
        spatial_df['max_val'] = max_val_list
        T.print_head_n(spatial_df)
        ## df to tif
        dic_label={'rainfall_seasonality_all_year_zscore': 1,

            'heavy_rainfall_days_zscore': 2,
                   'detrended_sum_rainfall_CV_zscore':3,
                   'Fire_sum_average_zscore':4,
                   'composite_LAI_beta_mean_zscore':5
        }

        spatial_df['max_label'] = spatial_df['max_key'].map(dic_label)
        # ## calculate _percentage
        #
        for ii in range(5):
            count=np.count_nonzero(spatial_df['max_label']==ii+1)
            percentage=count/len(spatial_df)*100

            plt.bar(ii+1,percentage)

        plt.show()
        exit()
        spatial_dict = T.df_to_spatial_dic(spatial_df,  'max_label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(spatial_dict, self.outdir + 'max_label.tif')



    def normalized_partial_corr(self,model):
        fdir=self.outdir
        spatial_dicts={}
        variables_list = []

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if 'max_label' in f:
                continue
            if 'Ternary_plot' in f:
                continue
            var_list=f.split('.')[0]
            print(f)
            fpath = join(fdir, f)
            fname=f.split('.')[0]
            spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[fname] = spatial_dict_i
            variables_list.append(fname)

        df = T.spatial_dics_to_df(spatial_dicts)
        df = df.dropna(subset=variables_list,how='any')
        # T.print_head_n(df);exit()
        df_abs= pd.DataFrame()
        df_abs['pix'] = df['pix'].tolist()
        for var_i in variables_list:
            abs_vals=np.array(df[var_i].tolist())
            abs_vals = np.abs(abs_vals)
            df_abs[var_i]=abs_vals
        # T.print_head_n(df_abs);exit()

        norm_dict = {}
        # T.add_dic_to_df()

        for i,row in tqdm(df_abs.iterrows(),total=len(df_abs)):
            # print(row[variables_list])
            sum_vals = row[variables_list].sum()
            # print(sum_vals)
            # if sum_vals == 0:
            #     sum_vals = np.nan
            pix = row['pix']
            norm_dict_i = {}
            for var_i in variables_list:
                var_i_norm = row[var_i] / sum_vals
                norm_dict_i[f'{var_i}_norm'] = var_i_norm
            norm_dict[pix] = norm_dict_i
        df_abs = T.add_dic_to_df(df_abs, norm_dict, 'pix')
        # T.print_head_n(df_abs);exit()
        for var_i in variables_list:

            dic_norm=T.df_to_spatial_dic(df_abs,f'{var_i}_norm',)
            DIC_and_TIF().pix_dic_to_tif(dic_norm,join(fdir,f'{var_i}_norm.tif'))
        # T.save_df(df_abs,join(fdir,'df_normalized.df'));exit()

        climate_weights_list = []
        for i,row in df_abs.iterrows():

            detrended_sum_rainfall_CV = row['detrended_sum_rainfall_CV_zscore_norm']
            CV_intraannual_rainfall_ecosystem_year = row['CV_intraannual_rainfall_ecosystem_year_zscore_norm']
            climate_sum =  detrended_sum_rainfall_CV + CV_intraannual_rainfall_ecosystem_year
            climate_weights_list.append(climate_sum)
        df_abs['climate_norm']=climate_weights_list
        rgb_arr = np.zeros((360, 720, 4))
        # Ter = xymap.Ternary_plot()
        Ter = xymap.Ternary_plot(
            top_color=(67, 198, 219),
            left_color=(255, 165, 00),
            # left_color=(119,0,188),
            right_color=(230, 0, 230),
            # center_color=(85,85,85),
            center_color=(230, 230, 230),
            # center_color=(255,255,255),
        )

        for i, row in df_abs.iterrows():
                pix = row['pix']
                r,c = pix
                climate_norm = row['climate_norm']
                Fire_sum_max_norm = row['Fire_sum_average_zscore_norm']
                composite_LAI_beta_mean_norm = row[f'{model}_sensitivity_zscore_norm']
                x,y,z = climate_norm, Fire_sum_max_norm, composite_LAI_beta_mean_norm
                color = Ter.get_color(x,y,z)
                color = color * 255
                color = np.array(color,dtype=np.uint8)
                alpha = 255
                color = np.append(color, alpha)
                # print(color);exit()

                rgb_arr[r][c] = color
        # xymap.GDAL_func().ar
        rgb_arr = np.array(rgb_arr, dtype=np.uint8)
        outtif = join(fdir, 'Ternary_plot.tif')
        tif_template = join(fdir,os.listdir(fdir)[0])
        print(rgb_arr)

        xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
        grid_triangle_legend = Ter.grid_triangle_legend()
        # plt.imshow(grid_triangle_legend)
        # plt.show()
        # T.open_path_and_file(fdir)
        # exit()

    def normalized_partial_corr_unpacked(self,model):
        fdir=self.outdir
        spatial_dicts={}
        variables_list = []

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if 'p_value' in f:
                continue
            if 'Ternary_plot' in f:
                continue
            var_list=f.split('.')[0]
            print(f)
            fpath = join(fdir, f)
            fname=f.split('.')[0]
            spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[fname] = spatial_dict_i
            variables_list.append(fname)

        df = T.spatial_dics_to_df(spatial_dicts)
        df = df.dropna(subset=variables_list,how='any')
        # T.print_head_n(df);exit()
        df_abs= pd.DataFrame()
        df_abs['pix'] = df['pix'].tolist()
        for var_i in variables_list:
            abs_vals=np.array(df[var_i].tolist())
            abs_vals = np.abs(abs_vals)
            df_abs[var_i]=abs_vals
        # T.print_head_n(df_abs);exit()

        norm_dict = {}
        # T.add_dic_to_df()

        for i,row in tqdm(df_abs.iterrows(),total=len(df_abs)):
            # print(row[variables_list])
            sum_vals = row[variables_list].sum()
            # print(sum_vals)
            # if sum_vals == 0:
            #     sum_vals = np.nan
            pix = row['pix']
            norm_dict_i = {}
            for var_i in variables_list:
                var_i_norm = row[var_i] / sum_vals
                norm_dict_i[f'{var_i}_norm'] = var_i_norm
            norm_dict[pix] = norm_dict_i


        df_abs = T.add_dic_to_df(df_abs, norm_dict, 'pix')
        for var_i in variables_list:

            dic_norm=T.df_to_spatial_dic(df_abs,f'{var_i}_norm',)
            DIC_and_TIF().pix_dic_to_tif(dic_norm,join(fdir,f'{var_i}_norm.tif'))
        ######T.save_df(df_abs,join(fdir,'df_normalized.df'));exit()

        ## df to dic

        # T.print_head_n(df_abs);exit()

        climate_weights_list = []
        # for i,row in df_abs.iterrows():
        #     VPD_detrend_CV = row['VPD_detrend_CV_norm']
        #     detrended_sum_rainfall_CV = row['detrended_sum_rainfall_CV_norm']
        #     CV_intraannual_rainfall_ecosystem_year = row['CV_intraannual_rainfall_ecosystem_year_norm']
        #     climate_sum = VPD_detrend_CV + detrended_sum_rainfall_CV + CV_intraannual_rainfall_ecosystem_year
        #     climate_weights_list.append(climate_sum)
        # df_abs['climate_norm']=climate_weights_list
        rgb_arr = np.zeros((360, 720, 4))
        # Ter = xymap.Ternary_plot()
        Ter = xymap.Ternary_plot(
            top_color=(67, 198, 219),
            left_color=(255, 165, 00),
            # left_color=(119,0,188),
            right_color=(230, 0, 230),
            # center_color=(85,85,85),
            center_color=(230, 230, 230),
            # center_color=(255,255,255),
        )

        for i, row in df_abs.iterrows():
            pix = row['pix']
            r,c = pix
            climate_norm = row[f'CV_intraannual_rainfall_ecosystem_year_zscore_norm']
            Fire_sum_max_norm = row[f'detrended_sum_rainfall_CV_zscore_norm']
            composite_LAI_beta_mean_norm = row[f'{model}_sensitivity_zscore_norm']
            x,y,z = climate_norm, Fire_sum_max_norm, composite_LAI_beta_mean_norm
            color = Ter.get_color(x,y,z)
            color = color * 255
            color = np.array(color,dtype=np.uint8)
            alpha = 255
            color = np.append(color, alpha)
            # print(color);exit()

            rgb_arr[r][c] = color
        # xymap.GDAL_func().ar
        rgb_arr = np.array(rgb_arr, dtype=np.uint8)
        ### - 蓝绿色（上）：CV_intraannual_rainfall（年内降雨变异）主导
# - 橙黄色（左下）：CV_interannual_rainfall（年际降雨变异）主导
# - 粉紫色（右下）：LAI_sensitivity（植被敏感性）主导
        outtif = join(fdir, 'Ternary_plot.tif')
        tif_template = join(fdir,os.listdir(fdir)[0])
        print(rgb_arr)

        xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
        grid_triangle_legend = Ter.grid_triangle_legend()
        plt.imshow(grid_triangle_legend)
        plt.show()
        T.open_path_and_file(fdir)
        exit()

    def normalized_partial_corr_ensemble(self, model):
        fdir = self.outdir
        spatial_dicts = {}
        variables_list = []

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if not 'norm' in f:
                continue
            if 'norm_2' in f:
                continue
            if 'Ternary_plot' in f:
                continue
            var_list = f.split('.')[0]
            print(f)
            fpath = join(fdir, f)
            fname = f.split('.')[0]
            spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[fname] = spatial_dict_i
            variables_list.append(fname)

        df = T.spatial_dics_to_df(spatial_dicts)
        df = df.dropna(subset=variables_list, how='any')
        # T.print_head_n(df);exit()
        df_abs = pd.DataFrame()
        df_abs['pix'] = df['pix'].tolist()
        for var_i in variables_list:
            abs_vals = np.array(df[var_i].tolist())
            abs_vals = np.abs(abs_vals)
            df_abs[var_i] = abs_vals
        # T.print_head_n(df_abs);exit()

        norm_dict = {}
        # T.add_dic_to_df()

        for i, row in tqdm(df_abs.iterrows(), total=len(df_abs)):
            # print(row[variables_list])
            sum_vals = row[variables_list].sum()
            # print(sum_vals)
            # if sum_vals == 0:
            #     sum_vals = np.nan
            pix = row['pix']
            norm_dict_i = {}
            for var_i in variables_list:
                var_i_norm = row[var_i] / sum_vals
                norm_dict_i[f'{var_i}_2'] = var_i_norm
            norm_dict[pix] = norm_dict_i

        df_abs = T.add_dic_to_df(df_abs, norm_dict, 'pix')


        ######T.save_df(df_abs,join(fdir,'df_normalized.df'));exit()

        ## df to dic

        # T.print_head_n(df_abs);exit()

        climate_weights_list = []
        # for i,row in df_abs.iterrows():
        #     VPD_detrend_CV = row['VPD_detrend_CV_norm']
        #     detrended_sum_rainfall_CV = row['detrended_sum_rainfall_CV_norm']
        #     CV_intraannual_rainfall_ecosystem_year = row['CV_intraannual_rainfall_ecosystem_year_norm']
        #     climate_sum = VPD_detrend_CV + detrended_sum_rainfall_CV + CV_intraannual_rainfall_ecosystem_year
        #     climate_weights_list.append(climate_sum)
        # df_abs['climate_norm']=climate_weights_list
        rgb_arr = np.zeros((360, 720, 4))
        # Ter = xymap.Ternary_plot()
        Ter = xymap.Ternary_plot(
            top_color=(67, 198, 219),
            left_color=(255, 165, 00),
            # left_color=(119,0,188),
            right_color=(230, 0, 230),
            # center_color=(85,85,85),
            center_color=(230, 230, 230),
            # center_color=(255,255,255),
        )

        for i, row in df_abs.iterrows():
            pix = row['pix']
            r, c = pix
            climate_norm = row[f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore_norm_2']
            Fire_sum_max_norm = row[f'{model}_detrended_sum_rainfall_CV_zscore_norm_2']
            composite_LAI_beta_mean_norm = row[f'{model}_sensitivity_zscore_norm_2']
            x, y, z = climate_norm, Fire_sum_max_norm, composite_LAI_beta_mean_norm
            color = Ter.get_color(x, y, z)
            color = color * 255
            color = np.array(color, dtype=np.uint8)
            alpha = 255
            color = np.append(color, alpha)
            # print(color);exit()

            rgb_arr[r][c] = color
        # xymap.GDAL_func().ar
        rgb_arr = np.array(rgb_arr, dtype=np.uint8)
        ### - 蓝绿色（上）：CV_intraannual_rainfall（年内降雨变异）主导
        # - 橙黄色（左下）：CV_interannual_rainfall（年际降雨变异）主导
        # - 粉紫色（右下）：LAI_sensitivity（植被敏感性）主导
        outtif = join(fdir, 'Ternary_plot.tif')
        tif_template = join(fdir, os.listdir(fdir)[0])
        print(rgb_arr)

        xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
        grid_triangle_legend = Ter.grid_triangle_legend()
        plt.imshow(grid_triangle_legend)
        plt.show()
        T.open_path_and_file(fdir)
        # exit()

    def plot_pdf(self):
        dff=result_root + rf'3mm\Multiregression\partial_correlation\Obs\obs_climate\Dataframe\\df_normalized.df'
        df=T.load_df(dff)
        df=self.df_clean(df)

        variables = {
            'sensitivity': df['composite_LAI_beta_mean'].to_list(),
            'inter_rainfall': df['detrended_sum_rainfall_CV'].to_list(),
            'intra_rainfall': df['CV_intraannual_rainfall_ecosystem_year'].to_list()
        }
        plt.figure()

        for var_name, values in variables.items():
            arr = np.array(values)
            arr=arr*100
            arr = arr[~np.isnan(arr)]


            sns.kdeplot(arr, fill=False, linewidth=2,label=var_name)

            plt.grid(True)

        plt.legend()
        plt.show()

        pass





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


    def statistic_corr(self):

        dff = result_root + rf'3mm\Multiregression\partial_correlation\\partial_correlation_df.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        df.dropna(inplace=True)

        variable_list=self.xvar_list

        for variable in variable_list:
            vals=df[variable].to_list()
            vals=np.array(vals)


            arr_corr=vals


            arr_pos=len(df[df[variable]>0])/len(df)*100
            arr_neg=len(df[df[variable]<0])/len(df)*100

            ## significant positive and negative
            ## 1 is significant and 2 positive or negative
            df_sig=df[df[f'{variable}_p_value']<0.05]
            arr_pos_sig=len(df_sig[df_sig[variable]>0])/len(df)*100
            arr_neg_sig=len(df_sig[df_sig[variable]<0])/len(df)*100


            result_dic={
                'Negative_sig': arr_neg_sig,
                'Negative': arr_neg,
                'Positive':arr_pos,
                'Positive_sig':arr_pos_sig,

            }
            # df_new=pd.DataFrame(result_dic,index=[variable])
            # ## plot
            # df_new=df_new.T
            # df_new=df_new.reset_index()
            # df_new.columns=['Variable','Percentage']
            # df_new.plot.bar(x='Variable',y='Percentage',rot=45,color='green')
            # plt.show()
            color_list = ['#d01c8b', '#f1b6da', '#b8e186', '#4dac26']
            width = 0.4
            alpha_list = [1, 0.5, 0.5, 1]

            # 逐个画 bar
            for i, (key, val) in enumerate(result_dic.items()):
                plt.bar(i , val, color=color_list[i], alpha=alpha_list[i], width=width)
                plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
                plt.ylabel('Percentage')
                plt.title(variable)

            plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
            plt.show()



    def statistic_trend(self):
        fdir = result_root + rf'\3mm\RF_Multiregression\trend\\'
        variable_list=['rainfall_intensity_trend','rainfall_frenquency_trend']
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            result_dic={}
            variable=f.split('.')[0]
            if not variable in variable_list:
                continue
            fpath_corr = join(fdir, f)
            fpath_pvalue=fdir+f.replace('trend','p_value')
            arr_corr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_corr)
            arr_pvalue, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_pvalue)
            arr_corr[arr_corr<-99]=np.nan
            arr_corr[arr_corr>99]=np.nan
            arr_corr=arr_corr[~np.isnan(arr_corr)]

            arr_pvalue[arr_pvalue<-99]=np.nan
            arr_pvalue[arr_pvalue>99]=np.nan
            arr_pvalue=arr_pvalue[~np.isnan(arr_pvalue)]
            ## corr negative and positive
            arr_corr = arr_corr.flatten()
            arr_pvalue = arr_pvalue.flatten()
            arr_pos=len(arr_corr[arr_corr>0])/len(arr_corr)*100
            arr_neg=len(arr_corr[arr_corr<0])/len(arr_corr)*100


            ## significant positive and negative
            ## 1 is significant and 2 positive or negative

            mask_pos = (arr_corr > 0) & (arr_pvalue < 0.05)
            mask_neg = (arr_corr < 0) & (arr_pvalue < 0.05)


            # 满足条件的像元数
            count_positive_sig = np.sum(mask_pos)
            count_negative_sig = np.sum(mask_neg)

            # 百分比
            significant_positive = (count_positive_sig / len(arr_corr)) * 100
            significant_negative = (count_negative_sig / len(arr_corr)) * 100
            result_dic = {

                'sig neg': significant_negative,
                'non sig neg': arr_neg,
                'non sig pos': arr_pos,
                'sig pos': significant_positive



            }
            # df_new=pd.DataFrame(result_dic,index=[variable])
            # ## plot
            # df_new=df_new.T
            # df_new=df_new.reset_index()
            # df_new.columns=['Variable','Percentage']
            # df_new.plot.bar(x='Variable',y='Percentage',rot=45,color='green')
            # plt.show()
            color_list = ['red', 'red', 'green', 'green']
            width = 0.4
            alpha_list = [1, 0.5, 0.5, 1]

            # 逐个画 bar
            for i, (key, val) in enumerate(result_dic.items()):
                plt.bar(i , val, color=color_list[i], alpha=alpha_list[i], width=width)
                plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
                plt.ylabel('Percentage')
                plt.title(variable)

            plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
            plt.show()


    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']


        return df

    def aridity_bin(self):

        dff = result_root + rf'\3mm\Multiregression\partial_correlation\Obs\obs_climate_fire\Dataframe\\Dataframe.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df=df[df['composite_LAI_CV_trend'] > 0]
        # df=df[df['composite_LAI_CV_p_value'] < 0.05]

        df.dropna(inplace=True)
        for column in df.columns:
            print(column)


        # 设置变量名
        target_var = 'composite_LAI_Fire_sum_average_zscore'
        pval_var = f'{target_var}_p_value'
        bin_var = 'Burn_area_mean'
        plt.hist(df[bin_var])
        plt.show()
        bin_edges = np.linspace(0,1000,11)
        bin_labels = [f'{round(bin_edges[i], 2)}-{round(bin_edges[i + 1], 2)}' for i in range(len(bin_edges) - 1)]

        df['bin'] = pd.cut(df[bin_var], bins=bin_edges, labels=bin_labels, include_lowest=True)

        # 初始化结果字典
        result_dic = {}

        for label in bin_labels:
            df_bin = df[df['bin'] == label][[target_var, pval_var]].dropna()

            if len(df_bin) == 0:
                result_dic[label] = [0, 0, 0, 0]
                continue

            pos = (df_bin[target_var] > 0)
            neg = (df_bin[target_var] < 0)
            sig = (df_bin[pval_var] < 0.05)

            pos_sig = (pos & sig).sum() / len(df_bin)
            pos_nonsig = (pos & ~sig).sum() / len(df_bin)
            neg_sig = (neg & sig).sum() / len(df_bin)
            neg_nonsig = (neg & ~sig).sum() / len(df_bin)

            # result_dic[label] = [pos_sig, pos_nonsig, neg_sig, neg_nonsig]
            result_dic[label] = [pos_sig, neg_sig, ]



        # 转换为 DataFrame
        # df_plot = pd.DataFrame(result_dic,
        #                        index=['Positive_sig', 'Positive_nonsig', 'Negative_sig', 'Negative_nonsig']).T

        df_plot = pd.DataFrame(result_dic,
                               index=['Positive_sig', 'Negative_sig', ]).T

        # 画图
        # colors = ['green', 'lightgreen', 'red', 'lightcoral']
        colors = ['green', 'red',  ]
        df_plot.plot(kind='bar', stacked=False, color=colors, figsize=(10, 6))

        plt.ylabel('Percentage')
        plt.title(f'{target_var} by {bin_var} bin')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()






    def pft_max_label(self):
        dff = result_root + rf'\3mm\Multiregression\partial_correlation\Obs\obs_climate_fire\Dataframe\\Dataframe.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        df.dropna(inplace=True)


        ## get uqniue pft
        # pft_unique=df['continent'].unique()
        # print(pft_unique);exit()
        pft_unique_list=['Global','Evergreen','Deciduous','Grass','Shrub',]
        continent_list=['Global','North_America','South_America','Asia','Africa','Australia']


        max_labels = [1, 2, 3, 4]
        dic_label={ 1:'CV_intraannual_rainfall',
                    2:'CV_interannual_rainfall',
                    3:'Fire',
                    4:'Gamma'


        }

        # 存储每个 pft 对应的 max_label 百分比分布
        all_results = []

        for pft in continent_list:
            if pft == 'Global':
                df_mask = df
            else:
                mask = (df['continent'] == pft)
                df_mask = df[mask]
            result_list = []

            for label in max_labels:
                df_temp = df_mask[df_mask['max_label'] == label]
                percentage = len(df_temp) / len(df_mask) * 100 if len(df_mask) > 0 else 0
                result_list.append(percentage)

            all_results.append(result_list)

        # 转换成 DataFrame，方便画图
        result_df = pd.DataFrame(all_results, index=continent_list, columns=dic_label.values())
        color_list=['#d7191c','#fec980','#c7e8ad','#2b83ba']

        # 画分组柱状图
        result_df.plot(kind='bar', stacked=False, figsize=(5, 3), rot=0, color=color_list)
        plt.ylabel("Percentage (%)")

        plt.tight_layout()
        plt.show()

    def pft_test2(self):
        dff = result_root + rf'\3mm\Multiregression\partial_correlation\Obs\obs_climate_fire\Dataframe\\Dataframe.df'
        df = T.load_df(dff)
        df = self.df_clean(df)



        ## get uqniue pft
        pft_unique = df['landcover_classfication'].unique()
        # print(pft_unique);exit()
        pft_unique_list = ['Global','Grass', 'Evergreen', 'Deciduous', 'Shrub']
        for pft in pft_unique_list:
            if pft == 'Global':
                df_mask=df
            else:
                mask = (df['landcover_classfication'] == pft)
                df_mask = df[mask]
            df_mask = df_mask.dropna()


            bins=np.arange(0,1.1,0.25)

            ## calculate each bin percentage
            result_len_dic={}


            for var in variables:
                var_name=label_dic[var]
                label_list=[]
                df_var = df_mask[[var]].dropna()
                total = len(df_var)
                result_len_dic[var_name]=[]
                for i in range(len(bins)-1):

                    if i < len(bins) - 2:
                        mask = (df_var[var] >= bins[i]) & (df_var[var] < bins[i + 1])
                    else:
                        mask = (df_var[var] >= bins[i]) & (df_var[var] <= bins[i + 1])
                    bins[i]=round(bins[i],2)
                    bins[i + 1] = round(bins[i + 1], 2)
                    label=f'{bins[i]}-{bins[i+1]}'
                    label_list.append(label)

                    percentage=len(df_mask[mask])/total*100
                    ## percentage.2f
                    percentage=round(percentage,2)

                    result_len_dic[var_name].append(percentage)


            df_new=pd.DataFrame(result_len_dic)
            df_new=df_new.T
            T.print_head_n(df_new)
            ax=df_new.plot.bar(stacked=True,colormap='Set2',width=0.5,)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)



            ## add legend as label list
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, label_list, loc='upper right', bbox_to_anchor=(1.1, 1.05))


            plt.title(pft)
            plt.show()













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


class partial_correlation_TRENDY():
    def __init__(self):


        self.fdirX = result_root + rf'\3mm\Multiregression\partial_correlation\TRENDY\Input\\X\\'
        self.fdirY = result_root + rf'\3mm\Multiregression\partial_correlation\TRENDY\Input\Y\\'



    def run(self):

        self.xvar_list = ['CV_intraannual_rainfall_ecosystem_year_zscore', 'detrended_sum_rainfall_CV_zscore',
                        'Fire_sum_average_zscore' ]

        self.model_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                           'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                           'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                           'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                           'ORCHIDEE_S2_lai',

                           'YIBs_S2_Monthly_lai',

                           ]
        self.model_list = ['TRENDY_mean' ]


        for model in self.model_list:
            self.outdir = result_root + rf'\3mm\Multiregression\partial_correlation\TRENDY\Result\climate_fire_sensitivity\\{model}\\'
            T.mk_dir(self.outdir, force=True)
            self.outpartial = self.outdir + rf'\partial_corr_{model}.npy'
            self.outpartial_pvalue = self.outdir + rf'\partial_pvalue_{model}.npy'

            y_var = f'{model}_detrend_CV_zscore.npy'
            x_var_list=self.xvar_list+[f'{model}_sensitivity_zscore']


            # df=self.build_df(self.fdirX,self.fdirY,x_var_list,y_var)
            # # # # #
            # self.cal_partial_corr(df,x_var_list)
            # # # # # # self.cal_single_correlation()
            # # # # # # self.cal_single_correlation_ly()
            # # # # # # self.check_data()
            # self.plot_partial_correlation()
            #
            # # self.maximum_partial_corr()
            # self.normalized_partial_corr(model)
            self.normalized_partial_corr_ensemble(model)
            # self.normalized_partial_corr_unpacked(model)
            # self.plot_pdf()
            # self.statistic_corr()
            # self.statistic_trend()

        # self.pft_test2()


    def check_data(self):
        f=result_root+rf'\3mm\Multiregression\zscore\\Fire_sum_max_zscore.npy'
        dic=T.load_npy(f)

        val_list=[]
        for pix in dic:
            val=dic[pix]
            val_list.append(val)
        val_list=np.array(val_list)
        val_list=val_list[~np.isnan(val_list)]
        val_list=val_list.flatten()
        plt.hist(val_list)
        plt.show()



        pass
    def build_df(self,fdir_X,fdir_Y,fx_list,fy):
        df = pd.DataFrame()

        filey = fdir_Y + fy
        print(filey)

        dic_y = T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix][0:22]

            if len(yvals) == 0:
                continue
            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
            yvals=yvals
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
            filex = fdir_X + xvar+'.npy'


            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
            dic_x = T.load_npy(filex)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                xvals = dic_x[pix][0:22]
                xvals = np.array(xvals)
                xvals = xvals
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
                # print(x_var_list_valid_new_cov)
                x_var_list_valid_new_cov.remove(x)
                # print(x_var_list_valid_new_cov)
                r, p = self.partial_corr(df_new, x, 'y', x_var_list_valid_new_cov)
                partial_correlation[x] = r
                partial_correlation_p_value[x] = p

            partial_correlation_dic[pix] = partial_correlation
            partial_p_value_dic[pix] = partial_correlation_p_value
        T.save_npy(partial_correlation_dic, outf_corr)
        T.save_npy(partial_p_value_dic, outf_pvalue)





            # print(df_new)


    def cal_single_correlation(self):
        f_x= result_root + rf'\3mm\Multiregression\input\\sum_rainfall.npy'
        f_y = result_root + rf'\3mm\Multiregression\input\\composite_LAI_beta_mean.npy'
        outdir=join(result_root, 'Multiregression', 'correlation')
        T.mk_dir(outdir, force=True)
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
            y_val = y_val[0:24]

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
        arr=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_r_dic)

        outf=outdir+'\\sum_rainfall.tif'
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr,outf)
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

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)



        f_partial = self.outpartial
        f_pvalue = self.outpartial_pvalue
        outdir= self.outdir


        partial_correlation_dic = np.load(f_partial, allow_pickle=True, encoding='latin1').item()
        # partial_correlation_p_value_dic = np.load(f_pvalue, allow_pickle=True, encoding='latin1').item()


        var_list = []
        for pix in partial_correlation_dic:



            vals = partial_correlation_dic[pix]
            # vals = partial_correlation_p_value_dic[pix]


            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in partial_correlation_dic:
                r, c = pix
                if r < 60:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                dic_i = partial_correlation_dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, self.outdir + f'{var_i}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            # plt.imshow(arr, vmin=-1, vmax=1)
            #
            # plt.title(var_i)
            # plt.colorbar()
        #
        # plt.show()










    def maximum_partial_corr(self):
        fdir=self.outdir
        array_list=[]
        array_arg={}
        var_list=[]

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if 'p_value' in f:
                continue
            if 'maximum_partial_corr' in f:
                continue
            var_list=f.split('.')[0]
            print(f)
            fpath = join(fdir, f)

            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr = arr.astype(np.float32)

            arr[arr <- 99] = np.nan
            array_list.append(arr)
        array_list = np.array(array_list)


        abs_array = np.abs(array_list)
        all_nan_mask = np.all(np.isnan(abs_array), axis=0)

        array_max = np.full(abs_array.shape[1:], np.nan)
        array_arg = np.full(abs_array.shape[1:], np.nan)

        # 对非全NaN的像元计算 max 和 argmax
        valid_mask = ~all_nan_mask
        array_max[valid_mask] = np.nanmax(abs_array[:, valid_mask], axis=0)
        array_arg[valid_mask] = np.nanargmax(abs_array[:, valid_mask], axis=0)

        plt.imshow(array_arg)
        plt.show()
        array_flatten=array_arg.flatten()
        array_flatten=array_flatten[~np.isnan(array_flatten)]

        array_flat = array_flatten.astype(int)  # 转为整数索引
        percentage=np.bincount(array_flat)/len(array_flat)*100
        plt.bar(np.arange(len(percentage)), percentage)
        plt.show()


        DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_arg, self.outdir + f'maximum_partial_corr.tif')


        pass

    def normalized_partial_corr(self, model):
        fdir = self.outdir
        spatial_dicts = {}
        variables_list = []

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if 'max_label' in f:
                continue
            if 'Ternary_plot' in f:
                continue
            var_list = f.split('.')[0]
            print(f)
            fpath = join(fdir, f)
            fname = f.split('.')[0]
            spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[fname] = spatial_dict_i
            variables_list.append(fname)

        df = T.spatial_dics_to_df(spatial_dicts)
        df = df.dropna(subset=variables_list, how='any')
        # T.print_head_n(df);exit()
        df_abs = pd.DataFrame()
        df_abs['pix'] = df['pix'].tolist()
        for var_i in variables_list:
            abs_vals = np.array(df[var_i].tolist())
            abs_vals = np.abs(abs_vals)
            df_abs[var_i] = abs_vals
        # T.print_head_n(df_abs);exit()

        norm_dict = {}
        # T.add_dic_to_df()

        for i, row in tqdm(df_abs.iterrows(), total=len(df_abs)):
            # print(row[variables_list])
            sum_vals = row[variables_list].sum()
            # print(sum_vals)
            # if sum_vals == 0:
            #     sum_vals = np.nan
            pix = row['pix']
            norm_dict_i = {}
            for var_i in variables_list:
                var_i_norm = row[var_i] / sum_vals
                norm_dict_i[f'{var_i}_norm'] = var_i_norm
            norm_dict[pix] = norm_dict_i
        df_abs = T.add_dic_to_df(df_abs, norm_dict, 'pix')
        # T.print_head_n(df_abs);exit()
        for var_i in variables_list:

            dic_norm=T.df_to_spatial_dic(df_abs,f'{var_i}_norm',)
            DIC_and_TIF().pix_dic_to_tif(dic_norm,join(fdir,f'{var_i}_norm.tif'))
        # T.save_df(df_abs,join(fdir,'df_normalized.df'))

        climate_weights_list = []
        for i, row in df_abs.iterrows():
            detrended_sum_rainfall_CV = row['detrended_sum_rainfall_CV_zscore_norm']
            CV_intraannual_rainfall_ecosystem_year = row['CV_intraannual_rainfall_ecosystem_year_zscore_norm']
            climate_sum = detrended_sum_rainfall_CV + CV_intraannual_rainfall_ecosystem_year
            climate_weights_list.append(climate_sum)
        df_abs['climate_norm'] = climate_weights_list
        rgb_arr = np.zeros((360, 720, 4))
        # Ter = xymap.Ternary_plot()
        Ter = xymap.Ternary_plot(
            top_color=(67, 198, 219),
            left_color=(255, 165, 00),
            # left_color=(119,0,188),
            right_color=(230, 0, 230),
            # center_color=(85,85,85),
            center_color=(230, 230, 230),
            # center_color=(255,255,255),
        )

        for i, row in df_abs.iterrows():
            pix = row['pix']
            r, c = pix
            climate_norm = row['climate_norm']
            Fire_sum_max_norm = row['Fire_sum_average_zscore_norm']
            composite_LAI_beta_mean_norm = row[f'{model}_sensitivity_zscore_norm']
            x, y, z = climate_norm, Fire_sum_max_norm, composite_LAI_beta_mean_norm
            color = Ter.get_color(x, y, z)
            color = color * 255
            color = np.array(color, dtype=np.uint8)
            alpha = 255
            color = np.append(color, alpha)
            # print(color);exit()

            rgb_arr[r][c] = color
        # xymap.GDAL_func().ar
        rgb_arr = np.array(rgb_arr, dtype=np.uint8)
        outtif = join(fdir, 'Ternary_plot.tif')
        tif_template = join(fdir, os.listdir(fdir)[0])
        print(rgb_arr)

        xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
        grid_triangle_legend = Ter.grid_triangle_legend()
        # plt.imshow(grid_triangle_legend)
        # plt.show()
        # T.open_path_and_file(fdir)
        # exit()
    def normalized_partial_corr_ensemble(self, model):
        fdir = self.outdir
        spatial_dicts = {}
        variables_list = []

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if not 'norm' in f:
                continue
            if 'norm_2' in f:
                continue
            if 'Ternary_plot' in f:
                continue
            var_list = f.split('.')[0]
            print(f)
            fpath = join(fdir, f)
            fname = f.split('.')[0]
            spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[fname] = spatial_dict_i
            variables_list.append(fname)

        df = T.spatial_dics_to_df(spatial_dicts)
        df = df.dropna(subset=variables_list, how='any')
        # T.print_head_n(df);exit()
        df_abs = pd.DataFrame()
        df_abs['pix'] = df['pix'].tolist()
        for var_i in variables_list:
            abs_vals = np.array(df[var_i].tolist())
            abs_vals = np.abs(abs_vals)
            df_abs[var_i] = abs_vals
        # T.print_head_n(df_abs);exit()

        norm_dict = {}
        # T.add_dic_to_df()

        for i, row in tqdm(df_abs.iterrows(), total=len(df_abs)):
            # print(row[variables_list])
            sum_vals = row[variables_list].sum()
            # print(sum_vals)
            # if sum_vals == 0:
            #     sum_vals = np.nan
            pix = row['pix']
            norm_dict_i = {}
            for var_i in variables_list:
                var_i_norm = row[var_i] / sum_vals
                norm_dict_i[f'{var_i}_2'] = var_i_norm
            norm_dict[pix] = norm_dict_i

        df_abs = T.add_dic_to_df(df_abs, norm_dict, 'pix')
        # for var_i in variables_list:
        #     dic_norm = T.df_to_spatial_dic(df_abs, f'{var_i}_2', )
        #     DIC_and_TIF().pix_dic_to_tif(dic_norm, join(fdir, f'{var_i}_2.tif'))

        ######T.save_df(df_abs,join(fdir,'df_normalized.df'));exit()

        ## df to dic

        # T.print_head_n(df_abs);exit()

        climate_weights_list = []
        for i,row in df_abs.iterrows():

            detrended_sum_rainfall_CV = row[rf'{model}_detrended_sum_rainfall_CV_zscore_norm_2']
            CV_intraannual_rainfall_ecosystem_year = row[rf'{model}_CV_intraannual_rainfall_ecosystem_year_zscore_norm_2']
            climate_sum =  detrended_sum_rainfall_CV + CV_intraannual_rainfall_ecosystem_year
            climate_weights_list.append(climate_sum)
        df_abs[f'{model}_climate_norm_2']=climate_weights_list

        new_var_list=[f'{model}_climate_norm_2',f'{model}_Fire_sum_average_zscore_norm_2',f'{model}_sensitivity_zscore_norm_2']
        for var_i in new_var_list:
            dic_norm = T.df_to_spatial_dic(df_abs, f'{var_i}', )
            DIC_and_TIF().pix_dic_to_tif(dic_norm, join(fdir, f'{var_i}.tif'))
        exit()
        rgb_arr = np.zeros((360, 720, 4))
        # Ter = xymap.Ternary_plot()
        Ter = xymap.Ternary_plot(
            top_color=(67, 198, 219),
            left_color=(255, 165, 00),
            # left_color=(119,0,188),
            right_color=(230, 0, 230),
            # center_color=(85,85,85),
            center_color=(230, 230, 230),
            # center_color=(255,255,255),
        )

        for i, row in df_abs.iterrows():
            pix = row['pix']
            r, c = pix
            climate_norm = row[f'{model}_climate_norm_2']
            Fire_sum_max_norm = row[f'{model}_Fire_sum_average_zscore_norm_2']
            composite_LAI_beta_mean_norm = row[f'{model}_sensitivity_zscore_norm_2']
            x, y, z = climate_norm, Fire_sum_max_norm, composite_LAI_beta_mean_norm
            color = Ter.get_color(x, y, z)
            color = color * 255
            color = np.array(color, dtype=np.uint8)
            alpha = 255
            color = np.append(color, alpha)
            # print(color);exit()

            rgb_arr[r][c] = color
        # xymap.GDAL_func().ar
        rgb_arr = np.array(rgb_arr, dtype=np.uint8)
        ### - 蓝绿色（上）：CV_intraannual_rainfall（年内降雨变异）主导
        # - 橙黄色（左下）：CV_interannual_rainfall（年际降雨变异）主导
        # - 粉紫色（右下）：LAI_sensitivity（植被敏感性）主导
        outtif = join(fdir, 'Ternary_plot.tif')
        tif_template = join(fdir, os.listdir(fdir)[0])
        print(rgb_arr)

        xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
        grid_triangle_legend = Ter.grid_triangle_legend()
        plt.imshow(grid_triangle_legend)
        plt.show()
        T.open_path_and_file(fdir)
        # exit()
    def normalized_partial_corr_unpacked(self,model):
        fdir=self.outdir
        spatial_dicts={}
        variables_list = []

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if not 'norm' in f:
                continue
            if 'norm_2' in f:
                continue
            if 'p_value' in f:
                continue
            if 'Ternary_plot' in f:
                continue
            var_list=f.split('.')[0]
            print(f)
            fpath = join(fdir, f)
            fname=f.split('.')[0]
            spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[fname] = spatial_dict_i
            variables_list.append(fname)

        df = T.spatial_dics_to_df(spatial_dicts)
        df = df.dropna(subset=variables_list,how='any')
        # T.print_head_n(df);exit()
        df_abs= pd.DataFrame()
        df_abs['pix'] = df['pix'].tolist()
        for var_i in variables_list:
            abs_vals=np.array(df[var_i].tolist())
            abs_vals = np.abs(abs_vals)
            df_abs[var_i]=abs_vals
        # T.print_head_n(df_abs);exit()

        norm_dict = {}
        # T.add_dic_to_df()

        for i,row in tqdm(df_abs.iterrows(),total=len(df_abs)):
            # print(row[variables_list])
            sum_vals = row[variables_list].sum()
            # print(sum_vals)
            # if sum_vals == 0:
            #     sum_vals = np.nan
            pix = row['pix']
            norm_dict_i = {}
            for var_i in variables_list:
                var_i_norm = row[var_i] / sum_vals
                norm_dict_i[f'{var_i}_2'] = var_i_norm
            norm_dict[pix] = norm_dict_i


        df_abs = T.add_dic_to_df(df_abs, norm_dict, 'pix')
        # for var_i in variables_list:
        #
        #     dic_norm=T.df_to_spatial_dic(df_abs,f'{var_i}_2',)
        #     DIC_and_TIF().pix_dic_to_tif(dic_norm,join(fdir,f'{var_i}_2.tif'))
        # T.save_df(df_abs,join(fdir,'df_normalized.df'));exit()

        ## df to dic

        # T.print_head_n(df_abs);exit()

        climate_weights_list = []
        # for i,row in df_abs.iterrows():
        #     VPD_detrend_CV = row['VPD_detrend_CV_norm']
        #     detrended_sum_rainfall_CV = row['detrended_sum_rainfall_CV_norm']
        #     CV_intraannual_rainfall_ecosystem_year = row['CV_intraannual_rainfall_ecosystem_year_norm']
        #     climate_sum = VPD_detrend_CV + detrended_sum_rainfall_CV + CV_intraannual_rainfall_ecosystem_year
        #     climate_weights_list.append(climate_sum)
        # df_abs['climate_norm']=climate_weights_list
        rgb_arr = np.zeros((360, 720, 4))
        # Ter = xymap.Ternary_plot()
        Ter = xymap.Ternary_plot(
            top_color=(67, 198, 219),
            left_color=(255, 165, 00),
            # left_color=(119,0,188),
            right_color=(230, 0, 230),
            # center_color=(85,85,85),
            center_color=(230, 230, 230),
            # center_color=(255,255,255),
        )

        for i, row in df_abs.iterrows():
            pix = row['pix']
            r,c = pix
            climate_norm = row[f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore_norm_2']
            Fire_sum_max_norm = row[f'{model}_detrended_sum_rainfall_CV_zscore_norm_2']
            composite_LAI_beta_mean_norm = row[f'{model}_sensitivity_zscore_norm_2']
            x,y,z = climate_norm, Fire_sum_max_norm, composite_LAI_beta_mean_norm
            color = Ter.get_color(x,y,z)
            color = color * 255
            color = np.array(color,dtype=np.uint8)
            alpha = 255
            color = np.append(color, alpha)
            # print(color);exit()

            rgb_arr[r][c] = color
        # xymap.GDAL_func().ar
        rgb_arr = np.array(rgb_arr, dtype=np.uint8)
        ### - 蓝绿色（上）：CV_intraannual_rainfall（年内降雨变异）主导
# - 橙黄色（左下）：CV_interannual_rainfall（年际降雨变异）主导
# - 粉紫色（右下）：LAI_sensitivity（植被敏感性）主导
        outtif = join(fdir, 'Ternary_plot.tif')
        tif_template = join(fdir,os.listdir(fdir)[0])
        print(rgb_arr)

        xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
        grid_triangle_legend = Ter.grid_triangle_legend()
        plt.imshow(grid_triangle_legend)
        plt.show()
        # T.open_path_and_file(fdir)
        # exit()

    def plot_pdf(self):
        dff=result_root + rf'3mm\Multiregression\partial_correlation\Obs\obs_climate\Dataframe\\df_normalized.df'
        df=T.load_df(dff)
        for col in df.columns:
            print(col)
        df=self.df_clean(df)
        flag=0



        fig, axes = plt.subplots(3, 6, figsize=(12, 18))  # Adjust figsize if too tight
        axes = axes.flatten()
        self.model_list = ['OBS_mean', 'GLOBMAP_LAI','LAI4g','SNU_LAI','TRENDY_mean',  'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                           'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                           'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                           'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                           'ORCHIDEE_S2_lai', 'YIBs_S2_Monthly_lai',

                           ]
        for model in self.model_list:
            if model=='OBS_mean' or model=='TRENDY_mean':
                variables = {
                    'Sensitivity': df[f'{model}_sensitivity_zscore_norm_2'].to_list(),
                    'CV_intra_rainfall': df[f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore_norm_2'].to_list(),
                    'CV_inter_rainfall': df[f'{model}_detrended_sum_rainfall_CV_zscore_norm_2'].to_list(),
                }
            else:


                variables = {
                    'Sensitivity': df[f'{model}_sensitivity_zscore_norm'].to_list(),
                    'CV_intra_rainfall': df[f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore_norm'].to_list(),
                    'CV_inter_rainfall': df[f'{model}_detrended_sum_rainfall_CV_zscore_norm'].to_list(),
                }


            ## all model plot in the same layout
            ax = axes[flag]


            for var_name, values in variables.items():
                if flag >= len(axes):
                    break

                arr = np.array(values)
                arr=arr*100
                arr = arr[~np.isnan(arr)]
                mean_val = np.mean(arr)
                # ax.axvline(mean_val, linestyle='--', linewidth=1, alpha=0.8)



                sns.kdeplot(arr, fill=False, linewidth=2,label=var_name,ax=ax)
                # sns.ecdfplot(arr, label=var_name, ax=ax, linewidth=2, )
            ax.set_xlim(0, 100)
            ax.set_ylabel('')
            ax.grid(True)
            ax.set_title(model)
            ax.legend(fontsize=6)

            # plt.grid(True)

            flag=flag+1


            #
            #
        plt.legend()
        plt.show()

        pass





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


    def statistic_corr(self):

        dff = result_root + rf'3mm\Multiregression\partial_correlation\\partial_correlation_df.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        df.dropna(inplace=True)

        variable_list=self.xvar_list

        for variable in variable_list:
            vals=df[variable].to_list()
            vals=np.array(vals)


            arr_corr=vals


            arr_pos=len(df[df[variable]>0])/len(df)*100
            arr_neg=len(df[df[variable]<0])/len(df)*100

            ## significant positive and negative
            ## 1 is significant and 2 positive or negative
            df_sig=df[df[f'{variable}_p_value']<0.05]
            arr_pos_sig=len(df_sig[df_sig[variable]>0])/len(df)*100
            arr_neg_sig=len(df_sig[df_sig[variable]<0])/len(df)*100


            result_dic={
                'Negative_sig': arr_neg_sig,
                'Negative': arr_neg,
                'Positive':arr_pos,
                'Positive_sig':arr_pos_sig,

            }
            # df_new=pd.DataFrame(result_dic,index=[variable])
            # ## plot
            # df_new=df_new.T
            # df_new=df_new.reset_index()
            # df_new.columns=['Variable','Percentage']
            # df_new.plot.bar(x='Variable',y='Percentage',rot=45,color='green')
            # plt.show()
            color_list = ['#d01c8b', '#f1b6da', '#b8e186', '#4dac26']
            width = 0.4
            alpha_list = [1, 0.5, 0.5, 1]

            # 逐个画 bar
            for i, (key, val) in enumerate(result_dic.items()):
                plt.bar(i , val, color=color_list[i], alpha=alpha_list[i], width=width)
                plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
                plt.ylabel('Percentage')
                plt.title(variable)

            plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
            plt.show()



    def statistic_trend(self):
        fdir = result_root + rf'\3mm\RF_Multiregression\trend\\'
        variable_list=['rainfall_intensity_trend','rainfall_frenquency_trend']
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            result_dic={}
            variable=f.split('.')[0]
            if not variable in variable_list:
                continue
            fpath_corr = join(fdir, f)
            fpath_pvalue=fdir+f.replace('trend','p_value')
            arr_corr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_corr)
            arr_pvalue, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_pvalue)
            arr_corr[arr_corr<-99]=np.nan
            arr_corr[arr_corr>99]=np.nan
            arr_corr=arr_corr[~np.isnan(arr_corr)]

            arr_pvalue[arr_pvalue<-99]=np.nan
            arr_pvalue[arr_pvalue>99]=np.nan
            arr_pvalue=arr_pvalue[~np.isnan(arr_pvalue)]
            ## corr negative and positive
            arr_corr = arr_corr.flatten()
            arr_pvalue = arr_pvalue.flatten()
            arr_pos=len(arr_corr[arr_corr>0])/len(arr_corr)*100
            arr_neg=len(arr_corr[arr_corr<0])/len(arr_corr)*100


            ## significant positive and negative
            ## 1 is significant and 2 positive or negative

            mask_pos = (arr_corr > 0) & (arr_pvalue < 0.05)
            mask_neg = (arr_corr < 0) & (arr_pvalue < 0.05)


            # 满足条件的像元数
            count_positive_sig = np.sum(mask_pos)
            count_negative_sig = np.sum(mask_neg)

            # 百分比
            significant_positive = (count_positive_sig / len(arr_corr)) * 100
            significant_negative = (count_negative_sig / len(arr_corr)) * 100
            result_dic = {

                'sig neg': significant_negative,
                'non sig neg': arr_neg,
                'non sig pos': arr_pos,
                'sig pos': significant_positive



            }
            # df_new=pd.DataFrame(result_dic,index=[variable])
            # ## plot
            # df_new=df_new.T
            # df_new=df_new.reset_index()
            # df_new.columns=['Variable','Percentage']
            # df_new.plot.bar(x='Variable',y='Percentage',rot=45,color='green')
            # plt.show()
            color_list = ['red', 'red', 'green', 'green']
            width = 0.4
            alpha_list = [1, 0.5, 0.5, 1]

            # 逐个画 bar
            for i, (key, val) in enumerate(result_dic.items()):
                plt.bar(i , val, color=color_list[i], alpha=alpha_list[i], width=width)
                plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
                plt.ylabel('Percentage')
                plt.title(variable)

            plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
            plt.show()


    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df



    def pft_test(self):
        dff = result_root + rf'3mm\Multiregression\partial_correlation\\partial_correlation_df.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        df.dropna(inplace=True)

        selected_vairables_list = [
            'composite_LAI_beta_mean',
            'detrended_sum_rainfall_CV',
            'Fire_sum_max',

        ]
        ## get uqniue pft
        pft_unique=df['landcover_classfication'].unique()
        # print(pft_unique);exit()
        pft_unique_list=['Grass','Evergreen','Deciduous','Shrub']

        for var in selected_vairables_list:
            df_temp=df[df[f'{var}_p_value']<0.05]

            result_list=[]

            for pft in pft_unique_list:


                mask=(df_temp['landcover_classfication']==pft)
                df_mask=df_temp[mask]
                df_mask=df_mask.dropna()
                vals=df_mask[var].tolist()
                result_list.append(vals)
            plt.boxplot(result_list)

            plt.xticks(range(len(pft_unique_list)),pft_unique_list)


            plt.title(var)
            plt.show()

    def pft_test2(self):
        dff = result_root + rf'3mm\Multiregression\partial_correlation\Obs\obs_climate\Dataframe\\df_normalized.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        variables = {
            'composite_LAI_beta_mean',
          'detrended_sum_rainfall_CV',
           'CV_intraannual_rainfall_ecosystem_year'
        }

        ## get uqniue pft
        pft_unique = df['landcover_classfication'].unique()
        # print(pft_unique);exit()
        pft_unique_list = ['Global','Grass', 'Evergreen', 'Deciduous', 'Shrub']
        for pft in pft_unique_list:
            if pft == 'Global':
                df_mask=df
            else:
                mask = (df['landcover_classfication'] == pft)
                df_mask = df[mask]
            df_mask = df_mask.dropna()


            bins=np.arange(0,1.1,0.25)

            ## calculate each bin percentage
            result_len_dic={}
            label_dic = {'composite_LAI_beta_mean': 'Sensitivity',
                         'detrended_sum_rainfall_CV': 'CV_interannual_rainfall',
                         'CV_intraannual_rainfall_ecosystem_year': 'CV_intraannual_rainfall'
                         }

            for var in variables:
                var_name=label_dic[var]
                label_list=[]
                df_var = df_mask[[var]].dropna()
                total = len(df_var)
                result_len_dic[var_name]=[]
                for i in range(len(bins)-1):

                    if i < len(bins) - 2:
                        mask = (df_var[var] >= bins[i]) & (df_var[var] < bins[i + 1])
                    else:
                        mask = (df_var[var] >= bins[i]) & (df_var[var] <= bins[i + 1])
                    bins[i]=round(bins[i],2)
                    bins[i + 1] = round(bins[i + 1], 2)
                    label=f'{bins[i]}-{bins[i+1]}'
                    label_list.append(label)

                    percentage=len(df_mask[mask])/total*100
                    ## percentage.2f
                    percentage=round(percentage,2)

                    result_len_dic[var_name].append(percentage)


            df_new=pd.DataFrame(result_len_dic)
            df_new=df_new.T
            T.print_head_n(df_new)
            ax=df_new.plot.bar(stacked=True,colormap='Set2',width=0.5,)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)



            ## add legend as label list
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, label_list, loc='upper right', bbox_to_anchor=(1.1, 1.05))


            plt.title(pft)
            plt.show()













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

class multi_regression_beta_TRENDY():
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'

        self.fdirX=self.result_root+rf'\3mm\moving_window_multi_regression\moving_window\window_detrend_ecosystem_year\\\\'
        self.fdir_Y=self.result_root+rf'\\\3mm\relative_change_growing_season\moving_window_extraction\\'

        self.xvar_list = ['sum_rainfall_detrend','Tmax_detrend','VPD_detrend']
        self.y_var_list =  ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',

                          'YIBs_S2_Monthly_lai']


    def run(self):
        # self.anomaly()
        # self.detrend()
        # self.moving_window_extraction()

        self.window = 38-15+1


        # # ####step 1 build dataframe
        for y_var in self.y_var_list:



            outdir = self.result_root + rf'3mm\moving_window_multi_regression\TRENDY\\multiresult_relative_change_detrend_ecosystem_year\{y_var}\\'
            # if os.path.isdir(outdir):
            #     continue
            T.mk_dir(outdir, force=True)
            # for i in range(self.window):
            #
            #     df_i = self.build_df(self.fdirX, self.fdir_Y, self.xvar_list, y_var,i)
            #     outf= outdir+rf'\\window{i:02d}.npy'
            #     if os.path.isfile(outf):
            #         continue
            #     print(outf)
            # # #
            #     self.cal_multi_regression_beta(df_i,self.xvar_list, outf)  # 修改参数
            ##step 2 crate individial files
            # self.plt_multi_regression_result(outdir,y_var)
#
        # ##step 3 covert to time series

            # self.convert_files_to_time_series(outdir,y_var) ## 这里乘以100
            # ## step 4 build dataframe using build Dataframe function and then plot here

            # # spatial trends of sensitivity
            # self.calculate_trend_trend(outdir)
            self.composite_beta()
            # plot robinson
            # self.plot_robinson()
            # self.plot_sensitivity_preicipation_trend()

    def anomaly(self):  ### anomaly GS

        fdir = rf'E:\Project3\Result\3mm\moving_window_multi_regression\original\\'

        outdir = rf'E:\Project3\Result\3mm\moving_window_multi_regression\\anomaly_ecosystem_year\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)
            # if os.path.isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')

            dic = np.load(fdir + f, allow_pickle=True, ).item()

            anomaly_dic = {}

            for pix in tqdm(dic):

                r, c = pix

                time_series = dic[pix]['ecosystem_year']
                print(len(time_series))

                time_series = np.array(time_series)

                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                # plt.plot(time_series)
                # plt.show()

                mean = np.nanmean(time_series)

                delta_time_series = (time_series - mean)

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

        fdir=result_root + rf'\3mm\moving_window_multi_regression\anomaly_ecosystem_year\\selected_variables\\'
        outdir=result_root + rf'\3mm\moving_window_multi_regression\anomaly_ecosystem_year\\selected_variables\\detrend\\'
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
                detrend_delta_time_series = signal.detrend(time_series)+np.nanmean(time_series)
                # plt.plot(time_series)
                # plt.plot(detrend_delta_time_series)
                # plt.show()

                detrend_zscore_dic[pix] = detrend_delta_time_series

            np.save(outf, detrend_zscore_dic)

    def moving_window_extraction(self):

        fdir_all = self.result_root + rf'3mm\moving_window_multi_regression\anomaly_ecosystem_year\selected_variables\detrend\\'

        outdir = self.result_root  + rf'\3mm\moving_window_multi_regression\moving_window\window_detrend_ecosystem_year\\'
        T.mk_dir(outdir, force=True)
        # outdir = self.result_root + rf'\3mm\extract_LAI4g_phenology_year\moving_window_extraction\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir_all):

            if not f.endswith('.npy'):
                continue

            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)

            # if os.path.isfile(outf):
            #     continue
            # if os.path.isfile(outf):
            #     continue

            dic = T.load_npy(fdir_all + f)
            window = 15

            new_x_extraction_by_window = {}
            for pix in tqdm(dic):

                time_series = dic[pix]
                # time_series = dic[pix]

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
            if i + window >= len(x)+1:  ####revise  here!!
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




    def build_df(self, fdir_X, fdir_Y, xvar_list,y_var,w):

        df = pd.DataFrame()
        dic_y=T.load_npy(fdir_Y+y_var+'_detrend.npy')
        pix_list = []
        y_val_list=[]

        for pix in dic_y:
            r,c= pix


            if len(dic_y[pix]) == 0:
                continue
            # print(len(dic_y[pix]))
            if len(dic_y[pix]) != self.window:
                continue
            vals = dic_y[pix][w]

            # exit()
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            vals = np.array(vals,dtype=float)


            vals[vals>999.0] = np.nan
            vals[vals<-999.0] = np.nan

            pix_list.append(pix)
            y_val_list.append(vals)

        df['pix'] = pix_list
        df['y'] = y_val_list

        ##df histogram



        # build x

        for xvar in xvar_list:


            x_val_list = []
            x_arr = T.load_npy(fdir_X+xvar+'.npy')
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in x_arr:
                    x_val_list.append([])
                    continue
                # print(len(x_arr[pix]))
                if len(x_arr[pix]) < self.window:
                    x_val_list.append([])
                    continue
                vals = x_arr[pix][w]
                vals = np.array(vals)
                vals = np.array(vals, dtype=float)
                vals[vals > 999] = np.nan
                vals[vals < -999] = np.nan
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
            r,c=pix

            y_vals = row['y']
            y_vals[y_vals<-999]=np.nan
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

            # print(y_vals)
            if len(y_vals) != 15:
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
            # print(df_new['y'])

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

    def plt_multi_regression_result(self, multi_regression_result_dir,y_var):
        fdir = multi_regression_result_dir
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if 'pvalue' in f:
                continue
            print(f)

            w=f.split('\\')[-1].split('.')[0][-2:]


            w=int(w)

            dic = T.load_npy(fdir+f)
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
                arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
                outdir=fdir+'TIFF\\'
                T.mk_dir(outdir)
                outf=outdir+f.replace('.npy','')

                DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, outf + f'_{var_i}.tif')
                std = np.nanstd(arr)
                mean = np.nanmean(arr)
                vmin = mean - std
                vmax = mean + std
                # plt.figure()
                # arr[arr > 0.1] = 1
                # plt.imshow(arr,vmin=-0.5,vmax=0.5)
                #
                # plt.title(var_i)
                # plt.colorbar()

            # plt.show()
    def convert_files_to_time_series(self, multi_regression_result_dir,y_var):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        # average_LAI_f = self.result_root + rf'state_variables\LAI4g_1982_2020.npy'
        # average_LAI_dic = T.load_npy(average_LAI_f)  ### normalized Co2 effect


        fdir = multi_regression_result_dir+'\\'+'TIFF\\'



        variable_list = ['sum_rainfall_detrend']



        for variable in variable_list:
            array_list = []

            for f in os.listdir(fdir):

                if not variable in f:
                    continue
                if not f.endswith('.tif'):
                    continue
                if 'pvalue' in f:
                    continue
                print(f)

                array= ToRaster().raster2array(fdir+f)[0]
                array = np.array(array)


                array_list.append(array)
            array_list=np.array(array_list)

            ## array_list to dic
            dic=DIC_and_TIF(pixelsize=0.5).void_spatial_dic()
            result_dic = {}
            for pix in dic:
                r, c = pix

                if r < 60:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue


                dic[pix]=array_list[:,r,c] ## extract time series




                time_series=dic[pix]
                time_series = np.array(time_series)
                time_series = time_series*100  ###currently no multiply %/100mm
                result_dic[pix]=time_series
                if np.nanmean(dic[pix])<=5:
                    continue
                # print(len(dic[pix]))
                # exit()
            outdir=multi_regression_result_dir+'\\'+'npy_time_series\\'
            print(outdir)
            # exit()
            T.mk_dir(outdir,force=True)
            outf=outdir+rf'\\{variable}.npy'
            np.save(outf,result_dic)

        pass

    def plot_moving_window_time_series(self):
        df= T.load_df(result_root + rf'\3mm\Dataframe\moving_window_multi_regression\\phenology_LAI_mean_trend.df')

        # variable_list = ['precip_detrend','rainfall_frenquency_detrend']
        variable_list = ['precip', 'rainfall_frenquency','rainfall_seasonality_all_year','rainfall_intensity']

        df=df.dropna()
        df=self.df_clean(df)

        fig = plt.figure()
        i = 1

        for variable in variable_list:

            ax = fig.add_subplot(2, 2, i)

            vals = df[f'{variable}'].tolist()

            vals_nonnan = []

            for val in vals:
                if type(val) == float:  ## only screening
                    continue
                if np.isnan(np.nanmean(val)):
                    continue
                if np.nanmean(val) <=-999:
                    continue

                vals_nonnan.append(val)
            ###### calculate mean
            vals_mean = np.array(vals_nonnan)  ## axis=0, mean of each row  竖着加
            vals_mean = np.nanmean(vals_mean, axis=0)
            vals_mean = vals_mean.tolist()
            plt.plot(vals_mean, label=variable)

            i = i + 1

        plt.xlabel('year')

        plt.ylabel(f'{variable}_LAI4g')
        # plt.legend()

        plt.show()
    def calculate_trend_trend(self,outdir):  ## calculate the trend of trend

    ## here input is the npy file
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)

        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=outdir+'\\npy_time_series\\'



        outdir_trend=outdir+'\\trend\\'
        T.mk_dir(outdir_trend,force=True)



        for f in os.listdir(fdir):
            if not f.endswith('npy'):
                continue

            if 'p_value' in f:
                continue


            dic = T.load_npy(fdir + f)

            outf = outdir + f.split('.')[0] + f'_trend.npy'
            print(outf)



            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):

                time_series_all = dic[pix]
                # print(time_series_all)

                threshold = 1e-28
                time_series_all[np.abs(time_series_all) < threshold] = 0.0
                if np.isnan(np.nanmean(time_series_all)):
                    continue
                MODID_LUCC_value = dic_modis_mask[pix]
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue

                if MODID_LUCC_value == 12:
                    continue

                dryland_value=dic_dryland_mask[pix]
                if np.isnan(dryland_value):
                    continue
                time_series_all = np.array(time_series_all)

                if len(time_series_all) < 24:
                    continue
                time_series_all[time_series_all < -999] = np.nan

                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series_all)), time_series_all)

                trend_dic[pix]=slope
                p_value_dic[pix]=p_value

            arr_trend=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(trend_dic)
            arr_p_value = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(p_value_dic)
            # plt.imshow(arr_trend)
            # plt.colorbar()
            # plt.show()
            outf = outdir_trend + f.split('.')[0] + f'_trend.tif'
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_trend,outf)
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_p_value, outf + '_p_value.tif')
                ## save
            # np.save(outf, trend_dic)
            # np.save(outf+'_p_value', p_value_dic)

            ##tiff

    def composite_beta(self):
        fdir=result_root+rf'3mm\moving_window_multi_regression\TRENDY\multiresult_relative_change_detrend_ecosystem_year\\'
        variables_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',
                          'YIBs_S2_Monthly_lai']
        outf=fdir+'TRENDY_ensemble_LAI_beta_mean.tif'

        average_list = []

        for va in variables_list:
            fdir_i=join(fdir,va,'trend')
            fpath=join(fdir_i,'sum_rainfall_detrend_trend.tif')


            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)

            average_list.append(arr)
        average_list=np.array(average_list)
        average_list=np.nanmean(average_list,axis=0)
        # plt.imshow(average_list)
        # plt.colorbar()
        # plt.show()
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(average_list,outf)









        # np.save(rf'D:\Project3\Result\3mm\moving_window_multi_regression\\multiresult_zscore_detrend\composite_LAI\\composite_LAI_beta_mean.npy',average_dic)



    def plot_robinson(self):

        fdir_trend = result_root+rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year\\\npy_time_series\trend\\'
        temp_root = result_root+rf'\3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year\\\npy_time_series\trend\\'
        outdir = result_root+rf'\3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year\\npy_time_series\\\trend_plot\\'
        T.mk_dir(outdir, force=True)
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
            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=-2, vmax=2, is_discrete=True, colormap_n=7, cmap='RdBu')

            Plot_Robinson().plot_Robinson_significance_scatter(m,p_value_f,temp_root,0.05, s=0.2, marker='.')
            plt.title(f'{fname}')
            # plt.show()
            outf = outdir + f+'1.pdf'
            plt.savefig(outf)
            plt.close()

    pass


    def df_clean(self,df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]
        df=df[df['LC_max']<10]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df
    def plot_significant_percentage_area(self):  ### insert bar plot for all spatial map to calculate percentage

        dff = result_root + rf'3mm\Dataframe\Trend\\Trend.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        ##plt histogram of LAI
        df=df[df['LAI4g_1983_2020_trend']<30]
        df=df[df['LAI4g_1983_2020_trend']>-30]

        vals_p_value = df['LAI4g_1983_2020_p_value'].values
        significant_browning_count = 0
        non_significant_browning_count = 0
        significant_greening_count = 0
        non_significant_greening_count = 0

        for i in range(len(vals_p_value)):
            if vals_p_value[i] < 0.05:
                if df['LAI4g_1983_2020_trend'].values[i] > 0:
                    significant_greening_count = significant_greening_count + 1
                else:
                    significant_browning_count = significant_browning_count + 1
            else:
                if df['LAI4g_1983_2020_trend'].values[i] > 0:
                    non_significant_browning_count = non_significant_browning_count + 1
                else:
                    non_significant_greening_count = non_significant_greening_count + 1
            ## plot bar
        ##calculate percentage
        significant_greening_percentage = significant_greening_count / len(vals_p_value)*100
        non_significant_greening_percentage = non_significant_greening_count / len(vals_p_value)*100
        significant_browning_percentage = significant_browning_count / len(vals_p_value)*100
        non_significant_browning_percentage = non_significant_browning_count / len(vals_p_value)*100

        count = [non_significant_browning_percentage,significant_browning_percentage, significant_greening_percentage,

                 non_significant_greening_percentage]
        print(count)
        labels = ['non_significant_browning','significant_browning', 'significant_greening',
                  'non_significant_greening']
        color_list=['navajowhite','chocolate','navy','lightblue',]
        ##gap = 0.1
        df_new=pd.DataFrame({'count':count})
        df_new_T=df_new.T


        df_new_T.plot.barh( stacked=True, color=color_list,legend=False,width=0.1,)
        ## add legend
        plt.legend(labels)

        plt.ylabel('Percentage (%)')
        plt.tight_layout()

        plt.show()

class Figure5():
    def __init__(self):
        self.map_width = 8.2 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
    def run(self):
        self.TRENDY_beta()
        pass


    def TRENDY_LAImin_LAImax(self):
        dff=result_root+rf'3mm\Dataframe\LAImin_LAImax_all_models\\Trend_all.df'
        df=T.load_df(dff)
        df=self.df_clean(df)

        variables_list = ['composite_LAI',
                          'TRENDY_ensemble', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',

                          'YIBs_S2_Monthly_lai']
        values_max_list=[]
        values_min_list=[]

        for variable in variables_list:
            values_min=df[f'{variable}_detrend_min_trend'].values
            values_max=df[f'{variable}_detrend_max_trend'].values
            values_max_list.append(values_max)
            values_min_list.append(values_min)
        values_min_list=np.array(values_min_list)
        values_max_list=np.array(values_max_list)
        values_max_list_mean=np.nanmean(values_max_list,axis=1)
        values_min_list_mean=np.nanmean(values_min_list,axis=1)
        ## add legend

        fig, ax = plt.subplots(figsize=(self.map_width*1.5, self.map_height))
        dic_label_name = {'composite_LAI': 'Composite LAI',
                          'TRENDY_ensemble': 'TRENDY ensemble',
                          'CABLE-POP_S2_lai': 'CABLE-POP',
                          'CLASSIC_S2_lai': 'CLASSIC',
                          'CLM5': 'CLM5',
                          'DLEM_S2_lai': 'DLEM',
                          'IBIS_S2_lai': 'IBIS',
                          'ISAM_S2_lai': 'ISAM',
                          'ISBA-CTRIP_S2_lai': 'ISBA-CTRIP',
                          'JSBACH_S2_lai': 'JSBACH',
                          'JULES_S2_lai': 'JULES',
                          'LPJ-GUESS_S2_lai': 'LPJ-GUESS',
                          'LPX-Bern_S2_lai': 'LPX-Bern',
                          'ORCHIDEE_S2_lai': 'ORCHIDEE',

                          'YIBs_S2_Monthly_lai': 'YIBs',

                          }

        plt.bar(variables_list,values_max_list_mean,color='#96cccb',width=0.7,edgecolor='black',label='Trend in LAImax',)
        plt.bar(variables_list,values_min_list_mean,color='#f6cae5',width=0.7,edgecolor='black',label='Trend in LAImin',)
        plt.legend()

        plt.xticks(np.arange(len(variables_list)),variables_list,rotation=45)
        ## add y=0
        plt.hlines(0, -0.5, len(variables_list) - 0.5, colors='black', linestyles='dashed')
        plt.ylabel('(%/yr)')
        plt.axhline(y=0, color='grey', linestyle='-')
        ax.set_xticks(range(len(variables_list)))
        ax.set_xticklabels(dic_label_name.values(), rotation=90, fontsize=10, font='Arial')
        plt.tight_layout()
        plt.show()
        print(values_max_list_mean)
        print(values_min_list_mean)
        print(variables_list)

    def TRENDY_beta(self):
        dff=result_root+rf'3mm\Dataframe\LAImin_LAImax_all_models_beta\\Trend_all.df'
        df=T.load_df(dff)
        df=self.df_clean(df)

        variables_list = ['composite_LAI',
                          'TRENDY_ensemble_LAI', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',

                          'YIBs_S2_Monthly_lai']
        result_dic={}
        result_stats={}
        values_beta_list=[]
        CI_list=[]


        for variable in variables_list:
            values_beta=df[f'{variable}_beta_trend'].values
            values_beta=np.array(values_beta)
            values_beta[values_beta>100]=np.nan
            values_beta[values_beta<-100]=np.nan

            n=len(values_beta)
            confidence=0.95
            std=np.nanstd(values_beta)
            t_critical=stats.t.ppf((1 + confidence) / 2., n - 1)
            margin_of_error=t_critical * std / np.sqrt(n)
            ci_lower=np.nanmean(values_beta)-margin_of_error
            ci_upper=np.nanmean(values_beta)+margin_of_error
            CI_list.append([ci_lower,ci_upper])


            values_beta_list.append(values_beta)
        CI_list=np.array(CI_list)
        CI_list_T=CI_list.T



        values_beta_list=np.array(values_beta_list)

        values_beta_list_mean=np.nanmean(values_beta_list,axis=1)
        values_beta_list_std=np.nanstd(values_beta_list,axis=1)

        # add legend
        df_new=pd.DataFrame(result_dic)

        fig, ax = plt.subplots(figsize=(self.map_width*1.5, self.map_height))
        dic_label_name = {'composite_LAI': 'Composite LAI',
                          'TRENDY_ensemble_LAI': 'TRENDY ensemble',
                          'CABLE-POP_S2_lai': 'CABLE-POP',
                          'CLASSIC_S2_lai': 'CLASSIC',
                          'CLM5': 'CLM5',
                          'DLEM_S2_lai': 'DLEM',
                          'IBIS_S2_lai': 'IBIS',
                          'ISAM_S2_lai': 'ISAM',
                          'ISBA-CTRIP_S2_lai': 'ISBA-CTRIP',
                          'JSBACH_S2_lai': 'JSBACH',
                          'JULES_S2_lai': 'JULES',
                          'LPJ-GUESS_S2_lai': 'LPJ-GUESS',
                          'LPX-Bern_S2_lai': 'LPX-Bern',
                          'ORCHIDEE_S2_lai': 'ORCHIDEE',

                          'YIBs_S2_Monthly_lai': 'YIBs',

                          }

        ## plot volin
        plt.bar(variables_list,values_beta_list_mean,color='#96cccb',width=0.7,edgecolor='black',
                label='Trend in Beta',yerr=CI_list_T[1]-CI_list_T[0],capsize=3)
        ## CI bar



        plt.xticks(np.arange(len(variables_list)),variables_list,rotation=45)
        plt.ylim(-0.2,0.3)
        ## add y=0
        plt.hlines(0, -0.5, len(variables_list) - 0.5, colors='black', linestyles='dashed')
        plt.ylabel('Beta (%/100ppm/yr)')
        plt.axhline(y=0, color='grey', linestyle='-')
        ax.set_xticks(range(len(variables_list)))
        ax.set_xticklabels(dic_label_name.values(), rotation=90, fontsize=10, font='Arial')
        plt.tight_layout()
        plt.show()


    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df


class multi_regression_anomaly():
    def __init__(self):

        self.fdirX = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\X\\'
        self.fdirY = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\Y\\'

        self.xvar = ['detrended_sum_rainfall_CV', 'composite_LAI_beta_mean',
                     'CV_intraannual_rainfall_ecosystem_year', 'Fire_sum_average', 'sum_rainfall']

        self.y_var = ['composite_LAI_detrend_CV']
        # self.y_var = ['TRENDY_ensemble_composite_time_series_detrend_CV']

        self.multi_regression_result_dir = result_root + rf'3mm\Multiregression\Multiregression_result_residual\OBS_fire_zscore\slope\\'
        T.mk_dir(self.multi_regression_result_dir, force=True)

        self.multi_regression_result_f = self.multi_regression_result_dir + f'{self.y_var[0]}.npy'

        pass

    def run(self):
        ### 0 this is for whole region training not pixel wised

        # self.cal_multi_regression_beta_whole_area()

        # step 1 build dataframe

        df=self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var)
        # #
        # # # # # step 2 cal correlation
        #
        # # self.cal_multi_regression_R2()
        # self.cal_multi_regression_beta_pixel_based(df)
        # #
        # # # # step 3 plot
        # self.plt_multi_regression_result(self.multi_regression_result_dir,self.y_var[0])
        # self.plt_multi_regression_result_p_value(self.multi_regression_result_dir, self.y_var[0])

        # self.normalized_multi_regression()
        # self.statistics_contribution()

        # step 5
        # self.calculate_trend_contribution()
        # self.statistic_contribution()
        self.statistic_Sensitivity()

        pass

    def build_df(self, fdir_X, fdir_Y, fx_list, fy):

        df = pd.DataFrame()

        filey = fdir_Y + fy[0] + '_zscore.npy'
        print(filey)

        dic_y = T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix][0:22]

            if len(yvals) == 0:
                continue
            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
            if yvals[0] == None:
                continue

            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df[self.y_var[0]] = y_val_list

        # build x

        for xvar in fx_list:

            # print(var_name)
            x_val_list = []
            filex = fdir_X + xvar + '_zscore.npy'
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
                xvals = dic_x[pix][0:22]
                xvals = np.array(xvals)
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
        ## save df
        T.save_df(df, self.multi_regression_result_dir + fy[0] + '.df')
        T.df_to_excel(df, self.multi_regression_result_dir + fy[0] + '.xlsx')

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

    def cal_multi_regression_beta_pixel_based(self, df):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd

        import joblib
        from sklearn.metrics import r2_score

        x_var_list = self.xvar

        outf = self.multi_regression_result_f

        multi_derivative = {}
        R2_result = {}
        p_value_result = {}

        for i, row in tqdm(df.iterrows(), total=len(df)):
            # print(row);exit()
            pix = row.pix

            y_vals = row[self.y_var[0]]
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

                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue

                df_new[x] = x_vals

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue
            if len(x_var_list_valid) < 2:
                continue
            # T.print_head_n(df_new)

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
            # x_var_list_valid_new.append('CO2:CRU')
            # # x_var_list_valid_new.append('tmax:CRU')

            df_new = df_new.dropna()
            ## build multiregression model and consider interactioon

            model = sm.OLS(df_new['y'], sm.add_constant(df_new[x_var_list_valid_new])).fit()

            # 获取回归系数和 p-value
            coef = model.params  # 系数
            pvals = model.pvalues  # 每个系数的 p-value
            rsq = model.rsquared  # R²

            # 保存结果
            coef_dic = dict(coef)  # 含常数项的系数
            pval_dic = dict(pvals)  # 含常数项的 p-value

            multi_derivative[pix] = coef_dic
            p_value_result[pix] = pval_dic
            R2_result[pix] = rsq

        T.save_npy(multi_derivative, outf)
        T.save_npy(p_value_result, outf.replace('.npy', '_p_value.npy'))
        outfR2 = outf.replace('.npy', '_R2.npy')

        DIC_and_TIF().pix_dic_to_tif(R2_result, outfR2.replace('.npy', '_R2.tif'))
        T.save_npy(R2_result, outfR2)

    pass

    def cal_multi_regression_R2(self):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd
        from sklearn.metrics import r2_score
        import joblib
        dff = result_root + rf'\3mm\SHAP_beta\Dataframe\\\moving_window_zscore.df'
        df = T.load_df(dff)

        df = self.df_clean(df)
        # print(df.columns);exit()
        x_var_list = ['composite_LAI_beta', 'CV_intraannual_rainfall_ecosystem_year_zscore',
                      'Fire_sum_average_zscore', 'detrended_sum_rainfall_CV_zscore', 'VPD_max_zscore',
                      'rainfall_seasonality_all_year_zscore']
        y_var = 'composite_LAI_CV_zscore'
        df = df.dropna()
        # for col in df.columns:
        #     print(col)
        # exit()
        T.print_head_n(df)

        X = df[x_var_list]
        Y = df[y_var]

        # print(X_train)
        linear_model = LinearRegression()
        linear_model.fit(X, Y)

        y_pred = linear_model.predict(X)

        ## calculate R2
        R2 = r2_score(Y, y_pred)
        plt.scatter(Y, y_pred)
        plt.show()
        print(R2);
        exit()

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir, y_var):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        f = self.multi_regression_result_f

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:

            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            # print(var_i)
            spatial_dic = {}
            for pix in dic:
                r, c = pix
                if r < 60:
                    continue

                landcover_value = crop_mask[pix]

                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            arr = arr * array_mask
            print(var_i)

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, f'{multi_regression_result_dir}\\{var_i}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr, vmin=-5, vmax=5)

            plt.title(var_i)
            plt.colorbar()

        plt.show()

    def plt_multi_regression_result_p_value(self, multi_regression_result_dir, y_var):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        f = self.multi_regression_result_f.replace('.npy', '_p_value.npy')

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:

            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            # print(var_i)
            spatial_dic = {}
            for pix in dic:
                r, c = pix
                if r < 60:
                    continue

                landcover_value = crop_mask[pix]

                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            arr = arr * array_mask
            print(var_i)

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, f'{multi_regression_result_dir}\\{var_i}_p_value.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr, vmin=-5, vmax=5)

            plt.title(var_i)
            plt.colorbar()

        plt.show()

    def plt_R2_result(self, multi_regression_result_dir, y_var):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        f = self.multi_regression_result_f

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:

            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            # print(var_i)
            spatial_dic = {}
            for pix in dic:
                r, c = pix
                if r < 60:
                    continue

                landcover_value = crop_mask[pix]

                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            arr = arr * array_mask
            print(var_i)

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, f'{multi_regression_result_dir}\\{var_i}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr, vmin=-5, vmax=5)

            plt.title(var_i)
            plt.colorbar()

        plt.show()

    def normalized_multi_regression(self):
        fdir = self.multi_regression_result_dir
        spatial_dicts = {}
        variables_list = []

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue

            if 'Ternary_plot' in f:
                continue
            var_list = f.split('.')[0]
            print(f)
            fpath = join(fdir, f)
            fname = f.split('.')[0]
            spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[fname] = spatial_dict_i
            variables_list.append(fname)

        df = T.spatial_dics_to_df(spatial_dicts)
        df = df.dropna(subset=variables_list, how='any')
        # T.print_head_n(df);exit()
        df_abs = pd.DataFrame()
        df_abs['pix'] = df['pix'].tolist()
        for var_i in variables_list:
            abs_vals = np.array(df[var_i].tolist())
            abs_vals = np.abs(abs_vals)
            df_abs[var_i] = abs_vals
        # T.print_head_n(df_abs);exit()

        norm_dict = {}
        # T.add_dic_to_df()

        for i, row in tqdm(df_abs.iterrows(), total=len(df_abs)):
            # print(row[variables_list])
            sum_vals = row[variables_list].sum()
            # print(sum_vals)
            # if sum_vals == 0:
            #     sum_vals = np.nan
            pix = row['pix']
            norm_dict_i = {}
            for var_i in variables_list:
                var_i_norm = row[var_i] / sum_vals
                norm_dict_i[f'{var_i}_norm'] = var_i_norm
            norm_dict[pix] = norm_dict_i
        df_abs = T.add_dic_to_df(df_abs, norm_dict, 'pix')
        # T.print_head_n(df_abs);exit()

        climate_weights_list = []
        # for i,row in df_abs.iterrows():
        #     # VPD_detrend_CV = row['VPD_detrend_CV_norm']
        #     detrended_sum_rainfall_CV = row['detrended_sum_rainfall_CV_norm']
        #     CV_intraannual_rainfall_ecosystem_year = row['CV_intraannual_rainfall_ecosystem_year_norm']
        #     climate_sum = detrended_sum_rainfall_CV + CV_intraannual_rainfall_ecosystem_year
        #     climate_weights_list.append(climate_sum)
        # df_abs['climate_norm']=climate_weights_list
        # T.save_df(df_abs, f'{self.multi_regression_result_dir}\\contributions.df')

        rgb_arr = np.zeros((360, 720, 4))
        # Ter = xymap.Ternary_plot()
        Ter = xymap.Ternary_plot(
            top_color=(67, 198, 219),
            left_color=(255, 165, 00),
            # left_color=(119,0,188),
            right_color=(230, 0, 230),
            # center_color=(85,85,85),
            center_color=(230, 230, 230),
            # center_color=(255,255,255),
        )

        for i, row in df_abs.iterrows():
            pix = row['pix']
            r, c = pix
            climate_norm = row['CV_intraannual_rainfall_ecosystem_year_norm']
            Fire_sum_max_norm = row['detrended_sum_rainfall_CV_norm']
            composite_LAI_beta_mean_norm = row['composite_LAI_beta_mean_norm']
            x, y, z = climate_norm, Fire_sum_max_norm, composite_LAI_beta_mean_norm
            color = Ter.get_color(x, y, z)
            color = color * 255
            color = np.array(color, dtype=np.uint8)
            alpha = 255
            color = np.append(color, alpha)
            # print(color);exit()

            rgb_arr[r][c] = color
        # xymap.GDAL_func().ar
        rgb_arr = np.array(rgb_arr, dtype=np.uint8)
        outtif = join(fdir, 'Ternary_plot.tif')
        tif_template = join(fdir, os.listdir(fdir)[0])
        print(rgb_arr)

        xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
        grid_triangle_legend = Ter.grid_triangle_legend()
        plt.imshow(grid_triangle_legend)
        plt.show()
        T.open_path_and_file(fdir)
        exit()

    def calculate_trend_contribution(self):
        ## here I would like to calculate the trend contribution of each variable
        ## the trend contribution is defined as the slope of the linear regression between the variable and the target variable mutiplied by trends of the variable
        ## load the trend of each variable
        ## load the trend of the target variable
        ## load multi regression result
        ## calculate the trend contribution
        trend_dir = result_root + rf'3mm\Multiregression\partial_correlation\Obs\obs_climate\input\X\\trend\\'

        selected_vairables_list = self.xvar

        trend_dict = {}
        for variable in selected_vairables_list:
            fpath = join(trend_dir, f'{variable}_zscore_trend.tif')
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array[array < -9999] = np.nan
            spatial_dict = D.spatial_arr_to_dic(array)
            for pix in tqdm(spatial_dict, desc=variable):
                r, c = pix
                if r < 60:
                    continue
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue
                if not pix in trend_dict:
                    trend_dict[pix] = {}
                key = variable
                trend_dict[pix][key] = spatial_dict[pix]

        f = self.multi_regression_result_f
        print(f)
        print(isfile(f))
        # exit()
        dic_multiregression = T.load_npy(f)
        var_list = []
        for pix in dic_multiregression:

            # landcover_value = crop_mask[pix]
            # if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
            #     continue

            vals = dic_multiregression[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        # print(var_list)
        # exit()
        for var_i in var_list:
            spatial_dic = {}
            for pix in dic_multiregression:
                if not pix in trend_dict:
                    continue

                dic_i = dic_multiregression[pix]
                if not var_i in dic_i:
                    continue
                val_multireg = dic_i[var_i]
                if var_i not in trend_dict[pix]:
                    continue

                val_trend = trend_dict[pix][var_i]
                val_contrib = val_multireg * val_trend
                spatial_dic[pix] = val_contrib
            arr_contrib = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr_contrib, cmap='RdBu', interpolation='nearest')
            plt.colorbar()
            plt.title(var_i)
            plt.show()
            outdir = join(self.multi_regression_result_dir, 'contribution')
            T.mk_dir(outdir, force=True)

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_contrib, join(outdir, f'{var_i}_contrib.tif'))

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]
        df = df[df['composite_LAI_detrend_CV_zscore_trend'] > 0]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def statistic_contribution(self):
        dff = result_root + rf'\3mm\Multiregression\Multiregression_result\OBS_fire_zscore\contribution\Dataframe\\Dataframe.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        df.dropna(inplace=True)

        selected_vairables_list = self.xvar
        selected_vairables_list += ['residual']
        result_stat_dict = {}

        for variable in selected_vairables_list:
            values = df[f'{variable}_contrib'].values  # df[variable].values
            values = np.array(values)
            values = values[values > -99]
            values = values[values < 99]
            values_average = np.nanmean(values)
            values_std = np.nanstd(values)
            values_CI = values_std * 1.96 / np.sqrt(len(values))
            result_stat_dict[variable] = [values_average, values_CI]

        ## plot

        for variable in selected_vairables_list:
            values_average, values_CI = result_stat_dict[variable]
            plt.bar(variable, values_average, yerr=values_CI, width=0.5)
        plt.show()

        # plt.savefig(result_root + rf'3mm\Multiregression\Multiregression_result\contribution\statistic.png')
        #

    def statistic_Sensitivity(self):
        dff = result_root + rf'3mm\Multiregression\Multiregression_result\OBS_fire_zscore\contribution\Dataframe\\Dataframe.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df = df[df['composite_LAI_detrend_CV_zscore_trend'] > 0]
        df.dropna(inplace=True)

        selected_vairables_list = self.xvar
        result_stat_dict = {}

        for variable in selected_vairables_list:
            values = df[variable].values
            values = np.array(values)
            values = values[values > -99]
            values = values[values < 99]
            values_average = np.nanmean(values)
            values_std = np.nanstd(values)
            values_CI = values_std * 1.96 / np.sqrt(len(values))
            result_stat_dict[variable] = [values_average, values_CI]

        ## plot
        fig, ax = plt.subplots(figsize=(8, 6))

        for variable in selected_vairables_list:
            values_average, values_CI = result_stat_dict[variable]

            bars = plt.bar(variable, values_average, width=0.5)

        # 美化坐标轴和标签
        ax.set_ylabel('Effect Size', fontsize=14)
        ax.axhline(0, color='black', linewidth=0.8)

        ax.set_xticklabels(selected_vairables_list, rotation=20, fontsize=12)

        ax.tick_params(axis='y', labelsize=12)

        plt.tight_layout()
        plt.show()

        # plt.savefig(result_root + rf'3mm\Multiregression\Multiregression_result\contribution\statistic.png')
        #


class multi_regression_zscore():
    def __init__(self):

        self.fdirX = result_root+rf'3mm\Multiregression\\zscore\\'
        self.fdirY = result_root+rf'\3mm\Multiregression\\zscore\\'

        self.xvar = ['detrended_sum_rainfall_CV', 'composite_LAI_beta_mean',
                          'CV_intraannual_rainfall_ecosystem_year', ]

        # self.y_var = ['composite_LAI_CV']
        self.y_var = ['TRENDY_ensemble_composite_time_series_detrend_CV']

        self.multi_regression_result_dir = result_root + rf'\3mm\\Multiregression\\Multiregression_result\\TRENDY\\'
        T.mk_dir(self.multi_regression_result_dir, force=True)

        self.multi_regression_result_f = self.multi_regression_result_dir+f'{self.y_var[0]}.npy'

        pass

    def run(self):

        # #step 1 build dataframe

        df=self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var)
        #
        # # # # step 2 cal correlation
        self.cal_multi_regression_beta(df)
        #
        # # # step 3 plot
        self.plt_multi_regression_result(self.multi_regression_result_dir,self.y_var[0])

        self.normalized_multi_regression()
        # self.statistics_contribution()
        ## step 4 convert m2/m2/ppm to %/100ppm
        # self.convert_CO2_sensitivity_unit()

        # step 5
        # self.calculate_trend_contribution()
        # self.statistic_contribution()
        # self.statistic_Sensitivity()

        pass

    def build_df(self, fdir_X, fdir_Y, fx_list, fy):

        df = pd.DataFrame()

        filey = fdir_Y + fy[0] + '_zscore.npy'
        print(filey)

        dic_y = T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix][0:22]

            if len(yvals) == 0:
                continue
            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
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
            filex = fdir_X + xvar + '_zscore.npy'
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
                xvals = dic_x[pix][0:22]
                xvals = np.array(xvals)
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
        ## save df
        T.save_df(df, self.multi_regression_result_dir + fy[0] + '.df')
        T.df_to_excel(df, self.multi_regression_result_dir + fy[0] + '.xlsx')

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

    def cal_multi_regression_beta(self,df):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd
        import joblib


        x_var_list = self.xvar

        outf = self.multi_regression_result_f

        multi_derivative = {}

        for i, row in tqdm(df.iterrows(), total=len(df)):
            # print(row);exit()
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

                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue

                df_new[x] = x_vals

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue
            if len(x_var_list_valid) < 2:
                continue
            # T.print_head_n(df_new)

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
            # x_var_list_valid_new.append('CO2:CRU')
            # # x_var_list_valid_new.append('tmax:CRU')

            df_new = df_new.dropna()
            ## build multiregression model and consider interactioon

            linear_model = LinearRegression()
            # print(df_new['y'])

            linear_model.fit(df_new[x_var_list_valid_new], df_new['y'])
            coef_ = np.array(linear_model.coef_)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))
            # print(df_new['y'])
            # exit()
            multi_derivative[pix] = coef_dic
        T.save_npy(multi_derivative, outf)

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir, y_var):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        f = self.multi_regression_result_f

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:


            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            # print(var_i)
            spatial_dic = {}
            for pix in dic:
                r, c = pix
                if r < 60:
                    continue

                landcover_value = crop_mask[pix]

                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            arr = arr * array_mask
            print(var_i)


            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, f'{multi_regression_result_dir}\\{var_i}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr, vmin=-5, vmax=5)

            plt.title(var_i)
            plt.colorbar()

        plt.show()


    def normalized_multi_regression(self):
        fdir=self.multi_regression_result_dir
        spatial_dicts={}
        variables_list = []

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue

            if 'Ternary_plot' in f:
                continue
            var_list=f.split('.')[0]
            print(f)
            fpath = join(fdir, f)
            fname=f.split('.')[0]
            spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
            spatial_dicts[fname] = spatial_dict_i
            variables_list.append(fname)

        df = T.spatial_dics_to_df(spatial_dicts)
        df = df.dropna(subset=variables_list,how='any')
        # T.print_head_n(df);exit()
        df_abs= pd.DataFrame()
        df_abs['pix'] = df['pix'].tolist()
        for var_i in variables_list:
            abs_vals=np.array(df[var_i].tolist())
            abs_vals = np.abs(abs_vals)
            df_abs[var_i]=abs_vals
        # T.print_head_n(df_abs);exit()

        norm_dict = {}
        # T.add_dic_to_df()

        for i,row in tqdm(df_abs.iterrows(),total=len(df_abs)):
            # print(row[variables_list])
            sum_vals = row[variables_list].sum()
            # print(sum_vals)
            # if sum_vals == 0:
            #     sum_vals = np.nan
            pix = row['pix']
            norm_dict_i = {}
            for var_i in variables_list:
                var_i_norm = row[var_i] / sum_vals
                norm_dict_i[f'{var_i}_norm'] = var_i_norm
            norm_dict[pix] = norm_dict_i
        df_abs = T.add_dic_to_df(df_abs, norm_dict, 'pix')
        # T.print_head_n(df_abs);exit()

        climate_weights_list = []
        # for i,row in df_abs.iterrows():
        #     # VPD_detrend_CV = row['VPD_detrend_CV_norm']
        #     detrended_sum_rainfall_CV = row['detrended_sum_rainfall_CV_norm']
        #     CV_intraannual_rainfall_ecosystem_year = row['CV_intraannual_rainfall_ecosystem_year_norm']
        #     climate_sum = detrended_sum_rainfall_CV + CV_intraannual_rainfall_ecosystem_year
        #     climate_weights_list.append(climate_sum)
        # df_abs['climate_norm']=climate_weights_list
        # T.save_df(df_abs, f'{self.multi_regression_result_dir}\\contributions.df')



        rgb_arr = np.zeros((360,720,4))
        # Ter = xymap.Ternary_plot()
        Ter = xymap.Ternary_plot(
            top_color=(67, 198, 219),
            left_color=(255, 165, 00),
            # left_color=(119,0,188),
            right_color=(230, 0, 230),
         # center_color=(85,85,85),
            center_color=(230, 230, 230),
            # center_color=(255,255,255),
        )

        for i, row in df_abs.iterrows():
            pix = row['pix']
            r,c = pix
            climate_norm = row['CV_intraannual_rainfall_ecosystem_year_norm']
            Fire_sum_max_norm = row['detrended_sum_rainfall_CV_norm']
            composite_LAI_beta_mean_norm = row['composite_LAI_beta_mean_norm']
            x,y,z = climate_norm, Fire_sum_max_norm, composite_LAI_beta_mean_norm
            color = Ter.get_color(x,y,z)
            color = color * 255
            color = np.array(color,dtype=np.uint8)
            alpha = 255
            color = np.append(color, alpha)
            # print(color);exit()

            rgb_arr[r][c] = color
        # xymap.GDAL_func().ar
        rgb_arr = np.array(rgb_arr, dtype=np.uint8)
        outtif = join(fdir, 'Ternary_plot.tif')
        tif_template = join(fdir,os.listdir(fdir)[0])
        print(rgb_arr)

        xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
        grid_triangle_legend = Ter.grid_triangle_legend()
        plt.imshow(grid_triangle_legend)
        plt.show()
        T.open_path_and_file(fdir)
        exit()

    def statistics_contribution(self):
        file=join(self.multi_regression_result_dir,'contributions.df')
        df=T.load_df(file)
        df=self.df_clean(df)

        pft_unique_list = ['Grass', 'Evergreen', 'Deciduous', 'Shrub']
        result_dict={}
        for pft in pft_unique_list:

            df_pft = df[df['landcover_classfication']==pft]
            climate_norm=df_pft['climate_norm'].tolist()
            ## calculate norm more than 0.5% pixels numbers
            climate_norm=np.array(climate_norm)
            climate_norm_percent=len(climate_norm[climate_norm>0.4])/len(df_pft)*100

            Fire_sum_max_norm=df_pft['Fire_sum_max_norm'].tolist()
            Fire_sum_max_norm=np.array(Fire_sum_max_norm)
            Fire_sum_max_norm_percent=len(Fire_sum_max_norm[Fire_sum_max_norm>0.4])/len(df_pft)*100

            composite_LAI_beta_mean_norm=df_pft['composite_LAI_beta_mean_norm'].tolist()
            composite_LAI_beta_mean_norm=np.array(composite_LAI_beta_mean_norm)
            composite_LAI_beta_mean_norm_percent=len(composite_LAI_beta_mean_norm[composite_LAI_beta_mean_norm>0.4])/len(df_pft)*100

            result_dict[pft]={'climate_norm':climate_norm_percent,'Fire_sum_max_norm':Fire_sum_max_norm_percent,
                              'composite_LAI_beta_mean_norm':composite_LAI_beta_mean_norm_percent}
        ## plot bar
        for pft in pft_unique_list:
            climate_norm_percent=result_dict[pft]['climate_norm']
            Fire_sum_max_norm_percent=result_dict[pft]['Fire_sum_max_norm']
            composite_LAI_beta_mean_norm_percent=result_dict[pft]['composite_LAI_beta_mean_norm']
            plt.bar([1,2,3], [climate_norm_percent,Fire_sum_max_norm_percent,composite_LAI_beta_mean_norm_percent], tick_label=['climate_norm','Fire_sum_max_norm','composite_LAI_beta_mean_norm'])
            plt.title(pft)
            plt.show()






    pass


    def convert_CO2_sensitivity_unit(self):
        period_list = ['1982_2020']
        for period in period_list:
            CO2_sensitivity_f = result_root + rf'multi_regression\\anomaly\\{period}\\CO2_LAI4g_{period}.tif'
            average_LAI4g_f = result_root + rf'\state_variables\\\\LAI4g_{period}.npy'
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(CO2_sensitivity_f)
            arr[arr < -99] = np.nan
            dic_CO2_sensitivity = DIC_and_TIF().spatial_arr_to_dic(arr)

            dic_LAI4g_average = T.load_npy(average_LAI4g_f)

            for pix in dic_CO2_sensitivity:
                CO2_sensitivity = dic_CO2_sensitivity[pix]
                CO2_sensitivity = np.array(CO2_sensitivity, dtype=float)
                if np.isnan(CO2_sensitivity):
                    continue
                if not pix in dic_LAI4g_average:
                    continue
                LAI_average = dic_LAI4g_average[pix]
                LAI_average = np.array(LAI_average, dtype=float)

                if np.isnan(LAI_average):
                    continue
                CO2_sensitivity = CO2_sensitivity / LAI_average * 100
                if CO2_sensitivity < -99999:
                    continue
                dic_CO2_sensitivity[pix] = CO2_sensitivity
            arr_new = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(dic_CO2_sensitivity)
            arr_new[arr_new < -99] = np.nan
            arr_new[arr_new > 99] = np.nan

            # plt.imshow(arr_new)
            # plt.colorbar()
            # plt.show()

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_new, f'{CO2_sensitivity_f.replace(".tif", "_scale.tif")}')
            # DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_CO2_sensitivity, f'{CO2_sensitivity_f.replace(".tif","_new.tif")}')
            # T.save_npy(dic_CO2_sensitivity, CO2_sensitivity_f.replace('.tif', '.npy'))

    def calculate_trend_contribution(self):
        ## here I would like to calculate the trend contribution of each variable
        ## the trend contribution is defined as the slope of the linear regression between the variable and the target variable mutiplied by trends of the variable
        ## load the trend of each variable
        ## load the trend of the target variable
        ## load multi regression result
        ## calculate the trend contribution
        trend_dir = result_root + rf'\3mm\Multiregression\Multiregression_result\Trend\\'

        selected_vairables_list = [
            'fire_ecosystem_year_average',
            'sum_rainfall_growing_season',
            'CV_intraannual_rainfall_growing_season',
            'CV_intraannual_rainfall_ecosystem_year',
        ]

        trend_dict = {}
        for variable in selected_vairables_list:
            fpath = join(trend_dir, f'{variable}.tif')
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array[array < -9999] = np.nan
            spatial_dict = D.spatial_arr_to_dic(array)
            for pix in tqdm(spatial_dict, desc=variable):
                r, c = pix
                if r < 60:
                    continue
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue
                if not pix in trend_dict:
                    trend_dict[pix] = {}
                key = variable
                trend_dict[pix][key] = spatial_dict[pix]

        f = self.multi_regression_result_f
        print(f)
        print(isfile(f))
        # exit()
        dic_multiregression = T.load_npy(f)
        var_list = []
        for pix in dic_multiregression:

            # landcover_value = crop_mask[pix]
            # if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
            #     continue

            vals = dic_multiregression[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        # print(var_list)
        # exit()
        for var_i in var_list:
            spatial_dic = {}
            for pix in dic_multiregression:
                if not pix in trend_dict:
                    continue

                dic_i = dic_multiregression[pix]
                if not var_i in dic_i:
                    continue
                val_multireg = dic_i[var_i]
                if var_i not in trend_dict[pix]:
                    continue

                val_trend = trend_dict[pix][var_i]
                val_contrib = val_multireg * val_trend
                spatial_dic[pix] = val_contrib
            arr_contrib = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr_contrib, cmap='RdBu', interpolation='nearest')
            plt.colorbar()
            plt.title(var_i)
            plt.show()
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_contrib,
                                                   f'{self.multi_regression_result_dir}\\{var_i}_trend_contribution.tif')
    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def statistic_contribution(self):
        dff=result_root + rf'3mm\Multiregression\Multiregression_result\contribution\Dataframe\\contribution.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        df = df[df['composite_LAI_beta_trend_growing_season'] > 0]
        df.dropna(inplace=True)

        selected_vairables_list = [
            'CV_intraannual_rainfall_ecosystem_year_composite_LAI_beta_senstivity',
            'CV_intraannual_rainfall_growing_season_composite_LAI_beta_senstivity',
            'fire_ecosystem_year_average_composite_LAI_beta_senstivity',
            'sum_rainfall_growing_season_composite_LAI_beta_senstivity',
        ]
        result_stat_dict={}



        for variable in selected_vairables_list:
            values=df[variable].values
            values=np.array(values)
            values=values[values>-99]
            values=values[values<99]
            values_average=np.nanmean(values)
            values_std=np.nanstd(values)
            values_CI=values_std*1.96/np.sqrt(len(values))
            result_stat_dict[variable]=[values_average,values_CI]

        ## plot


        for variable in selected_vairables_list:
            values_average,values_CI=result_stat_dict[variable]
            plt.bar(variable,values_average,yerr=values_CI, width=0.5)
        plt.show()

        # plt.savefig(result_root + rf'3mm\Multiregression\Multiregression_result\contribution\statistic.png')
        #

    def statistic_Sensitivity(self):
        dff = result_root + rf'3mm\Multiregression\partial_correlation\\partial_correlation_df.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df = df[df['composite_LAI_beta_trend'] > 0]
        df.dropna(inplace=True)

        selected_vairables_list = [
            'composite_LAI_beta_mean',
            'detrended_sum_rainfall_CV',
            'Fire_sum_max',

        ]
        result_stat_dict = {}

        for variable in selected_vairables_list:
            values = df[variable].values
            values = np.array(values)
            values = values[values > -99]
            values = values[values < 99]
            values_average = np.nanmean(values)
            values_std = np.nanstd(values)
            values_CI = values_std * 1.96 / np.sqrt(len(values))
            result_stat_dict[variable] = [values_average, values_CI]

        ## plot
        fig, ax = plt.subplots(figsize=(8, 6))

        for variable in selected_vairables_list:
            values_average, values_CI = result_stat_dict[variable]



            bars = plt.bar(variable, values_average,  width=0.5)



        # 美化坐标轴和标签
        ax.set_ylabel('Effect Size', fontsize=14)
        ax.axhline(0, color='black', linewidth=0.8)

        ax.set_xticklabels(selected_vairables_list, rotation=20, fontsize=12)

        ax.tick_params(axis='y', labelsize=12)

        plt.tight_layout()
        plt.show()

        # plt.savefig(result_root + rf'3mm\Multiregression\Multiregression_result\contribution\statistic.png')
        #




class GAM():
    def __init__(self):
        pass
    def run(self):
        # self.VIF()
        self.GAM_model()
        pass
    def df_clean(self, df):
        # T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        print('original len(df):',len(df))
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df=df[df['LC_max']<20]
        # df = df[df['extraction_mask'] == 1]



        df = df[df['MODIS_LUCC'] != 12]
        # df=df[df['landcover_classfication'] != 'Cropland']
        print('filtered len(df):',len(df))
        # exit()


        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def VIF(self):

        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        dff = rf'D:\Project3\Result\3mm\SHAP_beta\Dataframe\\moving_window_zscore.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        # df = df[df['wet_dry'] == 'wetting']

        # Example: Your data
        # Suppose you have a DataFrame with columns:
        # 'beta', 'landcover', 'aridity', 'sum_rainfall'

        df.dropna(inplace=True)



        X = df[['VPD', 'Aridity', 'sum_rainfall','heat_event_frenquency',
                'rooting_depth','nitrogen','sand','cwdx80_05','Burn_area_mean','FVC_average',
                'SM_average',]]
        X_const = add_constant(X)



        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X_const.values, i + 1) for i in range(len(X.columns))]  # skip intercept
        print(vif)

    def GAM_model(self):
        import pandas as pd
        import numpy as np
        from pygam import LinearGAM, s, f

        dff=rf'D:\Project3\Result\3mm\SHAP_beta\Dataframe\\moving_window_zscore.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        # for column in df.columns:
        #     print(column)
        # exit()

        # Example: Your data
        # Suppose you have a DataFrame with columns:
        # 'beta', 'landcover', 'aridity', 'sum_rainfall'

        df.dropna(inplace=True)

        df_sample = df.sample(n=10000, random_state=42)


        # Convert categorical variable to codes


        # X = df_sample[['composite_LAI_beta','CV_intraannual_rainfall_ecosystem_year_zscore',
        #                'Fire_sum_average_zscore','detrended_sum_rainfall_CV_zscore'
        #
        #
        #                ]].values

        X = df_sample[['composite_LAI_beta', 'CV_intraannual_rainfall_ecosystem_year_zscore',
                       'detrended_sum_rainfall_CV_zscore'

                       ]].values
        y=df_sample['composite_LAI_CV_zscore'].values



        gam = LinearGAM(
            f(0) +  # categorical: landcover_code
            s(1) +  # smooth term: aridity
            s(2) + # smooth term: sum_rainfall
            s(3)




        ).fit(X, y)

        gam.summary()

        fig, axs = plt.subplots(2, 3, figsize=(15, 4))
        titles = [ 'composite_LAI_beta', 'CV_intraannual_rainfall_ecosystem_year_zscore',
                   'Fire_sum_average_zscore','detrended_sum_rainfall_CV_zscore']

        for i in range(len(titles)):
            ax = axs.flatten()[i]
            XX = gam.generate_X_grid(term=i)
            pd_mean, pd_ci = gam.partial_dependence(term=i, X=XX, width=0.95)

            ax.plot(XX[:, i], pd_mean, label='Partial dependence')
            ax.plot(XX[:, i], pd_ci, c='r', ls='--', label='95% CI')
            ax.set_title(titles[i])
            ax.grid(True)

        plt.tight_layout()
        plt.show()




def main():
    # CCI_landcover_preprocess().run()

    # Figure1().run()
    # Figure2().run()
    # Figure3_beta().run()
    # Figure3_beta_2().run()
    # Figure4().run()
    # build_dataframe().run()
    # greening_CV_relationship().run()
    # multi_regression_beta().run()
    # multi_regression_beta_TRENDY().run()
    # multi_regression_anomaly().run()

    # Figure5().run()

     partial_correlation().run()
    # partial_correlation_TRENDY().run()
    # GAM().run()









if __name__ == '__main__':
    main()