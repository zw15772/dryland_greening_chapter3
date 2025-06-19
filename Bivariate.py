# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg
from openpyxl.styles.builtins import percent
# from green_driver_trend_contribution import *
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import t
from sympy.abc import alpha

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

        self.fdirX=self.result_root+rf'\3mm\moving_window_multi_regression\moving_window\window_detrend_ecosystem_year_zscore\\'
        self.fdir_Y=self.result_root+rf'\3mm\moving_window_multi_regression\moving_window\window_detrend_ecosystem_year_zscore\\'

        self.xvar_list = ['sum_rainfall_zscore_detrend','Tmax_zscore_detrend','VPD_zscore_detrend']
        self.y_var = ['GLOBALMAP_zscore_detrend']


    def run(self):
        # self.anomaly()
        # self.detrend()
        # self.moving_window_extraction()

        self.window = 38-15+1
        outdir = self.result_root + rf'3mm\moving_window_multi_regression\multiresult_raw_detrend\multi_regression_result_detrend_ecosystem_year_SNU_LAI\\'
        T.mk_dir(outdir, force=True)

        # # ####step 1 build dataframe
        # for i in range(self.window):
        #
        #     df_i = self.build_df(self.fdirX, self.fdir_Y, self.xvar_list, self.y_var,i)
        #     outf= outdir+rf'\\window{i:02d}.npy'
        #     if os.path.isfile(outf):
        #         continue
        #     print(outf)
        # # #
        #     self.cal_multi_regression_beta(df_i,self.xvar_list, outf)  # 修改参数
        # ##step 2 crate individial files
        # self.plt_multi_regression_result(outdir,self.y_var)
# #
        # ##step 3 covert to time series

        # self.convert_files_to_time_series(outdir,self.y_var) ## 这里乘以100
        ### step 4 build dataframe using build Dataframe function and then plot here
        # self.plot_moving_window_time_series() not use
        ## spatial trends of sensitivity
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



        variable_list = ['sum_rainfall_zscore_detrend']



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
                time_series = time_series  ###currently no multiply %/100mm
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

class bivariate_analysis():
    def __init__(self):
        pass
    def run(self):
        ## step1
        # self.generate_bivarite_map()
    ## Step2
        # self.generate_df()

        ## step3
        # build_dataframe().run()

        ## step4
        self.generate_three_dimension()

        # self.plot_figure1d_new()
        # self.plot_robinson()

    def generate_bivarite_map(self):  ##

        import xymap
        tif_rainfall = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\trend\detrended_sum_rainfall_CV_trend.tif'
        # tif_CV=  result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'
        tif_sensitivity = result_root + rf'3mm\moving_window_multi_regression\multiresult\multi_regression_result_detrend_ecosystem_year_composite_LAI\trend\\composite_LAI_beta_mean_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outdir = result_root + rf'3mm\\\bivariate_analysis\\composite_LAI\\'
        T.mk_dir(outdir, force=True)
        outtif = outdir + rf'\\interannual_CVrainfall_beta.tif'

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

        f_CVLAItiff=result_root+rf'3mm\extract_composite_phenology_year\trend\\composite_LAI_CV_trend.tif'
        array_CV_LAI,originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_CVLAItiff)
        dic_CV_LAI=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(array_CV_LAI)


        df=T.spatial_dics_to_df({'CVrainfall_beta':dic_beta_CVrainfall,f'{variable}_CV':dic_CV_LAI})

        T.save_df(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend_{variable}.df')
        T.df_to_excel(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend_{variable}.xlsx')
        exit()





    def generate_three_dimension(self):
        variable='composite_LAI'
        dff=result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend_{variable}.df'
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
        outf = outdir + rf'CV_rainfall_beta_LAI.tif'

        spatial_dic = T.df_to_spatial_dic(df, 'CV_rainfall_beta_LAI')
        DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(spatial_dic, outf)
        ##save pdf

        # plt.savefig(outf)
        # plt.close()



    def plot_figure1d_new(self):
        variable='GLOBMAP_CV'
        dff = rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\Trend_{variable}.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        df_sig=df[df['detrended_SNU_LAI_CV_p_value']<0.05]
        dic_label = {1: 'CVLAI+_CVrainfall+_posbeta', 2: 'CVLAI+_CVrainfall+_negbeta', 3: 'CVLAI+_CVrainfall-posbeta',
                     4: 'CVLAI+_CVrainfall-negbeta',
                     5: 'CVLAI-_CVrainfall+_posbeta', 6: 'CVLAI-_CVrainfall+_negbeta', 7: 'CVLAI-_CVrainfall-posbeta',
                     8: 'CVLAI-_CVrainfall-negbeta'}
        dic = {}

        greening_list = []
        browning_list = []


        df_greening=df_sig[df_sig['CV_rainfall_beta_LAI']<5]
        count = len(df_greening)
        greening_list.append(count)

        df_browning=df_sig[df_sig['CV_rainfall_beta_LAI']>=5]
        count = len(df_browning)
        browning_list.append(count)

        greening_sum = np.sum(greening_list)
        browning_sum = np.sum(browning_list)
        greening_percentage = greening_sum / len(df)
        browning_percentage = browning_sum / len(df)
        # print(greening_percentage,browning_percentage);exit()
        # print(greening_sum,browning_sum)

        color_list2 = [
            '#75b1d3',

            '#e7f4cb',

            '#fdba6e',
            '#d7191c',

        ]

        color_list1 = [
            '#2c7bb6',

            '#b7dee3',

            '#fee8a4',
            '#ed6e43',
        ]

        ## count the number of pixels
        for i in range(1, 9):

            if i < 5:
                df_i=df_sig[df_sig['CV_rainfall_beta_LAI']==i]
                count=len(df_i)
                dic[i]=count/greening_sum*100

            else:
                df_i=df_sig[df_sig['CV_rainfall_beta_LAI']==i]
                count=len(df_i)
                dic[i]=count/browning_sum*100
        pprint(dic)

        dic[5] = -dic[5]
        dic[6] = -dic[6]
        dic[7] = -dic[7]
        dic[8] = -dic[8]
        print(dic)

        group_labels = ['CVinterannualrainfall+', 'CVinterannualrainfall-', 'CVinterannualrainfall+',
                        'CVinterannualrainfall-']
        group_data = [
            [dic[1], dic[2]],
            [dic[3], dic[4]],
            [dic[5], dic[6]],
            [dic[7], dic[8]],
        ]

        # 拆分上下部分

        bottoms = [0 if v[0] >= 0 else v[0] for v in group_data]
        heights1 = [v[0] for v in group_data]
        heights2 = [v[1] for v in group_data]

        x = np.arange(len(group_labels))
        bar_width = 0.5

        fig, ax = plt.subplots(figsize=(12 * centimeter_factor, 8 * centimeter_factor))

        # 第一层

        bars1 = ax.bar(x, heights1, width=bar_width, color=color_list1)

        # 第二层堆叠
        bars2 = ax.bar(x, heights2, width=bar_width, bottom=heights1, color=color_list2)

        # 坐标轴 & 图例
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel('Percentage (%)')

        ax.axhline(0, color='black', linewidth=0.8)
        plt.show()

        # plt.tight_layout()
        # plt.savefig(rf'D:\Project3\Result\3mm\FIGURE\CV_rainfall_beta_LAI.pdf')
    def plot_robinson(self):

        fdir_trend = result_root + rf'\3mm\bivariate_analysis\\'
        temp_root = result_root + rf'\3mm\bivariate_analysis\\temp\\'
        outdir = result_root + rf'\3mm\bivariate_analysis\\ROBINSON\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):

            if not f.endswith('.tif'):
                continue

            fname = f.split('.')[0]

            # fpath = fdir_trend + f

            fpath = rf"D:\Project3\Result\3mm\bivariate_analysis\CV_rainfall_beta_LAI.tif"
            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            # m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=8, is_discrete=True, colormap_n=8, cmap='RdYlBu_r',)

            color_list1 = [
                '#006400',
                '#66CDAA',
                '#008080',
                '#AFEEEE',
                '#8B0000',
                '#FFA500',
                '#800080',
                '#FFFF00',
            ]



            my_cmap2 = T.cmap_blend(color_list1, n_colors=8)
            # arr = ToRaster().raster2array(fpath)[0]
            # arr[arr<-999]=np.nan
            # plt.imshow(arr,cmap=my_cmap,vmin=1,vmax=8,interpolation='nearest')
            # plt.colorbar()
            # plt.show()
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=9, is_discrete=True, colormap_n=9,
                                                   cmap='Paired', )

            plt.title(f'{fname}')
            plt.show()
            outf = outdir + 'CV_rainfall_beta_LAI_new.pdf'
            # plt.savefig(outf)
            # plt.close()
            # exit()

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
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
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
        #### CV list

        # color_list = [
        #     '#008837',
        #     '#a6dba0',
        #     '#f7f7f7',
        #     '#c2a5cf',
        #     '#7b3294',
        # ]
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

class greening_CV_relationship():
    def __init__(self):
        self.map_width = 15.3 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass

    def run(self):

        # self.generate_bivarite_map()
        # self.statistic_bar()
        self.heatmap_LAImin_max_CV()

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

    def generate_bivarite_map(self):  ##

        import xymap
        tif_rainfall = result_root + rf'3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\\\LAI4g_detrend_CV_trend.tif'
        # tif_CV=  result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'
        tif_sensitivity= result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\trend_analysis_simple_linear\\LAI4g_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outdir = result_root + rf'3mm\\\bivariate_analysis\\'
        T.mk_dir(outdir, force=True)
        outtif = outdir + rf'\\CV_greening_bivariate_LAI4g.tif'
        T.mk_dir(result_root + rf'bivariate_analysis\\')
        tif1 = tif_rainfall
        tif2 = tif_sensitivity

        dic1 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif1)
        dic2 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif2)
        dics = {'CV_LAI': dic1,
                'Trend_LAI': dic2}
        df = T.spatial_dics_to_df(dics)
        # print(df)
        df['CV_LAI_increase'] = df['CV_LAI'] > 0
        df['Trend_LAI_increase'] = df['Trend_LAI'] > 0

        print(df)
        label_list = []
        for i, row in df.iterrows():
            if row['CV_LAI_increase'] and row['Trend_LAI_increase']:
                label_list.append(1)
            elif row['CV_LAI_increase'] and not row['Trend_LAI_increase']:
                label_list.append(2)
            elif not row['CV_LAI_increase'] and row['Trend_LAI_increase']:
                label_list.append(3)
            elif not row['CV_LAI_increase'] and not row['Trend_LAI_increase']:
                label_list.append(4)
            else:
                raise

        df['label'] = label_list
        result_dic = T.df_to_spatial_dic(df, 'label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(result_dic, outtif)


    def heatmap_LAImin_max_CV(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        dff=rf'D:\Project3\Result\3mm\bivariate_analysis\Dataframe\\Trend_all.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        print(len(df))
        # df = df[df['detrended_SNU_LAI_CV_p_value'] < 0.05]
        # df = df[df['LAI4g_detrend_CV_p_value'] < 0.05]
        df = df[df['GlOBMAP_detrend_CV_p_value'] < 0.05]
        # # print(len(df));exit()


        # plt.show();exit()

        T.print_head_n(df)
        # x_var = 'SNU_LAI_relative_change_detrend_min_trend'
        # y_var = 'SNU_LAI_relative_change_detrend_max_trend'
        # z_var = 'SNU_LAI_CV'

        x_var = 'GLOBMAP_LAI_relative_change_detrend_min_trend'
        y_var = 'GLOBMAP_LAI_relative_change_detrend_max_trend'
        z_var = 'GlOBMAP_detrend_CV_trend'
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




class build_dataframe():


    def __init__(self):

        self.this_class_arr = (result_root+rf'\3mm\bivariate_analysis\Dataframe\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + rf'Trend_composite_LAI.df'

        pass

    def run(self):


        df = self.__gen_df_init(self.dff)
        # df=self.add_detrend_zscore_to_df(df)
        # df=self.append_attributes(df)
        df=self.add_trend_to_df(df)


        df=self.add_aridity_to_df(df)
        df=self.add_dryland_nondryland_to_df(df)
        df=self.add_MODIS_LUCC_to_df(df)
        df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        df=self.add_landcover_classfication_to_df(df)
        df=self.add_maxmium_LC_change(df)
        df=self.add_row(df)

        df=self.add_lat_lon_to_df(df)



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
        fdir=rf'D:\Project3\Result\3mm\extract_composite_phenology_year\trend\\'
        for f in os.listdir(fdir):
            if not 'composite_LAI_CV' in f:
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
        df = df.rename(columns={'detrended_GIMMS_plus_NDVI_CV_trend': 'GIMMS_plus_NDVI_detrend_CV_trend',
                                'detrended_GIMMS_plus_NDVI_CV_p_value': 'GIMMS_plus_NDVI_detrend_CV_p_value',
                                'detrended_NDVI4g_CV_p_value': 'NDVI4g_detrend_CV_p_value',
                                'detrended_NDVI4g_CV_trend': 'NDVI4g_detrend_CV_trend',
                                'detrended_NDVI_trend': 'NDVI_detrend_trend',
                                'detrended_NDVI_p_value': 'NDVI_detrend_p_value',




                            }

                               )



        return df
    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=[


                              'weighted_avg_GOSIF',

                              'weighted_avg_contribution_GOSIF',



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

        self.fdir_X = result_root + rf'\3mm\RF_Multiregression\\'
        self.fdir_Y = result_root + rf'\3mm\RF_Multiregression\\'
        # self.xvar_list = ['fire_ecosystem_year_average', 'pi_average',
        #                   'rainfall_frenquency', 'rainfall_intensity', 'rainfall_seasonality_all_year', ]
        self.xvar_list = [
                          'rainfall_frenquency', 'rainfall_intensity', ]

        self.y_var = ['composite_LAI_beta_mean']
        self.outdir = result_root + rf'\3mm\\RF_Multiregression\partial_correlation\\'
        T.mk_dir(self.outdir, force=True)

        self.outpartial = self.outdir + rf'\partial_corr.npy'
        self.outpartial_pvalue = self.outdir + rf'\partial_pvalue.npy'

    def run(self):
        # df=self.build_df(self.fdir_X,self.fdir_Y,self.xvar_list,self.y_var)
        # #
        # self.cal_partial_corr(df,self.xvar_list, )
        # self.cal_single_correlation()
        # self.cal_single_correlation_ly()
        # self.plot_partial_correlation()
        # self.maximum_partial_corr()
        # self.statistic_corr()
        self.statistic_trend()


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
            yvals=yvals[0:22]
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
                xvals = xvals[0:22]
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

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)



        f_partial = self.outpartial
        f_pvalue = self.outpartial_pvalue
        outdir= self.outdir


        # partial_correlation_dic = np.load(f_partial, allow_pickle=True, encoding='latin1').item()
        partial_correlation_p_value_dic = np.load(f_pvalue, allow_pickle=True, encoding='latin1').item()


        var_list = []
        for pix in partial_correlation_p_value_dic:

            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue
            modis_value= dic_modis_mask[pix]
            if modis_value==12:
                continue

            # vals = partial_correlation_dic[pix]
            vals = partial_correlation_p_value_dic[pix]


            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in partial_correlation_p_value_dic:

                dic_i = partial_correlation_p_value_dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr, self.outdir + f'partial_pvalue_{var_i}.tif')
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










    def maximum_partial_corr(self):
        fdir=self.outdir
        array_list=[]
        array_arg={}
        var_list=[]

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
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
        fdir = result_root + rf'3mm\RF_Multiregression\partial_correlation\\'
        variable_list=['partial_corr_rainfall_intensity','partial_corr_rainfall_frenquency']
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            result_dic={}
            variable=f.split('.')[0]
            if not variable in variable_list:
                continue
            fpath_corr = join(fdir, f)
            fpath_pvalue=fdir+f.replace('partial_corr','partial_pvalue')
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



def main():

    # bivariate_analysis().run()
    # build_dataframe().run()
    greening_CV_relationship().run()
    # multi_regression_beta().run()

    # partial_correlation().run()









if __name__ == '__main__':
    main()