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
class multiregression_intrasensitivity():
    def __init__(self):
        self.fdir=result_root+rf'\Multiregression_intrasensitivity\\input\\'
        self.xvar_list=[ 'Tmax_detrend_deseason','Precip_detrend_deseason','VPD_detrend_deseason']
        self.yvar='average_detrend_deseasonalized_composite_LAI'
        self.outdir=result_root+rf'\Multiregression_intrasensitivity\\result\\'
        T.mk_dir(self.outdir,force=True)

        pass
    def run(self):
        # self.detrend_deseasonalized_LAI()
        # self.detrend_deseasonalized_climate()
        ## calculating intrasensitivity

        # self.calculating_multiregression_intrasensitivity()
        # self.trend_analysis()
        # self.moving_window_extraction()
        self.moving_window_average_anaysis()


        pass
    def detrend_deseasonalized_LAI(self):
        fdir=rf'D:\Project3\Data\SNU_LAI\extract_phenology_monthly\\'
        outdir=rf'D:\Project3\Data\SNU_LAI\extract_phenology_monthly_detrend_deseason\\'
        T.mk_dir(outdir,force=True)
        dic=T.load_npy_dir(fdir)
        result_dic={}
        for pix in dic:
            vals=dic[pix]
            n_years, n_months = vals.shape
            vals_T = vals.T


            deseason_detrend_T = []
            for i in range(n_months):
                month_series = vals_T[i]
                # plt.plot(month_series)
                month_mean = np.nanmean(month_series)
                deseason = month_series - month_mean
                # plt.plot(deseason)
                detrend_deseason = T.detrend_vals(deseason)
                # plt.plot(detrend_deseason)
                # # plt.legend(['month_series','deseason','detrend_deseason'])
                # plt.show()
                deseason_detrend_T.append(detrend_deseason)
            # plt.imshow(deseason_detrend_T,interpolation='nearest',cmap='jet'
            #            )
            # plt.show()

            # 转回 (years, months)
            deseason_arr = np.array(deseason_detrend_T).T
            # plt.imshow(deseason_arr,interpolation='nearest',cmap='jet'
            #            )
            # plt.show()

            result_dic[pix] = deseason_arr
        outf=outdir+'extract_phenology_monthly_detrend_deseason.npy'

        T.save_npy(result_dic, outf)
        pass



    def detrend_deseasonalized_climate(self):
        fdir=rf'D:\Project3\Data\CRU_monthly\Tmax\extract_phenology_monthly\\'
        outdir=rf'D:\Project3\Data\CRU_monthly\Tmax\extract_phenology_monthly_detrend_deseason\\'
        T.mk_dir(outdir,force=True)
        dic=T.load_npy_dir(fdir)
        result_dic={}
        for pix in dic:
            vals=dic[pix]
            n_years, n_months = vals.shape
            vals_T = vals.T

            ## reshape 38 years or 39years
            deseason_T = []
            for i in range(n_months):
                month_series = vals_T[i]
                month_mean = np.nanmean(month_series)
                deseason = month_series - month_mean
                detrend_deseason = T.detrend_vals(deseason)
                deseason_T.append(detrend_deseason)

            # 转回 (years, months)
            deseason_arr = np.array(deseason_T).T
            plt.imshow(deseason_arr,interpolation='nearest',cmap='jet'
                       )
            plt.show()

            result_dic[pix] = deseason_arr
        outf=outdir+'VPD_detrend_deseason.npy'

        T.save_npy(result_dic, outf)



        pass

    def calculating_multiregression_intrasensitivity(self):
        import numpy as np
        import statsmodels.api as sm
        from tqdm import tqdm

        # 假设这些是每个像素对应的字典，键是 pix，值是 (year, month)
        fdir=rf'D:\Project3\Result\Nov\Multiregression_intrasensitivity\input\\'
        fLAI=fdir+rf'\\average_detrend_deseasonalized_composite_LAI.npy'
        f_temp=fdir+rf'\\Tmax_detrend_deseason.npy'
        f_precip=fdir+rf'\\Precip_detrend_deseason.npy'
        f_vpd=fdir+rf'\\VPD_detrend_deseason.npy'
        dic_LAI = T.load_npy(fLAI)
        dic_temp = T.load_npy(f_temp)
        dic_precip = T.load_npy(f_precip)
        # dic_vpd = T.load_npy(f_vpd)

        out_dic = {}

        for pix in tqdm(dic_LAI):
            vals_LAI = np.array(dic_LAI[pix], dtype=float)
            # print(vals_LAI[0, :])
            if vals_LAI.ndim != 2:
                continue

            n_years, n_months = vals_LAI.shape
            # 对南北半球长度不同自动处理
            # 确保其他变量长度匹配
            if pix not in dic_temp or pix not in dic_precip:
                continue
            vals_temp = np.array(dic_temp[pix], dtype=float)
            vals_precip = np.array(dic_precip[pix], dtype=float)
            # vals_vpd = np.array(dic_vpd[pix], dtype=float)

            betas = []  # 存储每年的回归系数
            pvals = []  # 存储每年的显著性

            for yr in range(n_years):
                y = vals_LAI[yr, :]
                # print(y);exit()

                x1 = vals_precip[yr, :]
                x2 = vals_temp[yr, :]
                # x3 = vals_vpd[yr, :]

                # 检查是否全为 NaN
                if np.isnan(y).all() or np.isnan(x1).all():
                    betas.append([np.nan] * 3)
                    pvals.append([np.nan] * 3)
                    continue

                # 拼接自变量矩阵
                X = np.column_stack([x1, x2,])
                mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
                if mask.sum() < 5:  # 有效样本太少跳过
                    betas.append([np.nan] * 3)
                    pvals.append([np.nan] * 3)
                    continue

                X_valid = X[mask]
                y_valid = y[mask]

                X_valid = sm.add_constant(X_valid)  # 加入截距项
                model = sm.OLS(y_valid, X_valid).fit()
                betas.append(model.params)  # β0, β1, β2, β3
                # print(model.params);exit()
                pvals.append(model.pvalues)
            # plt.plot(betas)
            # plt.legend(['β0', 'β1', 'β2',])
            # plt.show()



            out_dic[pix] = {
                'intrasensitivity_val': np.array(betas)[:, 1].astype(float),
                'intrasensitivity_pval': np.array(pvals)[:, 1].astype(float)
            }


        T.save_npy(out_dic, self.outdir + 'multiregression_intrasensitivity.npy')

        pass
    def trend_analysis(self):

        f=rf'D:\Project3\Result\Nov\Multiregression_intrasensitivity\result\\multiregression_intrasensitivity.npy'
        dic=T.load_npy(f)
        result_dic={}
        pvalue_result={}
        for pix in dic:
            vals=dic[pix]['intrasensitivity_val']

            slope, b, r, p_value = T.nan_line_fit(np.arange(len(vals)), vals)
            result_dic[pix]=slope
            pvalue_result[pix]=p_value
        DIC_and_TIF().pix_dic_to_tif(result_dic, self.outdir + 'multiregression_intrasensitivity_trend.tif')
        DIC_and_TIF().pix_dic_to_tif(pvalue_result, self.outdir + 'multiregression_intrasensitivity_pvalue.tif')


        pass

    def moving_window_extraction(self):


        fdir_all =result_root+ rf'\Multiregression_intrasensitivity\result\\'
        outdir = result_root + rf'\Multiregression_intrasensitivity\moving_window_extraction\\'

        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir_all):

            if not f.endswith('.npy'):
                continue

            #
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


                time_series = dic[pix]['intrasensitivity_val']
                print(len(time_series))


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

    def moving_window_average_anaysis(self): ## each window calculating the average

        fdir = result_root + rf'\Multiregression_intrasensitivity\moving_window_extraction\\'
        outdir = result_root + rf'\Multiregression_intrasensitivity\moving_window_extraction\\\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):


            dic = T.load_npy(fdir + f)
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

                    average=np.nanmean(time_series)
                    # print(average)

                    trend_list.append(average)
                plt.plot(trend_list)
                # plt.ylabel('burn area km2')
                # plt.show()

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)

class multiregression_intersensitivity():
    def __init__(self):


        self.fdirX= result_root+rf' \moving_window_multi_regression\obs\moving_window\detrend\\'
        self.fdir_Y=result_root+rf' \moving_window_multi_regression\obs\moving_window\detrend\\'

        self.xvar_list = ['rainfall_frenquency_growing_season_zscore_detrend',
                          'Tmax_growing_season_zscore_detrend','VPD_growing_season_zscore_detrend',
                          'sum_rainfall_growing_season_zscore_detrend']
        self.y_var_list = ['LAI4g_zscore_detrend','GLOBMAP_LAI_zscore_detrend','SNU_LAI_zscore_detrend']


    def run(self):
        # self.anomaly()
        # exit()
        # self.detrend()
        # exit()
        self.moving_window_extraction()
        exit()
        for y_var in self.y_var_list:
            # print(y_var)


            self.window = 38-15+1
            outdir = self.result_root + rf'\3mm\moving_window_multi_regression\obs\multi_regression_result\growing_season_detrend\\{y_var}\\'
            T.mk_dir(outdir, force=True)

            # # ####step 1 build dataframe
            # for i in range(self.window):
            #
            #     df_i = self.build_df(self.fdirX, self.fdir_Y, self.xvar_list, y_var,i)
            #     outf= outdir+rf'\\window{i:02d}.npy'
            #     # if os.path.isfile(outf):
            #     #     continue
            #     print(outf)
            # #
            #     self.cal_multi_regression_beta(df_i,self.xvar_list, outf)  # 修改参数
            ###step 2 crate individial files
            self.plt_multi_regression_result(outdir)  #### !!!
    # #
            # ##step 3 covert to time series

            self.convert_files_to_time_series(outdir,y_var) ## 这里乘以100
        ### step 4 build dataframe using build Dataframe function and then plot here
        # self.plot_moving_window_time_series() not use
        ## spatial trends of sensitivity
        # self.calculate_trend_trend(outdir)
        # self.composite_beta()
        # plot robinson
        # self.plot_robinson()
        # self.plot_sensitivity_preicipation_trend()

    def anomaly(self):  ### anomaly GS

        fdir = rf'D:\Project3\Result\Nov\CRU_monthly\\extract_annual_growing_season_mean\\'

        outdir = rf'D:\Project3\Result\Nov\CRU_monthly\\anomaly\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            if not f.endswith('.npy'):
                continue


            outf = outdir + f.split('.')[0] + '.npy'
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

                time_series = np.array(time_series,float)

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

        fdir=result_root + rf'\CRU_monthly\anomaly\\'
        outdir=result_root + rf'\CRU_monthly\\detrend\\'
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

    def moving_window_extraction(self):

        fdir_all = result_root + rf'\CRU_monthly\detrend\\'

        outdir = result_root  + rf'\CRU_monthly\detrend\\moving_window_extraction\\'
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
                print((len(time_series)))
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
        dic_y=T.load_npy(fdir_Y+y_var+'.npy')
        pix_list = []
        y_val_list=[]

        for pix in dic_y:
            r,c= pix


            if len(dic_y[pix]) == 0:
                continue
            vals = dic_y[pix][w]
            # plt.plot(vals)
            # plt.show()

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

    def plt_multi_regression_result(self, multi_regression_result_dir):
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



        variable_list = ['sum_rainfall_growing_season_zscore','rainfall_frenquency_growing_season_zscore']

        new_name={'sum_rainfall_growing_season_zscore': f'{y_var}_intersensitivity',
                  'rainfall_frenquency_growing_season_zscore':f'{y_var}_intrasensitivity'}


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
                time_series = time_series
                result_dic[pix]=time_series
                if np.nanmean(dic[pix])<=5:
                    continue
                # print(len(dic[pix]))
                # exit()
            outdir=multi_regression_result_dir+'\\'+'npy_time_series\\'
            print(outdir)
            # exit()
            T.mk_dir(outdir,force=True)
            outf=outdir+new_name[variable]+'.npy'
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


def main():
    # multiregression_intrasensitivity().run()
    multiregression_intersensitivity().run()

    pass

if __name__ == '__main__':
    main()

