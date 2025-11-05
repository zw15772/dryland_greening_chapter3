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
        self.detrend_deseasonalized_LAI()
        # self.detrend_deseasonalized_climate()
        ## calculating intrasensitivity

        # self.calculating_multiregression_intrasensitivity()
        # self.trend_analysis()
        # self.moving_window_extraction()
        # self.moving_window_average_anaysis()


        pass
    def detrend_deseasonalized_LAI(self):
        fdir=rf'D:\Project3\Data\LAI4g\extract_phenology_monthly\\'
        outdir=rf'D:\Project3\Data\LAI4g\extract_phenology_monthly_detrend_deseason\\'
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
                deseason = (month_series - month_mean)/month_mean
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
        outf=outdir+'extract_phenology_monthly_detrend_deseason_relative_change.npy'

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


        self.fdirX= result_root+rf' \Multiregression_intersensitivity\input\\'
        self.fdir_Y=result_root+rf' \Multiregression_intersensitivity\input\\'

        self.xvar_list = ['Precip_sum_anomaly_detrend',
                          'Tmax_anomaly_detrend',
                          'sum_rainfall_growing_season_zscore_detrend']
        self.y_var_list = ['composite_LAI_relative_change_detrend_mean','composite_LAI_relative_change_detrend_median',]


    def run(self):
        # self.anomaly()
        #
        # self.detrend()
        # exit()
        # self.moving_window_extraction()
        self.calculating_multiregression_intersensitivity()
        # self.trend_analysis()
        # self.plot_time_series()
        # exit()






    def calculating_multiregression_intersensitivity(self):
        import numpy as np
        import statsmodels.api as sm
        from tqdm import tqdm

        # 假设这些是每个像素对应的字典，键是 pix，值是 (year, month)
        fdir=rf'D:\Project3\Result\Nov\Multiregression_intersensitivity\input\\'
        fLAI=fdir+rf'\\composite_LAI_relative_change_detrend_median.npy'
        f_temp=fdir+rf'\\Tmax_anomaly_detrend.npy'
        f_precip=fdir+rf'\\Precip_sum_anomaly_detrend.npy'

        dic_LAI = T.load_npy(fLAI)
        dic_temp = T.load_npy(f_temp)
        dic_precip = T.load_npy(f_precip)

        out_beta={}

        for pix in tqdm(dic_LAI):
            if pix not in dic_temp or pix not in dic_precip:
                continue

            vals_LAI = np.array(dic_LAI[pix], dtype=float)
            vals_temp = np.array(dic_temp[pix], dtype=float)
            vals_precip = np.array(dic_precip[pix], dtype=float)

            # 要求二维 [n_windows, n_years_in_window]
            if vals_LAI.ndim != 2:
                continue

            n_windows, n_years = vals_LAI.shape

            beta_p_list = []
            beta_t_list = []
            p_p_list = []
            p_t_list = []

            for w in range(n_windows):
                y = vals_LAI[w, :]
                x1 = vals_precip[w, :]
                x2 = vals_temp[w, :]

                # 有效数据检查
                mask = ~np.isnan(y) & ~np.isnan(x1) & ~np.isnan(x2)
                print(mask.sum())
                if mask.sum() < 5:
                    beta_p_list.append(np.nan)
                    beta_t_list.append(np.nan)
                    p_p_list.append(np.nan)
                    p_t_list.append(np.nan)
                    continue

                X = np.column_stack([x1[mask], x2[mask]])
                X = sm.add_constant(X)
                y_valid = y[mask]

                try:
                    model = sm.OLS(y_valid, X).fit()
                    betas = model.params
                    pvals = model.pvalues
                    beta_p_list.append(betas[1])  # 降雨敏感性
                    beta_t_list.append(betas[2])  # 温度敏感性
                    p_p_list.append(pvals[1])
                    p_t_list.append(pvals[2])
                except:
                    beta_p_list.append(np.nan)
                    beta_t_list.append(np.nan)
                    p_p_list.append(np.nan)
                    p_t_list.append(np.nan)

            out_beta[pix] = {
                'intersensitivity_precip_val': np.array(beta_p_list),
                'intersensitivity_precip_pval': np.array(p_p_list),
                'intersensitivity_temp_val': np.array(beta_t_list),
                'intersensitivity_temp_pval': np.array(p_t_list)

            }


        # === 保存输出 ===
        outdir = r'D:\Project3\Result\Nov\Multiregression_intersensitivity\output\\composite_LAI_relative_change_detrend_median\\'
        T.mk_dir(outdir, force=True)

        T.save_npy(out_beta, outdir + 'multiregression_intrasensitivity.npy')

    def trend_analysis(self):
        outdir=result_root+r'\Multiregression_intersensitivity\output\\'

        f=result_root+rf'\Multiregression_intersensitivity\output\\multiregression_intrasensitivity.npy'
        dic=T.load_npy(f)
        result_dic={}
        pvalue_result={}
        for pix in dic:
            vals=dic[pix]['intersensitivity_precip_val']

            slope, b, r, p_value = T.nan_line_fit(np.arange(len(vals)), vals)
            result_dic[pix]=slope
            pvalue_result[pix]=p_value
        DIC_and_TIF().pix_dic_to_tif(result_dic, outdir + 'multiregression_intrasensitivity_trend.tif')
        DIC_and_TIF().pix_dic_to_tif(pvalue_result, outdir + 'multiregression_intrasensitivity_pvalue.tif')


        pass


    def plot_time_series(self):
        f=rf'D:\Project3\Result\Nov\Multiregression_intersensitivity\output\\multiregression_intrasensitivity.npy'
        out_beta_dic=T.load_npy(f)

        for pix in out_beta_dic:
            beta_p = np.array(out_beta_dic[pix]['intersensitivity_precip_val'])  # 降雨敏感性
            beta_t = np.array(out_beta_dic[pix]['intersensitivity_temp_val'])  # 温度敏感性


            # 去掉异常值
            beta_p[np.abs(beta_p) > 999] = np.nan
            beta_t[np.abs(beta_t) > 999] = np.nan

            # 绘制时间序列
            plt.figure(figsize=(8, 4))
            plt.plot(beta_p, marker='o', color='#1f78b4', label='β_precip (rainfall sensitivity)')
            plt.plot(beta_t, marker='s', color='#e31a1c', label='β_temp (temperature sensitivity)')
            plt.axhline(0, color='k', lw=1)
            plt.xlabel('Moving window index')
            plt.ylabel('Sensitivity coefficient')
            plt.title(f'Intersensitivity time series at pixel {pix}')
            plt.legend()
            plt.tight_layout()
            plt.show()














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

class multiregression_intersensitivity_TRENDY():
    def __init__(self):


        self.fdirX= result_root+rf' \Multiregression_intersensitivity\input\\'
        self.fdir_Y=result_root+rf' \Multiregression_intersensitivity\input\\'

        self.xvar_list = ['Precip_sum_anomaly_detrend',
                          'Tmax_anomaly_detrend',
                          'sum_rainfall_growing_season_zscore_detrend']
        self.y_var_list = ['composite_LAI_relative_change_detrend_mean','composite_LAI_relative_change_detrend_median',]


    def run(self):
        # self.anomaly()
        #
        # self.detrend()
        # exit()
        # self.moving_window_extraction()
        self.calculating_multiregression_intersensitivity()
        # self.trend_analysis()
        # self.plot_time_series()
        # exit()




    def calculating_multiregression_intersensitivity(self):
        import numpy as np
        import statsmodels.api as sm
        from tqdm import tqdm

        # 假设这些是每个像素对应的字典，键是 pix，值是 (year, month)
        fdir=rf'D:\Project3\Result\Nov\Multiregression_intersensitivity\input\\'
        fLAI=fdir+rf'\\composite_LAI_relative_change_detrend_median.npy'
        f_temp=fdir+rf'\\Tmax_anomaly_detrend.npy'
        f_precip=fdir+rf'\\Precip_sum_anomaly_detrend.npy'

        dic_LAI = T.load_npy(fLAI)
        dic_temp = T.load_npy(f_temp)
        dic_precip = T.load_npy(f_precip)

        out_beta={}

        for pix in tqdm(dic_LAI):
            if pix not in dic_temp or pix not in dic_precip:
                continue

            vals_LAI = np.array(dic_LAI[pix], dtype=float)
            vals_temp = np.array(dic_temp[pix], dtype=float)
            vals_precip = np.array(dic_precip[pix], dtype=float)

            # 要求二维 [n_windows, n_years_in_window]
            if vals_LAI.ndim != 2:
                continue

            n_windows, n_years = vals_LAI.shape

            beta_p_list = []
            beta_t_list = []
            p_p_list = []
            p_t_list = []

            for w in range(n_windows):
                y = vals_LAI[w, :]
                x1 = vals_precip[w, :]
                x2 = vals_temp[w, :]

                # 有效数据检查
                mask = ~np.isnan(y) & ~np.isnan(x1) & ~np.isnan(x2)
                print(mask.sum())
                if mask.sum() < 5:
                    beta_p_list.append(np.nan)
                    beta_t_list.append(np.nan)
                    p_p_list.append(np.nan)
                    p_t_list.append(np.nan)
                    continue

                X = np.column_stack([x1[mask], x2[mask]])
                X = sm.add_constant(X)
                y_valid = y[mask]

                try:
                    model = sm.OLS(y_valid, X).fit()
                    betas = model.params
                    pvals = model.pvalues
                    beta_p_list.append(betas[1])  # 降雨敏感性
                    beta_t_list.append(betas[2])  # 温度敏感性
                    p_p_list.append(pvals[1])
                    p_t_list.append(pvals[2])
                except:
                    beta_p_list.append(np.nan)
                    beta_t_list.append(np.nan)
                    p_p_list.append(np.nan)
                    p_t_list.append(np.nan)

            out_beta[pix] = {
                'intersensitivity_precip_val': np.array(beta_p_list),
                'intersensitivity_precip_pval': np.array(p_p_list),
                'intersensitivity_temp_val': np.array(beta_t_list),
                'intersensitivity_temp_pval': np.array(p_t_list)

            }


        # === 保存输出 ===
        outdir = r'D:\Project3\Result\Nov\Multiregression_intersensitivity\output\\composite_LAI_relative_change_detrend_median\\'
        T.mk_dir(outdir, force=True)

        T.save_npy(out_beta, outdir + 'multiregression_intrasensitivity.npy')

    def trend_analysis(self):
        outdir=result_root+r'\Multiregression_intersensitivity\output\\'

        f=result_root+rf'\Multiregression_intersensitivity\output\\multiregression_intrasensitivity.npy'
        dic=T.load_npy(f)
        result_dic={}
        pvalue_result={}
        for pix in dic:
            vals=dic[pix]['intersensitivity_precip_val']

            slope, b, r, p_value = T.nan_line_fit(np.arange(len(vals)), vals)
            result_dic[pix]=slope
            pvalue_result[pix]=p_value
        DIC_and_TIF().pix_dic_to_tif(result_dic, outdir + 'multiregression_intrasensitivity_trend.tif')
        DIC_and_TIF().pix_dic_to_tif(pvalue_result, outdir + 'multiregression_intrasensitivity_pvalue.tif')


        pass


    def plot_time_series(self):
        f=rf'D:\Project3\Result\Nov\Multiregression_intersensitivity\output\\multiregression_intrasensitivity.npy'
        out_beta_dic=T.load_npy(f)

        for pix in out_beta_dic:
            beta_p = np.array(out_beta_dic[pix]['intersensitivity_precip_val'])  # 降雨敏感性
            beta_t = np.array(out_beta_dic[pix]['intersensitivity_temp_val'])  # 温度敏感性


            # 去掉异常值
            beta_p[np.abs(beta_p) > 999] = np.nan
            beta_t[np.abs(beta_t) > 999] = np.nan

            # 绘制时间序列
            plt.figure(figsize=(8, 4))
            plt.plot(beta_p, marker='o', color='#1f78b4', label='β_precip (rainfall sensitivity)')
            plt.plot(beta_t, marker='s', color='#e31a1c', label='β_temp (temperature sensitivity)')
            plt.axhline(0, color='k', lw=1)
            plt.xlabel('Moving window index')
            plt.ylabel('Sensitivity coefficient')
            plt.title(f'Intersensitivity time series at pixel {pix}')
            plt.legend()
            plt.tight_layout()
            plt.show()














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

def main():
    # multiregression_intrasensitivity().run()
    multiregression_intersensitivity().run()

    pass

if __name__ == '__main__':
    main()

