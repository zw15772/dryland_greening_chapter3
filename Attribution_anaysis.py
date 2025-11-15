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
        self.trend_analysis()
        # self.moving_window_extraction()
        # self.moving_window_average_anaysis()


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

            # plt.imshow(vals_T,interpolation='nearest',cmap='jet')
            # plt.show()

            deseason_detrend_T = []
            for i in range(n_months):
                month_series = vals_T[i]
                # plt.plot(month_series)
                month_mean = np.nanmean(month_series)
                deseason = (month_series - month_mean)/month_mean*100
                # plt.plot(deseason)
                detrend_deseason = T.detrend_vals(deseason)
                # plt.plot(detrend_deseason)
                # plt.legend(['month_series','deseason','detrend_deseason'])
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
        fLAI=fdir+rf'\\composite_LAI_relative_change_detrend_median.npy'
        f_temp=fdir+rf'\\Tmax_detrend_deseason.npy'
        f_precip=fdir+rf'\\Precip_detrend_deseason.npy'

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



    def run(self):

        # self.calculating_multiregression_intersensitivity()
        self.trend_analysis()
        # self.plot_time_series()
        # exit()






    def calculating_multiregression_intersensitivity(self):
        import numpy as np
        import statsmodels.api as sm
        from tqdm import tqdm

        # 假设这些是每个像素对应的字典，键是 pix，值是 (year, month)
        yvar_list = ['composite_LAI_relative_change_detrend_median',
                     'GLOBMAP_LAI_relative_change_detrend',
                     'LAI4g_relative_change_detrend',
                     'SNU_LAI_relative_change_detrend',
                     'composite_LAI_relative_change_detrend_mean',]

        # 假设这些是每个像素对应的字典，键是 pix，值是 (year, month)
        fdir = result_root + rf'\Multiregression_intersensitivity\input_obs\\'
        for yvar in yvar_list:
            fLAI = fdir + rf'\\{yvar}.npy'
            f_temp = fdir + rf'\\Tmax_anomaly_detrend.npy'
            f_precip = fdir + rf'\\Precip_sum_anomaly_detrend.npy'
            f_VPD=fdir + rf'\\VPD_anomaly_detrend.npy'

            dic_LAI = T.load_npy(fLAI)
            dic_temp = T.load_npy(f_temp)
            dic_precip = T.load_npy(f_precip)
            dic_vpd=T.load_npy(f_VPD)

            out_beta = {}

            for pix in tqdm(dic_LAI):
                if pix not in dic_temp or pix not in dic_precip:
                    continue

                vals_LAI = np.array(dic_LAI[pix], dtype=float)
                vals_temp = np.array(dic_temp[pix], dtype=float)
                vals_precip = np.array(dic_precip[pix], dtype=float)
                vals_vpd = np.array(dic_vpd[pix], dtype=float)

                # 要求二维 [n_windows, n_years_in_window]
                if vals_LAI.ndim != 2:
                    continue

                n_windows, n_years = vals_LAI.shape

                beta_p_list = []
                beta_t_list = []
                beta_vpd_list=[]
                p_p_list = []
                p_t_list = []
                p_vpd_list=[]

                for w in range(n_windows):
                    y = vals_LAI[w, :]
                    x1 = vals_precip[w, :]
                    x2 = vals_temp[w, :]
                    x3=vals_vpd[w,:]

                    # 有效数据检查
                    mask = ~np.isnan(y) & ~np.isnan(x1) & ~np.isnan(x2) & ~np.isnan(x3)
                    # print(mask.sum())
                    if mask.sum() < 5:
                        beta_p_list.append(np.nan)
                        beta_t_list.append(np.nan)
                        beta_vpd_list.append(np.nan)
                        p_p_list.append(np.nan)
                        p_t_list.append(np.nan)
                        p_vpd_list.append(np.nan)

                        continue

                    X = np.column_stack([x1[mask], x2[mask],x3[mask]])
                    X = sm.add_constant(X)
                    y_valid = y[mask]

                    try:
                        model = sm.OLS(y_valid, X).fit()
                        betas = model.params
                        pvals = model.pvalues
                        beta_p_list.append(betas[1])  # 降雨敏感性
                        beta_t_list.append(betas[2])  # 温度敏感性
                        beta_vpd_list.append(betas[3])  ## vpd 敏感性

                        p_p_list.append(pvals[1])
                        p_t_list.append(pvals[2])
                        p_vpd_list.append(pvals[3])

                    except:
                        beta_p_list.append(np.nan)
                        beta_t_list.append(np.nan)
                        p_p_list.append(np.nan)
                        p_t_list.append(np.nan)

                print(len(beta_p_list))

                out_beta[pix] = {
                    'intersensitivity_precip_val': np.array(beta_p_list),
                    'intersensitivity_precip_pval': np.array(p_p_list),
                    'intersensitivity_temp_val': np.array(beta_t_list),
                    'intersensitivity_temp_pval': np.array(p_t_list),
                    'intersensitivity_vpd_val': np.array(beta_vpd_list),
                    'intersensitivity_vpd_pval': np.array(p_vpd_list),


                }


            # === 保存输出 ===
            outdir = r'D:\Project3\Result\Nov\Multiregression_intersensitivity\output_obs\\'
            T.mk_dir(outdir, force=True)

            T.save_npy(out_beta, outdir + f'{yvar}_sensitivity2.npy')

    def trend_analysis(self):
        fdir=result_root+r'\Multiregression_intersensitivity\output_obs\\'
        outdir=result_root+r'\Multiregression_intersensitivity\output_obs\\trend\\'
        T.mk_dir(outdir,force=True)
        for f in os.listdir(fdir):
            if not 'composite_LAI_relative_change_detrend_median_sensitivity2' in f:
                continue
            fname=f.split('.')[0]


            fpath=join(fdir,f)
            dic=T.load_npy(fpath)
            result_dic={}
            pvalue_result={}
            for pix in dic:
                vals=dic[pix]['intersensitivity_precip_val']

                slope, b, r, p_value = T.nan_line_fit(np.arange(len(vals)), vals)
                result_dic[pix]=slope
                pvalue_result[pix]=p_value
            DIC_and_TIF().pix_dic_to_tif(result_dic, outdir + f'{fname}_trend2.tif')
            DIC_and_TIF().pix_dic_to_tif(pvalue_result, outdir + f'{fname}_pvalue2.tif')


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




    def run(self):

        self.calculating_multiregression_intersensitivity()
        # self.trend_analysis()
        # self.plot_time_series()
        # exit()


    def calculating_multiregression_intersensitivity(self):
        import numpy as np
        import statsmodels.api as sm
        from tqdm import tqdm
        yvar_list = ['TRENDY_ensemble_mean',
                      'TRENDY_ensemble_median','CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',
                      'YIBs_S2_Monthly_lai',]

        # 假设这些是每个像素对应的字典，键是 pix，值是 (year, month)
        fdir=result_root+rf'\Multiregression_intersensitivity\input_TRENDY\\'
        for yvar in yvar_list:

            fLAI = fdir + rf'\\{yvar}_relative_change_detrend.npy'
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
            outdir = result_root + rf'Multiregression_intersensitivity\output_TRENDY\\'
            T.mk_dir(outdir, force=True)

            T.save_npy(out_beta, outdir + f'{yvar}_sensitivity.npy')

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


class partial_correlation_obs:
    def __init__(self):
        self.map_width = 8.2 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass

        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = rf'D:/Project3/Result/Nov/partial_correlation/Obs/'

        self.fdirX = self.result_root + rf'\input\\X\\'
        self.fdirY = self.result_root + rf'\input\\Y\\'
        # self.model_list = [
        #     'composite_LAI_median', 'LAI4g', 'GLOBMAP_LAI', 'SNU_LAI',
        #     'TRENDY_ensemble_median', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
        #     'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
        #     'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
        #     'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
        #     'ORCHIDEE_S2_lai',
        #
        #     'YIBs_S2_Monthly_lai',
        #
        # ]

        pass
    def run(self):
        # self.calculating_colinearity_pixel()
        # self.calc_colinearity_global()
        ##### step3 calculating
        self.xvar_list = [

            'Precip_sum_detrend_CV',
            'CV_daily_rainfall_average']
        self.model_list = ['SNU_LAI', 'GLOBMAP_LAI',
                           'LAI4g','composite_LAI_mean',
                          'composite_LAI_median' ]


        # for model in self.model_list:
        #         self.outdir=self.result_root+rf'\result\\'+model+'\\'
        #
        #         T.mk_dir(self.outdir, force=True)
        #         self.outpartial =  self.outdir + rf'\partial_corr_{model}.npy'
        #         self.outpartial_pvalue =  self.outdir + rf'\partial_pvalue_{model}.npy'
        # #
        #         y_var = f'{model}_detrend_CV.npy'
        #         x_var_list = self.xvar_list + [f'{model}_sensitivity']
        #
        #
        #         df=self.build_df(self.fdirX,self.fdirY,x_var_list,y_var)
        #         #
        #         self.cal_partial_corr(df,x_var_list, )
        # # #         #
        # #         # # # # # # self.check_data()
        #         self.plot_partial_correlation()
        #         self.plot_partial_correlation_p_value()


        # self.statistic_trend_bar()
        # self.plot_spatial_map_sig()
        self.statistic_corr_boxplot()

        # self.max_correlation_without_trend()
        # self.max_correlation_with_trend()
        # self.statistic_partial_trends_dorminant()
        pass


    def calculating_colinearity_pixel(self):

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from tqdm import tqdm

        fdir = r'D:\Project3\Result\Nov\partial_correlation\colinear_test\\'

        # === 1. 读取4个变量 ===
        var_names = [
            'composite_LAI_median_sensitivity',
            'Precip_sum_detrend_CV',
            'CV_daily_rainfall_average',
            'CV_monthly_rainfall_average'
        ]
        dic_all = {}
        for f in os.listdir(fdir):
            for name in var_names:
                if name in f:
                    dic_all[name] = T.load_npy(fdir + f)

        print(f"Loaded {len(dic_all)} variables:", list(dic_all.keys()))

        # === 2. 遍历所有像素 ===
        pix_list = list(dic_all[var_names[0]].keys())
        colinear_dic = {}

        for pix in tqdm(pix_list, desc='Checking colinearity'):
            # --- 检查像素是否在所有字典中 ---
            if not all(pix in dic_all[v] for v in var_names):
                continue

            vals = []
            for v in var_names:
                val=dic_all[v][pix]
                if isinstance(val, dict):
                    if 'intersensitivity_precip_val' in val:
                        arr = np.array(val['intersensitivity_precip_val'], dtype=float)
                    else:
                        continue
                else:
                    arr = np.array(val, dtype=float)
                vals.append(arr)

            vals = np.array(vals)  # shape: (4, T)
            if vals.shape[1] < 10:  # 太短不分析
                continue

            # --- 去nan并转置成 (T, 4) ---
            vals = vals.T
            if np.isnan(vals).any():
                vals = vals[~np.isnan(vals).any(axis=1)]

            if vals.shape[0] < 10:
                continue

            df = pd.DataFrame(vals, columns=var_names)

            # === (A) 计算相关矩阵 ===
            corr = df.corr()
            plt.imshow(corr, interpolation='nearest', cmap='jet',vmin=-1,vmax=1)
            plt.colorbar()
            plt.title(pix)
            plt.show()

            # === (B) 计算VIF ===
            X = df.values
            vif_values = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            vif_dic = dict(zip(var_names, vif_values))

            colinear_dic[pix] = {
                'corr': corr.values,
                'vif': vif_dic
            }

        print(f"Finished {len(colinear_dic)} valid pixels.")

        # === 3. 输出结果 ===
        # outf = r'D:\Project3\Result\Nov\partial_correlation\colinear_test\colinear_summary.npy'
        # T.save_npy(colinear_dic, outf)
        # print("Saved:", outf)

    import numpy as np
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import matplotlib.pyplot as plt

    def calc_colinearity_global(self):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from tqdm import tqdm
        fdir = r'D:\Project3\Result\Nov\partial_correlation\colinear_test\\'

        # === 1. 载入四个变量 ===
        var_names = [
            'composite_LAI_median_sensitivity',
            'Precip_sum_detrend_CV',
            'CV_daily_rainfall_average',
            'CV_monthly_rainfall_average'
        ]
        dic_all = {}
        for f in os.listdir(fdir):
            for name in var_names:
                if name in f:
                    dic_all[name] = T.load_npy(fdir + f)

        print(f"Loaded {len(dic_all)} variables:", list(dic_all.keys()))
        # print("dic_all keys:", list(dic_all.keys()))

        # === 2. 对齐像素 ===
        pix_all = set.intersection(*[set(dic_all[v].keys()) for v in var_names])

        all_data = {v: [] for v in var_names}

        # === 3. 提取所有像素的值并拼接 ===
        for pix in pix_all:
            vals = {}

            for v in var_names:
                val = dic_all[v][pix]
                if isinstance(val, dict):  # composite_LAI_median_sensitivity
                    val = val.get('intersensitivity_precip_val', np.nan)
                arr = np.array(val, dtype=float).ravel()
                if np.all(np.isnan(arr)):
                    continue
                vals[v] = arr

            # 检查长度一致
            if len({len(vals[k]) for k in vals}) != 1:
                continue

            for v in var_names:
                all_data[v].extend(vals[v])

        # === 4. 组合成 DataFrame ===
        df = pd.DataFrame(all_data)
        df = df.dropna()

        print(df.shape)  # (N, 4)

        # === 5. 计算相关矩阵 ===
        corr = df.corr()
        print("\nCorrelation Matrix:")
        print(corr)

        plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.xticks(np.arange(len(var_names)), var_names, rotation=45, ha='right')
        plt.yticks(np.arange(len(var_names)), var_names)
        plt.colorbar(label='Pearson r')
        plt.title('Global correlation matrix')
        plt.tight_layout()
        plt.show()

        # === 6. 计算 VIF ===
        X = df.values
        vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif_result = dict(zip(var_names, vif))
        print("\nVariance Inflation Factors (VIF):")
        for k, v in vif_result.items():
            print(f"{k}: {v:.2f}")

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
            yvals = dic_y[pix]

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
                if 'sensitivity' in xvar:
                    xvals = dic_x[pix].get('intersensitivity_precip_val', np.nan)
                else:
                    xvals = dic_x[pix]
                xvals = np.array(xvals)

                if len(xvals) == 0:
                    x_val_list.append([])
                    continue
                print(len(xvals))

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
        outf_corr=self.outpartial
        outf_pvalue=self.outpartial_pvalue

        partial_correlation_dic= {}
        partial_p_value_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']

            if len(y_vals) == 0:
                continue


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

    def plot_partial_correlation(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)



        f_partial = self.outpartial



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

    def plot_spatial_map_sig(self):

        model_list = self.model_list

        for model in model_list:
            variable_list = self.xvar_list
            fdir = self.result_root + rf'\result\\'+model+'\\'
            print(fdir)
            outdir = self.result_root + rf'\result\\{model}\\sig\\'
            T.mk_dir(outdir, True)
            new_variable_list = variable_list + [f'{model}_sensitivity']

            for variable in new_variable_list:
                f_trend_path = fdir + f'{variable}.tif'
                f_pvalue_path = fdir + f'{variable}_p_value.tif'

                arr_corr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_trend_path)
                arr_pvalue, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_pvalue_path)
                # plt.imshow(arr_corr)
                # plt.colorbar()
                # plt.show()
                arr_corr[arr_corr < -99] = np.nan
                arr_corr[arr_corr > 99] = np.nan
                arr_pvalue[arr_pvalue > 99] = np.nan
                arr_pvalue[arr_pvalue < -99] = np.nan
                arr_corr[arr_pvalue > 0.05] = np.nan

                # plt.imshow(arr_corr)
                # plt.colorbar()
                outf = outdir  + f'{variable}.tif'
                DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_corr, outf)

    def statistic_corr_boxplot(self):
        """
        绘制 partial correlation 的分布（仅针对 CVLAI 上升区域）
        显示 sensitivity (γ), CV_inter, CV_intra 的箱线图
        """

        # === 1. 读取数据 ===
        dff = result_root + rf'\partial_correlation\Dataframe\\Obs_TRENDY.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        # === 仅保留CVLAI显著上升的像素 ===
        df = df[df['composite_LAI_detrend_CV_median_trend'] > 0]
        df = df[df['composite_LAI_detrend_CV_median_p_value'] < 0.05]

        # === 2. 变量设置 ===
        variable_list = [
            'sensitivity',
            'Precip_sum_detrend_CV',
            'CV_daily_rainfall_average',
        ]

        label_dic = {
            'sensitivity': r'$\gamma$',
            'Precip_sum_detrend_CV': r'$CV_{inter}$',
            'CV_daily_rainfall_average': r'$CV_{intra}$',
        }



        # === 4. 数据提取 ===

        for model in self.model_list:
            result_dic = {}
            # if 'composite_LAI_median' not in model:
            #     continue

            for variable in variable_list:
                new_variable = f'{model}_{variable}'
                if new_variable not in df.columns:
                    continue

                vals = np.array(df[new_variable].tolist(), dtype=float)
                vals[(vals > 99) | (vals < -99)] = np.nan
                vals = vals[~np.isnan(vals)]
                result_dic[new_variable] = vals

        # === 5. 按 variable_list 顺序组织数据 ===
            data_list = []
            x_labels = []

            for var in variable_list:
                key = f'{model}_{var}'
                if key in result_dic:
                    data_list.append(result_dic[key])
                    x_labels.append(label_dic[var])

                    # 设置颜色
            color_list = ['#a577ad', 'yellowgreen', 'Pink', '#f599a1']
            dark_colors = ['#774685', 'Olive', 'Salmon', '#c3646f']  # 可以改为你自定义的 darken_color 函数

            # 绘图
            fig, ax = plt.subplots(figsize=(4, 3))

            box = ax.boxplot(
                data_list,
                patch_artist=True,
                widths=0.4,
                showfliers=False

            )

            # 自定义颜色
            # === 美化箱线图（让 median、whisker 与箱体颜色一致） ===
            for i, patch in enumerate(box['boxes']):
                face_color = color_list[i]
                edge_color = dark_colors[i]

                # 箱体
                patch.set_facecolor(face_color)
                patch.set_edgecolor(edge_color)
                patch.set_linewidth(1.5)

                # 中位线
                box['medians'][i].set_color(edge_color)
                box['medians'][i].set_linewidth(1.8)

                # 上下须（whisker）
                box['whiskers'][2 * i].set_color(edge_color)
                box['whiskers'][2 * i + 1].set_color(edge_color)
                box['whiskers'][2 * i].set_linewidth(1.2)
                box['whiskers'][2 * i + 1].set_linewidth(1.2)

                # 顶部和底部横线（caps）
                box['caps'][2 * i].set_color(edge_color)
                box['caps'][2 * i + 1].set_color(edge_color)
                box['caps'][2 * i].set_linewidth(1.2)
                box['caps'][2 * i + 1].set_linewidth(1.2)

            # 设置x轴

            plt.xticks(range(1, len(x_labels) + 1), x_labels, fontsize=10)
            plt.xlabel('')
            plt.ylabel('Partial correlation', fontsize=10)

            plt.axhline(0, color='gray', linestyle='--')
            # plt.tight_layout()
            # plt.show()

            outdir=result_root + rf'\FIGURE\Figure4\\'
            Tools().mk_dir(outdir, force=True)

            outf=join(outdir,f'{model}_partial_correlation_boxplot.pdf')
            plt.savefig(outf,bbox_inches='tight',dpi=300

            )
            plt.close()

    def darken_color(self, color, amount=0.7):
        """
        给颜色加深，amount 越小越深 (0~1之间)
        """
        import matplotlib.colors as mcolors
        c = mcolors.to_rgb(color)
        return tuple([max(0, x * amount) for x in c])

    def max_correlation_without_trend(self):
        dff = result_root + rf'\partial_correlation\Dataframe\\Obs_TRENDY.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df = df[df['composite_LAI_detrend_CV_median_trend'] > 0]
        df = df[df['composite_LAI_detrend_CV_median_p_value'] < 0.05]

        model_list = self.model_list

        var_list = [
            'sensitivity',
            'Precip_sum_detrend_CV',
            'CV_daily_rainfall_average',
        ]

        for model in tqdm(model_list):

            outdir = result_root + rf'\partial_correlation\Obs\result\\{model}\\'
            T.mk_dir(outdir, force=True)

            # === 拼接变量名称 ===
            var_list_sens = [f'{model}_' + v for v in var_list]

            max_var_list = []
            color_list = []
            trend_val_list = []

            for _, row in df.iterrows():
                # === 提取该像素下的 sensitivity 值 ===
                vals_sens = np.array([row[v] for v in var_list_sens], dtype=float)
                vals_sens[(vals_sens < -10) | (vals_sens > 10)] = np.nan

                if np.all(np.isnan(vals_sens)):
                    max_var_list.append(np.nan)
                    color_list.append(np.nan)
                    trend_val_list.append(np.nan)
                    continue

                # === 找最大绝对值 ===
                idx_max = np.nanargmax(np.abs(vals_sens))
                max_var = var_list[idx_max]  # 注意取原始名字 (e.g. 'sensitivity')

                # === 嵌套逻辑：dominant + trend方向 ===
                if 'sensitivity' in max_var:
                        color = 1
                elif 'Precip_sum_detrend_CV' in max_var:

                        color = 2

                elif 'CV_daily_rainfall_average' in max_var:

                        color = 3

                else:
                    color = np.nan

                max_var_list.append(max_var)
                color_list.append(color)


            df['max_var'] = max_var_list
            df['color'] = color_list



            # === 写出 color_map ===
            outdir= result_root + rf'\partial_correlation\Obs\result\\{model}\\'
            spatial_dic = T.df_to_spatial_dic(df, 'color')
            out_tif = join(outdir, 'dominant_color_map_without_trend.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dic, out_tif)

    def max_correlation_with_trend(self):
        dff = result_root + rf'\partial_correlation\Dataframe\\Obs_TRENDY.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df = df[df['composite_LAI_detrend_CV_median_trend'] > 0]
        df = df[df['composite_LAI_detrend_CV_median_p_value'] < 0.05]

        model_list = self.model_list

        var_list = [
            'sensitivity',
            'Precip_sum_detrend_CV',
            'CV_daily_rainfall_average',
        ]

        for model in tqdm(model_list):

            outdir = result_root + rf'\partial_correlation\Obs\result\\{model}\\'
            T.mk_dir(outdir, force=True)

            # === 拼接变量名称 ===
            var_list_sens = [f'{model}_' + v for v in var_list]

            max_var_list = []
            color_list = []
            trend_val_list = []

            for _, row in df.iterrows():
                # === 提取该像素下的 sensitivity 值 ===
                vals_sens = np.array([row[v] for v in var_list_sens], dtype=float)
                vals_sens[(vals_sens < -10) | (vals_sens > 10)] = np.nan

                if np.all(np.isnan(vals_sens)):
                    max_var_list.append(np.nan)
                    color_list.append(np.nan)
                    trend_val_list.append(np.nan)
                    continue

                # === 找最大绝对值 ===
                idx_max = np.nanargmax(np.abs(vals_sens))
                max_var = var_list[idx_max]  # 注意取原始名字 (e.g. 'sensitivity')

                # === 对应趋势列名 ===
                if 'sensitivity' in max_var:
                    trend_col = f'{model}_{max_var}_trend'
                else:
                    trend_col = f'{max_var}_trend'

                if trend_col in df.columns:
                    trend_val = row[trend_col]
                else:
                    trend_val = np.nan

                # === 嵌套逻辑：dominant + trend方向 ===
                if 'sensitivity' in max_var:
                    if trend_val > 999:
                        color=np.nan
                    elif trend_val < -999:
                        color=np.nan
                    elif trend_val > 0:
                        color = 2
                    elif trend_val < 0:
                        color = 1
                    else:
                        color = np.nan

                elif 'Precip_sum_detrend_CV' in max_var:
                    if trend_val > 999:
                        color=np.nan
                    elif trend_val < -999:
                        color=np.nan
                    elif trend_val > 0:
                        color = 4
                    elif trend_val < 0:
                        color = 3
                    else:
                        color = np.nan

                elif 'CV_daily_rainfall_average' in max_var:
                    if trend_val > 999:
                        color=np.nan
                    elif trend_val < -999:
                        color=np.nan
                    elif trend_val > 0:
                        color = 6
                    elif trend_val < 0:
                        color = 5
                    else:
                        color = np.nan

                else:
                    color = np.nan

                max_var_list.append(max_var)
                color_list.append(color)
                trend_val_list.append(trend_val)

            df['max_var'] = max_var_list
            df['color'] = color_list

            # === 写出 color_map ===
            outdir = result_root + rf'\partial_correlation\Obs\result\\{model}\\'
            spatial_dic = T.df_to_spatial_dic(df, 'color')
            out_tif = join(outdir, 'dominant_color_map_with_trend.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dic, out_tif)

    def statistic_partial_trends_dorminant(self):
        dff = result_root + rf'\partial_correlation\Dataframe\\Obs_TRENDY.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df=df[df['composite_LAI_detrend_CV_median_trend']>0]
        df=df[df['composite_LAI_detrend_CV_median_p_value']<0.05]


        for col in df.columns:
            print(col)

        model_list = self.model_list
        model_list=['composite_LAI_median']

        result_dic = {}

        # —— 统计：各模型在每个组 ii 的面积百分比（分母用各自非空的样本）——
        for ii in [1, 2, 3, 4, 5, 6, ]:
            percentage_list = []
            for model in model_list:
                col = f'{model}_dominant_color_map_with_trend'
                df_mask = df.dropna(subset=[col])  # 不要改写 df 本体
                df_ii = df_mask[df_mask[col] == ii]
                percent_ii = len(df_ii) / len(df_mask) * 100.0
                percentage_list.append(percent_ii)
            result_dic[ii] = percentage_list
        pprint(result_dic)

        # === 按变量分组 (+, -) ===
        group_pairs = {
            'γ': [1, 2],
            'CV_inter': [3, 4],
            'CV_intra': [5, 6],
        }

        # === 可视化 ===
        plt.figure(figsize=(3, 3))
        x_labels = list(group_pairs.keys())
        x = np.arange(len(x_labels))


        # 颜色设定：负号 = 蓝色，正号 = 红色
        color_minus = '#4A90E2'  # 蓝
        color_plus = '#E94E77'  # 红

        for i, model in enumerate(model_list):
            bar_bottom = np.zeros(len(group_pairs))
            for j, (group_name, pair) in enumerate(group_pairs.items()):
                val_minus = result_dic[pair[0]][i]
                val_plus = result_dic[pair[1]][i]
                # 绘制堆叠
                plt.bar(x[j] + i * 0.15, val_minus, width=0.7, color=color_minus)
                plt.bar(x[j] + i * 0.15, val_plus, bottom=val_minus, width=0.7, color=color_plus)

        # === 外观设置 ===
        plt.xticks(x + (len(model_list) - 1) * 0.15 / 2, x_labels, rotation=0)
        plt.ylabel('Area (%)')

        plt.legend(['negative trend', 'positive trend'], loc='upper right')
        plt.tight_layout()
        plt.show()



    def statistic_contribution_area_barplot(self):
        dff = result_root + rf'\partial_correlation\Dataframe\\Obs_TRENDY.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        for col in df.columns:
                print(col)

        model_list=self.model_list


        result_dic = {}

        # —— 统计：各模型在每个组 ii 的面积百分比（分母用各自非空的样本）——
        for ii in [1, 2, 3, 4, 5, 6, ]:
            percentage_list = []
            for model in model_list:
                col = f'{model}_dorminant_color_map'
                df_mask = df.dropna(subset=[col])  # 不要改写 df 本体
                df_ii = df_mask[df_mask[col] == ii]
                percent_ii = len(df_ii) / len(df_mask) * 100.0
                percentage_list.append(percent_ii)
            result_dic[ii] = percentage_list
        pprint(result_dic)

        dic_variable_name = {1: 'Trends gamma+',
                             2: 'Trends gamma-',
                             3:'Trends CV_intra+',
                             4:'Trends CV_intra-',
                             5:'Trends CV_inter+',
                             6:'Trends CV_inter-'


                             }

        # 颜色：前四个为 obs，第五个（如 TRENDY ensemble）单独色，其余为统一色
        color_list = ['#ADC9E4', '#EBF0FC', '#EBF0FC', '#EBF0FC', '#dd736c'] \
                     + ['#F7DAD4'] * (len(model_list) - 5)

        # 用模型名作为行索引，便于对齐
        df_new = pd.DataFrame(result_dic, index=model_list)

        # —— 画图：每个 ii 一张图，obs 与 models 留间隔，第一根柱子的高度画虚线（只跨 models）——
        for ii in [1, 2, 3, 4, 5, 6]:
            vals = df_new[ii].values
            n_all = len(vals)
            n_obs = 4  # 前 4 个是 obs
            gap = 1.2  # obs 与 models 间的空隙（单位≈一个柱宽）

            # 构造 x 位置：models 整体右移形成间隔
            x = np.arange(n_all, dtype=float)
            x[n_obs:] += gap

            fig, ax = plt.subplots(figsize=(self.map_width, self.map_height))
            ax.bar(x, vals, color=color_list[:n_all], edgecolor='black', width=0.8)

            # 在第一个柱子的高度画虚线（只跨 models 区域）
            y_ref = vals[0]  # 第一个柱子的高度
            xmin = x[0] - 0.4  # 第一个柱子的左边缘
            xmax = x[-1] + 0.4  # 最后一个柱子的右边缘
            ax.hlines(y_ref, xmin, xmax, colors='k', linestyles='--', linewidth=1.1, zorder=5)

                # 可选：标出 obs/models 分界
            ax.axvline(x[n_obs] - 0.9, color='0.75', linestyle=':', linewidth=1)

            # plt.ylabel('Area percentage (%)')
            plt.xticks([])
            ax.text(0.02, 0.98, dic_variable_name[ii],
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=12, fontfamily='Arial',
                    bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=1.5))
            ax.set_ylim(0, 20)
            plt.grid(axis='y', alpha=0.25)


            plt.show()

            #
            # plt.savefig(result_root + rf'\3mm\FIGURE\Figure5_comparison\barplot\\barplot_{ii}.pdf', dpi=300, bbox_inches='tight')
            # plt.close()

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

class partial_correlation_TRENDY():
    def __init__(self):
        self.map_width = 8.2 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass

        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = rf'D:/Project3/Result/Nov/partial_correlation/Obs/'

        self.fdirX = self.result_root + rf'\input\\X\\'
        self.fdirY = self.result_root + rf'\input\\Y\\'

        pass
    def run(self):
        # self.calculating_colinearity_pixel()
        # self.calc_colinearity_global()
        ##### step3 calculating
        self.xvar_list = [

            'Precip_sum_detrend_CV',
            'CV_daily_rainfall_average']
        self.model_list = ['SNU_LAI', 'GLOBMAP_LAI',
                           'LAI4g','composite_LAI_mean',
                          'composite_LAI_median' ]


        # for model in self.model_list:
        #         self.outdir=self.result_root+rf'\result\\'+model+'\\'
        #
        #         T.mk_dir(self.outdir, force=True)
        #         self.outpartial =  self.outdir + rf'\partial_corr_{model}.npy'
        #         self.outpartial_pvalue =  self.outdir + rf'\partial_pvalue_{model}.npy'
        # #
        #         y_var = f'{model}_detrend_CV.npy'
        #         x_var_list = self.xvar_list + [f'{model}_sensitivity']
        #
        #
        #         df=self.build_df(self.fdirX,self.fdirY,x_var_list,y_var)
        #         #
        #         self.cal_partial_corr(df,x_var_list, )
        # # #         #
        # #         # # # # # # self.check_data()
        #         self.plot_partial_correlation()
        #         self.plot_partial_correlation_p_value()
        # self.statistic_trend_bar()
        self.plot_spatial_map_sig()
        pass


    def calculating_colinearity_pixel(self):

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from tqdm import tqdm

        fdir = r'D:\Project3\Result\Nov\partial_correlation\colinear_test\\'

        # === 1. 读取4个变量 ===
        var_names = [
            'composite_LAI_median_sensitivity',
            'Precip_sum_detrend_CV',
            'CV_daily_rainfall_average',
            'CV_monthly_rainfall_average'
        ]
        dic_all = {}
        for f in os.listdir(fdir):
            for name in var_names:
                if name in f:
                    dic_all[name] = T.load_npy(fdir + f)

        print(f"Loaded {len(dic_all)} variables:", list(dic_all.keys()))

        # === 2. 遍历所有像素 ===
        pix_list = list(dic_all[var_names[0]].keys())
        colinear_dic = {}

        for pix in tqdm(pix_list, desc='Checking colinearity'):
            # --- 检查像素是否在所有字典中 ---
            if not all(pix in dic_all[v] for v in var_names):
                continue

            vals = []
            for v in var_names:
                val=dic_all[v][pix]
                if isinstance(val, dict):
                    if 'intersensitivity_precip_val' in val:
                        arr = np.array(val['intersensitivity_precip_val'], dtype=float)
                    else:
                        continue
                else:
                    arr = np.array(val, dtype=float)
                vals.append(arr)

            vals = np.array(vals)  # shape: (4, T)
            if vals.shape[1] < 10:  # 太短不分析
                continue

            # --- 去nan并转置成 (T, 4) ---
            vals = vals.T
            if np.isnan(vals).any():
                vals = vals[~np.isnan(vals).any(axis=1)]

            if vals.shape[0] < 10:
                continue

            df = pd.DataFrame(vals, columns=var_names)

            # === (A) 计算相关矩阵 ===
            corr = df.corr()
            plt.imshow(corr, interpolation='nearest', cmap='jet',vmin=-1,vmax=1)
            plt.colorbar()
            plt.title(pix)
            plt.show()

            # === (B) 计算VIF ===
            X = df.values
            vif_values = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            vif_dic = dict(zip(var_names, vif_values))

            colinear_dic[pix] = {
                'corr': corr.values,
                'vif': vif_dic
            }

        print(f"Finished {len(colinear_dic)} valid pixels.")

        # === 3. 输出结果 ===
        # outf = r'D:\Project3\Result\Nov\partial_correlation\colinear_test\colinear_summary.npy'
        # T.save_npy(colinear_dic, outf)
        # print("Saved:", outf)

    import numpy as np
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import matplotlib.pyplot as plt

    def calc_colinearity_global(self):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from tqdm import tqdm
        fdir = r'D:\Project3\Result\Nov\partial_correlation\colinear_test\\'

        # === 1. 载入四个变量 ===
        var_names = [
            'composite_LAI_median_sensitivity',
            'Precip_sum_detrend_CV',
            'CV_daily_rainfall_average',
            'CV_monthly_rainfall_average'
        ]
        dic_all = {}
        for f in os.listdir(fdir):
            for name in var_names:
                if name in f:
                    dic_all[name] = T.load_npy(fdir + f)

        print(f"Loaded {len(dic_all)} variables:", list(dic_all.keys()))
        # print("dic_all keys:", list(dic_all.keys()))

        # === 2. 对齐像素 ===
        pix_all = set.intersection(*[set(dic_all[v].keys()) for v in var_names])

        all_data = {v: [] for v in var_names}

        # === 3. 提取所有像素的值并拼接 ===
        for pix in pix_all:
            vals = {}

            for v in var_names:
                val = dic_all[v][pix]
                if isinstance(val, dict):  # composite_LAI_median_sensitivity
                    val = val.get('intersensitivity_precip_val', np.nan)
                arr = np.array(val, dtype=float).ravel()
                if np.all(np.isnan(arr)):
                    continue
                vals[v] = arr

            # 检查长度一致
            if len({len(vals[k]) for k in vals}) != 1:
                continue

            for v in var_names:
                all_data[v].extend(vals[v])

        # === 4. 组合成 DataFrame ===
        df = pd.DataFrame(all_data)
        df = df.dropna()

        print(df.shape)  # (N, 4)

        # === 5. 计算相关矩阵 ===
        corr = df.corr()
        print("\nCorrelation Matrix:")
        print(corr)

        plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.xticks(np.arange(len(var_names)), var_names, rotation=45, ha='right')
        plt.yticks(np.arange(len(var_names)), var_names)
        plt.colorbar(label='Pearson r')
        plt.title('Global correlation matrix')
        plt.tight_layout()
        plt.show()

        # === 6. 计算 VIF ===
        X = df.values
        vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif_result = dict(zip(var_names, vif))
        print("\nVariance Inflation Factors (VIF):")
        for k, v in vif_result.items():
            print(f"{k}: {v:.2f}")

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
            yvals = dic_y[pix]

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
                if 'sensitivity' in xvar:
                    xvals = dic_x[pix].get('intersensitivity_precip_val', np.nan)
                else:
                    xvals = dic_x[pix]
                xvals = np.array(xvals)

                if len(xvals) == 0:
                    x_val_list.append([])
                    continue
                print(len(xvals))

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
        outf_corr=self.outpartial
        outf_pvalue=self.outpartial_pvalue

        partial_correlation_dic= {}
        partial_p_value_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']

            if len(y_vals) == 0:
                continue


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

    def plot_partial_correlation(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)



        f_partial = self.outpartial



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

    def plot_spatial_map_sig(self):

        model_list = self.model_list

        for model in model_list:
            variable_list = self.xvar_list
            fdir = self.result_root + rf'\result\\'+model+'\\'
            print(fdir)
            outdir = self.result_root + rf'\result\\{model}\\sig\\'
            T.mk_dir(outdir, True)
            new_variable_list = variable_list + [f'{model}_sensitivity']

            for variable in new_variable_list:
                f_trend_path = fdir + f'{variable}.tif'
                f_pvalue_path = fdir + f'{variable}_p_value.tif'

                arr_corr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_trend_path)
                arr_pvalue, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_pvalue_path)
                # plt.imshow(arr_corr)
                # plt.colorbar()
                # plt.show()
                arr_corr[arr_corr < -99] = np.nan
                arr_corr[arr_corr > 99] = np.nan
                arr_pvalue[arr_pvalue > 99] = np.nan
                arr_pvalue[arr_pvalue < -99] = np.nan
                arr_corr[arr_pvalue > 0.05] = np.nan

                # plt.imshow(arr_corr)
                # plt.colorbar()
                outf = outdir  + f'{variable}.tif'
                DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_corr, outf)



def main():
    # multiregression_intrasensitivity().run()
    multiregression_intersensitivity().run()
    # multiregression_intersensitivity_TRENDY().run()
    # partial_correlation_obs().run()

    pass

if __name__ == '__main__':
    main()

