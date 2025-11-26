# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg

from upload_version_drrland_greening import trend_analysis

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
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lytools import *
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pprint import pprint
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

class PLOT_dataframe():  ## plot all time series, trends bar figure 1, figure 2 and figure 3
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run (self):
        # self.plot_CV_LAI()
        # self.plot_relative_change_LAI()
        # self.plot_std()
        # self.plot_LAImax_LAImin()
        # self.plot_rainfallmax_min()
        self.statistic_CV_trend_bar()
        # self.statistic_trend_bar()

        # self.plot_CV_trend_among_models2()
        # self.TRENDY_LAImin_LAImax_barplot() ## Figure3


        pass

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def plot_LAImax_LAImin(self):
        df = T.load_df(result_root + rf'\bivariate\Dataframe\\Dataframe.df')
        df = self.df_clean(df)
        print(len(df))

        variable_list = ['composite_LAImax_mean', 'composite_LAImin_mean']
        dic_label = {'composite_LAImax_mean': 'LAImax',
                     'composite_LAImin_mean': 'LAImin'}
        color_dic = {'composite_LAImax_mean': 'purple',
                     'composite_LAImin_mean': 'teal'}

        year_list = range(0, 25)
        result_dic = {}
        std_dic = {}

        # === 计算每个窗口的均值和标准差 ===
        for var in variable_list:
            mean_dic, std_dic_i = {}, {}
            for year in year_list:
                df_i = df[df['window'] == year]
                vals = np.array(df_i[var].tolist(), dtype=float)
                mean_dic[year] = np.nanmean(vals)
                std_dic_i[year] = np.nanstd(vals)
            result_dic[var] = mean_dic
            std_dic[var] = std_dic_i

        df_mean = pd.DataFrame(result_dic)
        df_std = pd.DataFrame(std_dic)

        # === 绘图 ===
        plt.figure(figsize=(self.map_width, self.map_height))

        for var in variable_list:
            color = color_dic[var]

            # 计算线和阴影区
            y = df_mean[var]
            yerr = df_std[var]
            years = list(year_list)

            # 背景阴影 (mean ± std)
            plt.fill_between(years,
                             y - yerr,
                             y + yerr,
                             color=color,
                             alpha=0.1)

            # 主趋势线
            plt.plot(years, y, color=color, linewidth=2.5,
                     label=dic_label[var], marker='o')

            # 拟合趋势线 + 注释
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, y)
            print(var, slope, p_value)
            x_pos = max(years) * 0.85
            y_pos = y.mean()
            # plt.text(x_pos, y_pos + 1.5, f'{dic_label[var]} slope={slope:.3f}', fontsize=10, color=color)
            # plt.text(x_pos, y_pos - 12, f'p={p_value:.3f}', fontsize=10, color=color)

        # === X轴标签（15年滑窗） ===
        window_size = 15
        year_range = range(1982, 2021)
        year_range_str = []
        for year in year_range:
            start_year = year
            end_year = year + window_size - 1
            if end_year > 2021:
                break
            year_range_str.append(f'{start_year}-{end_year}')

        plt.xticks(range(len(year_range_str))[::3], year_range_str[::3], rotation=45, ha='right')
        plt.ylabel('Relative change(%)', fontsize=12)
        plt.grid(alpha=0.4)
        plt.legend(loc='upper left')
        # plt.tight_layout()
        plt.show()

        # out_pdf_fdir = result_root + rf'FIGURE\\Figure2\\'
        # T.mk_dir(out_pdf_fdir)
        # plt.savefig(out_pdf_fdir + 'time_series_LAImin_LAImax_mean.pdf', dpi=300, bbox_inches='tight')
        # plt.close()

        pass

    def plot_rainfallmax_min(self):
        df = T.load_df(result_root + rf'\bivariate\Dataframe\\Dataframe.df')
        df = self.df_clean(df)
        print(len(df))

        variable_list = ['Precip_sum_relative_change_detrend_max', 'Precip_sum_relative_change_detrend_min']
        dic_label = {'Precip_sum_relative_change_detrend_max': 'IAV Precip max',
                     'Precip_sum_relative_change_detrend_min': 'IAV Precip min'}
        color_dic = {'Precip_sum_relative_change_detrend_max': 'purple',
                     'Precip_sum_relative_change_detrend_min': 'teal'}

        year_list = range(0, 25)
        result_dic = {}
        std_dic = {}

        # === 计算每个窗口的均值和标准差 ===
        for var in variable_list:
            mean_dic, std_dic_i = {}, {}
            for year in year_list:
                df_i = df[df['window'] == year]
                vals = np.array(df_i[var].tolist(), dtype=float)
                mean_dic[year] = np.nanmean(vals)
                std_dic_i[year] = np.nanstd(vals)
            result_dic[var] = mean_dic
            std_dic[var] = std_dic_i

        df_mean = pd.DataFrame(result_dic)
        df_std = pd.DataFrame(std_dic)

        # === 绘图 ===
        plt.figure(figsize=(self.map_width, self.map_height))

        for var in variable_list:
            color = color_dic[var]

            # 计算线和阴影区
            y = df_mean[var]
            yerr = df_std[var]
            years = list(year_list)

            # 背景阴影 (mean ± std)
            plt.fill_between(years,
                             y - yerr,
                             y + yerr,
                             color=color,
                             alpha=0.1)

            # 主趋势线
            plt.plot(years, y, color=color, linewidth=2.5,
                     label=dic_label[var], marker='o')

            # 拟合趋势线 + 注释
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, y)
            print(var, slope, p_value)
            x_pos = max(years) * 0.85
            y_pos = y.mean()
            # plt.text(x_pos, y_pos + 10, f'{dic_label[var]} slope={slope:.3f}', fontsize=10, color=color)
            # plt.text(x_pos, y_pos - 10, f'p={p_value:.3f}', fontsize=10, color=color)

        # === X轴标签（15年滑窗） ===
        window_size = 15
        year_range = range(1982, 2021)
        year_range_str = []
        for year in year_range:
            start_year = year
            end_year = year + window_size - 1
            if end_year > 2021:
                break
            year_range_str.append(f'{start_year}-{end_year}')

        plt.xticks(range(len(year_range_str))[::3], year_range_str[::3], rotation=45, ha='right')
        plt.ylabel('Relative change(%)', fontsize=12)
        plt.grid(alpha=0.4)
        plt.legend(loc='upper left')
        # plt.tight_layout()
        # plt.show()

        out_pdf_fdir = result_root + rf'FIGURE\\Figure2\\'
        T.mk_dir(out_pdf_fdir)
        plt.savefig(out_pdf_fdir + 'time_series_precipmax_precipmin.pdf', dpi=300, bbox_inches='tight')
        plt.close()


    def plot_CV_LAI(self):  ##### plot for 4 clusters

        df = T.load_df(
            result_root + rf'\Dataframe\\CVLAI\\CVLAI.df')
        print(len(df))
        df = self.df_clean(df)

        print(len(df))
        T.print_head_n(df)
        # exit()

        # create color list with one green and another 14 are grey



        color_list = ['black','green', 'blue',  'magenta', 'black','purple',  'purple', 'black', 'yellow', 'purple', 'pink', 'grey',
                      'brown', 'lime', 'teal', 'magenta']
        linewidth_list = [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]




        variable_list=['composite_LAI_mean',
                       'LAI4g','SNU_LAI',
            'GLOBMAP_LAI',]
        dic_label={'composite_LAI_mean':'Composite LAI',
                   'LAI4g':'GIMMS4g',

                   'GLOBMAP_LAI':'GLOBMAP',
                   'SNU_LAI':'SNU',}
        year_list=range(0,25)


        result_dic = {}
        CI_dic={}
        std_dic={}

        for var in variable_list:

            result_dic[var] = {}
            CI_dic[var] = {}
            data_dic = {}
            CI_dic_data={}
            std_dic_data={}

            for year in year_list:
                df_i = df[df['window'] == year]

                vals = df_i[f'{var}_detrend_CV'].tolist()
                SEM=stats.sem(vals)
                CI=stats.t.interval(0.95, len(vals)-1, loc=np.nanmean(vals), scale=SEM)
                std=np.nanstd(vals)
                CI_dic_data[year]=CI
                std_dic_data[year]=std

                data_dic[year] = np.nanmean(vals)




            result_dic[var] = data_dic
            CI_dic[var]=CI_dic_data
            std_dic[var]=std_dic_data
        ##dic to df

        df_new = pd.DataFrame(result_dic)

        flag = 0
        plt.figure(figsize=(self.map_width, self.map_height))

        for var in variable_list:
            if var == 'composite_LAI_mean':
                ## plot CI bar
                plt.plot(year_list, df_new[var], label=dic_label[var], linewidth=linewidth_list[flag], color=color_list[flag],
                         )
                ## fill std
                # std = [std_dic[var][y] for y in year_list]
                # plt.fill_between(year_list, df_new[var]-std, df_new[var]+std, alpha=0.3, color=color_list[flag])

                ## fill CI
                # ci_low = [CI_dic[var][y][0] for y in year_list]
                # ci_high = [CI_dic[var][y][1] for y in year_list]
                # plt.fill_between(year_list, ci_low, ci_high, color=color_list[flag], alpha=0.3, label='95% CI')
                slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, df_new[var])

                print(var, f'{slope:.2f}',  f'{p_value:.2f}')


                ## std



            else:
                plt.plot(year_list, df_new[var], label=dic_label[var], linewidth=linewidth_list[flag], color=color_list[flag])

                # std = [std_dic[var][y] for y in year_list]
                # plt.fill_between(year_list, df_new[var] - std, df_new[var] + std, alpha=0.3, color=color_list[flag])
                slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, df_new[var])
                print(var, f'{slope:.2f}',  f'{p_value:.2f}')

            flag = flag + 1
        ## if var == 'composite_LAI_CV': plot CI bar


        window_size = 15

        # set xticks with 1982-1997, 1998-2013,.. 2014-2020
        year_range = range(1982, 2021)
        year_range_str = []
        for year in year_range:

            start_year = year
            end_year = year + window_size - 1
            if end_year > 2021:
                break
            year_range_str.append(f'{start_year}-{end_year}')

        plt.xticks(range(len(year_range_str))[::3], year_range_str[::3], rotation=45, ha='right')
        plt.yticks(np.arange(5, 25, 5))




        plt.ylabel(f'CVLAI (%/yr)')
        plt.grid(True, axis='x')  # 只画竖线（随 x 刻度）



        plt.legend(loc='upper left')

        plt.show()
        # plt.tight_layout()
        out_pdf_fdir = result_root + rf'\FIGURE\Figure1b\\'
        T.mk_dir(out_pdf_fdir, force=True)
        # plt.savefig(out_pdf_fdir + 'time_series_CV_mean.pdf', dpi=300, bbox_inches='tight')
        # plt.close()


        #
        # plt.legend()
        # plt.show()

    def plot_std(self):  ##### plot for 4 clusters

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats

        # === 读数据 ===
        df = T.load_df(result_root + rf'\3mm\product_consistency\dataframe\moving_window.df')
        df = self.df_clean(df)

        # 用实际存在的 year 索引，避免和 x 轴标签长度不一致
        year_list = sorted(df['year'].astype(int).unique())
        n = len(year_list)
        x = np.arange(n)

        # 两条变量：左轴 std，右轴 mean
        std_var = 'composite_LAI_relative_change_detrend_std'
        mean_var = 'composite_LAI_relative_change_detrend_mean'

        dic_label = {
            mean_var: 'Composite LAI mean',
            std_var: 'Composite LAI std',
        }

        # 聚合到每个窗口（year）上的均值
        def agg_mean(var):
            vals = []
            for y in year_list:
                v = df.loc[df['year'] == y, var].to_numpy(dtype=float)
                v = v[~np.isnan(v)]
                vals.append(np.nan if len(v) == 0 else np.nanmean(v))
            return np.array(vals, dtype=float)

        y_std = agg_mean(std_var)
        y_mean = agg_mean(mean_var)

        # === 画图：双轴 ===
        fig, ax1 = plt.subplots(figsize=(self.map_width, self.map_height))
        ax2 = ax1.twinx()

        color_std = 'purple'
        color_mean = 'teal'
        lw_std = lw_mean = 2

        l1, = ax1.plot(x, y_std, label=dic_label[std_var], color=color_std, linewidth=lw_std)
        l2, = ax2.plot(x, y_mean, label=dic_label[mean_var], color=color_mean, linewidth=lw_mean)

        # 线性趋势（对索引做回归）
        s1, _, _, p1, _ = stats.linregress(x, y_std)
        s2, _, _, p2, _ = stats.linregress(x, y_mean)
        print(std_var, f'{s1:.4f}', f'{p1:.4f}')
        print(mean_var, f'{s2:.4f}', f'{p2:.4f}')

        # 可选：在右侧上/下各标注 slope / p（上下对齐，同一 x 位置）
        # x_pos = x.max() * 0.95
        # y1 = np.nanmean(y_std)
        # y2 = np.nanmean(y_mean)
        # ax1.text(x_pos, y1 + 0.05 * np.nanstd(y_std), f'slope={s1:.4f}', color=color_std, ha='center', va='bottom',
        #          fontsize=9)
        # ax1.text(x_pos, y1 - 0.05 * np.nanstd(y_std), f'p={p1:.3f}', color=color_std, ha='center', va='top', fontsize=9)
        # ax2.text(x_pos, y2 + 0.05 * np.nanstd(y_mean), f'slope={s2:.4f}', color=color_mean, ha='center', va='bottom',
        #          fontsize=9)
        # ax2.text(x_pos, y2 - 0.05 * np.nanstd(y_mean), f'p={p2:.3f}', color=color_mean, ha='center', va='top',
        #          fontsize=9)

        # x 轴标签：按移动窗口生成（与长度 n 对齐）
        window_size=15
        year_range = range(1983, 2021)
        year_range_str = []
        for year in year_range:

            start_year = year
            end_year = year + window_size - 1
            if end_year > 2021:
                break
            year_range_str.append(f'{start_year}-{end_year}')

        ax1.set_xticks(range(len(year_range_str))[::4], year_range_str[::4], rotation=45, ha='right')

        # 轴标签与颜色
        ax1.set_ylabel('Composite LAI std (%/year)', color=color_std)
        ax2.set_ylabel('Composite LAI mean (%/year)', color=color_mean)
        ax1.tick_params(axis='y', colors=color_std)
        ax2.tick_params(axis='y', colors=color_mean)
        ax1.spines['left'].set_color(color_std)
        ax2.spines['right'].set_color(color_mean)

        # 网格只用左轴
        ax1.grid(True, which='major', alpha=0.5)

        # 合并图例
        lines = [l1, l2]
        labels = [l.get_label() for l in lines]



        # plt.show()
        # # plt.tight_layout()
        # out_pdf_fdir = result_root + rf'\3mm\product_consistency\pdf\\'
        # plt.savefig(out_pdf_fdir + 'std_mean_time_series.pdf', dpi=300, bbox_inches='tight')
        # plt.close()


        #
        # plt.legend()
        plt.show()

    def plot_relative_change_LAI(self):  ##### plot for 4 clusters

        df = T.load_df(
            result_root + rf'\Dataframe\relative_change\\relative_change.df')
        print(len(df))
        df = self.df_clean(df)

        print(len(df))
        T.print_head_n(df)
        # exit()

        # create color list with one green and another 14 are grey

        color_list = ['black','green', 'blue',  'magenta', 'black','purple',  'purple', 'black', 'yellow', 'purple', 'pink', 'grey',
                      'brown', 'lime', 'teal', 'magenta']
        linewidth_list = [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


        fig = plt.figure()
        i = 1

        # variable_list = ['LAI4g', 'AVHRR_solely_relative_change','GEODES_AVHRR_LAI_relative_change',]
        # variable_list = ['NDVI', 'NDVI4g', 'GIMMS_plus_NDVI', ]
        #'detrended_SNU_LAI_CV','SNU_LAI_predict_detrend_CV','

        variable_list = [
                         'composite_LAI_mean','LAI4g', 'SNU_LAI',
            'GLOBMAP_LAI',
                         ]
        dic_label={'LAI4g':'LAI4g','SNU_LAI':'SNU_LAI',
                   'GLOBMAP_LAI':'GLOBMAP_LAI',
                   'composite_LAI_mean':'Composite LAI'}
        year_list=range(1982,2021)


        result_dic = {}
        for var in variable_list:

            result_dic[var] = {}
            data_dic = {}

            for year in year_list:
                df_i = df[df['year'] == year]

                vals = df_i[f'{var}_relative_change'].tolist()
                data_dic[year] = np.nanmean(vals)
            result_dic[var] = data_dic
        ##dic to df

        df_new = pd.DataFrame(result_dic)
        flag=0
        plt.figure(figsize=(self.map_width, self.map_height))

        for var in variable_list:
            plt.plot(year_list, df_new[var], label=dic_label[var],linewidth=linewidth_list[flag], color=color_list[flag])
            flag=flag+1
            slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, df_new[var])
            print(var, f'{slope:.2f}', f'{p_value:.2f}')
        plt.ylabel('Relative change LAI (%)')

        plt.grid(True, axis='x')   # 只画竖线（随 x 刻度）

        plt.legend()
        plt.show()
        # out_pdf_fdir = result_root + rf'\Figure\\Figure1a\\'
        # plt.savefig(out_pdf_fdir + 'time_series_relative_change_mean.pdf', dpi=300, bbox_inches='tight')
        # plt.close()


    def statistic_CV_trend_bar(self):
        dff=result_root+rf'\Dataframe\Trends_CV\\Trends_CV.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        T.print_head_n(df)
        variable_list=['composite_LAI_mean_detrend_CV',
                       'composite_LAI_median_detrend_CV',
                       'GLOBMAP_LAI_detrend_CV',
                       'LAI4g_detrend_CV',
                       'SNU_LAI_detrend_CV',]



        for variable in variable_list:
            df_var = df[[f'{variable}_trend', f'{variable}_p_value']].dropna()
            total_n = len(df_var)
            if total_n == 0:
                continue

            df_sig = df_var[df_var[f'{variable}_p_value'] < 0.05]
            sig_pos = len(df_sig[df_sig[f'{variable}_trend'] > 0])
            sig_neg = len(df_sig[df_sig[f'{variable}_trend'] < 0])
            pct_sig_pos = sig_pos / total_n * 100
            pct_sig_neg = sig_neg / total_n * 100

            # === Non-significant 部分 ===
            df_nonsig = df_var[df_var[f'{variable}_p_value'] >= 0.05]
            nonsig_pos = len(df_nonsig[df_nonsig[f'{variable}_trend'] > 0])
            nonsig_neg = len(df_nonsig[df_nonsig[f'{variable}_trend'] < 0])


            total_nonsig = len(df_nonsig)
            pct_nonsig_pos = nonsig_pos / total_nonsig * 100 if total_nonsig > 0 else np.nan
            pct_nonsig_neg = nonsig_neg / total_nonsig * 100 if total_nonsig > 0 else np.nan

            # === 结果 ===
            result_dic = {
                'Sig. negative': pct_sig_neg,
                'Non-sig. negative': pct_nonsig_neg,
                'Non-sig. positive': pct_nonsig_pos,
                'Sig. positive': pct_sig_pos,


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
                'lightgrey',

                'lightgrey',
                '#7b3294',
            ]



            width = 0.4
            alpha_list = [1, 0.5, 0.5, 1]
            plt.figure(figsize=(3, 3))

            # 逐个画 bar
            for i, (key, val) in enumerate(result_dic.items()):
                plt.bar(i, val, color=color_list[i], alpha=alpha_list[i], width=width)
                plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
                plt.ylabel('Percentage')
                plt.title(variable)

            plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
            # plt.show()
            plt.savefig(result_root + rf'Figure\Figure1b\{variable}_trend_bar.pdf')
            plt.close()

    def statistic_trend_bar(self):
        dff = result_root + rf'\Dataframe\Trends_CV\\Trends_CV.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        T.print_head_n(df)
        variable_list = ['composite_LAI_mean_relative_change',
                         'composite_LAI_median_relative_change',
                         'GLOBMAP_LAI_relative_change',
                         'LAI4g_relative_change',
                         'SNU_LAI_relative_change']

        for variable in variable_list:
            df_var = df[[f'{variable}_trend', f'{variable}_p_value']].dropna()
            total_n = len(df_var)
            if total_n == 0:
                continue

            df_sig = df_var[df_var[f'{variable}_p_value'] < 0.05]
            sig_pos = len(df_sig[df_sig[f'{variable}_trend'] > 0])
            sig_neg = len(df_sig[df_sig[f'{variable}_trend'] < 0])
            pct_sig_pos = sig_pos / total_n * 100
            pct_sig_neg = sig_neg / total_n * 100

            # === Non-significant 部分 ===
            df_nonsig = df_var[df_var[f'{variable}_p_value'] >= 0.05]
            nonsig_pos = len(df_nonsig[df_nonsig[f'{variable}_trend'] > 0])
            nonsig_neg = len(df_nonsig[df_nonsig[f'{variable}_trend'] < 0])

            total_nonsig = len(df_nonsig)
            pct_nonsig_pos = nonsig_pos / total_nonsig * 100 if total_nonsig > 0 else np.nan
            pct_nonsig_neg = nonsig_neg / total_nonsig * 100 if total_nonsig > 0 else np.nan

            # === 结果 ===
            result_dic = {
                'Sig. negative': pct_sig_neg,
                'Non-sig. negative': pct_nonsig_neg,
                'Non-sig. positive': pct_nonsig_pos,
                'Sig. positive': pct_sig_pos,

            }
            # df_new=pd.DataFrame(result_dic,index=[variable])
            # ## plot
            # df_new=df_new.T
            # df_new=df_new.reset_index()
            # df_new.columns=['Variable','Percentage']
            # df_new.plot.bar(x='Variable',y='Percentage',rot=45,color='green')
            # plt.show()
            color_list = [
                '#844000',
                'lightgray',

                'lightgray',
                '#064c6c',
            ]

            # color_list = [
            #     '#844000',
            #     '#fc9831',
            #     '#fffbd4',
            #     '#86b9d2',
            #     '#064c6c',
            # ]

            width = 0.4
            alpha_list = [1, 0.5, 0.5, 1]
            plt.figure(figsize=(3, 3))

            # 逐个画 bar
            for i, (key, val) in enumerate(result_dic.items()):
                plt.bar(i, val, color=color_list[i], alpha=alpha_list[i], width=width)
                plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
                plt.ylabel('Percentage')
                plt.title(variable)

            plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
            # plt.show()
            plt.savefig(result_root + rf'Figure\Figure1a\{variable}_trend_bar.pdf')
            plt.close()

    pass

    def statistic_trend_CV_bar_1(self):
        fdir = result_root + rf'\Composite_LAI\CV\trend\\'
        variable_list=['composite_LAI_detrend_CV_mean','composite_LAI_detrend_CV_median','GlOBMAP_LAI_detrend_CV','LAI4g_detrend_CV','SNU_LAI_detrend_CV']
        for variable in variable_list:
            f_trend_path=fdir+f'{variable}_trend.tif'
            f_pvalue_path=fdir+f'{variable}_p_value.tif'
            result_dic={}

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
                'silver',

                'silver',
                '#7b3294',
            ]
            width = 0.4
            alpha_list = [1, 0.5, 0.5, 1]
            plt.figure(figsize=(3, 3))

            # 逐个画 bar
            for i, (key, val) in enumerate(result_dic.items()):
                plt.bar(i , val, color=color_list[i], alpha=alpha_list[i], width=width)
                plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
                plt.ylabel('Percentage (%)')
                # plt.title(variable)

            plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
            # plt.tight_layout()
            # plt.show()
        ## save pdf
            plt.savefig(result_root + rf'Figure\Figure1b\{variable}_CV_bar.pdf')




    def plot_CV_trend_among_models2(self):  ##here not calculating mean in program

        color_list = ['black', 'black', 'black', 'black', '#E7483D', '#a1a9d0',
                      '#f0988c', '#b883d3', '#ffff33', '#c4a5de',
                      '#984ea3', 'yellow',
                      '#9e9e9e', '#cfeaf1', '#f6cae5',
                      '#98cccb', '#5867AF', '#e66d50', ]
        ## I want use set 3 color

        mark_size_list = [200] * 1+[50] * 3 +[200] * 1+ [50] * 13


        dff = result_root + rf'\Dataframe\\Trends_CV\\Trends_CV.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        T.print_head_n(df)
        print(df.columns.tolist())
        ## print column names
        # print(df.columns)
        # exit()
        marker_list = ['^', 's', 'P', 'X', 'D'] * 4

        variables_list = ['composite_LAI_mean','LAI4g', 'GLOBMAP_LAI',
                          'SNU_LAI',
                          'TRENDY_ensemble_median',
                          'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',

                          'YIBs_S2_Monthly_lai']
        vals_trend_list = []
        vals_CV_list = []
        err_trend_list = []
        err_CV_list = []



        for variable in variables_list:

            vals_trend = df[f'{variable}_relative_change_trend'].values
            vals_CV = df[f'{variable}_detrend_CV_trend'].values
            vals_trend[vals_trend > 999] = np.nan

            vals_CV[vals_CV > 999] = np.nan
            vals_CV[vals_CV < -999] = np.nan
            vals_trend[vals_trend < -999] = np.nan
            vals_trend[vals_trend > 999] = np.nan

            # vals_CV[vals_CV_p_value > 0.05] = np.nan
            vals_trend = vals_trend[~np.isnan(vals_trend)]
            print(variable,np.nanmean(vals_trend))
            # plt.hist(vals_trend,bins=100,color=color_list[0],alpha=0.5,edgecolor='k')
            # plt.title(variable)
            # plt.show()
            vals_CV = vals_CV[~np.isnan(vals_CV)]
            vals_trend_list.append(np.nanmean(vals_trend))
            vals_CV_list.append(np.nanmean(vals_CV))
        # print(vals_trend_list)

        print(vals_CV_list);exit()



        # exit()

        n = len(variables_list)
        mark_size_list = mark_size_list[:n]
        color_list = color_list[:n]
        marker_list = marker_list[:n]

        # plt.scatter(vals_CV_list,vals_trend_list,marker=marker_list,color=color_list[0],s=100)
        # plt.show()
        ##plot error bar
        # plt.figure(figsize=(self.map_width, self.map_height))

        plt.figure(figsize=(13 * centimeter_factor, 10 * centimeter_factor))

        # self.map_width = 13 * centimeter_factor
        # self.map_height = 8.2 * centimeter_factor

        err_trend_list = np.array(err_trend_list)
        err_CV_list = np.array(err_CV_list)
        for i, (x, y, marker, color, var, mark_size) in enumerate(
                zip(vals_trend_list, vals_CV_list, marker_list, color_list, variables_list, mark_size_list)):
            plt.scatter(y, x, marker=marker, color=color_list[i], label=var, s=mark_size, edgecolors='black', )
            # plt.errorbar(y, x, xerr=err_trend_list[i], yerr=err_CV_list[i], fmt='none', color='grey', capsize=2, capthick=0.3,alpha=1)

            ##markerborderwidth=1

            plt.ylabel('Trends in LAI (%/yr)', fontsize=12)
            plt.xlabel('Trends in CVLAI (%/yr)', fontsize=12)
            plt.ylim(-0.3, .9)
            plt.xlim(-0.2, 0.5)
            plt.xticks(fontsize=12)
            ## xticks gap 0.05
            plt.yticks(np.arange(-0.2, .9, 0.2), fontsize=12)
            plt.yticks(fontsize=12)
            # plt.legend()
        ## save imagine
        plt.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
        plt.axvline(x=0.0, color='k', linestyle='--', linewidth=1)
        # plt.savefig(result_root + rf'\FIGURE\\Figure3\\obs_TRENDY_CV_trends_mean.pdf',  bbox_inches='tight')

        #
        plt.show()




        pass

    def TRENDY_LAImin_LAImax_barplot(self):
        dff=result_root+rf'\bivariate\Dataframe\\Dataframe.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        for column in df.columns:
            print(column)
        # exit()

        variables_list = ['composite_LAI_mean', 'LAI4g',   'GLOBMAP','SNU_LAI',
                           'TRENDY_ensemble_median',
                          'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',

                          'YIBs_S2_Monthly_lai']
        values_max_list=[]
        values_min_list=[]

        for variable in variables_list:
            if variable in ['composite_LAI_mean','LAI4g',  'GLOBMAP','SNU_LAI','TRENDY_ensemble_median']:

                values_min = df[f'{variable}_min_trend'].values
                values_max = df[f'{variable}_max_trend'].values
            else:
                values_min=df[f'{variable}_relative_change_detrend_min_trend'].values
                values_max=df[f'{variable}_relative_change_detrend_max_trend'].values
            values_max_list.append(values_max)
            values_min_list.append(values_min)
        values_min_list=np.array(values_min_list)
        values_max_list=np.array(values_max_list)
        values_max_list_mean=np.nanmean(values_max_list,axis=1)
        values_min_list_mean=np.nanmean(values_min_list,axis=1)
        ## add legend

        fig, ax = plt.subplots(figsize=(self.map_width, self.map_height))
        dic_label_name = {'composite_LAI_mean': 'Composite',
                          'LAI4g':'LAI4g',

                          'GLOBMAP': 'GLOBMAP',
                          'SNU_LAI': 'SNU',

                          'TRENDY_ensemble_median': 'TRENDY ensemble',
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
                          'LAI4g': 'GIMMS4g',

                          }


        plt.bar(variables_list,values_max_list_mean,color='#96cccb',width=0.7,edgecolor='black',label='Trend in LAImax',)
        plt.bar(variables_list,values_min_list_mean,color='#f6cae5',width=0.7,edgecolor='black',label='Trend in LAImin',)
        plt.legend()

        # plt.xticks(np.arange(len(variables_list)),variables_list,rotation=45)

        ## add y=0
        plt.hlines(0, -0.5, len(variables_list) - 0.5, colors='black', linestyles='dashed')
        plt.ylabel('(%/yr)')
        plt.axhline(y=0, color='grey', linestyle='-')
        labels = [dic_label_name.get(v, v) for v in variables_list]
        ax.set_xticks(range(len(variables_list)))
        # ax.set_xticklabels(labels, rotation=90, fontsize=10, font='Arial')
        # plt.tight_layout()
        plt.show()
        print(values_max_list_mean)
        print(values_min_list_mean)
        print(variables_list)
        outdir=result_root+rf'\FIGURE\\Figure3\\'
        T.mk_dir(outdir,force=True)
        outf=outdir+rf'barplot_mean.pdf'
        # plt.savefig(outf,dpi=300,bbox_inches='tight')
        # plt.savefig(outf, dpi=300, )


class greening_analysis(): ##not used

    def __init__(self):
        self.map_width = 15.3 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor

        pass
    def run(self):


        self.greening_products_basemap_whole_time()
        # self.greening_products_basemap_whole_time_CV()
        # self.greening_products_basemap_whole_time_consistency()
        # self.greening_products_basemap_two_time_periods()

        # self.plot_spatial_histgram_period()
        pass
    def greening_products_basemap_whole_time(self):
        ## three products 3 time periods comparison
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np

        vmin_vmax = {
            'NDVI4g': (-0.1, 0.1),

            # 'GOSIF': (-1, 1)
            'GIMMS_plus_NDVI': (-0.1, 0.1),
            'NDVI': (-0.5, 0.5),
            'Landsat': (-0.3, 0.3),
        }

        # Create synthetic data (replace with actual data as needed)
        fdir=result_root+rf'\3mm\relative_change_growing_season\TRENDY\\trend_analysis\\'
        temp_root = result_root + rf'\3mm\relative_change_growing_season\TRENDY\\trend_analysis\\temp_root\\'
        T.mk_dir(temp_root)
        # T.open_path_and_file(fdir);exit()

        products = [ 'NDVI4g', 'GIMMS_plus_NDVI','NDVI',]


        # fig, axes = plt.subplots(1, 3, figsize=(self.map_width*2, self.map_height))

        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.95, bottom=0.08, left=0.05, right=0.95)

        # Loop through products and periods to create subplots

        for i, product in enumerate(products):  # Use 'i' to index the subplot
            ax = axes[i]

            f_trend = fdir + rf'\{product}_trend.tif'

            f_p_value = fdir + rf'\{product}_p_value.tif'

            array_trend, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_trend)
            array_p_value, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_p_value)
            array_trend[array_trend<-999]=np.nan
            array_p_value[array_p_value<-999]=np.nan


            # m = Basemap(projection='cyl', resolution='l', ax=ax)
            # ##不显示经纬度
            # m.drawcoastlines()
            # m.drawcountries()
            # m.drawparallels(np.arange(-90., 91., 30.), labels=[0, 0, 0, 0])  # 不显示纬度标签
            # m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 0])  # 不显示经度标签

            # Plot the data using pcolormesh
            arr_trend = Tools().mask_999999_arr(array_trend, warning=False)

            # arr_trend = array_trend[:60]

            lon_list = np.arange(originX, originX + pixelWidth * arr_trend.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr_trend.shape[0], pixelHeight)
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            ## create the basemap instance

            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180,
                        resolution='l', ax=ax)

            m.drawcoastlines(linewidth=0.2,color='k',zorder=11)
            # Plot the data using pcolormesh
            vmin, vmax = vmin_vmax[product]
            color_list = [
                '#844000',
                '#fc9831',
                '#fffbd4',
                '#86b9d2',
                '#064c6c',
            ]

            # m, ret = Plot_Robinson().plot_Robinson(f_trend, ax, cmap='RdBu', vmin=vmin, vmax=vmax, is_plot_colorbar=False)
            #

            cmap='RdBu'

            ret = m.pcolormesh(lon_list, lat_list, array_trend, cmap=cmap, vmin=vmin, vmax=vmax)
            m = self.plot_sig_scatter(
                m, f_p_value, temp_root, sig_level=0.05, ax=None, linewidths=0.2,
                s=3,
                c='k', marker='x',
                zorder=100, res=4
            )
            # plt.show()
            # Add p_value markers
            # significance_threshold = 0.05  # Define significance threshold
            # significant = array_p_value < significance_threshold  # Mask for significant values
            # lon_significant = lon_list[significant]
            # lat_significant = lat_list[significant]
            #
            # # Scatter plot for significant p-values
            # m.scatter(lon_significant, lat_significant, s=0.005, c='k', marker='.', alpha=0.8, zorder=2)

            # Add a colorbar for this subplot
            ## colorbar ticks .2f



            cbar = fig.colorbar(ret, ax=ax, orientation='vertical', fraction=0.06, pad=0.05,
                                location='left')
            # cbar.set_label(f'{product} Trend Value',y=1)
            cbar.set_label(f'{product} trend (%)',labelpad=-68)

            tick_labels = np.round(np.linspace(vmin, vmax, 5), 2)
            cbar.set_ticks(tick_labels)
            cbar.set_ticklabels([f"{tick:.2f}" for tick in tick_labels])
            # ax.set_title(product)
            # plt.show()
            outf = result_root + rf'\3mm\relative_change_growing_season\TRENDY\\trend_analysis\\greening_products_basemap_{product}.png'
            plt.savefig(outf, dpi=600)
            plt.close()
            # T.open_path_and_file(outf)

    def greening_products_basemap_whole_time_CV(self):
        ## three products 3 time periods comparison
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np

        vmin_vmax = {
            'detrended_NDVI4g_CV': (-0.5, 0.5),

            # 'GOSIF': (-1, 1)
            'detrended_GIMMS_plus_NDVI_CV': (-0.5, 0.5),
            'NDVI_detrend_CV': (-0.5, 0.5),
            'detrended_Landsat_CV_2020': (-0.5, 0.5),
        }

        # Create synthetic data (replace with actual data as needed)
        fdir=result_root+rf'3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\'
        temp_root = result_root + rf'3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\temp_root\\'
        T.mk_dir(temp_root)

        products = [ 'detrended_NDVI4g_CV', 'detrended_GIMMS_plus_NDVI_CV','NDVI_detrend_CV','detrended_Landsat_CV_2020']



        # fig, axes = plt.subplots(1, 3, figsize=(self.map_width*2, self.map_height))

        fig, axes = plt.subplots(4, 1, figsize=(12, 14))
        # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.95, bottom=0.08, left=0.05, right=0.95)

        # Loop through products and periods to create subplots

        for i, product in enumerate(products):  # Use 'i' to index the subplot
            ax = axes[i]

            f_trend = fdir + rf'\{product}_trend.tif'

            f_p_value = fdir + rf'\{product}_p_value.tif'

            array_trend, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_trend)
            array_p_value, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_p_value)
            array_trend[array_trend<-999]=np.nan
            array_p_value[array_p_value<-999]=np.nan


            # m = Basemap(projection='cyl', resolution='l', ax=ax)
            # ##不显示经纬度
            # m.drawcoastlines()
            # m.drawcountries()
            # m.drawparallels(np.arange(-90., 91., 30.), labels=[0, 0, 0, 0])  # 不显示纬度标签
            # m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 0])  # 不显示经度标签

            # Plot the data using pcolormesh
            arr_trend = Tools().mask_999999_arr(array_trend, warning=False)

            # arr_trend = array_trend[:60]

            lon_list = np.arange(originX, originX + pixelWidth * arr_trend.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr_trend.shape[0], pixelHeight)
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            ## create the basemap instance

            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180,
                        resolution='l', ax=ax)

            m.drawcoastlines(linewidth=0.2,color='k',zorder=11)
            # Plot the data using pcolormesh
            vmin, vmax = vmin_vmax[product]
            color_list = [
                '#844000',
                '#fc9831',
                '#fffbd4',
                '#86b9d2',
                '#064c6c',
            ]

            # m, ret = Plot_Robinson().plot_Robinson(f_trend, ax, cmap='RdBu', vmin=vmin, vmax=vmax, is_plot_colorbar=False)
            #

            cmap='RdBu'

            ret = m.pcolormesh(lon_list, lat_list, array_trend, cmap=cmap, vmin=vmin, vmax=vmax)
            m = self.plot_sig_scatter(
                m, f_p_value, temp_root, sig_level=0.05, ax=None, linewidths=0.2,
                s=3,
                c='k', marker='x',
                zorder=100, res=4
            )
            # plt.show()
            # Add p_value markers
            # significance_threshold = 0.05  # Define significance threshold
            # significant = array_p_value < significance_threshold  # Mask for significant values
            # lon_significant = lon_list[significant]
            # lat_significant = lat_list[significant]
            #
            # # Scatter plot for significant p-values
            # m.scatter(lon_significant, lat_significant, s=0.005, c='k', marker='.', alpha=0.8, zorder=2)

            # Add a colorbar for this subplot
            cbar = fig.colorbar(ret, ax=ax, orientation='vertical', fraction=0.06, pad=0.05,
                                location='left')
            # cbar.set_label(f'{product} Trend Value',y=1)
            cbar.set_label(f'{product} trend (%)',labelpad=-58)
            # ax.set_title(product)
            # plt.show()
        outf = result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\greening_products_basemap_whole_time_CV.png'
        plt.savefig(outf, dpi=600)
        plt.close()

    def greening_products_basemap_whole_time_consistency(self):
        ## three products 3 time periods comparison
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np


        # Create synthetic data (replace with actual data as needed)
        fdir=result_root+rf'\3mm\bivariate_analysis\CV_products_comparison_bivariate\\'
        temp_root = result_root + rf'3mm\bivariate_analysis\CV_products_comparison_bivariate\\temp_root\\'

        T.mk_dir(temp_root)

        products = [ 'LAI4g_NDVI4g','LAI4g_GIMMS_NDVI', 'LAI4g_NDVI',]

        # fig, axes = plt.subplots(1, 3, figsize=(self.map_width*2, self.map_height))

        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.95, bottom=0.08, left=0.05, right=0.95)

        # Loop through products and periods to create subplots

        for i, product in enumerate(products):  # Use 'i' to index the subplot
            ax = axes[i]

            f_trend = fdir + rf'\{product}.tif'


            array_trend, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_trend)

            array_trend[array_trend<-999]=np.nan



            # m = Basemap(projection='cyl', resolution='l', ax=ax)
            # ##不显示经纬度
            # m.drawcoastlines()
            # m.drawcountries()
            # m.drawparallels(np.arange(-90., 91., 30.), labels=[0, 0, 0, 0])  # 不显示纬度标签
            # m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 0])  # 不显示经度标签

            # Plot the data using pcolormesh
            arr_trend = Tools().mask_999999_arr(array_trend, warning=False)

            # arr_trend = array_trend[:60]

            lon_list = np.arange(originX, originX + pixelWidth * arr_trend.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr_trend.shape[0], pixelHeight)
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            ## create the basemap instance

            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180,
                        resolution='l', ax=ax)

            m.drawcoastlines(linewidth=0.2,color='k',zorder=11)
            # Plot the data using pcolormesh

            color_list = [
                 '#a86ee1',
                '#d6cb38',
                '#4defef',
                '#de6e13',
            ]

            my_cmap2 = T.cmap_blend(color_list, n_colors=5)

            ret = m.pcolormesh(lon_list, lat_list, array_trend, cmap=my_cmap2,)
            # cbar = fig.colorbar(ret, ax=ax, orientation='vertical', fraction=0.06, pad=0.05,
            #                     location='left')
            # # cbar.set_label(f'{product} Trend Value',y=1)
            # cbar.set_label(f'{product}', labelpad=-58)


        outf = result_root + rf'3mm\bivariate_analysis\CV_products_comparison_bivariate\whole_time_consistency.png'
        plt.savefig(outf, dpi=600)
        plt.close()





    def plot_sig_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5,
                                               s=20,
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
        # fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        # Plot().Robinson_reproj(fpath_resample, fpath_resample, res=res * 100000)

        # arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)
        arr = Tools().mask_999999_arr(arr, warning=False)
        #
        os.remove(fpath_clip)
        # os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)
        # exit()

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
        lon_list = np.array(lon_list) + res/2
        lat_list = np.array(lat_list) - res/2
        # lon_list = lon_list - originX
        # lat_list = lat_list + originY
        # lon_list = lon_list + pixelWidth / 2
        # lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        # print(lat_list);exit()
        m.scatter(lon_list, lat_list, latlon=True, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)
        # ax.scatter(lon_list,lat_list)
        # plt.show()

        return m


    def greening_products_basemap_two_time_periods(self):
        ## three products 3 time periods comparison
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np

        # Create synthetic data (replace with actual data as needed)
        fdir=result_root+rf'\3mm\relative_change_growing_season\trend_analysis\\'

        products = ['LAI4g', 'NDVI', 'NIRv']
        period_list = ['1983_2001', '2002_2020' ,'all']

        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.92, bottom=0.08, left=0.05, right=0.95)

        # Loop through products and periods to create subplots

        for i, period in enumerate(period_list):
            for product in products:
                ax=axes[i][products.index(product)]
                if period == 'all':
                    f_trend = fdir + rf'\{product}_trend.tif'
                    f_p_value = fdir + rf'\{product}_p_value.tif'
                else:
                    f_trend=fdir+rf'\{product}_{period}_trend.tif'
                    f_p_value=fdir+rf'\{product}_{period}_p_value.tif'
                array_trend, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_trend)
                array_p_value, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_p_value)
                array_trend[array_trend<-999]=np.nan




                m = Basemap(projection='cyl', resolution='l', ax=ax)
                ##不显示经纬度
                m.drawcoastlines()
                # m.drawcountries()
                # m.drawparallels(np.arange(-90., 91., 30.), labels=[0, 0, 0, 0])  # 不显示纬度标签
                # m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 0])  # 不显示经度标签

                # Plot the data using pcolormesh
                arr_trend = Tools().mask_999999_arr(array_trend, warning=False)

                # arr_trend = array_trend[:60]

                lon_list = np.arange(originX, originX + pixelWidth * arr_trend.shape[1], pixelWidth)
                lat_list = np.arange(originY, originY + pixelHeight * arr_trend.shape[0], pixelHeight)
                lon_list, lat_list = np.meshgrid(lon_list, lat_list)
                ## create the basemap instance

                m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180,
                            resolution='i', ax=ax)
                m.drawcoastlines(linewidth=0.1,color='grey',zorder=0)
                # m.drawcountries()
                # Plot the data using pcolormesh
                ret = m.pcolormesh(lon_list, lat_list, array_trend, cmap='RdBu', vmin=-0.3, vmax=0.3)
                # plt.show()

                # Set title
                # ax.set_title(f'{product} - {period}')
        cbar = fig.colorbar(ret, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08)
        cbar.set_label('Trend Value')

        plt.show()


    def plot_spatial_histgram_period(self):
        from scipy.stats import gaussian_kde
        dff=result_root+rf'3mm\Dataframe\Trend\\Trend.df'
        df=T.load_df(dff)
        df = self.df_clean(df)
        period_list=['1983_2001','2002_2020']
        # period_list = ['1983_2020', ]
        print(df.columns)
        ##plt histogram of LAI
        df=df[df['LAI4g_1983_2020_trend']<30]
        df=df[df['LAI4g_1983_2020_trend']>-30]
        product_list=['LAI4g','NDVI','NIRv']
        result_dict={}

        for period in period_list:
            product_dict={}

            for product in product_list:
                value=df[f'{product}_{period}_trend'].values
                value=np.array(value)
                value[value<-99]=np.nan
                value[value > 99] = np.nan
                value = value[~np.isnan(value)]

                product_dict[product]=value
            result_dict[period]=product_dict

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, product in zip(axes, product_list):
            color_list = [  'red', 'green',]
            flag = 0
            for period in period_list:

                value = result_dict[period][product]
                value = np.array(value)
                value = value[~np.isnan(value)]
                sns.kdeplot(data=value, shade=True, color=color_list[flag], label=period,ax=ax,legend=True)
                ax.set_xlim(-2,2)
                ax.set_ylabel('')
                ax.set_yticks([])
                ax.set_title(product)
                ##add line for zero
                ax.axvline(x=0, color='grey', linestyle='--')
                flag+=1


        plt.legend()


        plt.show()  #

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df


class climate_variables:
    def __init__(self):
        pass
    def run(self):
        # self.trend_analysis_simply_linear()
        # self.plot_climate_variable_trends()
        self.average_analysis()

    def trend_analysis_simply_linear(self):
        from scipy.stats import theilslopes
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\\'
        outdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\ecosystem_year\relative_trend\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'detrend' in f:
                continue



            if not f.endswith('.npy'):
                continue

            outf=outdir+f.split('.')[0]
            if os.path.isfile(outf+'_trend.tif'):
                continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r,c=pix
                if r<60:
                    continue
                landcover_value=crop_mask[pix]
                if landcover_value==16 or landcover_value==17 or landcover_value==18:
                    continue
                if dic_modis_mask[pix]==12:
                    continue

                time_series=dic[pix]
                # print(time_series)


                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                if len(set(time_series))==1:
                    continue
                # print(time_series)


                if np.nanstd(time_series) == 0:
                    continue
                average_value=np.nanmean(time_series)
                try:

                        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                        slope,b,r,p_value=T.nan_line_fit(np.arange(len(time_series)), time_series)
                        trend_dic[pix] = slope/average_value*100
                        p_value_dic[pix] = p_value


                except:
                    continue



            arr_trend = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(trend_dic)
            arr_trend_dryland = arr_trend * array_mask

            p_value_arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(p_value_dic)
            p_value_arr_dryland = p_value_arr * array_mask


            # plt.imshow(arr_trend_dryland, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()
            # exit()

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_trend_dryland, outf + '_trend.tif')
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(p_value_arr_dryland, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend_dryland)
            np.save(outf + '_p_value', p_value_arr_dryland)

    def average_analysis(self):
        from scipy.stats import theilslopes
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\\'
        outdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\ecosystem_year\average\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if  'detrend' in f:
                continue

            if not f.endswith('.npy'):
                continue

            outf=outdir+f.split('.')[0]
            if os.path.isfile(outf+'_trend.tif'):
                continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r,c=pix
                if r<60:
                    continue
                landcover_value=crop_mask[pix]
                if landcover_value==16 or landcover_value==17 or landcover_value==18:
                    continue
                if dic_modis_mask[pix]==12:
                    continue

                time_series=dic[pix]['ecosystem_year']
                # print(time_series)


                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                if len(set(time_series))==1:
                    continue
                # print(time_series)


                if np.nanstd(time_series) == 0:
                    continue
                average_value=np.nanmean(time_series)
                try:

                        trend_dic[pix] = np.nanmean(time_series)

                except:
                    continue

            arr_trend = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(trend_dic)
            arr_trend_dryland = arr_trend * array_mask

            # plt.imshow(arr_trend_dryland, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()
            # exit()

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_trend_dryland, outf + '_trend.tif')

            np.save(outf + '_trend', arr_trend_dryland)


    def plot_climate_variable_trends(self):
        ## three products 3 time periods comparison
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np

        vmin_vmax = {
            'NDVI4g': (-0.1, 0.1),

            # 'GOSIF': (-1, 1)
            'rainfall_seasonality_all_year': (-0.2, 0.2),
            'rainfall_intensity': (-0.2, 0.2),
            'heat_event_frenquency': (-1, 1),
            'rainfall_frenquency': (-0.5, 0.5),
            'detrended_sum_rainfall_CV': (-1, 1),

        }

        # Create synthetic data (replace with actual data as needed)
        fdir=result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\relative_trend\\'
        temp_root = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\relative_trend\\temp_root\\'
        T.mk_dir(temp_root)
        # T.open_path_and_file(fdir);exit()

        products = [ 'rainfall_intensity', 'rainfall_frenquency','detrended_sum_rainfall_CV',]
        products = ['heat_event_frenquency', 'rainfall_seasonality_all_year', ]

        # fig, axes = plt.subplots(1, 3, figsize=(self.map_width*2, self.map_height))

        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.95, bottom=0.08, left=0.05, right=0.95)

        # Loop through products and periods to create subplots

        for i, product in enumerate(products):  # Use 'i' to index the subplot
            # ax = axes[i // 2, i % 2]
            ax=axes[i]


            f_trend = fdir + rf'\{product}_trend.tif'

            f_p_value = fdir + rf'\{product}_p_value.tif'

            array_trend, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_trend)
            array_p_value, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_p_value)
            array_trend[array_trend<-999]=np.nan
            array_p_value[array_p_value<-999]=np.nan


            # m = Basemap(projection='cyl', resolution='l', ax=ax)
            # ##不显示经纬度
            # m.drawcoastlines()
            # m.drawcountries()
            # m.drawparallels(np.arange(-90., 91., 30.), labels=[0, 0, 0, 0])  # 不显示纬度标签
            # m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 0])  # 不显示经度标签

            # Plot the data using pcolormesh
            arr_trend = Tools().mask_999999_arr(array_trend, warning=False)



            # arr_trend = array_trend[:60]

            lon_list = np.arange(originX, originX + pixelWidth * arr_trend.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr_trend.shape[0], pixelHeight)
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            ## create the basemap instance

            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180,
                        resolution='l', ax=ax)

            # m.drawcoastlines(linewidth=0.2,color='k',zorder=11)
            # Plot the data using pcolormesh
            vmin, vmax = vmin_vmax[product]


            cmap='RdBu'

            ret = m.pcolormesh(lon_list, lat_list, array_trend, cmap=cmap, vmin=vmin, vmax=vmax)
            m = self.plot_sig_scatter(
                m, f_p_value, temp_root, sig_level=0.05, ax=None, linewidths=0.2,
                s=3,
                c='k', marker='x',
                zorder=100, res=4
            )
            # plt.show()
            # Add p_value markers
            # significance_threshold = 0.05  # Define significance threshold
            # significant = array_p_value < significance_threshold  # Mask for significant values
            # lon_significant = lon_list[significant]
            # lat_significant = lat_list[significant]
            #
            # # Scatter plot for significant p-values
            # m.scatter(lon_significant, lat_significant, s=0.005, c='k', marker='.', alpha=0.8, zorder=2)

            # Add a colorbar for this subplot
            ## colorbar ticks .2f



            cbar = fig.colorbar(ret, ax=ax, orientation='vertical', fraction=0.06, pad=0.05,
                                location='left')
            # cbar.set_label(f'{product} Trend Value',y=1)
            cbar.set_label(f'{product} trend (%/yr)',labelpad=-68)

            tick_labels = np.round(np.linspace(vmin, vmax, 5), 2)
            cbar.set_ticks(tick_labels)
            cbar.set_ticklabels([f"{tick:.2f}" for tick in tick_labels])
            # ax.set_title(product)
            # plt.show()
        outf = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\\climate_variable_trends.png'
        plt.savefig(outf, dpi=600)
        plt.close()
        T.open_path_and_file(outf)

    def plot_sig_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5,
                                               s=20,
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
        # fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        # Plot().Robinson_reproj(fpath_resample, fpath_resample, res=res * 100000)

        # arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)
        arr = Tools().mask_999999_arr(arr, warning=False)
        #
        os.remove(fpath_clip)
        # os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)
        # exit()

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
        lon_list = np.array(lon_list) + res/2
        lat_list = np.array(lat_list) - res/2
        # lon_list = lon_list - originX
        # lat_list = lat_list + originY
        # lon_list = lon_list + pixelWidth / 2
        # lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        # print(lat_list);exit()
        m.scatter(lon_list, lat_list, latlon=True, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)
        # ax.scatter(lon_list,lat_list)
        # plt.show()

        return m

class Plot_Robinson_remote_sensing:
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

        # Blue represents high values, and red represents low values.
        plt.figure(figsize=(self.map_width, self.map_height))
        m = Basemap(projection='robin', lon_0=0, lat_0=90., resolution='c')

        # m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        # m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        # meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        # meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        # for obj in meridict:
        #     line = meridict[obj][0][0]
        # coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        # polys = m.fillcontinents(color='#FFFFFF', lake_color='#EFEFEF', zorder=90)
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
        self.Robinson_reproj(fpath_resample, fpath_resample_ortho, res=res * 10000)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)
        # lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
        # lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        # arr_reproj, originX_reproj, originY_reproj, pixelWidth_reproj, pixelHeight_reproj = ToRaster().raster2array(fpath_resample_ortho)
        # lon_list_reproj = np.arange(originX_reproj, originX_reproj + pixelWidth_reproj * arr_reproj.shape[1], pixelWidth_reproj)
        # lat_list_reproj = np.arange(originY_reproj, originY_reproj + pixelHeight_reproj * arr_reproj.shape[0], pixelHeight_reproj)
        # arr = m.transform_scalar(arr, lon_list, lat_list[::-1], len(lon_list_reproj), len(lat_list_reproj))
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr[arr > sig_level] = np.nan
        # plt.figure()
        # plt.imshow(arr,interpolation='nearest',cmap='jet')
        # plt.show()
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        #
        os.remove(fpath_clip)
        os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)

        spatial_dict = D_resample.spatial_arr_to_dic(arr)
        lon_lat_pix_dict = D_resample.spatial_tif_to_lon_lat_dic(temp_root)
        # keys = spatial_dict.keys()

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
        # lon_list = lon_list - originX
        # lat_list = lat_list + originY
        lon_list = lon_list + pixelWidth / 2
        lat_list = lat_list + pixelHeight / 2
        # print(lon_list)
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        m.scatter(lon_list, lat_list, latlon=True, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
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
        ### CV list

        # color_list = [
        #     '#008837',
        #     '#a6dba0',
        #     '#f7f7f7',
        #     '#c2a5cf',
        #     '#7b3294',
        # ]
        # std_list=[ '#e66101',
        #            '#fdb863',
        #            '#f7f7f7',
        #            '#b2abd2',
        #            '#5e3c99',
        #
        # ]
        # Blue represents high values, and red represents low values.
        if ax == None:
            # plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list)
        elif type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
        lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        # print(np.shape(arr))
        # plt.imshow(arr)
        # plt.show()
        if not is_reproj:
            arr_reproj, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            lon_list_reproj = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list_reproj = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        else:
            fpath_robinson = self.Robinson_reproj(fpath, fpath + '_robinson-reproj.tif', res=res)
            arr_reproj, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_robinson)
            lon_list_reproj = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list_reproj = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
            # print(originX, originY, pixelWidth, pixelHeight)
            arr_reproj[arr_reproj<-9999] = np.nan
            # plt.imshow(arr_reproj,interpolation='nearest')
            # plt.show()
            os.remove(fpath_robinson)
            # print(fpath_robinson)
            # exit()
        # originY1 = copy.copy(originY)
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr_m = ma.masked_where(np.isnan(arr), arr)
        # originX = 0
        # originY = originY * 2
        # originY = 0

        # lon_list, lat_list = np.meshgrid(lon_list, lat_list)
        # print(lon_list.shape)
        # plt.imshow(arr_m)
        # plt.show()
        # exit()
        m = Basemap(projection='robin', lon_0=0, lat_0=90., ax=ax, resolution='c')
        # print(lon_list)
        # print(lat_list)
        # m = Basemap(projection='robin', lon_0=0,ax=ax, resolution='c')
        arr_m = m.transform_scalar(arr_m,lon_list,lat_list[::-1],len(lon_list_reproj)*1,len(lat_list_reproj)*1,order=0)
        # m.transform_vector()
        # plt.imshow(arr_m,interpolation='nearest')
        # plt.show()

        # ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax, )
        ret = m.imshow(arr_m[::-1], cmap=cmap, zorder=99, vmin=vmin, vmax=vmax,interpolation='nearest')
        # m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        # m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        # plt.show()
        # meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        # meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        # for obj in meridict:
        #     line = meridict[obj][0][0]
        # coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        # polys = m.fillcontinents(color='whitesmoke', lake_color='#EFEFEF', zorder=90)
        # plt.show()
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


class Plot_Robinson_TRENDY:
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
        self.Robinson_reproj(fpath_resample, fpath_resample_ortho, res=res * 10000)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)
        # lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
        # lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        # arr_reproj, originX_reproj, originY_reproj, pixelWidth_reproj, pixelHeight_reproj = ToRaster().raster2array(fpath_resample_ortho)
        # lon_list_reproj = np.arange(originX_reproj, originX_reproj + pixelWidth_reproj * arr_reproj.shape[1], pixelWidth_reproj)
        # lat_list_reproj = np.arange(originY_reproj, originY_reproj + pixelHeight_reproj * arr_reproj.shape[0], pixelHeight_reproj)
        # arr = m.transform_scalar(arr, lon_list, lat_list[::-1], len(lon_list_reproj), len(lat_list_reproj))
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr[arr > sig_level] = np.nan
        # plt.figure()
        # plt.imshow(arr,interpolation='nearest',cmap='jet')
        # plt.show()
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        #
        os.remove(fpath_clip)
        os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)

        spatial_dict = D_resample.spatial_arr_to_dic(arr)
        lon_lat_pix_dict = D_resample.spatial_tif_to_lon_lat_dic(temp_root)
        # keys = spatial_dict.keys()

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
        # lon_list = lon_list - originX
        # lat_list = lat_list + originY
        lon_list = lon_list + pixelWidth / 2
        lat_list = lat_list + pixelHeight / 2
        # print(lon_list)
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        m.scatter(lon_list, lat_list, latlon=True, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
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
        # color_list = [
        #     '#844000',
        #     '#fc9831',
        #     '#fffbd4',
        #     '#86b9d2',
        #     '#064c6c',
        # ]
        ### CV list

        color_list = [
            '#008837',
            '#a6dba0',
            '#f7f7f7',
            '#c2a5cf',
            '#7b3294',
        ]
        # std_list=[ '#e66101',
        #            '#fdb863',
        #            '#f7f7f7',
        #            '#b2abd2',
        #            '#5e3c99',
        #
        # ]
        # Blue represents high values, and red represents low values.
        if ax == None:
            # plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list)
        elif type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
        lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        # print(np.shape(arr))
        # plt.imshow(arr)
        # plt.show()
        if not is_reproj:
            arr_reproj, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            lon_list_reproj = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list_reproj = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        else:
            fpath_robinson = self.Robinson_reproj(fpath, fpath + '_robinson-reproj.tif', res=res)
            arr_reproj, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_robinson)
            lon_list_reproj = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list_reproj = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
            # print(originX, originY, pixelWidth, pixelHeight)
            arr_reproj[arr_reproj<-9999] = np.nan
            # plt.imshow(arr_reproj,interpolation='nearest')
            # plt.show()
            os.remove(fpath_robinson)
            # print(fpath_robinson)
            # exit()
        # originY1 = copy.copy(originY)
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr_m = ma.masked_where(np.isnan(arr), arr)
        # originX = 0
        # originY = originY * 2
        # originY = 0

        # lon_list, lat_list = np.meshgrid(lon_list, lat_list)
        # print(lon_list.shape)
        # plt.imshow(arr_m)
        # plt.show()
        # exit()
        m = Basemap(projection='robin', lon_0=0, lat_0=90., ax=ax, resolution='c')
        # print(lon_list)
        # print(lat_list)
        # m = Basemap(projection='robin', lon_0=0,ax=ax, resolution='c')
        arr_m = m.transform_scalar(arr_m,lon_list,lat_list[::-1],len(lon_list_reproj)*1,len(lat_list_reproj)*1,order=0)
        # m.transform_vector()
        # plt.imshow(arr_m,interpolation='nearest')
        # plt.show()

        # ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax, )
        ret = m.imshow(arr_m[::-1], cmap=cmap, zorder=99, vmin=vmin, vmax=vmax,interpolation='nearest')
        m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        # plt.show()
        meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        for obj in meridict:
            line = meridict[obj][0][0]
        coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        polys = m.fillcontinents(color='#eeeeee', lake_color='#EFEFEF', zorder=90)
        # plt.show()
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


class Colormap():
    def __init__(self):
        pass
    def run(self):
        # self.colormap()
        self.statistic_contribution_area_individual_model()
        pass

    def colormap(self):
        outdir = result_root + rf'\FIGURE\\Colormap\\'
        T.mk_dir(outdir, True)
        temp_root = result_root + rf'\3mm\relative_change_growing_season\TRENDY\trend_analysis\\temp_root\\'

        model_list = [

                      'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',
                      'YIBs_S2_Monthly_lai']

        model_list = [

            'SNU_LAI', 'GLOBMAP_LAI',
            'LAI4g',

            ]

        dic_name = {'SNU_LAI': 'SNU',
                    'GLOBMAP_LAI': 'GLOBMAP',
                    'composite_LAI_median': 'Composite',
                    'composite_LAI': 'Composite',

                    'LAI4g': 'GIMMS4g',
                    'TRENDY_ensemble_median': 'TRENDY_ensemble',
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
                    'ORCHIDEE_S2_lai': 'ORCHIDEE',
                    'SDGVM_S2_lai': 'SDGVM',
                    'YIBs_S2_Monthly_lai': 'YIBs',
                    'LPX-Bern_S2_lai': 'LPX-Bern',
                    }


        fdir_all = result_root + rf'\partial_correlation\Obs\result\\'
        for model in model_list:
            fdir = fdir_all + rf'{model}\\'
            temp_fdir = temp_root + rf'{model}\\'
            T.mk_dir(temp_fdir, True)


            color_list= ['#a577ad',

            '#dae67a', '#f599a1',]

            my_cmap2 = T.cmap_blend(color_list, n_colors=6)


            fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.19))
            fpath = fdir + f'dominant_color_map_without_sign.tif'


            # 画 Robinson 投影 + 栅格
            m, mappable = Plot_Robinson_TRENDY().plot_Robinson(
                fpath, ax=ax, cmap=my_cmap2, vmin=1, vmax=3, colormap_n=4,is_discrete=True
            )


            # 裁剪显示范围
            lat_min, lat_max = -60, 60
            lon_min, lon_max = -125, 155
            x_min, _ = m(lon_min, 0)
            x_max, _ = m(lon_max, 0)
            _, y_min = m(0, lat_min)
            _, y_max = m(0, lat_max)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_title(dic_name.get(model, model), fontsize=8, font='Arial')
            ax.set_xticks([])
            ax.set_yticks([])

            # if mappable_for_cbar is None:
            #     mappable_for_cbar = mappable  # 只取第一个用于共享色标

            # 共享色标（水平放在底部）
            # cbar = fig.colorbar(
            #     mappable_for_cbar, ax=[ax for ax in axes if ax.has_data()],
            #     orientation='horizontal', fraction=0.035, pad=0.04
            # )
            # cbar.set_label('Trend (% per year)', fontsize=11)

            # 紧凑布局与间距
            # plt.subplots_adjust(hspace=0.08, wspace=0.02)
            outf = outdir + '\\' + model + '.png'
            # plt.show()
            plt.savefig(outf, dpi=600, bbox_inches='tight')
            plt.close()

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


    def statistic_contribution_area_individual_model(self):
        dff = result_root + rf'\partial_correlation\Dataframe\\Obs_TRENDY_comparison.df'
        df = T.load_df(dff)
        df = self.df_clean(df)

        model_list = ['TRENDY_ensemble_median_2',
            'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
            'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
            'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
            'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
            'ORCHIDEE_S2_lai',

            'YIBs_S2_Monthly_lai']

        # model_list = [
        #     'composite_LAI_median',
        #
        #     'SNU_LAI', 'GLOBMAP_LAI',
        #     'LAI4g',
        #
        # ]

        for model in model_list:
            percentage_list = []
            sum = 0
            df = df.dropna(subset=[f'{model}_dominant_color_map_without_sign'])


            for ii in [1, 2, 3,]:
                df_ii = df[df[f'{model}_dominant_color_map_without_sign'] == ii]
                # df_ii = df[df['composite_LAI_median_color_map'] == ii]

                percent = len(df_ii) / len(df) * 100
                sum = sum + percent


                percentage_list.append(percent)
            # print(percentage_list)
            # print(sum);


            color_list = ['#a577ad',

                          '#dae67a', '#f599a1', ]

            plt.figure(figsize=(1.2,1.2))
            plt.bar([1, 2, 3, ], percentage_list, color=color_list)
            plt.ylim(0, 50)
            plt.xticks([])

            plt.ylabel('Area(%)')
            # plt.show()
            outdir = result_root + rf'\FIGURE\Colormap\\'
            plt.savefig(outdir + f'\\statistics_contribution_area_{model}.pdf', dpi=300, bbox_inches='tight')
            plt.close()


class LAImax_LAImin():
    def __init__(self):
        pass
    def run(self):
        self.LAImax_LAImin_all()

        pass

    def LAImax_LAImin_all(self):
        outdir = result_root + rf'\FIGURE\\Colormap\\'
        T.mk_dir(outdir, True)
        temp_root = result_root + rf'\3mm\relative_change_growing_season\TRENDY\trend_analysis\\temp_root\\'

        model_list = [

                      'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',
                      'YIBs_S2_Monthly_lai']

        model_list = [

            'SNU_LAI', 'GLOBMAP_LAI',
            'LAI4g',

            ]

        dic_name = {'SNU_LAI': 'SNU',
                    'GLOBMAP_LAI': 'GLOBMAP',
                    'composite_LAI_median': 'Composite',
                    'composite_LAI': 'Composite',

                    'LAI4g': 'GIMMS4g',
                    'TRENDY_ensemble_median': 'TRENDY_ensemble',
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
                    'ORCHIDEE_S2_lai': 'ORCHIDEE',
                    'SDGVM_S2_lai': 'SDGVM',
                    'YIBs_S2_Monthly_lai': 'YIBs',
                    'LPX-Bern_S2_lai': 'LPX-Bern',
                    }


        fdir_all = result_root + rf'\partial_correlation\Obs\result\\'
        for model in model_list:
            fdir = fdir_all + rf'{model}\\'
            temp_fdir = temp_root + rf'{model}\\'
            T.mk_dir(temp_fdir, True)


            color_list= ['#a577ad',

            '#dae67a', '#f599a1',]

            my_cmap2 = T.cmap_blend(color_list, n_colors=6)


            fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.19))
            fpath = fdir + f'dominant_color_map_without_sign.tif'


            # 画 Robinson 投影 + 栅格
            m, mappable = Plot_Robinson_TRENDY().plot_Robinson(
                fpath, ax=ax, cmap=my_cmap2, vmin=1, vmax=3, colormap_n=4,is_discrete=True
            )


            # 裁剪显示范围
            lat_min, lat_max = -60, 60
            lon_min, lon_max = -125, 155
            x_min, _ = m(lon_min, 0)
            x_max, _ = m(lon_max, 0)
            _, y_min = m(0, lat_min)
            _, y_max = m(0, lat_max)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_title(dic_name.get(model, model), fontsize=8, font='Arial')
            ax.set_xticks([])
            ax.set_yticks([])

            # if mappable_for_cbar is None:
            #     mappable_for_cbar = mappable  # 只取第一个用于共享色标

            # 共享色标（水平放在底部）
            # cbar = fig.colorbar(
            #     mappable_for_cbar, ax=[ax for ax in axes if ax.has_data()],
            #     orientation='horizontal', fraction=0.035, pad=0.04
            # )
            # cbar.set_label('Trend (% per year)', fontsize=11)

            # 紧凑布局与间距
            # plt.subplots_adjust(hspace=0.08, wspace=0.02)
            outf = outdir + '\\' + model + '.png'
            # plt.show()
            plt.savefig(outf, dpi=600, bbox_inches='tight')
            plt.close()

    pass

    def Figure_robinson_reprojection(self):  # convert figure to robinson and no need to plot robinson again

        fdir_trend = result_root + rf'\bivariate\\rainfall\\'
        temp_root = result_root + rf'\bivariate\\rainfall\\'
        outdir = result_root + rf'\\bivariate\\ROBINSON\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):


            if not f.endswith('.tif'):
                continue

            fname = f.split('.')[0]

            fpath = fdir_trend + f
            outf=outdir + fname + '.tif'
            srcSRS=self.wkt_84()
            dstSRS=self.wkt_robinson()

            ToRaster().resample_reproj(fpath,outf, 5000, srcSRS=srcSRS, dstSRS=dstSRS)

            T.open_path_and_file(outdir)




class Partial_correlation():
    def __init__(self):
        pass
    def run(self):
        self.partial_correlation()
        pass

    def partial_correlation(self):
        outdir = result_root + rf'\3mm\FIGURE\\patial_corr\\Sensitivity\\'
        T.mk_dir(outdir, True)
        temp_root = result_root + rf'\3mm\relative_change_growing_season\TRENDY\trend_analysis\\temp_root\\'

        model_list = [
                      'TRENDY_ensemble_median',
                      'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',
                      'YIBs_S2_Monthly_lai']

        # model_list = [
        #     'composite_LAI_median',
        #     'SNU_LAI', 'GLOBMAP_LAI',
        #     'LAI4g',
        #     'composite_LAI',
        #     ]

        dic_name = {'SNU_LAI': 'SNU',
                    'GLOBMAP_LAI': 'GLOBMAP',
                    'composite_LAI_median': 'Composite',
                    'composite_LAI': 'Composite',

                    'LAI4g': 'GIMMS4g',
                    'TRENDY_ensemble_median': 'TRENDY_ensemble',
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
                    'ORCHIDEE_S2_lai': 'ORCHIDEE',
                    'SDGVM_S2_lai': 'SDGVM',
                    'YIBs_S2_Monthly_lai': 'YIBs',
                    'LPX-Bern_S2_lai': 'LPX-Bern',
                    }

        fdir_all = result_root + rf'\3mm\Multiregression\partial_correlation\Obs\result\\partial_corr2\\'
        fdir_all = result_root + rf'\3mm\Multiregression\partial_correlation\TRENDY\\partial_corr2\\'
        for model in model_list:
            fdir = fdir_all + rf'{model}\\sig\\'
            temp_fdir = temp_root + rf'{model}\\'
            T.mk_dir(temp_fdir, True)


            my_cmap2 ='Spectral'


            fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.19))
            # fpath = fdir + f'CV_intraannual_rainfall_ecosystem_year_zscore.tif'
            # fpath = fdir + f'detrended_sum_rainfall_ecosystem_year_CV_zscore.tif'
            fpath = fdir + f'{model}_sensitivity_zscore.tif'

            # 画 Robinson 投影 + 栅格
            m, mappable = Plot().plot_Robinson(
                fpath, ax=ax, cmap=my_cmap2, vmin=-1, vmax=1
            )
            # 叠加显著性
            # if not model == 'TRENDY_ensemble_mean':
            #     Plot().plot_Robinson_significance_scatter(
            #         m, f_pvalue, temp_root, sig_level=0.05, ax=ax,
            #         linewidths=0.5, s=1, c='k', marker='x', zorder=111, res=4
            #     )
            # else:
            #     continue

            # 裁剪显示范围
            lat_min, lat_max = -60, 60
            lon_min, lon_max = -125, 155
            x_min, _ = m(lon_min, 0)
            x_max, _ = m(lon_max, 0)
            _, y_min = m(0, lat_min)
            _, y_max = m(0, lat_max)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_title(dic_name.get(model, model), fontsize=8, font='Arial')
            ax.set_xticks([])
            ax.set_yticks([])

            # if mappable_for_cbar is None:
            #     mappable_for_cbar = mappable  # 只取第一个用于共享色标
            #
            # # 共享色标（水平放在底部）
            # cbar = fig.colorbar(
            #     mappable_for_cbar, ax=[ax for ax in axes if ax.has_data()],
            #     orientation='horizontal', fraction=0.035, pad=0.04
            # )
            # plot colorbar
            cbar = fig.colorbar(
                mappable, ax=ax,
                orientation='horizontal', fraction=0.035, pad=0.04
            )
            cbar.set_label('Partial correlation', fontsize=11)

            # 紧凑布局与间距
            # plt.subplots_adjust(hspace=0.08, wspace=0.02)
            outf = outdir + '\\' + model +'colorbar.pdf'
            # plt.show()
            plt.savefig(outf, dpi=600, bbox_inches='tight')
            plt.close()

    pass

    def plot_sig_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5,
                         s=20,
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
        # fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        # Plot().Robinson_reproj(fpath_resample, fpath_resample, res=res * 100000)

        # arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)
        arr = Tools().mask_999999_arr(arr, warning=False)
        #
        os.remove(fpath_clip)
        # os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)
        # exit()

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
        lon_list = np.array(lon_list) + res / 2
        lat_list = np.array(lat_list) - res / 2
        # lon_list = lon_list - originX
        # lat_list = lat_list + originY
        # lon_list = lon_list + pixelWidth / 2
        # lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        # print(lat_list);exit()
        m.scatter(lon_list, lat_list, latlon=True, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)
        # ax.scatter(lon_list,lat_list)
        # plt.show()

        return m

class Fire():
    def __init__(self):
        pass
    def run(self):
        # self.plot_spatial()
        self.histogram()
        # self.plot_spatial_SV()

        pass
    def histogram(self):
        tiff=result_root + rf'\3mm\Fire\moving_window_extraction\\Fire_percentage_annual_mean.tif'
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr[arr==999999]=np.nan
        arr = arr[~np.isnan(arr)]
        # arr_stastis=arr[arr>2]
        # print(len(arr_stastis)/len(arr));exit()

        fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.19))
        ## y is percentage of pixels


        n, bins = np.histogram(arr, bins=100)

        # 转换为百分比
        total = n.sum()
        n_percent = (n / total) * 100

        # 重新绘制百分比直方图
        bin_width = np.diff(bins)
        ax.bar(bins[:-1], n_percent, width=bin_width * 0.9,  # 0.9 = 10% gap
               align="edge", alpha=0.75, color='grey')

        ax.set_xlabel('Fraction of pixel area burned (%)',fontsize=10)
        ax.set_ylabel('Area (%)',fontsize=10)
        ax.set_xlim(0, 1)


        # plt.show()
        plt.savefig(result_root + rf'\3mm\FIGURE\\FIRE_SV\\Histogram.png', dpi=600, bbox_inches='tight')
        plt.close()


    def plot_spatial(self):
        outdir = result_root + rf'\3mm\FIGURE\\FIRE\\'
        T.mk_dir(outdir,True)
        temp_root = result_root + rf'3mm\Fire\moving_window_extraction\\temp_root\\'



        fdir = result_root + rf'\3mm\Fire\moving_window_extraction\\'






        fig,ax=plt.subplots(1,1,figsize=(3.35,2.19))
        fpath = fdir + f'Fire_percentage_annual_mean.tif'


        # 画 Robinson 投影 + 栅格
        m, mappable = Plot().plot_Robinson(
            fpath, ax=ax, cmap='rocket_r', vmin=0, vmax=1
        )


        # 裁剪显示范围
        lat_min, lat_max = -60, 60
        lon_min, lon_max = -125, 155
        x_min, _ = m(lon_min, 0)
        x_max, _ = m(lon_max, 0)
        _, y_min = m(0, lat_min)
        _, y_max = m(0, lat_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


        ax.set_xticks([])
        ax.set_yticks([])

        ##plot colorbar
        cbar = fig.colorbar(
            mappable, ax=ax,
            orientation='horizontal', fraction=0.035, pad=0.04
        )
        cbar.set_label('Burned area (%/yr)', fontsize=11)




    # 紧凑布局与间距
        # plt.subplots_adjust(hspace=0.08, wspace=0.02)
        outf = outdir + '\\' + 'Fire' + '.png'
        # plt.show()
        plt.savefig(outf, dpi=600, bbox_inches='tight')
        plt.close()


    def plot_spatial_SV(self):
        outdir = result_root + rf'\3mm\FIGURE\\FIRE_SV\\'
        T.mk_dir(outdir,True)
        temp_root = rf'D:\Project3\Data\VCF\dryland_tiff\dic_interpolation\mean\\temp_root\\'



        fdir =rf'D:\Project3\Data\VCF\dryland_tiff\dic_interpolation\mean\\'






        fig,ax=plt.subplots(1,1,figsize=(3.35,2.19))
        fpath = fdir + f'Non tree vegetation_mean.tif'


        # 画 Robinson 投影 + 栅格
        m, mappable = Plot().plot_Robinson(
            fpath, ax=ax, cmap='Spectral', vmin=0, vmax=100
        )


        # 裁剪显示范围
        lat_min, lat_max = -60, 60
        lon_min, lon_max = -125, 155
        x_min, _ = m(lon_min, 0)
        x_max, _ = m(lon_max, 0)
        _, y_min = m(0, lat_min)
        _, y_max = m(0, lat_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)


        ax.set_xticks([])
        ax.set_yticks([])

        ##plot colorbar
        cbar = fig.colorbar(
            mappable, ax=ax,
            orientation='horizontal', fraction=0.035, pad=0.04
        )
        cbar.set_label('Mean short vegetation (%)', fontsize=11)




    # 紧凑布局与间距
        # plt.subplots_adjust(hspace=0.08, wspace=0.02)
        outf = outdir + '\\' + 'Mean short vegetation' + '.png'
        # plt.show()
        plt.savefig(outf, dpi=600, bbox_inches='tight')
        plt.close()

class Trends_obs_and_model():
    def run(self):
        self.plot_remote_sensing()
        # self.plot_TRENDY()

    def plot_remote_sensing(self): ## put main ms. so pdf


        fdir_trend = result_root+rf'\Composite_LAI\relative_change\trend\\'
        temp_root = result_root+rf'\TRENDY\S2\relative_change\relative_change\trend_analysis_relative_change\\temp_plot\\'
        outdir = result_root+rf'FIGURE\\Figure1\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):

            if not f.endswith('.tif'):
                continue
            if 'p_value' in f:
                continue


            fname = f.split('.')[0]
            fname_p_value = fname.replace('trend', 'p_value')
            print(fname_p_value)
            fpath = fdir_trend + f
            # print(fpath);exit()
            p_value_f = fdir_trend + fname_p_value+'.tif'
            print(p_value_f)
            # exit()
            plt.figure(figsize=(Plot_Robinson_remote_sensing().map_width, Plot_Robinson_remote_sensing().map_height))
            m, ret = Plot_Robinson_remote_sensing().plot_Robinson(fpath, vmin=-1, vmax=1, is_discrete=True, colormap_n=9,)

            Plot_Robinson_remote_sensing().plot_Robinson_significance_scatter(m,p_value_f,temp_root,0.05, s=0.5, marker='.')
            # plt.title(f'{fname}')
            # plt.show()
            outf = outdir + f+'.pdf'
            plt.savefig(outf)
            plt.close()
            # T.open_path_and_file(outdir)
            # exit()


    def plot_TRENDY(self):
        outdir = result_root + rf'\FIGURE\Figure1\\'
        T.mk_dir(outdir,True)
        temp_root = result_root + rf'\TRENDY\S2\relative_change\relative_change\\temp_root\\'

        model_list = ['SNU_LAI', 'GLOBMAP_LAI', 'composite_LAI_median',
                      'LAI4g',
                      'TRENDY_ensemble_median',
                      'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai','ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai','LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai', 'SDGVM_S2_lai',
                      'YIBs_S2_Monthly_lai']



        dic_name = {'SNU_LAI_relative_change':'SNU',
                    'GLOBMAP_LAI_relative_change':'GLOBMAP',
                    'composite_LAI_relative_change_mean': 'Composite',


            'LAI4g': 'GIMMS4g',
                    'TRENDY_ensemble_median':'TRENDY_ensemble',
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
                    'ORCHIDEE_S2_lai': 'ORCHIDEE',
                    'SDGVM_S2_lai': 'SDGVM',
                    'YIBs_S2_Monthly_lai': 'YIBs',
                    'LPX-Bern_S2_lai': 'LPX-Bern',
                    }





        fdir = result_root + rf'\TRENDY\S2\relative_change\relative_change\trend_analysis_relative_change\\'

        color_list = ['#844000', '#fc9831', '#fffbd4', '#86b9d2', '#064c6c']
        my_cmap2 = T.cmap_blend(color_list, n_colors=5)

        # 7×3 画布（共 21 格，当前有 19 个模型）
        # fig, axes = plt.subplots(10, 2, figsize=(10, 25), sharex=False, sharey=False)
        # axes = axes.ravel()


        mappable_for_cbar = None  # 用于共享色标

        for i, model in enumerate(model_list):
            fig,ax=plt.subplots(1,1,figsize=(3.35,2.19))
            fpath = fdir + f'{model}_relative_change_trend.tif'
            f_pvalue = fdir + f'{model}_relative_change_p_value.tif'

            # 画 Robinson 投影 + 栅格
            m, mappable = Plot_Robinson_TRENDY().plot_Robinson(
                fpath, ax=ax, cmap=my_cmap2, vmin=-1, vmax=1
            )
            #叠加显著性
            if not model=='TRENDY_ensemble_median1':
                Plot_Robinson_TRENDY().plot_Robinson_significance_scatter(
                    m, f_pvalue, temp_root, sig_level=0.05, ax=ax,
                    linewidths=0.5, s=.5, c='k', marker='.', zorder=111, res=2
                )
            else:
                continue

            # 裁剪显示范围
            lat_min, lat_max = -60, 60
            lon_min, lon_max = -125, 155
            x_min, _ = m(lon_min, 0)
            x_max, _ = m(lon_max, 0)
            _, y_min = m(0, lat_min)
            _, y_max = m(0, lat_max)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_title(dic_name.get(model, model), fontsize= 8, font='Arial')
            ax.set_xticks([])
            ax.set_yticks([])

            # if mappable_for_cbar is None:
            #     mappable_for_cbar = mappable  # 只取第一个用于共享色标



        # 共享色标（水平放在底部）
        # cbar = fig.colorbar(
        #     mappable_for_cbar, ax=[ax for ax in axes if ax.has_data()],
        #     orientation='horizontal', fraction=0.035, pad=0.04
        # )
        # cbar.set_label('Trend (% per year)', fontsize=11)

        # 紧凑布局与间距
            # plt.subplots_adjust(hspace=0.08, wspace=0.02)
            outf = outdir + '\\' + model + '.png'
            # plt.show()
            plt.savefig(outf, dpi=600, bbox_inches='tight')
            plt.close()

    pass

    def plot_sig_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5,
                                               s=20,
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
        # fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        # Plot().Robinson_reproj(fpath_resample, fpath_resample, res=res * 100000)

        # arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)
        arr = Tools().mask_999999_arr(arr, warning=False)
        #
        os.remove(fpath_clip)
        # os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)
        # exit()

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
        lon_list = np.array(lon_list) + res/2
        lat_list = np.array(lat_list) - res/2
        # lon_list = lon_list - originX
        # lat_list = lat_list + originY
        # lon_list = lon_list + pixelWidth / 2
        # lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        # print(lat_list);exit()
        m.scatter(lon_list, lat_list, latlon=True, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)
        # ax.scatter(lon_list,lat_list)
        # plt.show()

        return m

class TRENDY_CV_moving_window_robust():

    def trend_analysis_plot(self):
        outdir = result_root + rf'3mm\FIGURE\moving_window_robust_test\10_year\\'
        T.mk_dir(outdir,True)
        temp_root = result_root + rf'\3mm\moving_window_robust_test\moving_window_extraction_average\\temp_root\\'
        model_list=['composite_LAI_median','GLOBMAP_LAI','LAI4g','SNU_LAI',]


        dic_name = {'SNU_LAI':'SNU',
                    'GLOBMAP_LAI':'GLOBMAP',
                    'composite_LAI': 'Composite',


            'LAI4g': 'GIMMS4g',
                    'TRENDY_ensemble_composite_time_series': 'TRENDY_ensemble',
                    'TRENDY_ensemble_median':'TRENDY_ensemble',
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
                    'ORCHIDEE_S2_lai': 'ORCHIDEE',
                    'SDGVM_S2_lai': 'SDGVM',
                    'YIBs_S2_Monthly_lai': 'YIBs',
                    'LPX-Bern_S2_lai': 'LPX-Bern',
                    }





        fdir = result_root + rf'\3mm\moving_window_robust_test\moving_window_extraction_average\10_year\trend\\\\'



        # 7×3 画布（共 21 格，当前有 19 个模型）
        # fig, axes = plt.subplots(10, 2, figsize=(10, 25), sharex=False, sharey=False)
        # axes = axes.ravel()


        mappable_for_cbar = None  # 用于共享色标

        for i, model in enumerate(model_list):
            fig,ax=plt.subplots(1,1,figsize=(3.35,2.19))

            fpath = fdir + f'{model}_detrend_CV_trend.tif'
            f_pvalue = fdir + f'{model}_detrend_CV_p_value.tif'

            # 画 Robinson 投影 + 栅格
            m, mappable = Plot().plot_Robinson(
                fpath, ax=ax, cmap='PRGn_r', vmin=-1, vmax=1
            )
            # 叠加显著性

            Plot().plot_Robinson_significance_scatter(
                m, f_pvalue, temp_root, sig_level=0.05, ax=ax,
                linewidths=0.5, s=.2, c='k', marker='.', zorder=111, res=2
            )




            # 裁剪显示范围
            lat_min, lat_max = -60, 60
            lon_min, lon_max = -125, 155
            x_min, _ = m(lon_min, 0)
            x_max, _ = m(lon_max, 0)
            _, y_min = m(0, lat_min)
            _, y_max = m(0, lat_max)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_title(dic_name.get(model, model), fontsize= 8, font='Arial')
            ax.set_xticks([])
            ax.set_yticks([])

            ## cbar

         ###plot colorbar
            # cbar = fig.colorbar(
            #     mappable, ax=ax,
            #     orientation='horizontal', fraction=0.035, pad=0.04
            # )
            # cbar.set_label('Trends in CVLAI (%/yr)', fontsize=11)

        # 紧凑布局与间距
            # plt.subplots_adjust(hspace=0.08, wspace=0.02)
            outf = outdir + '\\' + model + '.png'
            # plt.show()
            plt.savefig(outf, dpi=600, bbox_inches='tight')
            plt.close()

    pass



    def plot_sig_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5,
                                               s=10,
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
        # fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        # Plot().Robinson_reproj(fpath_resample, fpath_resample, res=res * 100000)

        # arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)
        arr = Tools().mask_999999_arr(arr, warning=False)
        #
        os.remove(fpath_clip)
        # os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)
        # exit()

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
        lon_list = np.array(lon_list) + res/2
        lat_list = np.array(lat_list) - res/2
        # lon_list = lon_list - originX
        # lat_list = lat_list + originY
        # lon_list = lon_list + pixelWidth / 2
        # lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        # print(lat_list);exit()
        m.scatter(lon_list, lat_list, latlon=True, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)
        # ax.scatter(lon_list,lat_list)
        # plt.show()

        return


class Trends_CV_obs_and_model():
    def __init__(self):
        pass
    def run(self):
        self.plot_remote_sensing()
        # self.plot_TRENDY()

    def plot_remote_sensing(self): ## put main ms. so pdf


        fdir_trend = result_root+rf'\Composite_LAI\CV\trend\\'
        temp_root = result_root+rf'\TRENDY\S2\relative_change\relative_change\trend_analysis_relative_change\\temp_plot\\'
        outdir = result_root+rf'FIGURE\\Figure1b\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):
            if not 'median' in f:
                continue

            if not f.endswith('.tif'):
                continue
            if 'p_value' in f:
                continue


            f_trend=fdir_trend+'composite_LAI_detrend_CV_median_trend.tif'
            f_p_value = fdir_trend + 'composite_LAI_detrend_CV_median_p_value.tif'
            print(f_p_value)



            # exit()
            plt.figure(figsize=(Plot_Robinson_remote_sensing().map_width, Plot_Robinson_remote_sensing().map_height))
            m, ret = Plot_Robinson_remote_sensing().plot_Robinson(f_trend, vmin=-1, vmax=1, is_discrete=True, colormap_n=9,)

            Plot_Robinson_remote_sensing().plot_Robinson_significance_scatter(m,f_p_value,temp_root,0.05, s=0.5, marker='.')
            # plt.title(f'{fname}')
            # plt.show()

            outf = outdir + f+'.pdf'
            plt.savefig(outf)
            plt.close()
            # T.open_path_and_file(outdir)
            # exit()

    def plot_TRENDY(self):
        outdir = result_root + rf'\FIGURE\Figure2\\'
        T.mk_dir(outdir,True)
        temp_root = result_root + rf'\TRENDY\S2\relative_change\relative_change\\temp_root\\'

        model_list = ['SNU_LAI', 'GLOBMAP_LAI',
                      'LAI4g',
                      'TRENDY_ensemble_median',
                      'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai','ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai','LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai', 'SDGVM_S2_lai',
                      'YIBs_S2_Monthly_lai']



        dic_name = {'SNU_LAI_relative_change':'SNU',
                    'GLOBMAP_LAI_relative_change':'GLOBMAP',
                    'composite_LAI_relative_change_mean': 'Composite',


            'LAI4g': 'GIMMS4g',
                    'TRENDY_ensemble_median':'TRENDY_ensemble',
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
                    'ORCHIDEE_S2_lai': 'ORCHIDEE',
                    'SDGVM_S2_lai': 'SDGVM',
                    'YIBs_S2_Monthly_lai': 'YIBs',
                    'LPX-Bern_S2_lai': 'LPX-Bern',
                    }





        fdir = result_root + rf'\TRENDY\S2\15_year\moving_window_extraction_CV\trend_analysis\\'


        color_list = [
            '#008837',
            '#a6dba0',
            '#f7f7f7',
            '#c2a5cf',
            '#7b3294',
        ]
        my_cmap2 = T.cmap_blend(color_list, n_colors=5)

        # 7×3 画布（共 21 格，当前有 19 个模型）
        # fig, axes = plt.subplots(10, 2, figsize=(10, 25), sharex=False, sharey=False)
        # axes = axes.ravel()


        mappable_for_cbar = None  # 用于共享色标

        for i, model in enumerate(model_list):
            fig,ax=plt.subplots(1,1,figsize=(3.35,2.19))
            fpath = fdir + f'{model}_detrend_CV_trend.tif'
            f_pvalue = fdir + f'{model}_detrend_CV_p_value.tif'

            # 画 Robinson 投影 + 栅格
            m, mappable = Plot_Robinson_TRENDY().plot_Robinson(
                fpath, ax=ax, cmap=my_cmap2, vmin=-1, vmax=1
            )
            #叠加显著性
            if not model=='TRENDY_ensemble_median1':
                Plot_Robinson_TRENDY().plot_Robinson_significance_scatter(
                    m, f_pvalue, temp_root, sig_level=0.05, ax=ax,
                    linewidths=0.5, s=.5, c='k', marker='.', zorder=111, res=2
                )
            else:
                continue

            # 裁剪显示范围
            lat_min, lat_max = -60, 60
            lon_min, lon_max = -125, 155
            x_min, _ = m(lon_min, 0)
            x_max, _ = m(lon_max, 0)
            _, y_min = m(0, lat_min)
            _, y_max = m(0, lat_max)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_title(dic_name.get(model, model), fontsize= 8, font='Arial')
            ax.set_xticks([])
            ax.set_yticks([])

            # if mappable_for_cbar is None:
            #     mappable_for_cbar = mappable  # 只取第一个用于共享色标



        # 共享色标（水平放在底部）
        # cbar = fig.colorbar(
        #     mappable_for_cbar, ax=[ax for ax in axes if ax.has_data()],
        #     orientation='horizontal', fraction=0.035, pad=0.04
        # )
        # cbar.set_label('Trend (% per year)', fontsize=11)

        # 紧凑布局与间距
            # plt.subplots_adjust(hspace=0.08, wspace=0.02)
            outf = outdir + '\\' + model + '.png'
            # plt.show()
            plt.savefig(outf, dpi=600, bbox_inches='tight')
            plt.close()

    pass



    def plot_sig_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5,
                                               s=20,
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
        # fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        # Plot().Robinson_reproj(fpath_resample, fpath_resample, res=res * 100000)

        # arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)
        arr = Tools().mask_999999_arr(arr, warning=False)
        #
        os.remove(fpath_clip)
        # os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)
        # exit()

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
        lon_list = np.array(lon_list) + res/2
        lat_list = np.array(lat_list) - res/2
        # lon_list = lon_list - originX
        # lat_list = lat_list + originY
        # lon_list = lon_list + pixelWidth / 2
        # lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        # print(lat_list);exit()
        m.scatter(lon_list, lat_list, latlon=True, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)
        # ax.scatter(lon_list,lat_list)
        # plt.show()

        return

class LAImax_LAImin_models():
    def run(self):
        # self.bivariate_map()
        self.Figure_robinson_reprojection()

        pass

    def bivariate_map(self):  ## figure 1  ## LAImin and LAImax bivariate
        import xymap


        fdir_max =result_root + rf'3mm\relative_change_growing_season\moving_window_min_max_anaysis\max\trend_analysis\\'
        fdir_min =result_root + rf'3mm\relative_change_growing_season\moving_window_min_max_anaysis\min\trend_analysis\\'

        outdir =result_root + rf'\\3mm\FIGURE\\bivariate\\'

        T.mkdir(outdir, force=True)
        model_list = [
            'TRENDY_ensemble',
            'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
            'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
            'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
            'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
            'ORCHIDEE_S2_lai',
            'YIBs_S2_Monthly_lai']

        for model in model_list:
            outtif = join(outdir, f'{model}.tif')

            fpath1 = join(fdir_max,f'{model}_detrend_max_trend.tif')

            fpath2 = join(fdir_min,f'{model}_detrend_min_trend.tif')


            #1
            tif1_label, tif2_label = 'LAImax_trend','LAImin_trend'
            #2
            # tif1_label, tif2_label = 'LAI_CV_trend','LAI_relative_change_mean_trend'

            #1
            min1, max1 = -1, 1
            min2, max2 = -1, 1

            #2
            # min1, max1 = -.3, .3
            # min2, max2 = -.5, .5

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
            upper_left_color = (0, 0, 110)
            upper_right_color =(112, 196, 181)
            lower_left_color = (237, 125, 49)

            lower_right_color = (193, 92, 156)
            center_color = (240, 240, 240)

            ## CV greening option
            #
            # upper_left_color = (194, 0, 120)
            # upper_right_color = (0,170,237)
            # lower_left_color = (233, 55, 43)
            # # lower_right_color = (160, 108, 168)
            # lower_right_color = (234, 233, 46)
            # center_color = (240, 240, 240)


            xymap.Bivariate_plot_1(res = 11,
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
                                                                        n_x = 5, n_y =5
                                                                        )

            T.open_path_and_file(outdir)
    def Figure_robinson_reprojection(self):  # convert figure to robinson and no need to plot robinson again

        fdir_trend = result_root + rf'\3mm\FIGURE\bivariate\\'
        temp_root = result_root + rf'\3mm\FIGURE\bivariate\bivariate\\temp_root\\'
        outdir = result_root + rf'\3mm\FIGURE\bivariate\\ROBINSON\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):


            if not f.endswith('.tif'):
                continue

            fname = f.split('.')[0]

            fpath = fdir_trend + f
            outf=outdir + fname + '.tif'
            srcSRS=self.wkt_84()
            dstSRS=self.wkt_robinson()

            ToRaster().resample_reproj(fpath,outf, 5000, srcSRS=srcSRS, dstSRS=dstSRS)

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




class PLOT_Climate_factors():
    def run(self):
        self.trend_analysis_plot()
        pass
    def trend_analysis_plot(self):
        outdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\Robinson\\'

        T.mk_dir(outdir,True)
        temp_root = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\temp_root\\'

        model_list = [  'heavy_rainfall_days', 'rainfall_seasonality_all_year',
                      'sum_rainfall', 'VPD',

                   ]

        dic_name = {'pi_average': 'SM-T coupling',
                    'heavy_rainfall_days': 'Heavy rainfall days',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality',
                    'sum_rainfall': 'Total rainfall',
                    'VOD_detrend_min': 'VOD_min',
                    'fire_ecosystem_year_average': 'Fire burn area',
                    'VPD': 'VPD',
                    'CV_intraannual_rainfall': 'CV_intraannual_rainfall_trend',

                    'Non tree vegetation': 'Non tree vegetation',
                    }



        fdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\trend\\'
        count = 1
        for model in model_list:


            fpath= fdir+f'{model}_trend.tif'
            f_p_value = fdir + f'{model}_p_value.tif'


            # ax = plt.subplot(3, 2, count)
            # print(count, f)
            count = count + 1
            # if not count == 9:
            #     continue

            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr = Tools().mask_999999_arr(arr, warning=False)
            arr_m = ma.masked_where(np.isnan(arr), arr)
            lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            plt.figure(figsize=(5, 3))

            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180,
                        resolution='l',)

            m = self.plot_sig_scatter(
                m, f_p_value, temp_root, sig_level=0.05, ax=None, linewidths=0.2,
                s=1,
                c='k', marker='x',
                zorder=100, res=4
            )
            plt.title(f'{dic_name[model]}', fontsize=10,font='Arial')

            m.drawcoastlines(linewidth=0.2, color='k', zorder=11)
            dic_vmin={'pi_average': -0.1,
                       'heavy_rainfall_days': -0.3,
                       'rainfall_seasonality_all_year': -0.5,
                       'sum_rainfall': -5,
                      'VOD_detrend_min': -0.005,
                      'fire_ecosystem_year_average': -1,
                      'VPD': -0.01,
                      'Non tree vegetation': -0.5,
                      'CV_intraannual_rainfall': -0.1


            }

            dic_vmax={'pi_average': 0.1,
                       'heavy_rainfall_days': .3,
                       'rainfall_seasonality_all_year': .5,
                       'sum_rainfall': 5,
                      'VOD_detrend_min': 0.005,
                      'fire_ecosystem_year_average': 1,
                      'VPD': 0.01,
                      'Non tree vegetation': 0.5,
                      'CV_intraannual_rainfall': 0.1}

            ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap='PRGn', zorder=-1, vmin=dic_vmin[model], vmax=dic_vmax[model])


            # plt.show()
        # cax = plt.axes([0.5 - 0.15, 0.05, 0.3, 0.02])
        # cb = plt.colorbar(ret, ax=ax, cax=cax, orientation='horizontal')
        # # cb.set_label(f'{model}_CV_trend', labelpad=-40)
            outf = outdir + rf'trend_analysis_plot_{model}.png'

            plt.subplots_adjust(hspace=0.055, wspace=0.038)
            #
            plt.savefig(outf, dpi=600)
            plt.close()
        # plt.show()
        T.open_path_and_file(outdir)
        # exit()
    pass

    def plot_sig_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5,
                                               s=20,
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
        # fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        # Plot().Robinson_reproj(fpath_resample, fpath_resample, res=res * 100000)

        # arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample)
        arr = Tools().mask_999999_arr(arr, warning=False)
        #
        os.remove(fpath_clip)
        # os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)
        # exit()

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
        lon_list = np.array(lon_list) + res/2
        lat_list = np.array(lat_list) - res/2
        # lon_list = lon_list - originX
        # lat_list = lat_list + originY
        # lon_list = lon_list + pixelWidth / 2
        # lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        # print(lat_list);exit()
        m.scatter(lon_list, lat_list, latlon=True, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)
        # ax.scatter(lon_list,lat_list)
        # plt.show()

        return m







class Bivariate_analysis():
    def __init__(self):
        pass
    def run (self):
        # self.plot_robinson()
        self.statistis()

    def plot_robinson(self):

        fdir_trend = result_root + rf'\3mm\bivariate_analysis\\'
        temp_root = result_root + rf'\3mm\bivariate_analysis\\temp\\'
        outdir = result_root + rf'\3mm\FIGURE\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):

            if not f.endswith('.tif'):
                continue

            fname = f.split('.')[0]


            fpath = rf"D:\Project3\Result\3mm\bivariate_analysis\heat_events_CVinternnaul.tif"
            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            # m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=8, is_discrete=True, colormap_n=8, cmap='RdYlBu_r',)

            color_list1 = [
                '#d01c8b',
                '#f3d705',
                '#3d46e8',
                '#4dac26',

            ]

            my_cmap2 = T.cmap_blend(color_list1, n_colors=5)
            # arr = ToRaster().raster2array(fpath)[0]
            # arr[arr<-999]=np.nan
            # plt.imshow(arr,cmap=my_cmap,vmin=1,vmax=8,interpolation='nearest')
            # plt.colorbar()
            # plt.show()
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=5, is_discrete=True, colormap_n=5,
                                                   cmap=my_cmap2, )

            plt.title(f'{fname}')
            # plt.show()
            outf = outdir + 'heat_events_CVinternnaul.pdf'
            plt.savefig(outf)
            plt.close()
            exit()

    def statistis(self):
        tiff=result_root+rf'\3mm\bivariate_analysis\heat_events_CVinternnaul.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        whole_number_pixels =array[array>-999].size

        dic={}
        for i in range(1,5):
            array[array==i]=i
            count=np.count_nonzero(array==i)
            percentage=count/whole_number_pixels*100
            dic[i]=percentage
        color_list = [
            '#d01c8b',
            '#f3d705',
            '#3d46e8',
            '#4dac26',


        ]

        print(dic)
        label_list=['Inc_Inc','Inc_Dec','Dec_Inc','Dec_Dec']
        plt.bar(dic.keys(),dic.values(),color=color_list)
        plt.xticks(range(1,5),label_list,size=12)
        plt.ylabel('Percentage (%)', size=12)
        # plt.show()
        plt.savefig(result_root + rf'\3mm\FIGURE\heatmap.pdf')


class calculate_longterm_CV():
    def __init__(self):
        pass
    def run (self):
        self.calculate_long_term_CV()
        pass
    def calculate_long_term_CV(self):
        fpath=result_root+rf'3mm\extract_LAI4g_phenology_year\dryland\detrend_TRENDY\\LAI4g_detrend.npy'
        dic=T.load_npy(fpath)
        result_dic={}
        for pix in dic:
            vals=dic[pix]
            mean=np.nanmean(vals)
            std=np.nanstd(vals)
            CV=std/mean*100
            result_dic[pix]=CV
        outf=result_root+rf'3mm\extract_LAI4g_phenology_year\dryland\detrend_TRENDY\\long_term_LAI4g_detrend_CV.npy'
        np.save(outf,result_dic)

        pass

def main():


    # Trends_obs_and_model().run()  ## Figure 1
    # Trends_CV_obs_and_model().run()  ## Figure 2
    PLOT_dataframe().run()
    # TRENDY_CV_moving_window_robust().trend_analysis_plot()
    # TRENDY_CV().trend_analysis_plot()
    # Fire().run()
    # Colormap().run()
    # Partial_correlation().run()
    # LAImax_LAImin_models().run()
    # PLOT_Climate_factors().run()
    # calculate_longterm_CV().run()
    # SHAP_CV().run()
    # SHAP_rainfall_seasonality().run()
    # SHAP_CO2_interaction().run()
    # Bivariate_analysis().run()
    # Trend_CV().run()


    pass

if __name__ == '__main__':
    main()