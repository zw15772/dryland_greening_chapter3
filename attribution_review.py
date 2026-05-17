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
            'CV_daily_rainfall_3mm_average']
        self.model_list = ['SNU_LAI', 'GLOBMAP_LAI',
                           'LAI4g','composite_LAI_mean',
                          'composite_LAI_median' ]


        # for model in self.model_list:
        #         self.outdir=self.result_root+rf'\result\\1mm_new\\'+model+'\\'
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
        # #
        # #

        # self.plot_spatial_map_sig()


        self.statistic_corr_boxplot()
        # self.statistic_percentage()
        #

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
            fdir = self.result_root + rf'\result\\5mm\\'+model+'\\'
            print(fdir)
            # outdir = self.result_root + rf'\result\\1mm_new\\{model}\\\sig_nomask\\'
            outdir = self.result_root + rf'\result\\5mm\\{model}\\\sig\\'
            T.mk_dir(outdir, True)
            new_variable_list = variable_list + [f'{model}_sensitivity']

            fdir_Y = result_root + rf'\Multiregression_contribution\Obs\input\Y\zscore\\trend\\'
            fy_trend = join(fdir_Y, f'{model}_detrend_CV_zscore_trend.tif')
            fy_trend_p_value = join(fdir_Y, f'{model}_detrend_CV_zscore_p_value.tif')

            arr_y_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
               fy_trend)
            arr_y_trend_p_value, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
                fy_trend_p_value)

            ## mask
            mask = np.ones_like(arr_y_trend)
            mask[(arr_y_trend_p_value > 0.05) & (arr_y_trend <= 0)] = np.nan
            # plt.imshow(arr_y_trend)
            # plt.colorbar()
            # plt.show()


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

                # === ★ 叠加 LAI 正趋势掩膜 ★
                arr_corr[np.isnan(mask)] = np.nan


                #
                # plt.imshow(arr_corr)
                # plt.colorbar()
                # plt.show()
                outf = outdir  + f'{variable}.tif'
                DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_corr, outf)

    def statistic_percentage(self):
        dff = result_root + rf'\partial_correlation\Dataframe\\Obs.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        print(len(df))

        # === 仅保留CVLAI显著上升的像素 ===

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
            if not 'composite_LAI_median' in model:
                continue

            result_dic = {}

            # print(len(df));exit()
            for variable in variable_list:
                new_variable = f'{model}_{variable}'
                if new_variable not in df.columns:
                    continue

                vals = np.array(df[new_variable].tolist(), dtype=float)
                vals[(vals > 99) | (vals < -99)] = np.nan
                vals = vals[~np.isnan(vals)]
                vals_pos = vals[vals > 0]
                vals_neg = vals[vals < 0]
                vals_pos_percent = len(vals_pos) / len(vals)
                vals_neg_percent = len(vals_neg) / len(vals)
                result_dic[new_variable] = [vals_pos_percent, vals_neg_percent]
        pprint(result_dic)


    def statistic_corr_boxplot(self):
        """
        绘制 partial correlation 的分布（仅针对 CVLAI 上升区域）
        显示 sensitivity (γ), CV_inter, CV_intra 的箱线图
        """

        # === 1. 读取数据 ===
        dff = result_root + rf'\partial_correlation\Dataframe\\1mm_new\\Obs.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        print(len(df))

        # === 仅保留CVLAI显著上升的像素 ===


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
            if not 'composite_LAI_median' in model:
                continue

            result_dic = {}


            # print(len(df));exit()
            for variable in variable_list:
                new_variable = f'{model}_{variable}_sig'
                if new_variable not in df.columns:
                    continue


                vals = np.array(df[new_variable].tolist(), dtype=float)
                vals[(vals > 99) | (vals < -99)] = np.nan
                vals = vals[~np.isnan(vals)]
                print(f'{variable}', len(vals))

            #     plt.hist(vals, bins=30)
            #     plt.axvline(np.mean(vals), color='g', label='Mean')
            #     plt.axvline(np.median(vals), color='r', label='Median')
            #     plt.legend()
            #     plt.show()

                # vals_mean=np.nanmean(vals)
                # print(vals_mean)
                result_dic[new_variable] = vals

        # === 5. 按 variable_list 顺序组织数据 ===
            data_list = []
            x_labels = []

            for var in variable_list:
                key = f'{model}_{var}_sig'
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
                showfliers=False,

                showmeans=False,

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

            outdir=result_root + rf'\FIGURE\SI\\'
            Tools().mk_dir(outdir, force=True)

            # outf=join(outdir,f'{model}_partial_correlation_boxplot_3mm.pdf')
            # plt.savefig(outf,bbox_inches='tight',dpi=300
            #
            # )
            # plt.close()



    def darken_color(self, color, amount=0.7):
        """
        给颜色加深，amount 越小越深 (0~1之间)
        """
        import matplotlib.colors as mcolors
        c = mcolors.to_rgb(color)
        return tuple([max(0, x * amount) for x in c])




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

class Delta_regression:

    def __init__(self):
        self.xvar = ['Precip_sum_detrend_CV_zscore',
                     'CV_daily_rainfall_average_zscore',
                     'VPD_detrend_average_zscore']

        self.model_list = ['composite_LAI_median','composite_LAI_mean']

        # self.model_list = ['composite_LAI_mean','composite_LAI_median', 'SNU_LAI', 'GLOBMAP_LAI', 'LAI4g' ]

        self.outdir = rf'D:\Project3\Result\Nov\Multiregression_contribution\Obs\review\\'
        T.mkdir(self.outdir, True)

        pass


    def run(self):

        ## step 1 zscore
        # self.zscore()
        # # step 2 build dataframe manually
        # df=self.build_df()
        # self.append_attributes(df)

        ##### step 1

        for model in self.model_list:
            x_list=self.xvar+[model+'_sensitivity_zscore']
        #     self.do_multi_regression(model, x_list)
            ## not using below function
            # self.do_multi_regression_control_experiment(model,x_list) ## not use this but the result is the same


            # self.calculate_trend_contribution(model,x_list)

        ## step 2
        ### before calculating contribution, build dataframe
        self.statistic_contribution_no_residual()  ## use this


        ###########################################3



        # self.dominant_region_trends() not use




        pass

    def zscore(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root + rf'\Multiregression_contribution\Obs\\review\X_review\\'
        outdir = result_root + rf'\Multiregression_contribution\Obs\review\X_review\\zscore\\'
        T.mk_dir(outdir, force=True)
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):



            dic = T.load_npy(fdir + f)
            outf = outdir + f.split('.')[0] + '_zscore.npy'

            zscore_dic = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue

                # time_series = dic[pix]['intersensitivity_precip_val']
                time_series = dic[pix]

                # # 检查 time_series 是否为 list 或 array（防止是 float/NaN）

                if not isinstance(time_series, (list, np.ndarray)):
                    print(f"{pix}: invalid time_series (not iterable): {time_series}")
                    continue

                time_series = np.array(time_series, dtype=float)
                # time_series = time_series[3:37]

                print(len(time_series))
                ## exclude nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                # if np.nanmean(time_series) >999:
                #     continue
                if np.nanmean(time_series) < -999:
                    continue
                time_series = time_series
                mean = np.nanmean(time_series)
                zscore = (time_series - mean) / np.nanstd(time_series)

                zscore_dic[pix] = zscore

                # plt.plot(time_series)
                # plt.legend(['raw'])
                # # plt.show()
                #
                # #
                # plt.plot(zscore)
                # plt.legend(['zscore'])
                # # # plt.legend(['raw','zscore'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)

    def build_df(self,):

        fdir = result_root+rf'\Multiregression_contribution\Obs\review\X_review\zscore\\'
        all_dic = {}

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'zscore' in f:
                continue

            fname = f.split('.')[0]

            fpath = fdir + f

            dic = T.load_npy(fpath)
            key_name = fname

            all_dic[key_name] = dic
        # print(all_dic.keys())
        df = T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)


        return df

    def append_attributes(self, df):  ## add attributes
        fdir = result_root+rf'\Multiregression_contribution\Obs\review\Y\zscore\\'

        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
                continue


            # array=np.load(fdir+f)
            # dic = DIC_and_TIF().spatial_arr_to_dic(array)
            dic = T.load_npy(fdir + f)
            key_name = f.split('.')[0]
            # if not key_name in var_list:
            #     continue
            print(key_name)

            # df[key_name] = df['pix'].map(dic)
            # T.print_head_n(df)
            df = T.add_spatial_dic_to_df(df, dic, key_name)
        outdir=result_root+rf'\Multiregression_contribution\Obs\review\\Dataframe\\'
        T.mk_dir(outdir, True)
        T.save_df(df, outdir + rf'\Dataframe.df')
        T.df_to_excel(df, outdir + rf'\Dataframe.xlsx')



        return df


    # def append_value(self, df):  ##补齐
    #
    #
    #
    #     for col in df.columns:
    #
    #         vals_new = []
    #
    #         for i, row in tqdm(df.iterrows(), total=len(df), desc=f'append {col}'):
    #             pix = row['pix']
    #             r, c = pix
    #             if r<60:
    #                 continue
    #             vals = row[col]
    #             print(vals)
    #             if type(vals) == float:
    #                 vals_new.append(np.nan)
    #                 continue
    #             vals = np.array(vals)
    #             print(len(vals))
    #
    #             if len(vals) == 25:
    #
    #                 vals = np.append(vals, np.nan)
    #                 vals_new.append(vals)
    #
    #             vals_new.append(vals)
    #
    #             # exit()
    #         df[col] = vals_new

        # T.save_df(df, result_root + rf'\Multiregression_contribution\Obs\Dataframe\Dataframe.df')
        # T.df_to_excel(df, result_root + rf'\Multiregression_contribution\Obs\Dataframe\Dataframe.xlsx')

    def do_multi_regression(self,mode_name,x_list):
        outdir = self.outdir + f'{mode_name}\\'
        T.mk_dir(outdir, force=True)

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        import statsmodels.api as sm
        df = self.load_df()
        T.print_head_n(df)

        var_list = x_list  + [mode_name+'_detrend_CV_zscore']
        spatial_dict = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']

            # ---------- 1. 收集所有变量 ----------
            series_dict = {}
            valid = True

            for var in var_list:
                val = row[var]
                if not isinstance(val, (list, np.ndarray)) or len(val) < 10:
                    valid = False
                    break
                series_dict[var] = np.array(val)

            if not valid:
                continue

            # ---------- 2. 对齐公共长度（关键） ----------
            min_len = min(len(v) for v in series_dict.values())

            for var in series_dict:
                series_dict[var] = series_dict[var][-min_len:]

            # ---------- 3. 构造 DataFrame（一次性） ----------
            df_i = pd.DataFrame(series_dict)

            # ---------- 4. 标准化（和 Methods 一致） ----------


            # ---------- 5. 回归 ----------
            X = df_i[x_list]
            y = df_i[mode_name + '_detrend_CV_zscore']

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            # ---------- 6. 提取 β ----------
            beta_dict = {var: model.params[var] for var in x_list}
            spatial_dict[pix] = beta_dict
        df_beta = T.dic_to_df(spatial_dict, 'pix')


        for x_var in x_list:
            spatial_dict_i = T.df_to_spatial_dic(df_beta,x_var)
            outf = join(outdir,f'{x_var}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_i,outf)
        T.open_path_and_file(self.outdir)

        pass

    def do_multi_regression_control_experiment(self,mode_name,x_list):
        outdir = self.outdir+f'{mode_name}\\'
        T.mk_dir(outdir,force=True)

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        import statsmodels.api as sm
        df = self.load_df()
        T.print_head_n(df)

        var_list = x_list  + [mode_name+'_detrend_CV_zscore']
        spatial_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']

            # === 检查变量是否都存在并获取长度 ===
            length_dict = {}
            valid = True
            for var_i in var_list:
                val = row[var_i]
                if isinstance(val, float) or len(val) == 0:
                    valid = False
                    break
                length_dict[var_i] = len(val)

            # === 如果长度不匹配，则跳过该像素 ===
            if len(set(length_dict.values())) > 1:
                print(f"Length mismatch at pixel {row['pix']}:")
                for k, v in length_dict.items():
                    print(f"   {k}: length={v}")
                continue


            df_i = pd.DataFrame()
            success = 0

            for var_i in var_list:
                if type(row[var_i]) == float:
                    success = 0
                    break
                if len(row[var_i]) == 0:
                    success = 0
                    break
                else:
                    success = 1
                df_i[var_i] = row[var_i]

            if not success:
                continue
            print(df_i[x_list])
            X = df_i[x_list]

            y = df_i[mode_name+'_detrend_CV_zscore']

            X=sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            y_pred = model.predict(X)
            # delta_y = y - y_pred

            dict_i = {}

            for x_var in x_list:
                X_constant = copy.copy(X)


                X_constant[x_var] = X_constant[x_var][0]
                y_pred_i = model.predict(X_constant)
                delta_y = y_pred - y_pred_i
                delta_i = X[x_var] - X[x_var][0]
                model_i = sm.OLS(delta_y, sm.add_constant(delta_i)).fit()
                beta = model_i.params[1]
                dict_i[x_var] = beta
            spatial_dict[pix] = dict_i
        df_beta = T.dic_to_df(spatial_dict,'pix')


        for x_var in x_list:
            spatial_dict_i = T.df_to_spatial_dic(df_beta,x_var)
            outf = join(outdir,f'{x_var}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_i,outf)
        T.open_path_and_file(outdir)

    def calculate_trend_contribution(self, y_variable, x_list):
        """
        Calculate the trend contribution of each variable:
        contribution = slope(x, y) * trend(x) / trend(y) * 100
        """

        trend_dir = result_root + rf'\Multiregression_contribution\Obs\review\X_review\\zscore\\trend\\'
        trend_dict = {}

        # === Load trend for each X variable ===
        for variable in x_list:
            fpath = join(trend_dir, f'{variable}_trend.tif')


            array, _, _, _, _ = ToRaster().raster2array(fpath)
            array[array < -9999] = np.nan
            spatial_dict = DIC_and_TIF().spatial_arr_to_dic(array)

            for pix, val in tqdm(spatial_dict.items(), desc=f'Loading {variable} trend'):
                if np.isnan(val):
                    continue
                if pix[0] < 60:  # skip high latitude
                    continue
                if pix not in trend_dict:
                    trend_dict[pix] = {}
                trend_dict[pix][variable] = val

        # === Load multiregression slopes ===
        fdir_slope = self.outdir + f'\\{y_variable}\\'

        multiregression_dic = {}
        for f in os.listdir(fdir_slope):
            if not f.endswith('.tif') or 'contrib' in f:
                continue
            arr, _, _, _, _ = ToRaster().raster2array(join(fdir_slope, f))
            multiregression_dic[f.split('.')[0]] = DIC_and_TIF().spatial_arr_to_dic(arr)


        # === Load Y trend and p-value ===
        fdir_Y = result_root + rf'\Multiregression_contribution\Obs\input\Y\zscore\trend\\'
        fy_trend = join(fdir_Y, f'{y_variable}_detrend_CV_zscore_trend.tif')
        fy_pval = join(fdir_Y, f'{y_variable}_detrend_CV_zscore_p_value.tif')

        arr_y_trend, _, _, _, _ = ToRaster().raster2array(fy_trend)
        arr_y_pval, _, _, _, _ = ToRaster().raster2array(fy_pval)

        dic_y_trend = DIC_and_TIF().spatial_arr_to_dic(arr_y_trend)
        dic_y_pval = DIC_and_TIF().spatial_arr_to_dic(arr_y_pval)

        # === Calculate contribution ===
        for var_i in x_list:
            if var_i not in multiregression_dic:
                print(f"Missing slope for {var_i}")
                continue

            spatial_dic = {}
            for pix in tqdm(multiregression_dic[var_i], desc=f'Calculating {var_i} contribution'):
                if pix not in trend_dict or var_i not in trend_dict[pix]:
                    continue
                if pix not in dic_y_trend or pix not in dic_y_pval:
                    continue

                trend_y = dic_y_trend[pix]
                p_value = dic_y_pval[pix]
                if np.isnan(trend_y) or np.isnan(p_value):
                    continue
                if p_value > 0.05 or abs(trend_y) < 1e-6:
                    continue
                if trend_y < 0:
                    continue

                val_multireg = multiregression_dic[var_i][pix]
                if np.isnan(val_multireg) or val_multireg < -9999:
                    continue

                val_trend = trend_dict[pix][var_i]
                val_contrib = val_multireg * val_trend / trend_y * 100
                spatial_dic[pix] = val_contrib

            # === Output contribution map ===
            arr_contrib = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            outpath = join(fdir_slope, f'{var_i}_trend_contrib.tif')
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_contrib, outpath)





    def statistic_contribution_no_residual(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        dff = result_root + rf'Multiregression_contribution\Obs\review\Dataframe\\Statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)


        # === 配色方案 ===
        color_list = ['#a577ad', 'yellowgreen', 'Pink', '#f69E57']
        dark_colors = ['#774685', 'Olive', 'Salmon', '#c3646f']  # 可以改为你自定义的 darken_color 函数


        for model in self.model_list:
            print(model)
            # if not 'median' in model:
            #     continue
            # === 变量名 ===
            fixed_order = [
                f'{model}_sensitivity_zscore_trend_contrib',
                f'{model}_CV_daily_rainfall_average_zscore_trend_contrib',
                f'{model}_Precip_sum_detrend_CV_zscore_trend_contrib',

                f'{model}_VPD_detrend_average_zscore_trend_contrib'
            ]

            label_map = {
                f'{model}_sensitivity_zscore_trend_contrib': 'γ',
                f'{model}_Precip_sum_detrend_CV_zscore_trend_contrib': 'CV_inter',
                f'{model}_VPD_detrend_average_zscore_trend_contrib': 'VPD',
                f'{model}_CV_daily_rainfall_average_zscore_trend_contrib':'CV_intra'

            }

            means, sems, labels = [], [], []
            print(len(df))



            df = df[df[f'{model}_detrend_CV_zscore_trend'] > 0]
            df = df[df[f'{model}_detrend_CV_zscore_p_value'] < 0.05]
            #
            # print(len(df));exit()

            # === 计算平均值和标准误差 ===
            for var in fixed_order:
                if var not in df.columns:
                    continue
                vals = np.array(df[var].values, dtype=float)
                vals[(vals > 99) | (vals < -99)] = np.nan
                vals = vals[~np.isnan(vals)]
                # print(vals);exit()
                if len(vals) == 0:
                    continue

                mean_val = np.nanmean(vals)
                # print(np.std(vals));exit()
                sem_val = np.nanstd(vals) / np.sqrt(len(vals))  # 标准误差

                means.append(mean_val)
                sems.append(sem_val)
                labels.append(label_map[var])
            print(sems)
            # print(means);exit()
            # print(f'{model}:', means)

            # === 绘图 ===
            fig, ax = plt.subplots(figsize=(4, 3))
            x = np.arange(len(labels))
            colors = color_list
            edges = dark_colors

            bars = ax.bar(
                x, means, width=0.4,
                color=colors, edgecolor=edges, linewidth=1.2, zorder=2
            )

            # 误差线
            for xi, mean, sem, edge in zip(x, means, sems, edges):
                ax.errorbar(
                    xi, mean, yerr=sem,
                    fmt='none', ecolor=edge, elinewidth=1.2, capsize=4, zorder=3
                )

            # 美化
            ax.axhline(0, color='gray', linestyle='--', lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_ylabel('Attribution of CVLAI (%)', fontsize=12)
            ax.set_yticklabels(ax.get_yticks(), fontsize=12)
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['left'].set_linewidth(1)
            # ax.spines['bottom'].set_linewidth(1)
            ax.tick_params(axis='y', width=1, length=3)
            ax.tick_params(axis='x', width=1, length=0)
            # plt.tight_layout()

            # === 输出保存 ===
            outdir =result_root + rf'Multiregression_contribution\Obs\review\\Figure\\'
            print(outdir)
            #
            Tools().mk_dir(outdir, force=True)
            outf = os.path.join(outdir, f'{model}_relative_contribution_1mm.pdf')
            plt.savefig(outf, bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close()


    def darken_color(self,color, amount=0.7):
        """
        给颜色加深，amount 越小越深 (0~1之间)
        """
        import matplotlib.colors as mcolors
        c = mcolors.to_rgb(color)
        return tuple([max(0, x * amount) for x in c])


    def statistic_contribution_area(self):
        dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_5\statistics\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df = df.dropna(subset=['composite_LAI_color_map'])

        percentage_list=[]
        sum=0


        for ii in [1,2,3,4,5,6]:
            df_ii=df[df['composite_LAI_median_color_map']==ii]
            percent=len(df_ii)/len(df)*100
            sum=sum+percent
            percentage_list.append(percent)
        print(percentage_list)
        print(sum)

        ## plot

        color_list = [

            '#f599a1', '#fcd590',
            '#e73618', '#dae67a',
            '#9fd7e9', '#a577ad',

        ]



        plt.figure(figsize=(4,3))
        plt.bar([1,2,3,4,5,6], percentage_list, color=color_list)
        ## save fig
        plt.ylabel('Area precentage (%)')

        # plt.tight_layout()
        outdir=result_root + (rf'\3mm\FIGURE\\Robinson\\')
        T.mk_dir(outdir,force=True)
        plt.show()

        # plt.savefig(outdir+'Area_precentage_composite_LAI_mean.pdf',dpi=300)





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


    def load_df(self):
        dff=result_root+rf'\Multiregression_contribution\Obs\review\Dataframe\\Dataframe.df'

        df = T.load_df(dff)
        # exit()
        # start_year = 0
        # end_year = 21
        # variable_list = self.xvar + self.y_var
        # df = Dataframe_per_value_transform(df, variable_list, start_year, end_year).df
        # T.print_head_n(df)
        return df

    class Delta_regression_TRENDY:

        def __init__(self):
            self.map_width = 8.2 * centimeter_factor
            self.map_height = 8.2 * centimeter_factor
            self.xvar = ['Precip_sum_detrend_CV_zscore',
                         'CV_daily_rainfall_average_zscore',
                         'VPD_detrend_average_zscore']

            self.model_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                               'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                               'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                               'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                               'ORCHIDEE_S2_lai',

                               'YIBs_S2_Monthly_lai',

                               ]
            #
            # self.model_list = ['TRENDY_ensemble_median',
            #
            #                    ]

            # self.model_list = ['composite_LAI_median', 'LAI4g',
            #                    'GLOBMAP_LAI', 'SNU_LAI',
            #
            #                    'TRENDY_ensemble_median', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
            #                    'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
            #                    'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
            #                    'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
            #                    'ORCHIDEE_S2_lai',
            #
            #                    'YIBs_S2_Monthly_lai',
            #
            #                    ]

            self.outdir = rf'D:\Project3\Result\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg_3\\'
            T.mkdir(self.outdir, force=True)

            pass

        def run(self):
            df = self.load_df()
            # ##self.do_multi_regression()

            # for model in self.model_list:
            #     x_list=self.xvar+[model+'_sensitivity_zscore']
            #
            #     self.do_multi_regression_control_experiment(model,x_list) ## not use this but the result is the same
            # # #
            # # # #
            #     self.calculate_trend_contribution(model,x_list)
            # self.statistic_contribution()
            # self.ensemble_trend_contribution()
            # self.normalized_contribution()
            # self.Ternary_plot()
            # self.plot_pdf()

            # self.max_correlation_with_sign()
            # self.statistic_contribution_area_heatmap()
            # self.statistic_contribution_area_barplot()

            # self.TRENDY_barplot()
            # self.TRENDY_barplot2()

            # self.maximum_contribution()
            # self.dominant_region_trends()
            # self.statistic_max_correlation()
            # self.Figure2_robinson()
            self.statistic_contribution_area()

            # self.sensitivity_vs_climate_factors()
            # self.statistic_contribution_area_barplot()

            # self.percentage_pft()
            # self.sensitivity_vs_climate_factors()
            # self.sensitivity_vs_climate_factors_2()

            # self.normalized_partial_corr()

            # self.heatmap2()
            # self.calculate_mean()

            pass

        def do_multi_regression(self):
            self.outdir = self.outdir

            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            import statsmodels.api as sm
            df = self.load_df()
            T.print_head_n(df)

            var_list = self.xvar + y_var
            spatial_dict = {}
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                df_i = pd.DataFrame()

                # 构造时序 DataFrame
                valid = True
                for var in var_list:
                    val = row[var]
                    if not isinstance(val, (list, np.ndarray)) or len(val) == 0:
                        valid = False
                        break
                    df_i[var] = val

                if not valid:
                    continue

                X = df_i[self.xvar]
                y = df_i[self.y_var[0]]

                # 标准多元回归
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()

                # 提取 β 系数（剔除常数项）
                beta_dict = {var: model.params[var] for var in self.xvar}
                spatial_dict[pix] = beta_dict

            df_beta = T.dic_to_df(spatial_dict, 'pix')

            for x_var in self.xvar:
                spatial_dict_i = T.df_to_spatial_dic(df_beta, x_var)
                outf = join(self.outdir, f'{x_var}.tif')
                DIC_and_TIF().pix_dic_to_tif(spatial_dict_i, outf)
            T.open_path_and_file(self.outdir)

            pass

        def do_multi_regression_control_experiment(self, mode_name, x_list):
            outdir = self.outdir + f'{mode_name}\\'
            T.mk_dir(outdir, force=True)

            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            import statsmodels.api as sm
            df = self.load_df()
            T.print_head_n(df)

            var_list = x_list + [mode_name + '_detrend_CV_zscore']
            spatial_dict = {}
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                df_i = pd.DataFrame()
                success = 0
                for var_i in var_list:
                    if type(row[var_i]) == float:
                        success = 0
                        break
                    if len(row[var_i]) == 0:
                        success = 0
                        break
                    else:
                        success = 1
                    if len(row[var_i]) != 24:
                        success = 0
                        break

                    df_i[var_i] = row[var_i]
                if not success:
                    continue
                X = df_i[x_list]
                ### add interaction terms

                y = df_i[mode_name + '_detrend_CV_zscore']

                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                y_pred = model.predict(X)
                # delta_y = y - y_pred

                dict_i = {}

                for x_var in x_list:
                    X_constant = copy.copy(X)

                    X_constant[x_var] = X_constant[x_var][0]
                    y_pred_i = model.predict(X_constant)
                    delta_y = y_pred - y_pred_i
                    delta_i = X[x_var] - X[x_var][0]
                    model_i = sm.OLS(delta_y, sm.add_constant(delta_i)).fit()
                    beta = model_i.params[1]
                    dict_i[x_var] = beta
                spatial_dict[pix] = dict_i
            df_beta = T.dic_to_df(spatial_dict, 'pix')

            for x_var in x_list:
                spatial_dict_i = T.df_to_spatial_dic(df_beta, x_var)
                outf = join(outdir, f'{x_var}.tif')
                DIC_and_TIF().pix_dic_to_tif(spatial_dict_i, outf)
            T.open_path_and_file(outdir)

        def calculate_trend_contribution(self, y_variable, x_list):
            ## here I would like to calculate the trend contribution of each variable
            ## the trend contribution is defined as the slope of the linear regression between the variable and the target variable mutiplied by trends of the variable
            ## load the trend of each variable
            ## load the trend of the target variable
            ## load multi regression result
            ## calculate the trend contribution
            trend_dir = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\Input\X_new\trend\\'

            selected_vairables_list = x_list

            trend_dict = {}
            for variable in selected_vairables_list:
                fpath = join(trend_dir, f'{variable}_trend.tif')
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

            outdir = self.outdir + f'\\{y_variable}\\'

            fdir_slope = outdir

            multiregression_dic = {}
            for f in os.listdir(fdir_slope):
                if not f.endswith('.tif'):
                    continue
                if 'contrib' in f:
                    continue

                arr_multiregression, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
                    join(fdir_slope, f))
                dic_multiregression = DIC_and_TIF().spatial_arr_to_dic(arr_multiregression)
                multiregression_dic[f.split('.')[0]] = dic_multiregression

                # exit()
            for var_i in x_list:
                spatial_dic = {}
                for pix in tqdm(dic_multiregression, desc=var_i):
                    if not pix in trend_dict:
                        continue

                    vals = multiregression_dic[var_i][pix]
                    if vals < -9999:
                        continue

                    val_multireg = vals
                    if var_i not in trend_dict[pix]:
                        continue

                    val_trend = trend_dict[pix][var_i]
                    val_contrib = val_multireg * val_trend
                    spatial_dic[pix] = val_contrib
                arr_contrib = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
                plt.imshow(arr_contrib, cmap='RdBu', interpolation='nearest')
                plt.colorbar()
                # plt.title(var_i)
                # plt.show()

                DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_contrib, join(outdir, f'{var_i}_contrib.tif'))

        def ensemble_trend_contribution(self):

            model_list = self.model_list

            fdir = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\X\\'
            arr_list = []

            arr_sensitivity = []

            for model in model_list:
                fdir_i = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\X\\' + model + '\\'
                for f in os.listdir(fdir_i):
                    if not f.endswith('.tif'):
                        continue
                    if 'sensitivity_zscore' in f:
                        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_i, f))
                        arr_sensitivity.append(arr)
            arr_sensitivity = np.nanmedian(arr_sensitivity, axis=0)
            arr_sensitivity[arr_sensitivity > 99] = np.nan
            arr_sensitivity[arr_sensitivity < -99] = np.nan
            plt.imshow(arr_sensitivity, cmap='RdYlGn')
            plt.colorbar()
            plt.show()
            outdir = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg_2\\TRENDY_ensemble_median\\'
            T.mk_dir(outdir, force=True)
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_sensitivity, outdir + f'TRENDY_ensemble_sensitivity_zscore.tif')
            #

            for variable in self.xvar:
                for model in model_list:
                    fpath = join(fdir, model, f'{variable}.tif')
                    arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                    arr[arr > 99] = np.nan
                    arr[arr < -99] = np.nan

                    arr_list.append(arr)

                arr_ensemble = np.nanmedian(arr_list, axis=0)
                arr_ensemble[arr_ensemble > 99] = np.nan
                arr_ensemble[arr_ensemble < -99] = np.nan
                plt.imshow(arr_ensemble, cmap='RdYlGn')
                plt.colorbar()
                plt.show()
                outdir = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg_2\\TRENDY_ensemble_median\\'
                T.mk_dir(outdir, force=True)

                DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_ensemble,
                                                      outdir + f'TRENDY_ensemble_{variable}.tif')

        def normalized_contribution(self):

            fdir_all = self.outdir

            for model in self.model_list:

                spatial_dicts = {}
                variables_list = []

                outdir = join(fdir_all, model)

                for f in os.listdir(join(fdir_all, model)):
                    if not f.endswith('.tif'):
                        continue
                    if 'contrib' in f:
                        continue
                    if 'norm' in f:
                        continue

                    if 'Ternary_plot' in f:
                        continue

                    print(f)
                    fpath = join(fdir_all, model, f)
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

                for var_i in variables_list:
                    dic_norm = T.df_to_spatial_dic(df_abs, f'{var_i}_norm', )
                    DIC_and_TIF().pix_dic_to_tif(dic_norm, join(outdir, f'{var_i}_norm.tif'))
                ######T.save_df(df_abs,join(fdir,'df_normalized.df'));exit()

                ## df to dic

                # T.print_head_n(df_abs);exit()

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
                    CV_IAV_norm = row[f'detrended_sum_rainfall_growing_season_zscore_norm']
                    rainfall_frequency_norm = row[f'rainfall_frenquency_zscore_norm']
                    composite_LAI_beta_mean_norm = row[f'{model}_sensitivity_zscore_norm']
                    x, y, z = CV_IAV_norm, rainfall_frequency_norm, composite_LAI_beta_mean_norm
                    color = Ter.get_color(x, y, z)
                    color = color * 255
                    color = np.array(color, dtype=np.uint8)
                    alpha = 255
                    color = np.append(color, alpha)
                    # print(color);exit()

                    rgb_arr[r][c] = color
                # xymap.GDAL_func().ar
                rgb_arr = np.array(rgb_arr, dtype=np.uint8)
                ### - 蓝绿色（上）： 主导
                # - 橙黄色（左下）： 主导
                # - 粉紫色（右下）：LAI_sensitivity（植被敏感性）主导
                outtif = join(outdir, 'Ternary_plot.tif')
                # tif_template = join(fdir_all, model, f'rainfall_frenquency_zscore.tif')
                tif_template = result_root + rf'3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg\CABLE-POP_S2_lai\\rainfall_frenquency_zscore.tif'
                print(rgb_arr)

                xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
                grid_triangle_legend = Ter.grid_triangle_legend()
                plt.imshow(grid_triangle_legend)
                plt.show()
                # # T.open_path_and_file(fdir)
                # exit()

        def max_correlation_with_sign(self):

            dff = result_root + rf'3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_5\\statistics\\statistics.df'
            df = T.load_df(dff)
            df = self.df_clean(df)
            model_list = self.model_list
            # for col in df.columns:
            #     print(col)
            # exit()

            var_list = ['detrended_sum_rainfall_ecosystem_year_CV_zscore',
                        'CV_intraannual_rainfall_ecosystem_year_zscore',
                        'sensitivity_zscore', ]

            for model in tqdm(model_list):

                outdir = self.outdir + model + '\\'
                T.mk_dir(outdir, force=True)
                var_list_contrib = [f'{model}_' + i + '_contrib' for i in var_list]

                var_list_sens = [f'{model}_' + i for i in var_list]

                max_var_list = []
                max_var_sign_list = []
                color_list = []
                df_continent = df
                for i, row in df_continent.iterrows():
                    vals_contrib = row[var_list_contrib].tolist()
                    vals_contrib = np.array(vals_contrib)
                    vals_contrib[vals_contrib < -10] = np.nan
                    vals_contrib[vals_contrib > 10] = np.nan

                    vals_sens = row[var_list_sens].tolist()
                    vals_sens = np.array(vals_sens)
                    vals_sens[vals_sens < -10] = np.nan
                    vals_sens[vals_sens > 10] = np.nan

                    if True in np.isnan(vals_contrib):
                        max_var_list.append(np.nan)
                        max_var_sign_list.append(np.nan)
                        color_list.append(np.nan)
                        continue
                    vals_contri_abs = np.abs(vals_contrib)
                    vals_contrib_dict = T.dict_zip(var_list_contrib, vals_contri_abs)
                    vals_sens_dic = T.dict_zip(var_list_sens, vals_sens)

                    max_var = T.get_max_key_from_dict(vals_contrib_dict)
                    max_var_new_dict = {f'{model}_detrended_sum_rainfall_ecosystem_year_CV_zscore_contrib':
                                            f'{model}_detrended_sum_rainfall_ecosystem_year_CV_zscore',
                                        f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore_contrib':
                                            f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore',
                                        f'{model}_sensitivity_zscore_contrib':
                                            f'{model}_sensitivity_zscore',
                                        }

                    max_val = vals_sens_dic[max_var_new_dict[max_var]]

                    if max_val > 0:
                        max_var_sign = '+'
                    else:
                        max_var_sign = '-'

                    if 'sensitivity' in max_var:
                        if max_var_sign == '-':
                            color = 1
                        else:
                            color = 6
                    elif 'CV_intraannual_rainfall_ecosystem_year_zscore' in max_var:
                        if max_var_sign == '-':
                            color = 2
                        else:
                            color = 5
                    elif 'detrended_sum_rainfall_ecosystem_year_CV_zscore' in max_var:
                        if max_var_sign == '-':
                            color = 3
                        else:
                            color = 4
                    else:

                        continue
                    max_var_list.append(max_var)
                    max_var_sign_list.append(max_var_sign)
                    color_list.append(color)

                df_continent['max_var'] = max_var_list

                df_continent['max_var_sign'] = max_var_sign_list
                df_continent['color'] = color_list

                ## to tiff
                spatial_dic = T.df_to_spatial_dic(df, 'color')
                outtif = join(outdir, 'color_map.tif')
                array = DIC_and_TIF().pix_dic_to_tif(spatial_dic, outtif)
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                # plt.imshow(arr, interpolation='nearest')
                # plt.colorbar()
                # plt.show()

        def statistic_contribution_area_heatmap(self):
            dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_4\\Dataframe\\statistics.df'
            df = T.load_df(dff)
            df = self.df_clean(df)
            for col in df.columns:
                print(col)
            # df = df.dropna(subset=['composite_LAI_color_map'])
            model_list = self.model_list

            heatmap_cols = ['TRENDY_ensemble'] + model_list

            row_order = [1, 6, 2, 5, 3, 4]
            heatmap_data = pd.DataFrame(index=[f'Group {i}' for i in row_order], columns=heatmap_cols)

            for ii in [1, 6, 2, 5, 3, 4]:
                percentage_list = []

                for model in model_list:
                    sum = 0

                    df_mask = df.dropna(subset=['composite_LAI_color_map'])
                    # df_mask=df_mask.dropna(subset=[f'{model}_color_map'])

                    df_mask1 = df.dropna(subset=[f'{model}_color_map'])

                    # tmp = df_mask[[f'{model}_color_map', 'composite_LAI_color_map']]
                    # ### filter data

                    df_ii = df_mask1[df_mask1[f'{model}_color_map'] == ii]
                    df_obs = df_mask[df_mask['composite_LAI_color_map'] == ii]

                    percent_ii = len(df_ii) / len(df_mask1) * 100
                    percent_obs = len(df_obs) / len(df_mask) * 100
                    percent_diff = percent_ii - percent_obs

                    percentage_list.append(percent_diff)

                ensemble_mean = float(np.nanmean(percentage_list)) if len(percentage_list) else np.nan
                heatmap_data.loc[f'Group {ii}'] = [ensemble_mean] + percentage_list

            ## plot

            heatmap_data = heatmap_data.astype(float)

            # 设置颜色顺序（可选）

            plt.figure(figsize=(9, 4))

            sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdBu",
                        cbar_kws={'label': 'Models-obs difference (%)'},
                        linewidths=0.3, vmin=-15, vmax=15)

            dic_label_name = {

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
            dic_variable_name = {1: '$\gamma$-',
                                 6: '$\gamma$+',
                                 2: 'Fq of rainfall-',
                                 5: 'Fq of rainfall+',

                                 3: 'CV Interannual rainfall-',

                                 4: 'CV Interannual rainfall+',

                                 }

            ax = plt.gca()
            ax.set_xticklabels([dic_label_name.get(k, k) for k in heatmap_data.columns],
                               rotation=90, fontsize=10, font='Arial')
            ax.set_yticklabels([dic_variable_name[i] for i in row_order],
                               rotation=0, fontsize=10, font='Arial')
            # plt.tight_layout()
            plt.show()
            # plt.savefig(result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_3\\heatmap_contribution_area.pdf')

        def statistic_contribution_area_barplot(self):
            dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_5\\statistics\\statistics.df'
            df = T.load_df(dff)
            df = self.df_clean(df)
            for col in df.columns:
                print(col)

            model_list = self.model_list

            result_dic = {}

            # —— 统计：各模型在每个组 ii 的面积百分比（分母用各自非空的样本）——
            for ii in [1, 6, 2, 5, 3, 4]:
                percentage_list = []
                for model in model_list:
                    col = f'{model}_color_map'
                    df_mask = df.dropna(subset=[col])  # 不要改写 df 本体
                    df_ii = df_mask[df_mask[col] == ii]
                    percent_ii = len(df_ii) / len(df_mask) * 100.0
                    percentage_list.append(percent_ii)
                result_dic[ii] = percentage_list
            pprint(result_dic)

            dic_variable_name = {1: '$\gamma$-',
                                 6: '$\gamma$+',
                                 2: 'CV rainfall intra-',
                                 5: 'CV rainfall intra+',

                                 3: 'CV rainfall inter-',

                                 4: 'CV rainfall inter+',

                                 }

            # 颜色：前四个为 obs，第五个（如 TRENDY ensemble）单独色，其余为统一色
            color_list = ['#ADC9E4', '#EBF0FC', '#EBF0FC', '#EBF0FC', '#dd736c'] \
                         + ['#F7DAD4'] * (len(model_list) - 5)

            # 用模型名作为行索引，便于对齐
            df_new = pd.DataFrame(result_dic, index=model_list)

            # —— 画图：每个 ii 一张图，obs 与 models 留间隔，第一根柱子的高度画虚线（只跨 models）——
            for ii in [1, 6, 2, 5, 3, 4]:
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
                ax.set_ylim(0, 45)
                plt.grid(axis='y', alpha=0.25)

                plt.show()

                #
                # plt.savefig(result_root + rf'\3mm\FIGURE\Figure5_comparison\barplot\\barplot_{ii}.pdf', dpi=300, bbox_inches='tight')
                # plt.close()

        def Figure2_robinson(self):

            fdir_trend = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg_3\\TRENDY_ensemble_median\\'
            temp_root = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg_3\\TRENDY_ensemble_median'
            outdir = result_root + rf'3mm\FIGURE\Robinson\\'
            T.mk_dir(outdir, force=True)
            T.mk_dir(temp_root, force=True)

            for f in os.listdir(fdir_trend):

                if not f.endswith('.tif'):
                    continue

                if not 'color_map' in f:
                    continue
                fpath = fdir_trend + f

                plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
                m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=6, is_discrete=True, colormap_n=7, )

                # plt.show()
                outf = outdir + f + '.pdf'
                plt.savefig(outf)
                plt.close()

        def TRENDY_barplot(self):
            dff = result_root + rf'3mm\Multiregression\partial_correlation\Obs\obs_climate\Dataframe\\partial_correlation.df'
            df = T.load_df(dff)
            df = self.df_clean(df)
            # df=df.dropna(axis=0, how='any')

            variables_list = ['composite_LAI',
                              'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                              'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                              'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                              'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                              'ORCHIDEE_S2_lai',

                              'YIBs_S2_Monthly_lai']
            result_dic = {}
            result_stats = {}
            values_beta_list = []
            CI_list = []

            for variable in variables_list:
                values_beta = df[f'{variable}_detrended_sum_rainfall_growing_season_zscore'].values
                values_beta = np.array(values_beta)
                values_beta[values_beta > 100] = np.nan
                values_beta[values_beta < -100] = np.nan

                n = len(values_beta)
                confidence = 0.95
                std = np.nanstd(values_beta)
                t_critical = stats.t.ppf((1 + confidence) / 2., n - 1)
                margin_of_error = t_critical * std / np.sqrt(n)
                ci_lower = np.nanmean(values_beta) - margin_of_error
                ci_upper = np.nanmean(values_beta) + margin_of_error
                CI_list.append([ci_lower, ci_upper])

                values_beta_list.append(values_beta)
            CI_list = np.array(CI_list)
            CI_list_T = CI_list.T

            values_beta_list = np.array(values_beta_list)

            values_beta_list_mean = np.nanmean(values_beta_list, axis=1)
            values_beta_list_std = np.nanstd(values_beta_list, axis=1)

            # add legend
            df_new = pd.DataFrame(result_dic)

            fig, ax = plt.subplots(figsize=(self.map_width * 1.5, self.map_height))
            dic_label_name = {'composite_LAI': 'Composite LAI',

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
            plt.bar(variables_list, values_beta_list_mean, color='#96cccb', width=0.7, edgecolor='black',
                    label='Trend in Beta', yerr=CI_list_T[1] - CI_list_T[0], capsize=3)
            ## CI bar

            plt.xticks(np.arange(len(variables_list)), variables_list, rotation=45)
            plt.ylim(-.5, .5)
            ## add y=0
            plt.hlines(0, -0.5, len(variables_list) - 0.5, colors='black', linestyles='dashed')
            plt.ylabel('Beta (%/100ppm/yr)')
            plt.axhline(y=0, color='grey', linestyle='-')
            ax.set_xticks(range(len(variables_list)))
            ax.set_xticklabels(dic_label_name.values(), rotation=90, fontsize=10, font='Arial')
            plt.tight_layout()
            plt.show()

        def Ternary_plot(self):
            import numpy as np
            import matplotlib.pyplot as plt
            import mpltern

            dff = result_root + rf'3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\Dataframe\\statistic.df'
            df = T.load_df(dff)
            df = self.df_clean(df)
            # df=df.dropna(axis=0, how='any')
            obs_list = ['composite_LAI', 'SNU_LAI', 'GLOBMAP_LAI', 'LAI4g']
            new_model_list = self.model_list + obs_list

            result = []
            for model in new_model_list:
                gamma = df[f'{model}_sensitivity_zscore_contrib'].abs()  # gamma
                cv_iav = df[f'{model}_rainfall_frenquency_zscore_contrib'].abs()  # CV_IAV rainfall
                rfq = df[f'{model}_detrended_sum_rainfall_growing_season_zscore_contrib'].abs()  # rainfall frequency

                gamma_mean = gamma.mean()  # gamma
                cv_iav_mean = cv_iav.mean()  # CV_IAV rainfall
                rfq_mean = rfq.mean()  # rainfall frequency
                result.append([model, gamma_mean, rfq_mean, cv_iav_mean])

            df_summary = pd.DataFrame(result, columns=["model", "Gamma", "rainfall_frenquency", "CV_IAV"])

            # ------- 取绝对值 -------
            cols = ["Gamma", "rainfall_frenquency", "CV_IAV"]
            df_abs = df_summary[cols].abs()

            # ------- 归一化 -------
            sums = df_abs.sum(axis=1).replace(0, np.nan)  # 避免除0
            df_summary["Gamma_normalize"] = df_abs["Gamma"] / sums
            df_summary["rainfall_frenquency_normalize"] = df_abs["rainfall_frenquency"] / sums
            df_summary["CV_IAV_normalize"] = df_abs["CV_IAV"] / sums

            # 把 sum=0 的情况填回 0
            df_summary[["Gamma_normalize", "rainfall_frenquency_normalize", "CV_IAV_normalize"]] = \
                df_summary[["Gamma_normalize", "rainfall_frenquency_normalize", "CV_IAV_normalize"]].fillna(0)

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(projection="ternary")

            # 轴标签
            ax.set_tlabel("CV IAV rainfall")  # 顶边
            ax.set_llabel("rainfall frequency")  # 左边
            ax.set_rlabel("Gamma")  # 右边

            ax.grid(True, linestyle=":", alpha=0.5)

            # 按照 (t, l, r) 顺序传值
            ax.scatter(
                df_summary["CV_IAV_normalize"],  # t = 顶边
                df_summary["rainfall_frenquency_normalize"],  # l = 左边
                df_summary["Gamma_normalize"],  # r = 右边
                s=90, c="C0", edgecolor="k"
            )

            # 标注模型名
            for i, row in df_summary.iterrows():
                ax.text(
                    row["CV_IAV_normalize"],
                    row["rainfall_frenquency_normalize"],
                    row["Gamma_normalize"],
                    row["model"], fontsize=9,
                    ha="center", va="center"
                )

            plt.show()

        def plot_pdf(self):
            dff = result_root + rf'3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\Dataframe\\statistic.df'
            df = T.load_df(dff)
            df = self.df_clean(df)
            # df=df.dropna(axis=0, how='any')
            obs_list = ['composite_LAI', 'SNU_LAI', 'GLOBMAP_LAI', 'LAI4g']
            new_model_list = self.model_list + obs_list
            flag = 0

            fig, axes = plt.subplots(3, 6, figsize=(12, 18))  # Adjust figsize if too tight
            axes = axes.flatten()
            self.model_list = ['composite_LAI', 'GLOBMAP_LAI', 'LAI4g', 'SNU_LAI', 'TRENDY_ensemble',
                               'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                               'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                               'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                               'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                               'ORCHIDEE_S2_lai', 'YIBs_S2_Monthly_lai',

                               ]

            for model in new_model_list:

                result = {'gamma': df[f'{model}_sensitivity_zscore_contrib'].to_list(),
                          'cv_iav': df[f'{model}_rainfall_frenquency_zscore_contrib'].to_list(),  # CV_IAV rainfall
                          'rfq': df[f'{model}_detrended_sum_rainfall_growing_season_zscore_contrib'].to_list(),
                          # rainfall frequency
                          }

                ## all model plot in the same layout
                ax = axes[flag]

                for var_name, values in result.items():
                    if flag >= len(axes):
                        break

                    arr = np.array(values)
                    arr[arr > 99] = np.nan
                    arr[arr < -99] = np.nan
                    arr = arr * 100
                    arr = arr[~np.isnan(arr)]
                    mean_val = np.mean(arr)
                    # ax.axvline(mean_val, linestyle='--', linewidth=1, alpha=0.8)

                    # sns.kdeplot(arr, fill=False, linewidth=2,label=var_name,ax=ax)
                    sns.ecdfplot(arr, label=var_name, ax=ax, linewidth=2, )
                ax.set_xlim(-30, 30)
                ax.set_ylabel('')
                ax.grid(True)
                ax.set_title(model)
                ax.legend(fontsize=6)

                # plt.grid(True)

                flag = flag + 1

                #
                #
            plt.legend()
            plt.show()

            pass

        def statistic_contribution_area(self):
            dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_5\\statistics\\statistics.df'
            df = T.load_df(dff)
            df = self.df_clean(df)

            percentage_list = []
            sum = 0

            for ii in [1, 2, 3, 4, 5, 6]:
                # df = df.dropna(subset=['TRENDY_ensemble_median_color_map'])
                # df_ii = df[df['TRENDY_ensemble_median_color_map'] == ii]
                # df = df.dropna(subset=['composite_LAI_median_color_map'])
                # df_ii = df[df['composite_LAI_median_color_map'] == ii]

                percent = len(df_ii) / len(df) * 100
                sum = sum + percent
                percentage_list.append(percent)
            print(percentage_list)
            print(sum);
            # exit()

            ## plot

            color_list = [

                '#f599a1', '#fcd590',
                '#e73618', '#dae67a',
                '#9fd7e9', '#a577ad',

            ]

            plt.figure(figsize=(3, 3))
            plt.bar([1, 2, 3, 4, 5, 6], percentage_list, color=color_list)

            plt.ylabel('Area precentage (%)')
            # plt.show()
            outdir = result_root + rf'\3mm\FIGURE\Robinson\\'
            plt.savefig(outdir + '\\statistics_contribution_area_model.pdf', dpi=300)
            plt.close()

            pass

            pass

        def sensitivity_vs_climate_factors(self):
            dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_3\Dataframe\\statistics.df'
            df = T.load_df(dff)
            df = self.df_clean(df)
            df.dropna(inplace=True)

            # 设置变量名
            target_var_list = [
                'composite_LAI_detrended_sum_rainfall_growing_season_zscore',
                'composite_LAI_rainfall_frenquency_zscore']

            color_list = ['#dd492c', '#c05f77', '#b5869a',
                          '#bfa8b1', '#d5c1ca', '#e7dce1',
                          ]

            sand_color_list = ['#ffffe5', '#fffaca', '#fff0ae', '#fee391', '#fece65',
                               '#feb642', '#fe9929', '#f27e1b', '#e1640e', '#cc4c02', '#aa3c03', '#882f05', '#662506']
            aridity_color_list = ['#d73027', '#ea6f44', '#f48e52', '#fec279', '#fed690',
                                  '#ffffbf', '#f7fccd', '#f0f9dc', '#e8f6ea', '#e0f3f8']
            root_depth_list = [
                '#f1e0b6',
                '#e6c981',
                '#d9b15a',
                '#c89b43',
                '#a88432',
                '#7f6d28',
                '#4f7f3b',
                '#2e7031',
                '#00441b'  # 深根
            ]
            short_vegetation_cover = ['#ffffcc',  # 极低覆盖
                                      '#d9f0a3',
                                      '#addd8e',
                                      '#78c679',
                                      '#41ab5d',
                                      '#238443',
                                      '#006837',
                                      '#004529']

            for target_var in target_var_list:

                # bin_var = 'Burn_area_sum'
                # bin_var = 'S_SAND'

                bin_var = 'SOC'
                # bin_var = 'sum_rainfall_mean'
                # bin_var = 'Tree cover_mean'
                # bin_var='Non_tree_vegetation_mean'
                # bin_var = 'rooting_depth_05'
                plt.hist(df[bin_var])
                plt.show()
                # bin_edges = np.arange(0, 101, 10)
                # bin_edges = np.arange(200,1201,100)
                # bin_edges = np.arange(0,501,50)
                # bin_edges = np.arange(0, 4000, 500)
                # bin_edges=np.arange(10,91,10)
                # bin_edges = np.arange(0.2, 0.66, 0.05)
                # bin_edges = np.arange(150,850,50)
                bin_edges = np.arange(0, 0.5, 0.05)
                # bin_edges = np.quantile(df[bin_var], np.linspace(0, 0.66, 11))
                bin_labels = [f'{round(bin_edges[i], 2)}-{round(bin_edges[i + 1], 2)}' for i in
                              range(len(bin_edges) - 1)]
                # bin_labels = [f'{round(bin_edges[i ], 2)}' for i in range(len(bin_edges) - 1)]

                df['bin'] = pd.cut(df[bin_var], bins=bin_edges, labels=bin_labels, include_lowest=True)

                # 初始化结果字典
                result_dic = {}
                count_list = []

                for label in bin_labels:
                    df_bin = df[df['bin'] == label][[target_var]].dropna()

                    if len(df_bin) == 0:
                        result_dic[label] = [0, 0, 0, 0]
                        continue

                    mean_val = np.nanmean(df_bin[target_var])
                    std_err = np.nanstd(df_bin[target_var]) / np.sqrt(len(df_bin))  # 标准误差
                    result_dic[label] = [mean_val, std_err]
                    count_list.append(len(df_bin))

                # 构造 DataFrame
                result_df = pd.DataFrame(result_dic).T
                result_df.columns = ['mean', 'std_err']
                result_df.index = bin_labels

                # 画图

                ax = result_df['mean'].plot(
                    kind='bar',
                    yerr=result_df['std_err'],
                    figsize=(4, 3),
                    color=aridity_color_list,
                    capsize=3,
                    error_kw={'elinewidth': 1, 'ecolor': 'gray'},
                    edgecolor='gray',
                )

                ## add count

                for i, bar in enumerate(ax.patches):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            height + 0.02,  # 微调高度

                            f'{count_list[i]}',
                            ha='center', va='bottom', fontsize=10, rotation=90)

                plt.axhline(y=0, color='gray', linestyle='-')
                ## xtick every 100 label
                xticks = ax.get_xticks()
                # xticklabels = [label.get_text() for label in ax.get_xticklabels()]
                # new_labels = [label if i % 2 == 0 else '' for i, label in enumerate(xticklabels)]
                # ax.set_xticklabels(new_labels, rotation=0)

                if target_var == 'detrended_sum_rainfall_CV_zscore_sensitivity':
                    plt.ylabel('CV Interannual Rainfall (zscore)')


                elif target_var == 'rainfall_frenquency_zscore_sensitivity':
                    plt.ylabel('Fq Rainfall(zscore)')

                if target_var == 'composite_LAI_beta_mean_zscore_contrib':
                    plt.xticks([])
                elif target_var == 'rainfall_frenquency_zscore_contrib':
                    plt.xticks([])
                else:
                    plt.xticks(rotation=45)

                #
                plt.tight_layout()
                plt.show()
                # ## save pdf
                # fig = ax.get_figure()
                # outdir=result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_2\figure\\'
                # T.mk_dir(outdir, force=True)
                # fig.savefig(outdir + f'{target_var}_{bin_var}.pdf', dpi=300, bbox_inches='tight')
                # plt.close(fig)

        def cohens_d(self, x, y):
            nx, ny = len(x), len(y)
            if nx < 2 or ny < 2:
                return np.nan
            pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
            return (np.nanmean(x) - np.nanmean(y)) / pooled if pooled > 0 else np.nan

        def box_plot_test(self):
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
            from scipy import stats

            # ---------- load & clean ----------
            dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_3\Dataframe\statistics.df'
            df0 = T.load_df(dff)
            df0 = self.df_clean(df0)

            # variables to compare across panels
            var_list = [
                'sum_rainfall_trend', 'SM_ecosystem_year_trend', 'pi_growing_season_trend',
                'sand_soil_grid', 'fire_ecosystem_year_sum', 'rooting_depth_05', ]
            # df_postive=df0[df0['composite_LAI_detrended_sum_rainfall_growing_season_zscore_correlation']>0]
            # df_negative=df0[df0['composite_LAI_detrended_sum_rainfall_growing_season_zscore_correlation']<0]

            df_postive = df0[df0['composite_LAI_rainfall_frenquency_zscore_correlation'] > 0]
            df_negative = df0[df0['composite_LAI_rainfall_frenquency_zscore_correlation'] < 0]
            ## ttest
            for var in var_list:
                vals_pos = df_postive[var].tolist()
                vals_pos_arr = np.array(vals_pos)
                vals_pos_arr = vals_pos_arr[~np.isnan(vals_pos_arr)]
                vals_neg = df_negative[var].tolist()
                vals_neg_arr = np.array(vals_neg)
                vals_neg_arr = vals_neg_arr[~np.isnan(vals_neg_arr)]
                t, p = stats.ttest_ind(vals_pos_arr, vals_neg_arr, equal_var=False)
                print(f'{var}: t={t:.3f}, p={p:.3f}')

        def filter_data(self, df):
            self.x_variable_range_dict_global_CRU = {
                'pi_growing_season_trend': [0, 7],
                'sum_rainfall_trend': [0, 7],
                'SM_ecosystem_year_trend': [0, 1500],
                'sand_soil_grid': [0, 800],
                'rooting_depth_05': [0, 25],
                'cwdx80_05_soil_grid': [0, 50], }

        def maximum_contribution(self):
            fdir = self.outdir
            array_dic_all = {}
            array_arg = {}

            var_name_list = []
            for f in os.listdir(fdir):
                if not f.endswith('.tif'):
                    continue
                if not 'contrib' in f:
                    continue
                if 'max_label' in f:
                    continue
                if 'Ternary_plot' in f:
                    continue
                var_name = f.split('.')[0]
                var_name_list.append(var_name)
                print(f)
                fpath = join(fdir, f)
                spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
                array_dic_all[var_name] = spatial_dict

            spatial_df = T.spatial_dics_to_df(array_dic_all)
            max_key_list = []
            max_val_list = []
            for i, row in spatial_df.iterrows():
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
            spatial_df.dropna()
            ## df to tif
            dic_label = {'CV_intraannual_rainfall_ecosystem_year_contrib': 2,
                         'detrended_sum_rainfall_CV_contrib': 3,
                         'composite_LAI_beta_mean_contrib': 1,

                         }

            spatial_df['max_label'] = spatial_df['max_key'].map(dic_label)
            # # ## calculate _percentage
            # for ii in range(1, 5):
            #     percent=spatial_df[spatial_df['max_label']==ii].shape[0]/spatial_df.shape[0]*100
            #     percent=round(percent,2)
            #     print(ii,percent)
            #
            #     plt.bar(ii,percent)
            # # plt.show()
            #
            #
            spatial_dict = T.df_to_spatial_dic(spatial_df, 'max_label')
            DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(spatial_dict, self.outdir + 'max_label.tif')

            dff_new = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\Dataframe\\Dataframe.df'
            df = T.load_df(dff_new)
            df = self.df_clean(df)
            df = df.dropna()

            pix_list = df['pix'].tolist()
            unique_pix_list = list(set(pix_list))
            spatial_dic = {}

            for pix in unique_pix_list:
                spatial_dic[pix] = 1
            arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
            plt.colorbar()
            plt.show()

            for ii in range(1, 4):
                percent = df[df['max_label'] == ii].shape[0] / df.shape[0] * 100
                percent = round(percent, 2)
                print(ii, percent)

                plt.bar(ii, percent)
            plt.show()

        def heatmap2(self):  ## plot trend as function of Aridity and precipitation trend
            ## plot trends as function of inter precipitaiton CV and intra precipitation CV
            dff = result_root + rf'3mm\Multiregression\partial_correlation\Obs\Dataframe\\statistics.df'
            df = T.load_df(dff)
            df = self.df_clean(df)
            # df.dropna(inplace=True)
            ###df =color map ==3 and 4
            # df = df[df['color_map'].isin([2, 5])]

            # df=df.dropna()
            T.print_head_n(df)
            x_var = 'sum_rainfall_trend'

            y_var = 'SM_trend'
            plt.hist(df[y_var])
            plt.show()
            plt.hist(df[x_var])
            plt.show()
            z_var = 'composite_LAI_detrended_sum_rainfall_growing_season_CV_zscore'
            # z_var='Fire_sum_average'

            bin_x = np.linspace(-4, 4, 6)
            bin_y = np.linspace(-0.001, 0.001, 6)

            # percentile_list=np.linspace(0,100,9)
            # bin_x=np.percentile(df[x_var],percentile_list)
            # print(bin_x)
            # bin_y=np.percentile(df[y_var],percentile_list)
            plt.figure()

            matrix_dict, x_ticks_list, y_ticks_list = T.df_bin_2d(df, val_col_name=z_var,
                                                                  col_name_x=x_var,
                                                                  col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
            self.plot_df_bin_2d_matrix(matrix_dict, -.5, .5, x_ticks_list, y_ticks_list, cmap='Viridis',
                                       is_only_return_matrix=False)
            plt.colorbar()
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.show()

            # plt.figure()

            matrix_dict, x_ticks_list, y_ticks_list = self.df_bin_2d_sample_size(df, val_col_name=z_var,
                                                                                 col_name_x=x_var,
                                                                                 col_name_y=y_var, bin_x=bin_x,
                                                                                 bin_y=bin_y)
            T.plot_df_bin_2d_matrix(matrix_dict, 0, 100, x_ticks_list, y_ticks_list, cmap='RdBu',
                                    is_only_return_matrix=False)

            plt.colorbar()
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.show()

        def plot_df_bin_2d_matrix(self, matrix_dict, vmin, vmax, x_ticks_list, y_ticks_list, cmap='RdBu',
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
            plt.imshow(matrix, cmap='RdBu', vmin=vmin, vmax=vmax)
            plt.xticks(range(len(c_list)), x_ticks_list)
            plt.yticks(range(len(r_list)), y_ticks_list[::-1])

        def df_bin_2d_sample_size(self, df, val_col_name, col_name_x, col_name_y, bin_x, bin_y, round_x=2, round_y=2):
            df_group_y, _ = T.df_bin(df, col_name_y, bin_y)
            matrix_dict = {}
            y_ticks_list = []
            x_ticks_dict = {}
            flag1 = 0
            for name_y, df_group_y_i in df_group_y:
                matrix_i = []
                y_ticks = (name_y[0].left + name_y[0].right) / 2
                y_ticks = np.round(y_ticks, round_y)
                y_ticks_list.append(y_ticks)
                df_group_x, _ = T.df_bin(df_group_y_i, col_name_x, bin_x)
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
            return matrix_dict, x_ticks_list, y_ticks_list

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

        def load_df(self):
            dff = rf'D:\Project3\Result\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\Dataframe\Dataframe.df'
            df = T.load_df(dff)
            # exit()
            # start_year = 0
            # end_year = 21
            # variable_list = self.xvar + self.y_var
            # df = Dataframe_per_value_transform(df, variable_list, start_year, end_year).df
            # T.print_head_n(df)
            return df


def main():
    Delta_regression().run()


    # partial_correlation_obs().run()


    pass

if __name__ == '__main__':
    main()
