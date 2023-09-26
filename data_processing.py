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
import datetime
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
class statistic_analysis():
    def __init__(self):
        pass
    def run(self):
        # self.trend_analysis()
        # self.detrend_zscore()
        # self.detrend_zscore_monthly()
        self.zscore()


    def trend_analysis(self):

        dic_mask_lc_file = data_root+'Base_data/LC_reclass2.npy'
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = data_root+'Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)



        period_list = ['early', 'peak', 'late', 'early_peak', 'early_peak_late']



        for period in period_list:

            fdir = data_root + rf'extraction_original_val\LAI\{period}\\'
            outdir = result_root + rf'trend_analysis\\original\\{period}\\'
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

    pass

    def detrend_zscore(self): #

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

    def detrend_zscore_monthly(self): #

        dic_mask_lc_file = data_root+'Base_data/LC_reclass2.npy'
        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = data_root+'/Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(
            tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)


        product_list = ['precip','Temp','SMroot','SMsurf','Et']


        for period in ['early', 'peak', 'late', 'early_peak']:

            for product in product_list:
                outdir = result_root + rf'detrend_zscore\\climate_monthly\{period}\\'
                outf=outdir+product+'.npy'
                # print(outf)
                # exit()
                f = result_root + rf'\extraction_original_val\Climate_monthly\\during_{period}_{product}.npy'

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
                    time_series = dic[pix][:19]
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

    def zscore(self):

        dic_mask_lc_file = data_root+'/Base_data/LC_reclass2.npy'

        dic_mask_lc = T.load_npy(dic_mask_lc_file)

        tiff_mask_landcover_change = data_root+'Base_data/lc_trend/max_trend.tif'

        array_mask_landcover_change, originX, originY, pixelWidth, pixelHeight = ToRaster(
            ).raster2array(tiff_mask_landcover_change)
        array_mask_landcover_change[array_mask_landcover_change * 20 > 10] = np.nan
        array_mask_landcover_change = DIC_and_TIF().spatial_arr_to_dic(array_mask_landcover_change)

        product_list = ['LAI4g','NDVI4g','GPP_CFE','GPP_baseline']

        outdir = result_root + rf'zscore\\'
        for product in product_list:
            outf = outdir + product + '.npy'
            # print(outf)
            # exit()
            f = data_root + rf'Extraction\\{product}.npy'

            Tools().mk_dir(outdir, force=True)
            dic = {}

            dic = dict(np.load(f, allow_pickle=True, ).item())

            zscore_dic = {}

            for pix in tqdm(dic):
                if pix not in dic_mask_lc:
                    continue

                val_lc_change = array_mask_landcover_change[pix]
                if val_lc_change < -9999:
                    continue
                if pix not in dic_mask_lc:
                    continue
                if dic_mask_lc[pix] == 'Crop':
                    continue
                if array_mask_landcover_change[pix] == np.nan:
                    continue
                # print(len(dic[pix]))
                time_series = dic[pix]
                # print(time_series)

                time_series = np.array(time_series)
                # plt.plot(time_series)
                # plt.show()

                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.nanmean(time_series) <= 0.:
                    continue

                delta_time_series = []
                mean = np.nanmean(time_series)
                std = np.nanstd(time_series)
                if std == 0:
                    continue
                delta_time_series = (time_series - mean) / std

                # plt.plot(delta_time_series)
                # plt.title(len(delta_time_series))
                # plt.show()

                zscore_dic[pix] = delta_time_series

            T.save_npy(zscore_dic, outf)

        pass
class build_dataframe():



    def __init__(self):

        self.this_class_arr = result_root + 'Dataframe\zscore\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'zscore.df'

        pass

    def run(self):

        df = self.__gen_df_init(self.dff)
        df=self.foo1(df)
        df = self.add_detrend_zscore_to_df(df)
        df=self.add_AI_classfication(df)
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
        f_name = f.split('\\')[-2] + '_' + f.split('\\')[-1].split('.')[0]
        print(f_name)

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 2002
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
                    v1 = vals[year - 1982]
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

        f = data_root + rf'\\Base_data\dryland_AI.tif\\dryland_classfication.npy'

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

        df['AI_classfication'] = val_list
        return df

def main():
    # statistic_analysis().run()
    build_dataframe().run()





    pass

if __name__ == '__main__':
    main()