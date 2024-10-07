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
result_root = 'E:\western US\Result/'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

class data_processing():
    def __init__(self):
        pass
    def run(self):
        # self.tif_to_dic()

        # self.extract_water_year_LAI4g()
        # self.extract_water_year_GPCC()
        self.relative_change()

        pass

    def tif_to_dic(self):

        fdir_all = rf'D:\Project3\Data\monthly_data\GPCC\\'


        year_list = list(range(1982, 2021))


        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            if not 'TIFF' in fdir:
                continue

            outdir=rf'E:\western US\Data\\monthly_data\\GPCC\\'
            T.mk_dir(outdir, force=True)
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




                all_array.append(array_unify)
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

    def extract_water_year_LAI4g(self):
        fdir_all = rf'E:\western US\Data\monthly_data\\'

        outdir = result_root + f'extract_water_year\\'
        Tools().mk_dir(outdir, force=True)


        for fdir in os.listdir(fdir_all):
            if not 'LAI4g' in fdir:
                continue

            spatial_dict = {}
            outf = outdir +'\\' + 'LAI4g_water_year.npy'

            if os.path.isfile(outf):
                continue
            print(outf)

            for f in os.listdir(fdir_all + fdir):
                spatial_dict_i = dict(np.load(fdir_all + fdir + '\\' + f, allow_pickle=True, ).item())
                spatial_dict.update(spatial_dict_i)

            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict):
                r, c = pix
                if r>240:
                    continue

                time_series = spatial_dict[pix]

                ##the same as GPCC
                time_series = time_series[10:466]
                time_series = np.array(time_series)
                time_series_reshape = np.reshape(time_series, (-1, 12))
                annual_mean_list = []
                for i in range(time_series_reshape.shape[0]):

                    annual_mean_list.append(np.mean(time_series_reshape[i, :]))

                annual_spatial_dict[pix] = annual_mean_list

            np.save(outf, annual_spatial_dict)





        pass



    def extract_water_year_GPCC(self):
        fdir_all = rf'E:\western US\Data\monthly_data\\'

        outdir = result_root + f'extract_water_year\\'
        Tools().mk_dir(outdir, force=True)


        for fdir in os.listdir(fdir_all):
            if not 'GPC' in fdir:
                continue

            spatial_dict = {}
            outf = outdir +'raw\\' + 'GPCC_water_year.npy'

            if os.path.isfile(outf):
                continue
            print(outf)

            for f in os.listdir(fdir_all + fdir):
                spatial_dict_i = dict(np.load(fdir_all + fdir + '\\' + f, allow_pickle=True, ).item())
                spatial_dict.update(spatial_dict_i)

            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict):
                r, c = pix
                if r>240:
                    continue

                time_series = spatial_dict[pix]
                ### from 1982 Nov to 2020 Oct
                time_series = time_series[10:466]
                time_series = np.array(time_series)
                time_series_reshape = np.reshape(time_series, (-1, 12))
                annual_sum_list = []
                for i in range(time_series_reshape.shape[0]):

                    annual_sum_list.append(np.sum(time_series_reshape[i, :]))

                annual_spatial_dict[pix] = annual_sum_list

            np.save(outf, annual_spatial_dict)

    def relative_change(self):

        # NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        # array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        # dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root + rf'\extract_water_year\raw\\'
        outdir = result_root + rf'extract_water_year\\relative_change\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            outf = outdir + f.split('.')[0] + '.npy'
            # if isfile(outf):
            #     continue
            print(outf)

            dic = T.load_npy(fdir + f)

            zscore_dic = {}

            for pix in tqdm(dic):
                delta_time_series_list = []
                # if pix not in dic_dryland_mask:
                #     continue

                # print(len(dic[pix]))
                time_series = dic[pix]
                # print(len(time_series))

                time_series = np.array(time_series)
                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue

                time_series = time_series
                mean = np.nanmean(time_series)
                relative_change = (time_series - mean) / abs(mean) * 100

                zscore_dic[pix] = relative_change
                ## plot
                # plt.plot(time_series)
                # plt.show()
                # plt.plot(relative_change)
                # plt.legend(['original','relative_change'])
                plt.show()

                ## save
            np.save(outf, zscore_dic)

        pass
class build_dataframe():


    def __init__(self):



        self.this_class_arr = (rf'E:\western US\Result\DataFrame\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'Dataframe.df'

        pass

    def run(self):


        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        # df=self.foo2(df)
        # df=self.add_multiregression_to_df(df)
        df=self.build_df(df)
        # df=self.build_df_monthly(df)
        # df=self.append_attributes(df)  ## 加属性
        # df=self.append_cluster(df)  ## 加属性
        # df=self.append_value(df)   ## insert or append value


        # df = self.add_detrend_zscore_to_df(df)
        # df=self.add_GPCP_lagged(df)
        # df=self.add_rainfall_characteristic_to_df(df)
        # df=self.add_lc_composition_to_df(df)


        # df=self.add_trend_to_df_scenarios(df)  ### add different scenarios of mild, moderate, extreme
        # df=self.add_trend_to_df(df)
        # df=self.add_mean_to_df(df)
        #
        df=self.add_AI_classfication(df)
        #
        df=self.add_aridity_to_df(df)
        # # # # # # #
        df=self.add_MODIS_LUCC_to_df(df)
        df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        df=self.add_landcover_classfication_to_df(df)
        df=self.add_maxmium_LC_change(df)
        df=self.add_row(df)
        df=self.add_continent_to_df(df)
        df=self.add_lat_lon_to_df(df)
        # df=self.add_soil_texture_to_df(df)
        # #
        # df=self.add_rooting_depth_to_df(df)
        #
        # df=self.add_area_to_df(df)


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
    def build_df(self, df):


        fdir=rf'E:\western US\Result\extract_water_year\\relative_change\\'
        all_dic= {}
        for f in os.listdir(fdir):


            fname= f.split('.')[0]

            fpath=fdir+f

            dic = T.load_npy(fpath)
            key_name=fname
            print(key_name)
            all_dic[key_name]=dic
        # print(all_dic.keys())
        df=T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df




    def append_attributes(self, df):  ## add attributes
        fdir = result_root+ rf'\extract_water_year\relative_change\\'
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
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


    def append_cluster(self, df):  ## add attributes
        dic_label = {'sig_greening_sig_wetting': 1, 'sig_browning_sig_wetting': 2, 'non_sig_greening_sig_wetting': 3,

                     'non_sig_browning_sig_wetting': 4, 'sig_greening_sig_drying': 5, 'sig_browning_sig_drying': 6,

                     'non_sig_greening_sig_drying': 7, 'non_sig_browning_sig_drying': 8, np.nan: 0}

        #### reverse
        dic_label = {v: k for k, v in dic_label.items()}


        fdir = result_root+rf'Dataframe\anomaly_trends\\'
        for f in os.listdir(fdir):
            if not f.endswith('tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)

            # array=np.load(fdir+f)
            dic = DIC_and_TIF().spatial_arr_to_dic(array)

            key_name='label'
            for k in dic:
                if dic[k] <-99:
                    continue
                dic[k]=dic_label[dic[k]]

            df=T.add_spatial_dic_to_df(df,dic,key_name)

        return df






    def append_value(self, df):  ##补齐
        fdir = result_root + rf'growth_rate\\\growth_rate_raw\\'
        col_list=[]
        for f in os.listdir(fdir):
            if not 'LAI4g' in f:
                continue

            if not f.endswith('.npy'):
                continue


            col_name=f.split('.')[0]+'_growth_rate_raw'

            col_list.append(col_name)

        for col in col_list:
            vals_new=[]

            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'append {col}'):
                pix = row['pix']
                r, c = pix
                vals=row[col]
                if type(vals)==float:
                    vals_new.append(np.nan)
                    continue
                vals=np.array(vals)
                # if len(vals)==23:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     # print(len(vals))
                # elif len(vals)==38:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     print(len(vals))
                if len(vals)==38:

                    # vals=np.append(vals,np.nan)
                    ## append at the beginning
                    vals = np.insert(vals, 0, np.nan)


                vals_new.append(vals)

                # exit()
            df[col]=vals_new

        return df

        pass


    def foo1(self, df):

        f = result_root + rf'growth_rate\growth_rate_trend_method2\\LAI4g.npy'
        # array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        # array = np.array(array, dtype=float)
        # dic = DIC_and_TIF().spatial_arr_to_dic(array)


        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 1983
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y)
                y += 1


        df['pix'] = pix_list

        df['year'] = year
        # df['window'] = 'VPD_LAI4g_00'
        df['LAI4g_growth_rate'] = change_rate_list
        return df

    def foo2(self, df):  # 新建trend

        f = result_root + rf'Result\monte_carlo_trend\difference\\monte_carlo_cold_dry_year_LAI_diff.tif'
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

    def add_multiregression_to_df(self, df):
        fdir = result_root + rf'multi_regression\1982_2020\\'
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if not 'CO2' in f:
                continue
            print(f.split('.')[0])

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
            array = np.array(array, dtype=float)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                val_list.append(val)
            df[f.split('.')[0]] = val_list
        return df



        pass

    def add_GPCP_lagged(self,df): ##
        fdir=result_root+rf'\extract_GS\OBS_LAI_extend\\'
        for f in os.listdir(fdir):
            if f.split('.')[0] not in ['GPCC','CRU','tmax']:
                continue

            spatial_dic = T.load_npy(fdir+f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year

                pix = row['pix']
                r, c = pix

                if not pix in spatial_dic:
                    NDVI_list.append(np.nan)
                    continue

                vals = spatial_dic[pix][1:]


                v1=vals[year-1983]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)
            col_name=f.split('.')[0]
            print(col_name)
            df[col_name]=NDVI_list

            return df
            pass

    def add_detrend_zscore_to_df(self, df):

        fdir=result_root+rf'\relative_change\OBS_LAI_extend\\'
        for f in os.listdir(fdir):
            if not 'CO2' in f:
                continue



            variable= f.split('.')[0]
            # if variable not in ['CRU','GLEAM_SMroot','VPD','tmax']:
            #     continue

            print(variable)


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
                # vals[vals<0]=np.nan
                # vals[vals>1500]=np.nan


                # print(len(vals))
                ##### if len vals is 38, the end of list add np.nan

                # if len(vals) == 38:
                #     vals=np.append(vals,np.nan)
                #     v1 = vals[year - 1982]
                #     NDVI_list.append(v1)
                # if len(vals)==39:
                # v1 = vals[year - 1982]
                # v1 = vals[year - 1982]
                # if year < 2000:  ## fillwith nan
                #     NDVI_list.append(np.nan)
                #     continue



                v1= vals[year - 1982]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)


            df[f'{variable}_relative_change'] = NDVI_list
        # exit()
        return df

    def add_rainfall_characteristic_to_df(self, df):
        fdir = rf'E:\Data\ERA5_precip\ERA5_daily\dict\dry_spell\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            variable= f.split('.')[0]

            print(variable)


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

                # print(len(vals))
                ##### if len vals is 38, the end of list add np.nan

                # if len(vals) == 38:
                #     vals=np.append(vals,np.nan)
                #     v1 = vals[year - 1982]
                #     NDVI_list.append(v1)
                # if len(vals)==39:
                # v1 = vals[year - 1982]
                # v1 = vals[year - 1982]
                # if year < 2000:  ## fillwith nan
                #     NDVI_list.append(np.nan)
                #     continue


                v1= vals[year - 1982]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)


            df[f'{variable}'] = NDVI_list
        # exit()
        return df



    def add_lc_composition_to_df(self, df):  ##add landcover composition to df


        fdir_all = data_root + rf'landcover_composition_DIC\\'

        all_dic = {}
        for fdir in os.listdir(fdir_all):
            fname=fdir.split('.')[0]

            dic={}
            for f in os.listdir(fdir_all+fdir):


                dicii = T.load_npy(fdir_all+fdir+'\\'+f)
                dic.update(dicii)


            all_dic[fname] = dic
        # print(all_dic.keys())
        df = T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df
        # exit()

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
        tiff = rf'D:\Project3\Data\Base_data\\continent.tif'
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
        D=DIC_and_TIF(pixelsize=0.25)
        df=T.add_lon_lat_to_df(df,D)
        return df



    def add_SOC_to_df(self, df):
        tiff=rf'D:\Project3\Data\Base_data\SOC\tif_sum\\SOC_sum.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'SOC_sum'
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

    def add_soil_texture_to_df(self, df):
        tiff=rf'D:\Project3\Data\Base_data\HWSD\tif_025\\S_SILT.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'silt'
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

    def add_rooting_depth_to_df(self, df):
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
        fdir=rf'D:\Project3\Result\extract_window\extract_detrend_original_window_CV\\'
        for f in os.listdir(fdir):
            # print(f)
            # exit()
            if not 'LAI4g_CV' in f:
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
                if val > 99:
                    val_list.append(np.nan)
                    continue
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
        df = df.rename(columns={'GPCC_LAI4g_p_value': 'GPCC_LAI4g_p_value_mm_unit',


                            }

                               )



        return df
    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=[rf'monte_carlo_relative_change_normal_slope',

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
        f = data_root + rf'\Base_data\MODIS_LUCC\\MODIS_LUCC_resample.tif'
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

        f = data_root + rf'\Base_data\\glc_025\\glc2000_025.tif'

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

        f = rf'E:\CCI_landcover\trend_analysis_LC\\LC_max.tif'

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

        f=data_root+rf'Base_data\dryland_AI.tif\\dryland.tif'

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

    def add_wetting_drying_transition_to_df(self, df):
        f=result_root+rf'classification\classification.npy'
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
            print(val)
            val_list.append(val)
        df['wetting_drying_transition'] = val_list
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
            elif val==2:
                label='Sub-Humid'
            elif val<-99:
                label=np.nan
            else:
                raise




            val_list.append(label)

        df['AI_classfication'] = val_list
        return df

    def classfy_greening_browning(self):


        df=T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes.df')

            ## sig greeing, browning
        for i, row in df.iterrows():
            if row[rf'LAI4g_trend'] > 0 and row[rf'LAI4g_p_value'] < 0.05:
                df.at[i, rf'LAI4g_trend_class'] = 'sig_greening'
            elif row[rf'LAI4g_trend'] < 0 and row[rf'LAI4g_p_value'] < 0.05:
                df.at[i, rf'LAI4g_trend_class'] = 'sig_browning'
                ## non sig greening, browning
            elif row[rf'LAI4g_trend'] > 0 and row[rf'LAI4g_p_value'] > 0.05:
                df.at[i, rf'LAI4g_trend_class'] = 'non_sig_greening'
            elif row[rf'LAI4g_trend'] < 0 and row[rf'LAI4g_p_value'] > 0.05:
                df.at[i, rf'LAI4g_trend_class'] = 'non_sig_browning'
            else:
                df.at[i, 'LAI4g_trend_class'] = 'other'

        T.save_df(df, result_root + rf'Dataframe\relative_changes\\relative_changes.df')
        T.df_to_excel(df, result_root + rf'Dataframe\relative_changes\\relative_changes')


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

    def show_field(self, df):
        for col in df.columns:
            print(col)
        return df
        pass

class check_data():
    def run(self):
        self.plot_sptial()

        # self.testrobinson()
        # self.plot_time_series()
        # self.plot_bar()

        pass

    def plot_sptial(self):

        fdir = rf'E:\western US\Result\extract_water_year\\raw\\\\GPCC_water_year.npy'

        dic = T.load_npy(fdir)
        # dic=T.load_npy_dir(fdir)

        # for f in os.listdir(fdir):
        #     if not f.endswith(('.npy')):
        #         continue
        #
        #     dic_i=T.load_npy(fdir+f)
        #     dic.update(dic_i)

        len_dic = {}

        for pix in dic:
            r, c = pix
            # if r<480:
            #     continue
            vals = dic[pix]
            if len(vals) < 1:
                continue
            if np.isnan(np.nanmean(vals)):
                continue

            # plt.plot(vals)
            # plt.show()

            # len_dic[pix]=np.nanmean(vals)
            # len_dic[pix]=np.nanstd(vals)

            len_dic[pix] = len(vals)
        arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(len_dic)

        plt.imshow(arr, cmap='RdBu', interpolation='nearest', vmin=0, vmax=1000)
        plt.colorbar()
        plt.title(fdir)
        plt.show()

    def testrobinson(self):
        fdir = rf'D:\Project3\Result\multi_regression_moving_window\window15_anomaly_GPCC\trend_analysis\100mm_unit\\'
        period = '1982_2020'
        fpath_p_value = result_root + rf'D:\Project3\Result\multi_regression_moving_window\window15_anomaly_GPCC\trend_analysis\100mm_unit\\\\GPCC_LAI4g_p_value.tif'
        temp_root = r'trend_analysis\anomaly\\temp_root\\'
        T.mk_dir(temp_root, force=True)

        # f =  rf'D:\Project3\Data\Base_data\dryland_AI.tif\dryland.tif'
        for f in os.listdir(fdir):
            if not f.endswith('GPCC_LAI4g_trend.tif'):
                continue

            fname = f.split('.')[0]
            print(fname)

            m, ret = Plot().plot_Robinson(fdir + f, vmin=-5, vmax=5, is_discrete=True, colormap_n=7, )
            # Plot().plot_Robinson_significance_scatter(m,fpath_p_value,temp_root,0.05)

            fname = 'ERA5 Precipitation CV (%)'
            plt.title(f'{fname}')

            plt.show()

    def plot_time_series(self):
        f = rf'D:\Project3\Result\zscore\GPCC.npy'
        f2 = rf'D:\Project3\Result\anomaly\OBS_extend\GPCC.npy'

        # f=result_root + rf'extract_GS\OBS_LAI_extend\\Tmax.npy'
        # f=data_root + rf'monthly_data\\Precip\\DIC\\per_pix_dic_004.npy'
        # f= result_root+ rf'detrend_zscore_Yang\LAI4g\\1982_2000.npy'
        # dic=T.load_npy(f)
        dic = T.load_npy(f)
        dic2 = T.load_npy(f2)

        for pix in dic:
            if not pix in dic2:
                continue
            vals = dic[pix]
            vals2 = dic2[pix]

            print(vals)

            vals = np.array(vals)
            vals2 = np.array(vals2)
            print(vals)
            print(len(vals))

            # if not len(vals)==19*12:
            #     continue
            # if True in np.isnan(vals):
            #     continue
            # print(len(vals))
            if np.isnan(np.nanmean(vals)):
                continue
            # if np.nanmean(vals)<-20:
            #     continue
            plt.plot(vals)
            plt.twinx()
            plt.plot(vals2)

            plt.title(pix)
            plt.legend([f'{f}'])
            plt.show()

    def plot_bar(self):
        fdir = rf'D:\Project3\Result\trend_analysis\original\\\OBS\\'
        variable_list = ['LAI4g_trend', 'LAI4g_xcludion_LaNina_trend', 'LAI4g_excludion_LaNina_EINino_trend']
        GPP_list = ['GPP_CFE_trend', 'GPP_CFE_excludion_LaNina_trend', 'GPP_CFE_eexcludion_LaNina_EINino_trend']
        average_dic = {}

        for f in os.listdir(fdir):
            if not f.endswith(('.npy')):
                continue
            if not 'LAI4g' in f:
                continue
            if 'tiff' in f:
                continue
            if 'p_value' in f:
                continue

            array = np.load(fdir + f)
            array = np.array(array)
            array[array < -99] = np.nan

            array = array[array != np.nan]
            ## calculate mean
            average_dic[f.split('.')[0]] = np.nanmean(array)

        #     average_dic[f.split('.')[0]]=np.nanmean(array,axis=0)
        df = pd.DataFrame(average_dic, index=['OBS'])
        df.plot.bar()
        #
        plt.show()

    pass


    def __del__(self):

        pass
class PLOT:
    def __init__(self):
        pass

    def __del__(self):
        pass
    def run(self):
        self.plot_anomaly_LAI_based_on_cluster()
        # self.trend_analysis()
        # self.spatial_average()
        # self.diff_spatial_average()


    def clean_df(self, df):
        # df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        # df = df[df['Aridity'] < 0.65]

        # df = df[df['MODIS_LUCC'] != 12]

        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 0]
        df = df[df['lat'] < 45]

        return df
    def plot_anomaly_LAI_based_on_cluster(self):  ##### plot for 4 clusters

        df = T.load_df(result_root + rf'\DataFrame\\Dataframe.df')
        print(len(df))


        print(len(df))
        T.print_head_n(df)
        df=self.clean_df(df)
        print(len(df))
        # exit()
        ## plot spatial
        pix_list = T.get_df_unique_val_list(df, 'pix')
        spatial_dict = {}
        for pix in pix_list:
            spatial_dict[pix] = 1
        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,interpolation='nearest')
        plt.show()

        #create color list with one green and another 14 are grey

        # color_list=['grey']*16

        # color_list=['green','blue','red','orange','aqua','brown','cyan', 'black', 'yellow', 'purple', 'pink', 'grey', 'brown','lime','teal','magenta']
        color_list=['blue']*16
        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=3
        # linewidth_list[2]=2

        fig = plt.figure()
        i = 1
        # variable_list=['GPCC_water_year',]
        # variable_list=['CRU','GPCC']

        variable_list=['LAI4g_water_year']
        scenario='S2'
        # variable_list= ['LAI4g',f'CABLE-POP_{scenario}_lai', f'CLASSIC_{scenario}_lai', 'CLM5',  f'IBIS_{scenario}_lai', f'ISAM_{scenario}_lai',
        #      f'ISBA-CTRIP_{scenario}_lai', f'JSBACH_{scenario}_lai', f'JULES_{scenario}_lai',  f'LPJ-GUESS_{scenario}_lai', f'LPX-Bern_{scenario}_lai',
        #      f'ORCHIDEE_{scenario}_lai', f'SDGVM_{scenario}_lai', f'YIBs_{scenario}_Monthly_lai']
        region_unique = T.get_df_unique_val_list(df, 'landcover_classfication')
        print(region_unique)


        for continent in ['Grass', 'Evergreen', 'Shrub', 'North_America']:
            ax = fig.add_subplot(2, 2, i)
            if continent=='North_America':
                df_continent=df
            else:

                df_continent = df[df['landcover_classfication'] == continent]
            pixel_number = len(df_continent)

            for product in variable_list:


                vals = df_continent[product].tolist()
                # pixel_area_sum = df_continent['pixel_area'].sum()

                # print(vals)

                vals_nonnan=[]
                for val in vals:


                    if type(val)==float: ## only screening
                        continue
                    if len(val) ==0:
                        continue
                    val=np.array(val)
                    # print(val)
                    val[val<-99]=np.nan


                    vals_nonnan.append(list(val))


                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)

                val_std=np.nanstd(vals_mean,axis=0)

                # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
                plt.plot(vals_mean,label=product,color=color_list[variable_list.index(product)],linewidth=linewidth_list[variable_list.index(product)])
                # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])
                a,b,r,p,se=stats.linregress(range(len(vals_mean)),vals_mean)

                plt.plot(range(len(vals_mean)),a*range(len(vals_mean))+b,color='r',linewidth=1)
                plt.text(2,40,f'r={r:.2f} p={p:.2f}',fontsize=12)

            i=i+1

            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1983, 2021, 4), rotation=45, fontsize=12)
            # plt.ylim(-0.2, 0.2)
            plt.ylim(-15, 15)


            # plt.xlabel('year')

            plt.ylabel(f'LAI4g(%)', fontsize=15)
            # plt.ylabel(f'precip(%/year)', fontsize=15)
            plt.title(f'{continent}_{pixel_number}_pixels')
            plt.grid(which='major', alpha=0.5)
        plt.legend()
        plt.show()

    def trend_analysis(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        fdir = result_root+rf'\extract_water_year\raw\\'
        outdir = result_root + rf'trend_analysis\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'GPCC' in f:
                continue



            outf=outdir+f.split('.')[0]+'_second'


            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}

            for pix in tqdm(dic):
                r,c=pix


                landcover_value=crop_mask[pix]
                if landcover_value==16 or landcover_value==17 or landcover_value==18:
                    continue
                if dic_modis_mask[pix]==12:
                    continue
                # time_series = dic[pix]

                time_series=dic[pix][0:17] ## 1983-1999
                # time_series = dic[pix][17:]
                # print(len(time_series))
                # exit()


                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                if len(set(time_series))==1:
                    continue
                # print(time_series)

                if np.nanstd(time_series) == 0:
                    continue
                try:
                    slope,b,r,p_value=T.nan_line_fit(np.arange(len(time_series)), time_series)
                    # slope = 1
                    # p_value=0
                    trend_dic[pix] = slope
                    p_value_dic[pix] = p_value
                except:
                    continue


            all_dict = {'trend':trend_dic,'p_value':p_value_dic}
            df = T.spatial_dics_to_df(all_dict)
            df = T.add_lon_lat_to_df(df, DIC_and_TIF(pixelsize=0.25))
            df = df[df['lon'] > -125]
            df = df[df['lon'] < -105]
            df = df[df['lat'] > 30]
            df = df[df['lat'] < 45]

            arr_trend_dict_roi = T.df_to_spatial_dic(df, 'trend')
            p_value_arr_dict_roi = T.df_to_spatial_dic(df, 'p_value')

            arr_trend_roi = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(arr_trend_dict_roi)
            p_value_arr_roi = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_arr_dict_roi)
            plt.imshow(arr_trend_roi, cmap='RdBu', vmin=-5, vmax=5)
            # plt.colorbar(label='LAI_trend (m2/m2/year)')


            # plt.ylabel(f'precip(mm/year)')
            # plt.title(f'{pixel_number}_pixels')
            plt.grid(which='major', alpha=0.5)
            plt.grid(which='minor', alpha=0.2)
            ## turn off ticks
            # plt.xticks([])
            # plt.yticks([])
            #

            plt.colorbar(label='Precip_trend (mm/year)')
            # plt.colorbar(label='LAI4g_trend (m2/m2/year)')

            significant_point = np.where(p_value_arr_roi < 0.05)
            plt.scatter(significant_point[1], significant_point[0], s=1, c='black',label='p<0.05',marker='*',alpha=0.5)


            # plt.title(f.split('.')[0])
            plt.show()

            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend_roi, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr_roi, outf + '_p_value.tif')
            # #
            # np.save(outf + '_trend', arr_trend_roi)
            # np.save(outf + '_p_value', p_value_arr_roi)



    def spatial_average(self):
        # NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        # array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        fdir = result_root+rf'\extract_water_year\raw\\'
        outdir = result_root + rf'spatial_average\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'GPCC' in f:
                continue

            outf=outdir+f.split('.')[0]+'_second.tif'


            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            average_dic = {}

            for pix in tqdm(dic):
                r,c=pix
                landcover_value=crop_mask[pix]
                if landcover_value==16 or landcover_value==17 or landcover_value==18:
                    continue
                if dic_modis_mask[pix]==12:
                    continue


                # time_series=dic[pix]
                # time_series=dic[pix][0:17]  # 1983-1999

                time_series=dic[pix][17:] # 2002-2020
                # print(len(time_series))


                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                if len(set(time_series))==1:
                    continue
                # print(time_series)

                if np.nanstd(time_series) == 0:
                    continue
                try:

                        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    average = np.nanmean(time_series)

                    average_dic[pix] = average

                except:
                    continue

            arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(average_dic)


        ## extract the region of interest

            arr_trend_dict = DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(arr_trend)


            all_dict = {'trend':arr_trend_dict}
            df = T.spatial_dics_to_df(all_dict)
            df = T.add_lon_lat_to_df(df, DIC_and_TIF(pixelsize=0.25))
            df = df[df['lon'] > -125]
            df = df[df['lon'] < -105]
            df = df[df['lat'] > 30]
            df = df[df['lat'] < 45]

            arr_trend_dict_roi = T.df_to_spatial_dic(df, 'trend')


            arr_trend_roi = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(arr_trend_dict_roi)

            plt.imshow(arr_trend_roi,  vmin=200, vmax=500, cmap='Spectral')
            # plt.colorbar(label='LAI_trend (m2/m2/year)')


            # plt.ylabel(f'precip(mm/year)')
            # plt.title(f'{pixel_number}_pixels')
            plt.grid(which='major', alpha=0.5)
            plt.grid(which='minor', alpha=0.2)

            plt.colorbar(label='Precip_average (mm/year)')
            # plt.colorbar(label='LAI4g_trend (m2/m2/)')



            plt.title(f.split('.')[0])
            plt.show()

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend_roi, outf + '_trend.tif')

            np.save(outf + '_trend', arr_trend_roi)

    def diff_spatial_average(self):
        fdir = result_root+rf'spatial_average\\'

        arr1, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + 'GPCC_water_year_second.tif_trend.tif')

        arr2, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + 'GPCC_water_year_first.tif_trend.tif')

        arr = arr1-arr2
        arr[arr==0]=np.nan
        plt.imshow(arr,  vmin=-100, vmax=100, cmap='RdBu')

        plt.colorbar(label='Precip_average (mm/year)')

        ## fontsize

        plt.grid(which='major', alpha=0.5)
        plt.grid(which='minor', alpha=0.2)


        plt.show()

        DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, fdir + 'GPCC_water_year_diff.tif')







def main():

    # data_processing().run()
    # build_dataframe().run()
    # check_data().run()
    PLOT().run()

    pass

if __name__ == '__main__':
    main()