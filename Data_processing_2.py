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
result_root = 'D:/Project3/Result/'
class Data_processing_2:

    def __init__(self):

        pass

    def run(self):

        # self.dryland_mask()
        # self.test_histogram()
        self.resampleSOC()

        pass
    def dryland_mask(self):
        ## here we extract dryland tif
        fpath=rf'D:\Project3\Data\Base_data\aridity_index05.tif\\aridity_index.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        NDVI_mask_f = rf'D:\Greening\Data\Base_data\\NDVI_mask.tif'
        array_NDVI_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)

        array[array_NDVI_mask < 0.1] = np.nan


        array[array >= 0.65] = np.nan


        outf=rf'D:\Project3\Data\Base_data\aridity_index05.tif\\dryland_mask.tif'
        DIC_and_TIF().arr_to_tif(array, outf)

    def test_histogram(self):

        fpath = rf'E:\Project3\Data\ERA5_daily\dict\moving_window_average_anaysis\\detrended_annual_LAI4g_CV.npy'
        spatial_dic=np.load(fpath, allow_pickle=True, encoding='latin1').item()
        data_list=[]

        for pix in spatial_dic:
            val=spatial_dic[pix]
            data_list.append(val)

        data_list=np.array(data_list)
        ## histogram

        plt.hist(data_list,bins=50)

        plt.show()




    def resampleSOC(self):
        f=rf'E:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth.tif'

        outf = rf'E:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth_05.tif'

        dataset = gdal.Open(f)



        try:
            gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
        # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
        # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
        except Exception as e:
            pass






class Phenology():  ### plot site based phenology curve
    ## this function is to see phenology of NH, SH and tropical
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass
    def run(self):

        self.phenology()
        pass

    def read_shp(self):
        fpath = r"C:\Users\wenzhang1\Desktop\point2.shp"
        df=T.read_point_shp(fpath, )
        return df

    def phenology(self):
        fdir_all = rf'E:\Project3\Data\LAI4g\\dic\\'
        spatial_dic=T.load_npy_dir(fdir_all)
        result_dic={}
        shp_df=self.read_shp()
        print(shp_df)
        lon_list=shp_df['point_x_pos'].to_list()
        lat_list=shp_df['point_y_pos'].to_list()
        pix_list=DIC_and_TIF().lon_lat_to_pix(lon_list, lat_list)


        for pix in pix_list:
            lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
            r,c=pix
            # if r<60:
            #     continu
            val=spatial_dic[pix]
            if T.is_all_nan(val):
                continue

            vals_reshape=val.reshape(-1,24)
            # plt.imshow(vals_reshape, interpolation='nearest', cmap='jet')
            # plt.colorbar()
            # plt.show()
            multiyear_mean = np.nanmean(vals_reshape, axis=0)
            #
            result_dic[pix]=multiyear_mean
            x=np.arange(0,24)
            xtick = [str(i) for i in x]
            plt.plot(x, multiyear_mean)
            plt.xticks(x, xtick)
            plt.xlabel('biweekly')
            plt.ylabel('LAI4g (m2/m2)')
            plt.title(f'lat:{lat},lon:{lon}')
            plt.show()
        # array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(result_dic)
        # plt.imshow(array, interpolation='nearest', cmap='jet')
        # plt.colorbar()
        #
        # plt.show()
            ## plt.plot(multiyear_mean)
            ## plt.show()





        pass

class build_moving_window_dataframe():
    def __init__(self):
        self.this_class_arr = (rf'D:\Project3\ERA5_025\Dataframe\moving_window_3mm\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'moving_window_3mm.df'
    def run(self):
        df = self.__gen_df_init(self.dff)
        # df=self.build_df(df)
        # self.append_value(df)
        # df=self.append_attributes(df)
        # df=self.add_trend_to_df(df)
        # df=self.foo1(df)
        df=self.add_window_to_df(df)
        # df=self.add_columns(df)
        #self.show_field()

        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff)
    def show_field(self):
        df = T.load_df(self.dff)
        for col in df.columns:
            print(col)



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

        fdir = rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\\'
        all_dic = {}
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
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

    def append_value(self, df):  ##补齐

        ## extract LAI4g

        for col in df.columns:
            if not 'LAI4g' in col:
                continue
            if 'CV' in col:
                continue

            vals_new = []


            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'append {col}'):
                pix = row['pix']
                r, c = pix
                if r<480:
                    continue
                vals = row[col]
                print(vals)
                if type(vals) == float:
                    vals_new.append(np.nan)
                    continue
                vals = np.array(vals)
                print(len(vals))
                # if len(vals)==23:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     # print(len(vals))
                # elif len(vals)==38:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     print(len(vals))
                if len(vals) == 23:

                    vals = np.append(vals, np.nan)
                    vals_new.append(vals)

                vals_new.append(vals)

                # exit()
            df[col] = vals_new

        return df

        pass

    def foo1(self, df):

        f = rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\\detrended_growing_season_LAI_mean_CV.npy'
        # array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        # array = np.array(array, dtype=float)
        # dic = DIC_and_TIF().spatial_arr_to_dic(array)

        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []

        for pix in tqdm(dic):
            time_series = dic[pix]
            y = 0

            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                window=y
                # print(window)
                year.append(window)
                y += 1

        df['pix'] = pix_list



        df['window'] = year

        df['LAI4g_CV_growing_season'] = change_rate_list
        return df
    def add_window_to_df(self, df):
        threshold = '3mm'

        fdir=rf'D:\Project3\ERA5_025\extract_rainfall_phenology_year\moving_window_average_anaysis\{threshold}\selected_variables\\'
        print(fdir)
        print(self.dff)
        variable_list = [
                         'rainfall_seasonality_all_year_ecosystem_year',
                       'heat_event_frenquency_growing_season','rainfall_frenquency_non_growing_season', 'rainfall_frenquency_growing_season',
                          'dry_spell_non_growing_season', 'dry_spell_growing_season',
                         'rainfall_intensity_growing_season',
                          'rainfall_intensity_non_growing_season',

                        ]

        variable_list = ['detrended_sum_rainfall_growing_season_CV', ]
        # variable_list = ['CO2_ecosystem_year', ]


        for f in os.listdir(fdir):
            variable = f.split('.')[0]

            if not variable in variable_list:
                continue



            variable= f.split('.')[0]

            print(variable)


            if not f.endswith('.npy'):
                continue



            val_dic = T.load_npy(fdir + f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                window = row.window
                # pix = row.pix
                pix = row['pix']
                r, c = pix


                if not pix in val_dic:
                    NDVI_list.append(np.nan)
                    continue

                y = window

                vals = val_dic[pix]
                vals=np.array(vals)
                # print(vals)
                vals[vals>250] = np.nan
                vals[vals<-250] = np.nan



                ##### if len vals is 38, the end of list add np.nan

                if len(vals) == 22:
                    vals=np.append(vals,np.nan)
                    v1 = vals[y-0]
                    NDVI_list.append(v1)
                elif len(vals)==23:
                    v1= vals[y-0]
                    NDVI_list.append(v1)
                else:
                    NDVI_list.append(np.nan)
                # print(y)
                # print(vals)
                # print(vals)

                # if  len(vals) != 23:
                #     NDVI_list.append(np.nan)
                #     continue
                # v1= vals[y-0]
                # NDVI_list.append(v1)


                # print(v1,year,len(vals))
            # plt.hist(NDVI_list)
            # plt.show()


            df[f'{variable}'] = NDVI_list
            # df[f'{variable}_ecosystem_year'] = NDVI_list
        # exit()
        return df
    def append_attributes(self, df):  ## add attributes
        fdir =  result_root + rf'\\moving_window_extraction\wet_year_moving_window_extraction\\'
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
    def add_columns(self, df):
        df['window'] = df['window'].str.extract(r'(\d+)').astype(int)




        return df

    def add_trend_to_df(self, df):
        fdir=result_root+rf'multi_regression_moving_window\window15_anomaly\\TIFF\\'
        for f in os.listdir(fdir):
            if not 'CO2' in f:
                continue
            if not f.endswith('.tif'):
                continue
            print(f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
            array = np.array(array, dtype=float)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
            f_name = f.split('.')[0]+'_LAI4g'
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

class build_dataframe():


    def __init__(self):



        self.this_class_arr = (rf'D:\Project3\ERA5_025\Dataframe\moving_window_3mm\\\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'moving_window_3mm.df'

        pass

    def run(self):


        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        # df=self.foo2(df)
        # df=self.add_multiregression_to_df(df)
        # df=self.build_df(df)
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
        # #
        # df=self.add_AI_classfication(df)
        #
        df=self.add_aridity_to_df(df)
        df=self.add_MODIS_LUCC_to_df(df)
        df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        df=self.add_landcover_classfication_to_df(df)
        # df=self.add_maxmium_LC_change(df)
        df=self.add_row(df)
        df=self.add_continent_to_df(df)
        df=self.add_lat_lon_to_df(df)
        df=self.add_soil_texture_to_df(df)
        # df=self.add_SOC_to_df(df)
        # # #
        df=self.add_rooting_depth_to_df(df)
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


        fdir=rf'E:\Project3\Data\ERA5_daily\dict\extract_rainfall_annual\rainfall_seasonality_unpack\\'
        all_dic= {}
        for f in os.listdir(fdir):


            fname= f.split('.')[0]
            if 'CV' not in fname:
                continue
            fpath=fdir+f

            dic = T.load_npy(fpath)
            key_name=fname
            print(key_name)
            all_dic[key_name]=dic
        # print(all_dic.keys())
        df=T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df

    def build_df_monthly(self, df):


        fdir = result_root+rf'extract_GS_return_monthly_data\individual_month_relative_change\X\\'
        all_dic= {}

        for fdir_ii in os.listdir(fdir):

            dic=T.load_npy(fdir+fdir_ii)

            key_name=fdir_ii.split('.')[0]
            all_dic[key_name] = dic
                # print(all_dic.keys())
        df = T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df



    def append_attributes(self, df):  ## add attributes
        fdir = result_root+ rf'\relative_change\OBS_LAI_extend\\'
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
                continue
            if not 'LAI4g' in f:
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

        f = rf'E:\Project3\Result\extract_rainfall_phenology_year\CRU-JRA\extraction_rainfall_characteristic\\CV_intraannual_rainfall.npy'
        # array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        # array = np.array(array, dtype=float)
        # dic = DIC_and_TIF().spatial_arr_to_dic(array)


        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []

        for pix in tqdm(dic):
            time_series = dic[pix]['ecosystem_year']

            y = 1982
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y)
                y += 1


        df['pix'] = pix_list

        df['year'] = year
        # df['window'] = 'VPD_LAI4g_00'
        df['intraannual_rainfall_CV'] = change_rate_list
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

        fdir=rf'E:\Project3\Result\extract_rainfall_phenology_year\CRU-JRA\extraction_rainfall_characteristic\\'
        for f in os.listdir(fdir):


            variable= f.split('.')[0]
            if variable not in ['rainfall_frenquency','rainfall_seasonality_all_year','heavy_rainfall_days','rainfall_intensity']:
                continue

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


                vals = val_dic[pix]['ecosystem_year']
                # print(len(vals))
                if len(vals)<38:

                    NDVI_list.append(np.nan)

                    continue

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


            df[f'{variable}'] = NDVI_list
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

                print(len(vals))
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
        f = data_root + rf'\Base_data\MODIS_LUCC\\MODIS_LUCC_resample_025.tif'
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

        f = rf'E:\Project3\Data\Base_data\lc_trend\\max_trend.tif'

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

        f=data_root+rf'Base_data\\aridity_index_025\\aridity_index.tif'

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
class CO2_processing():  ## here CO2 processing

    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass
    def run(self):

        # self.plot_CO2()
        # self.interpolate()
        # self.per_pix()
        self.interpolate1()

        # self.rename()
        pass
    def plot_CO2(self):
        fdir=rf'D:\Project3\Data\CO2\dic\monthly_historic\\'
        result_dic={}
        len_dic = {}

        dic = T.load_npy_dir(fdir)
        for pix in dic:
            vals = dic[pix]
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            len_vals = len(vals)

            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(vals)), vals)
            result_dic[pix] = slope
            len_dic[pix] = len_vals

        arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(len_dic)
        plt.imshow(arr, interpolation='nearest', cmap='jet')
        plt.colorbar()
        plt.show()

    def interpolate(self):

        import numpy as np
        import os
        from scipy.interpolate import interp1d

        # 设置数据路径
        data_path = rf'D:\Project3\Data\CO2\CO2_TIFF\unify\historic_SSP245\\'
        years = list(range(1982, 2021))
        months=list(range(1,13))

        # 获取所有缺失年月份

        missing_year_month_date = [f'{year}{month:02d}01' for year in years for month in months if not os.path.exists(os.path.join(data_path, f'{year}{month:02d}01.tif'))]
        valid_year_month_date = [f'{year}{month:02d}01' for year in years for month in months if os.path.exists(os.path.join(data_path, f'{year}{month:02d}01.tif'))]

        # 读取所有年份的数据
        raster_data = {}
        profile = None  # 用于存储栅格文件的元数据
        for year in years:
            for month in months:
                year_month_date = f"{year}{month}01"
                month_format = '{:02d}'.format(month)
                file_path = os.path.join(data_path, f"{year}{month_format}01.tif")
                if not os.path.exists(file_path):
                    ## create 720*1440 nan array
                    array = np.full((720, 1440), np.nan)
                    raster_data[year_month_date] = array
                else:

                    array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(file_path)
                    raster_data[f'{year}{month_format}01'] = array



        # 将数据堆叠成三维数组（time, row, col）
        all_data = sorted(raster_data.keys())
        stacked_data = np.stack([raster_data[date] for date in all_data], axis=0)
        ## mask for valid data



        # 创建插值函数
        interp_func = interp1d(valid_year_month_date, valid_data, axis=0,kind='linear', fill_value='extrapolate')


        # 生成插值后的数据
        interpolated_data = interp_func(missing_year_month_date)

        # 将插值后的数据保存到磁盘

    def per_pix(self):

        # 设置数据路径
        tif_dir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify\historic_SSP245\\'
        outdir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify\historic_SSP245_perpix\\'
        T.mkdir(outdir)
        Pre_Process().data_transform(tif_dir,outdir)

    def interpolate1(self):
        perpix_fdir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify\historic_SSP245_perpix\\'
        tif_dir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify\historic_SSP245\\'
        outdir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify\historic_SSP245_interpoolation\\'
        T.mkdir(outdir)
        year_list = []
        for f in T.listdir(tif_dir):
            if not f.endswith('.tif'):
                continue
            year = f.split('.')[0][0:4]
            year = int(year) - 1982
            year_list.append(year)
        year_list = sorted(list(set(year_list)))
        year_list = np.array(year_list)
        # print(date_list);exit()
        param_list = []

        for f in T.listdir(perpix_fdir):
            params = (perpix_fdir, outdir, f, year_list)
            param_list.append(params)
        MULTIPROCESS(self.kernel_interpolate1, param_list).run(process=16)


    def kernel_interpolate1(self,params):
        perpix_fdir,outdir,f,year_list = params
        fpath = join(perpix_fdir, f)
        outpath = join(outdir, f)
        spatial_dict = T.load_npy(fpath)
        K = KDE_plot()
        spatial_dict_interp = {}

        for pix in spatial_dict:
            vals = spatial_dict[pix]
            vals = np.array(vals)
            vals_reshape = vals.reshape(-1, 12)
            # print(len(vals_reshape))
            vals_reshape_T = vals_reshape.T
            vals_mon_interp = []
            for mon in range(12):
                vals_mon = vals_reshape_T[mon]
                # interp = interpolate.interp1d(year_list, vals_mon, kind='linear')
                # a,b,r,p = T.nan_line_fit(year_list, vals_mon)
                a, b, r, p = K.linefit(year_list, vals_mon)
                y = a * (2014 - 1982) + b
                vals_mon_interp.append(y)

                # KDE_plot().plot_fit_line(a, b, r, p, year_list)
                # plt.scatter(year_list, vals_mon)
                # vals_dict = T.dict_zip(year_list,vals_mon)
                # pprint(vals_dict)
                # print(interp(2014-1982))
                # print((vals_dict[2015-1982]+vals_dict[2013-1982])/2)
                # print(y)
                # interpolate = interp1d.
                # plt.show()
            vals_reshape = np.insert(vals_reshape, (2014 - 1982), vals_mon_interp, axis=0)
            vals_interp = vals_reshape.flatten()
            spatial_dict_interp[pix] = vals_interp
            # date_list = []
            # for year in range(1982, 2021):
            #     for mon in range(1, 13):
            #         date = datetime.datetime(year, mon, 1)
            #         date_list.append(date)
            # plt.plot(date_list,vals_interp)
            # plt.show
        T.save_npy(spatial_dict_interp, outpath)

        # 将插值后的数据保存到磁盘

    def rename(self):
        fdir=rf'D:\Project3\Data\CO2\CO2_TIFF\unify\historic_SSP245\\'

        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            date=f.split('.')[0][6:]
            if  date=='01':
                continue
            else:
                # print(f)

                ## replace other number with 01
                fnew=join(f.split('.')[0][0:6]+'01.'+f.split('.')[1])
                print(fnew)




            os.rename(os.path.join(fdir, f), os.path.join(fdir, fnew))

        pass



class PLOT():
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass

    def run(self):

        self.plot_anomaly_LAI_based_on_cluster()
        pass

    def plot_anomaly_LAI_based_on_cluster(self):  ##### plot for 4 clusters

        df = T.load_df(rf'D:\Project3\ERA5_025\Dataframe\LAI4g_CV_continent\\LAI4g_CV_continent.df')
        print(len(df))
        df=self.df_clean(df)

        print(len(df))
        T.print_head_n(df)
        # exit()




        #create color list with one green and another 14 are grey

        color_list=['grey']*16
        color_list[0]='green'

        color_list=['green','blue','red','orange','aqua','purple', 'black', 'yellow', 'purple', 'pink', 'grey', 'brown','lime','teal','magenta']
        linewidth_list=[1]*16
        linewidth_list[0]=2


        fig = plt.figure()
        i=1


        variable_list=['detrended_growing_season_LAI_mean_CV']


        for continent in ['Africa', 'Asia', 'Australia', 'South_America', 'North_America',  'global']:


            if continent=='global':
                df_continent=df
            else:

                df_continent = df[df['continent'] == continent]



            for product in variable_list:
                # print('=========')
                # print(product)
                # print(df_continent.columns.tolist())
                # exit()
                # T.print_head_n(df_continent)
                # exit()


                vals = df_continent[product].tolist()
                # pixel_area_sum = df_continent['pixel_area'].sum()


                # print(vals)

                vals_nonnan=[]
                for val in vals:


                    if type(val)==float: ## only screening
                        continue
                    if len(val) ==0:
                        continue
                    # val[val<-99]=np.nan

                    if not len(val) == 23:
                        ## add nan to the end of the list
                        for j in range(1):
                            val=np.append(val,np.nan)
                        # print(val)
                        # print(len(val))


                    vals_nonnan.append(list(val))

                        # exit()
                    # print(type(val))
                    # print(len(val))
                    # print(vals)

                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)
                # vals_mean=vals_mean/pixel_area_sum

                val_std=np.nanstd(vals_mean,axis=0)

                # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
                plt.plot(vals_mean,label=continent,color=color_list[i],linewidth=linewidth_list[variable_list.index(product)])
                i = i + 1
                plt.legend()

        plt.xticks(range(0, 23, 4))
        window_size = 15

        # set xticks with 1982-1997, 1998-2013,.. 2014-2020
        year_range = range(1982, 2021)
        year_range_str = []
        for year in year_range:

            start_year = year
            end_year = year + window_size - 1
            if end_year > 2020:
                break
            year_range_str.append(f'{start_year}-{end_year}')
        plt.xticks(range(len(year_range_str))[::3], year_range_str[::3], rotation=45, ha='right')
        plt.xticks(range(0, 23, 3))



        plt.xlabel('window')

        plt.ylabel(f'LAI CV (%)')


        plt.grid(which='major', alpha=0.5)

        # plt.show()
        out_pdf_fdir=rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\trend\\pdf\\'
        plt.savefig(out_pdf_fdir + 'time_series.pdf', dpi=300, bbox_inches='tight')
        plt.close()




    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 120]
        df = df[df['Aridity'] < 0.65]
        # df = df[df['LC_max'] < 20]

        df = df[df['MODIS_LUCC'] != 12]


        df = df[df['landcover_classfication'] != 'Cropland']

        return df


def main():
    # Data_processing_2().run()
    # Phenology().run()
    build_moving_window_dataframe().run()
    # CO2_processing().run()
    # build_dataframe().run()
    # PLOT().run()



    pass

if __name__ == '__main__':
    main()