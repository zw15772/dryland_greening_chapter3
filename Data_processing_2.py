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
D = DIC_and_TIF(pixelsize=0.5)
centimeter_factor = 1/2.54


this_root = 'E:\Project3\\'
data_root = 'E:/Project3/Data/'
result_root = 'E:/Project3/Result/'
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

    def nc_to_tif(self):

        f=rf'D:\Project3\Data\Base_data\HWSD\nc\\S_SAND.nc'
        outdir=rf'D:\Project3\Data\Base_data\HWSD\nc\\tif\\'
        Tools().mk_dir(outdir,force=True)

        nc = Dataset(fdir + f, 'r')

        print(nc)
        print(nc.variables.keys())
        # t = nc['time']
        # print(t)
        # start_year = int(t.units.split(' ')[-1].split('-')[0])

        # basetime = datetime.datetime(start_year, 1, 1)  # 告诉起始时间
        lat_list = nc['lat']
        lon_list = nc['lon']
        # lat_list=lat_list[::-1]  #取反
        print(lat_list[:])
        print(lon_list[:])

        origin_x = lon_list[0]  # 要为负数-180
        origin_y = lat_list[0]  # 要为正数90
        pix_width = lon_list[1] - lon_list[0]  # 经度0.5
        pix_height = lat_list[1] - lat_list[0]  # 纬度-0.5
        print(origin_x)
        print(origin_y)
        print(pix_width)
        print(pix_height)
        # SIF_arr_list = nc['SIF']
        SPEI_arr_list = nc['tmin']
        print(SPEI_arr_list.shape)
        print(SPEI_arr_list[0])
        # plt.imshow(SPEI_arr_list[5])
        # # plt.imshow(SPEI_arr_list[::])
        # plt.show()

        year = int(f.split('.')[-3][0:4])
        month = int(f.split('.')[-3][4:6])

        fname = '{}{:02d}.tif'.format(year, month)
        print(fname)
        newRasterfn = outdir + fname
        print(newRasterfn)
        longitude_start = origin_x
        latitude_start = origin_y
        pixelWidth = pix_width
        pixelHeight = pix_height
        # array = val
        # array=SPEI_arr_list[::-1]
        array = SPEI_arr_list

        array = np.array(array)
        # method 2
        array = array.T
        array = array * 0.001  ### GPP need scale factor
        array[array < 0] = np.nan

        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()
        ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, )


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
        self.threshold = '5mm'
        self.this_class_arr = (rf'E:\Project3\Result\3mm\Dataframe\\moving_window_CV\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + rf'moving_window_CV.df'
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

        fdir = rf'E:\Project3\Result\3mm\moving_window_multi_regression\with_trend\multi_regression_result\npy_time_series\\'
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

        f = rf'E:\Project3\Result\3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\detrended_growing_season_LAI_mean_CV.npy'
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
        threshold = self.threshold

        fdir=rf'E:\Project3\Result\3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\'
        print(fdir)
        print(self.dff)
        # variable_list = [
        #                  'rainfall_seasonality_all_year_ecosystem_year',
        #                'heat_event_frenquency_growing_season','rainfall_frenquency_non_growing_season', 'rainfall_frenquency_growing_season',
        #
        #                  'rainfall_intensity_growing_season',
        #                   'rainfall_intensity_non_growing_season','detrended_sum_rainfall_growing_season_CV'
        #
        #                 ]

        for f in os.listdir(fdir):
            if not 'sum_rainfall' in f:

                continue
            if 'detrended' in f:
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
                vals[vals>999] = np.nan
                vals[vals<-999] = np.nan

                ##### if len vals is 38, the end of list add np.nan

                # if len(vals) == 22:
                #     vals=np.append(vals,np.nan)
                #     v1 = vals[y-0]
                #     NDVI_list.append(v1)
                # elif len(vals)==23:
                #     v1= vals[y-0]
                #     NDVI_list.append(v1)
                # else:
                #     NDVI_list.append(np.nan)
                if len(vals) ==0:
                    NDVI_list.append(np.nan)
                    continue

                v1= vals[y-0]
                NDVI_list.append(v1)


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



        self.this_class_arr = (rf'E:\Project3\Result\3mm\Dataframe\\moving_window_CV\\\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'moving_window_CV.df'

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
        # df=self.add_aridity_to_df(df)
        # df=self.add_MODIS_LUCC_to_df(df)
        # df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        # df=self.add_landcover_classfication_to_df(df)
        # df=self.add_maxmium_LC_change(df)
        # df=self.add_row(df)
        # df=self.add_continent_to_df(df)
        # df=self.add_lat_lon_to_df(df)
        df=self.add_soil_texture_to_df(df)

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

        fdir=rf'E:\Project3\Result\3mm\moving_window_multi_regression\multi_regression_result\npy_time_series\\'
        all_dic= {}
        for f in os.listdir(fdir):

            fname= f.split('.')[0]
            if 'LAI' not in fname:
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
        fdir = result_root+ rf'3mm\moving_window_multi_regression\multi_regression_result\npy_time_series\\'
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
                continue
            if not 'rainfall_frenquency_phenology_LAI_mean' in f:
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

        f = rf'E:\Project3\Result\3mm\relative_change_growing_season\\phenology_LAI_mean.npy'
        # array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        # array = np.array(array, dtype=float)
        # dic = DIC_and_TIF().spatial_arr_to_dic(array)


        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 1982
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y)
                y += 1


        df['pix'] = pix_list

        df['year'] = year
        # df['window'] = 'VPD_LAI4g_00'
        df['LAI_relative_change'] = change_rate_list
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
        D=DIC_and_TIF(pixelsize=0.5)
        df=T.add_lon_lat_to_df(df,D)
        return df


    def add_soil_texture_to_df(self, df):
        fdir = data_root + rf'\Base_data\HWSD\tif\05\\'
        for f in (os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            if 'silt' in f:
                continue
            tiff = fdir + f

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
            array = np.array(array, dtype=float)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
            fname=f.split('.')[0]
            # print(fname)
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
        df = df.rename(columns={'weighted_avg_LAI_x': 'weighted_avg_LAI',


                            }

                               )



        return df
    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=[rf'weighted_avg_LAI_y',

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

        # self.resample_CO2()
        # self.unify_TIFF()

        # self.plot_CO2()

        # self.per_pix()
        # self.interpolate1()
        self.check_spatial()

        # self.rename()
        pass

    def resample_CO2(self):
        fdir_all = rf'D:\Project3\Data\CO2\CO2_TIFF\original\\'

        outdir_all = rf'D:\Project3\Data\CO2\CO2_TIFF\resample_05\\'


        year = list(range(1982, 2021))
        # print(year)
        # exit()
        for fdir in tqdm(os.listdir(fdir_all)):
            outdir = outdir_all + fdir + '\\'
            T.mk_dir(outdir, force=True)
            for f in os.listdir(fdir_all + fdir):
                if not f.endswith('.tif'):
                    continue
                outf = outdir + f
                if os.path.isfile(outf):
                    continue

                if f.startswith('._'):
                    continue


                print(f)
                # exit()
                date = f.split('.')[0]

                # print(date)
                # exit()
                dataset = gdal.Open(fdir_all + fdir + '/' + f)
                # print(dataset.GetGeoTransform())
                original_x = dataset.GetGeoTransform()[1]
                original_y = dataset.GetGeoTransform()[5]

                # band = dataset.GetRasterBand(1)
                # newRows = dataset.YSize * 2
                # newCols = dataset.XSize * 2
                try:
                    gdal.Warp(outdir + '{}.tif'.format(date), dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
                # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
                # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
                except Exception as e:
                    pass
    def unify_TIFF(self):
        fdir_all=rf'D:\Project3\Data\CO2\CO2_TIFF\\resample_05\\'
        outdir_all=rf'D:\Project3\Data\CO2\CO2_TIFF\\unify_05\\'

        T.mk_dir(outdir_all, force=True)


        for fdir in os.listdir(fdir_all):
            outdir = outdir_all + fdir + '\\'
            Tools().mk_dir(outdir, force=True)

            for f in tqdm(os.listdir(fdir_all+fdir)):
                fpath=join(fdir_all,fdir,f)

                outpath=join(outdir,f)

                if not f.endswith('.tif'):
                    continue
                if f.startswith('._'):
                    continue
                unify_tiff=DIC_and_TIF().unify_raster1(fpath,outpath,0.5)


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


    def per_pix(self):

        # 设置数据路径
        tif_dir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\historic_SSP245\\'
        outdir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\historic_SSP245_perpix\\'
        T.mkdir(outdir)
        Pre_Process().data_transform(tif_dir,outdir)

    def interpolate1(self):
        perpix_fdir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\historic_SSP245_perpix\\'
        tif_dir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\historic_SSP245\\'
        outdir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\historic_SSP245_interpoolation\\'
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
            # self.kernel_interpolate1(params)
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
            vals[vals<-999] = np.nan
            if np.isnan(np.nanmean(vals)):
                continue
            if len(vals) == 0:
                continue



            vals_reshape = vals.reshape(-1, 12)
            # print(len(vals_reshape))
            vals_reshape_T = vals_reshape.T
            vals_mon_interp = []
            for mon in range(12):
                vals_mon = vals_reshape_T[mon]
                # print(len(vals_mon))
                # interp = interpolate.interp1d(year_list, vals_mon, kind='linear')
                # a,b,r,p = T.nan_line_fit(year_list, vals_mon)
                # print(year_list)
                # print(len(year_list))
                # print(len(vals_mon));exit()
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
    def check_spatial(self):
        fdir=rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\phenology_year_extraction\\'

        dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in dic:
            vals = dic[pix]['ecosystem_year']
            if np.isnan(np.nanmean(vals)):
                print(pix)
            average = np.nanmean(vals)
            result_dic[pix] = average
        array=DIC_and_TIF().pix_dic_to_spatial_arr(result_dic)
        plt.imshow(array)
        plt.colorbar()
        plt.show()



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


class visualize_SHAP():

    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'

        self.spatial_dic = {0: 'CO2_ecosystem_year_shap',
                       1: 'detrended_sum_rainfall_growing_season_CV_shap',
                       2: 'heat_event_frenquency_growing_season_shap',
                       3: 'rainfall_frenquency_growing_season_shap',
                       4: 'rainfall_frenquency_non_growing_season_shap',
                       5: 'rainfall_intensity_growing_season_shap',
                       6: 'rainfall_intensity_non_growing_season_shap',
                       7: 'rainfall_seasonality_all_year_ecosystem_year_shap',
                       8: 'rooting_depth_shap',
                       9: 'silt_shap'}

        self.regroup_dic = {0: 'CO2',
                       1: 'Interanual_rainfall_CV',
                       2: 'Heat_event_frenquency',
                       3: 'Intraanual_rainfall',
                       4: 'Intraanual_rainfall',
                       5: 'Intraanual_rainfall',
                       6: 'Intraanual_rainfall',
                       7: 'Intraanual_rainfall',
                       8: 'Rooting_depth',
                       9: 'Silt'}

        pass

    def run(self):

        self.dominant_factor_shifting()
        pass
    def important_bar_ploting(self):  ##### plot for 4 clusters
        pass
    def SHAP_ploting(self):
        pass

    def spatial_importance_factor(self):  ##### plot for 4 clusters
        pass

    def dominant_factor_shifting(self):
        ## dominant factor spatial map regroup in 4 groups 1 interannual rainfall CV
        ## 2 intraannual rainfall 3 silt 5 root depth

        regroup_dic = {}

        period_list = ['first', 'second']
        result_period={}
        for period in period_list:
            dic = {}
            fpath=rf'D:\Project3\Result\ERA5_025\1mm\SHAP_CO2_{period}_Decade\png\pdp_shap_CV\shapely_contribution\\max_variable.tif'


            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)


            dic['CO2']=0
            dic['Interanual_rainfall_CV']=0
            dic['Heat_event_frenquency']=0
            dic['Intraanual_rainfall']=0
            dic['Silt']=0
            dic['Rooting_depth']=0

            array = np.array(array)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)


            for pix in val_dic:
                vals = val_dic[pix]
                if vals<-99:
                    continue
                result_dic = self.regroup_dic[vals]

                dic[result_dic] = dic[result_dic] + 1

            result_period[period]=dic

        ## dic to df
        df=pd.DataFrame(result_period)
        ##

        df.plot.barh()

        plt.show()












        fdir = rf'D:\Project3\Data\CO2\CO2_TIFF\unify_05\phenology_year_extraction\\'

        pass

class PLOT_dataframe():

    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass

    def run(self):

        # self.plot_anomaly_LAI_based_on_cluster()
        self.plot_time_series()
        pass

    def plot_anomaly_LAI_based_on_cluster(self):  ##### plot for 4 clusters

        df = T.load_df(result_root+rf'\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season.df')
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


        variable_list=['phenology_LAI_mean']

        for product in variable_list:
            # print('=========')
            # print(product)
            # print(df_continent.columns.tolist())
            # exit()
            # T.print_head_n(df_continent)
            # exit()


            vals = df[product].tolist()
            # pixel_area_sum = df_continent['pixel_area'].sum()


            # print(vals)

            vals_nonnan=[]
            for val in vals:

                if type(val)==float: ## only screening
                    continue
                if len(val) ==0:
                    continue


                vals_nonnan.append(list(val))


            ###### calculate mean
            vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
            vals_mean=np.nanmean(vals_mean,axis=0)
            # vals_mean=vals_mean/pixel_area_sum

            val_std=np.nanstd(vals_mean,axis=0)

            # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
            plt.plot(vals_mean,color=color_list[i],linewidth=linewidth_list[variable_list.index(product)])
            i = i + 1
            plt.legend()


        year_range = range(1983, 2021)
        xticks=range(0,len(year_range),4)
        plt.xticks(xticks,year_range[::4])
        plt.ylim(-8, 8)
        plt.xlabel('year')
        plt.ylabel(f'relative change (%)')
        plt.grid(which='major', alpha=0.5)
        plt.show()
        # out_pdf_fdir=rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\trend\\pdf\\'
        # plt.savefig(out_pdf_fdir + 'time_series.pdf', dpi=300, bbox_inches='tight')
        # plt.close()


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



class PLOT_data():
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
class greening_analysis():
    def __init__(self):
        pass

    def run(self):

        # self.relative_change()
        # self.weighted_average_LAI()
        # self.plot_time_series()
        # self.plot_time_series_spatial()
        # self.annual_growth_rate()
        # self.trend_analysis()
        # self.heatmap()
        self.heatmap()
        # self.testrobinson()

        pass

    def relative_change(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir = result_root + rf'\3mm\extract_LAI4g_phenology_year\extraction_LAI4g\\'
        outdir = result_root + rf'\3mm\\relative_change_growing_season\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'phenology_LAI_mean' in f:
                continue

            outf = outdir + f.split('.')[0] + '.npy'
            # if isfile(outf):
            #     continue
            print(outf)

            dic = T.load_npy(fdir + f)

            zscore_dic = {}

            for pix in tqdm(dic):
                delta_time_series_list = []
                if pix not in dic_dryland_mask:
                    continue

                # print(len(dic[pix]))
                time_series = dic[pix]['growing_season']
                # print(len(time_series))

                time_series = np.array(time_series)
                # time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue

                time_series = time_series
                mean = np.nanmean(time_series)
                relative_change = (time_series - mean) / mean * 100

                zscore_dic[pix] = relative_change
                # plot
                plt.plot(time_series)

                # plt.plot(relative_change)
                # plt.legend(['original','relative_change'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)

    def weighted_average_LAI(self):  ###add weighted average LAI in dataframe
        df =result_root+rf'\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season_yearly.df'
        df = T.load_df(df)
        df_clean = self.df_clean(df)
        # print(len(df_clean))

        # 去除异常值（根据业务需求设定阈值）
        df_clean_ii = df_clean[(df_clean['LAI_relative_change'] > -50) & (df_clean['LAI_relative_change'] < 50)]
        # print(len(df_clean_ii));exit()
        # Step 1: 计算纬度权重
        df_clean_ii['latitude_weight'] = np.cos(np.radians(df_clean_ii['lat']))
        # Step 2: 按年份对权重进行归一化
        # 确保每一年干旱区的权重总和为1
        df_clean_ii['normalized_weight'] = df_clean_ii.groupby('year')['latitude_weight'].transform(lambda x: x / x.sum())

        # print(df_clean_ii.groupby('year')['normalized_weight'].sum())
        # exit()

        # Step 3: 计算加权平均LAI
        # weighted_avg_lai = df_clean_ii.groupby('year')['LAI_relative_change'].apply(lambda x: (x * df_clean_ii['normalized_weight']).sum())
        df_clean_ii['weighted_avg_lai_contribution'] = df_clean_ii['LAI_relative_change'] * df_clean_ii['normalized_weight']

        weighted_avg_lai_per_year = (
            df_clean_ii.groupby('year')['weighted_avg_lai_contribution'].sum().reset_index(name='weighted_avg_LAI')
        )

        df_clean_ii = df_clean_ii.merge(weighted_avg_lai_per_year, on='year', how='left')
        T.print_head_n(df_clean_ii)

        # exit()
        # T.print_head_n(df_clean_ii);exit()
        outf=result_root+rf'\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season_yearly.df'
        T.save_df(df_clean_ii,outf)
        T.df_to_excel(df_clean_ii, outf)

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
    def plot_time_series(self):
        df=T.load_df(rf'E:\Project3\Result\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season_yearly.df')
        df=self.df_clean(df)
        weight_effect = df['normalized_weight'].mean()
        # print(f"Average Weight Effect: {weight_effect}")
        # exit()
        year_range = range(1983, 2021)
        result_dic = {}
        for year in year_range:
            df_i = df[df['year'] == year]
            vals = df_i['weighted_avg_LAI'].tolist()
            result_dic[year] = np.nanmean(vals)


        plt.plot(result_dic.values(),'g')
        year_range = range(1983, 2021)
        xticks = range(0, len(year_range), 4)
        plt.xticks(xticks, year_range[::4], fontsize=10, rotation=45)
        plt.ylim(-4, 4)
        plt.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4], fontsize=10)
        plt.xlabel('year', fontsize=10)
        plt.ylabel(f'Relative change (%)', fontsize=10)
        ##add line
        # plt.axhline(y=0, color='grey', linestyle='-',alpha=0.2)
        plt.grid(which='major', alpha=0.2)

        plt.show()




        pass


    def plot_time_series_spatial(self):
        fdir = result_root + rf'3mm\relative_change_growing_season\\'

        for f in os.listdir(fdir):
            if not 'LAI' in f:
                continue
            outf = fdir + f
            print(outf)
            dic = np.load(outf, allow_pickle=True).item()
            all_data = []


            for pix in dic:
                time_series = dic[pix]
                # print(time_series)
                if np.isnan(np.nanmean(time_series)):
                    continue

                all_data.append(time_series)

            all_data = np.array(all_data)
            average_data = np.nanmean(all_data, axis=0)


            plt.plot(average_data)
            plt.show()





    def annual_growth_rate(self):
        ## input raw data becuase time2-time1
        fdir=result_root + rf'\3mm\extract_LAI4g_phenology_year\extraction_LAI4g\\'
        outdir=result_root + rf'annual_growth_rate\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if 'detrend' in f:
                continue

            dict=np.load(fdir+f,allow_pickle=True).item()
            growth_rate_dic={}
            for pix in tqdm(dict):
                time_series=dict[pix]['growing_season']
                # print(len(time_series))
                if len(time_series)==0:
                    continue
                # print(time_series)
                growth_rate_time_series=np.zeros(len(time_series)-1)
                for i in range(len(time_series)-1):

                    growth_rate_time_series[i]=(time_series[i+1]-time_series[i])/time_series[i]*100
                growth_rate_dic[pix]=growth_rate_time_series
            np.save(outdir+f,growth_rate_dic)

        pass

    def trend_analysis(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        fdir = result_root+rf'\3mm\CRU_JRA_monthly\extract_rainfall_phenology_year\\'
        outdir = result_root + rf'\3mm\CRU_JRA_monthly\extract_rainfall_phenology_year\\trend_analysis\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            if not f.endswith('.npy'):
                continue

            outf=outdir+f.split('.')[0]
            # if os.path.isfile(outf+'_trend.tif'):
            #     continue
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

                time_series=dic[pix]['growing_season']
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
                try:

                        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    slope,b,r,p_value=T.nan_line_fit(np.arange(len(time_series)), time_series)
                    trend_dic[pix] = slope
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

    def gen_robinson_template(self):
        pass
    def testrobinson(self):

        fdir_trend = result_root+rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result\npy_time_series\trend\\'
        temp_root = result_root+rf'\3mm\moving_window_multi_regression\moving_window\multi_regression_result\npy_time_series\trend\\'
        outdir = result_root+rf'\3mm\moving_window_multi_regression\moving_window\multi_regression_result\npy_time_series\trend_plot\\'
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
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=-2, vmax=2, is_discrete=True, colormap_n=7,)

            Plot_Robinson().plot_Robinson_significance_scatter(m,p_value_f,temp_root,0.05, s=0.2, marker='.')
            plt.title(f'{fname}')
            # plt.show()
            outf = outdir + f+'2.pdf'
            plt.savefig(outf)
            plt.close()

    def heatmap(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        f_LAI_trend = result_root + rf'3mm\relative_change_growing_season\trend\\phenology_LAI_mean_trend.tif'
        f_LAI_sensitivity_precipitation=result_root+rf'\3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend\npy_time_series\\trend\\\sum_rainfall_trend.tif'
        f_precip_trend=result_root + rf'\3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\trend_ecosystem_year\\sum_rainfall_trend.tif'

        arr_LAI_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_LAI_trend)

        arr_LAI_trend[arr_LAI_trend < -999] = np.nan

        arr_LAI_sensitivity_precip, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_LAI_sensitivity_precipitation)
        arr_precip_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_precip_trend)
        arr_LAI_sensitivity_precip[arr_LAI_sensitivity_precip < -999] = np.nan
        arr_precip_trend[arr_precip_trend < -999] = np.nan
        arr_LAI_trend=np.array(arr_LAI_trend)
        arr_LAI_sensitivity_precip=np.array(arr_LAI_sensitivity_precip)
        arr_precip_trend=np.array(arr_precip_trend)

        dic_LAI_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_trend)
        dic_arr_LAI_sensitivity_precip=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_sensitivity_precip)
        dic_precip_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_precip_trend)

        result_dic={
            'LAI_trend':dic_LAI_trend,
            'LAI_sensitivity_precip_trend':dic_arr_LAI_sensitivity_precip,
            'intra_Preci_CV_trend':dic_precip_trend
        }
        # plt.hist(result_dic['LAI_trend'].values())
        # plt.show()
        # plt.hist(result_dic['LAI_sensitivity_precip_trend'].values())
        # plt.show()
        # plt.hist(result_dic['intra_Preci_CV_trend'].values())
        # plt.show()


        df=T.spatial_dics_to_df(result_dic)
        T.print_head_n(df)
        x_var = 'LAI_sensitivity_precip_trend'
        y_var = 'intra_Preci_CV_trend'
        z_var = 'LAI_trend'
        bin_x = [ -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]
        bin_y = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure()

        matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)

        T.plot_df_bin_2d_matrix(matrix_dict,-.3,.3,x_ticks_list,y_ticks_list,cmap='RdBu',
                              is_only_return_matrix=False)

        plt.colorbar()
        plt.show()
        matrix_dict, x_ticks_list, y_ticks_list = self.df_bin_2d_sample_size(df, val_col_name=z_var,
                                                                             col_name_x=x_var,
                                                                             col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        T.plot_df_bin_2d_matrix(matrix_dict, 0, 300, x_ticks_list, y_ticks_list, cmap='RdBu',
                                is_only_return_matrix=False)

        plt.colorbar()
        plt.show()


    #     plt.savefig(result_root + rf'Data_frame\\Frequency\\Trendy_{region}.pdf', dpi=300, )
    #     plt.close()

    def heatmap2(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        f_LAI_trend = result_root + rf'\3mm\relative_change_growing_season\trend\\phenology_LAI_mean_trend.tif'
        f_aridity=data_root+rf'Base_data\aridity_index_05\dryland_mask05.tif'
        f_precip_trend=result_root + rf'\3mm\CRU_JRA\extract_rainfall_phenology_year\trend_ecosystem_year\\sum_rainfall_trend.tif'

        arr_LAI_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_LAI_trend)

        arr_LAI_trend[arr_LAI_trend<-999]=np.nan
        # plt.imshow(arr_LAI_trend)
        # plt.show()

        aridity, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_aridity)
        aridity[aridity<-999]=np.nan
        aridity[aridity>999]=np.nan
        # plt.imshow(aridity)
        # plt.show()

        precip_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_precip_trend)
        precip_trend[precip_trend<-999]=np.nan
        precip_trend[precip_trend>999]=np.nan
        # plt.imshow(precip_trend)
        # plt.show()


        dic_LAI_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_trend)
        aridty_dic=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(aridity)
        precip_trend_dic=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(precip_trend)

        result_dic={
            'LAI_trend':dic_LAI_trend,
            'aridity':aridty_dic,
            'precip_trend':precip_trend_dic
        }
        df=T.spatial_dics_to_df(result_dic)
        df=df.dropna()
        T.print_head_n(df)
        x_var = 'aridity'
        y_var = 'precip_trend'
        z_var = 'LAI_trend'
        bin_x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
        bin_x=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
        bin_y = [-1,-0.5,0,0.5,1]

        # percentile_list=np.linspace(0,100,9)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure()


        matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)
        T.plot_df_bin_2d_matrix(matrix_dict, -.2, .2, x_ticks_list, y_ticks_list, cmap='RdBu',
                                is_only_return_matrix=False)
        plt.colorbar()
        plt.figure()


        matrix_dict, x_ticks_list, y_ticks_list = self.df_bin_2d_sample_size(df, val_col_name=z_var,
                                                              col_name_x=x_var,
                                                              col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)


        T.plot_df_bin_2d_matrix(matrix_dict,0,300,x_ticks_list,y_ticks_list,cmap='RdBu',
                              is_only_return_matrix=False)

        plt.colorbar()
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.show()
    def df_bin_2d_sample_size(self,df,val_col_name,col_name_x,col_name_y,bin_x,bin_y,round_x=2,round_y=2):
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
        return matrix_dict,x_ticks_list,y_ticks_list





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
        '''
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        # Blue represents high values, and red represents low values.
        if ax == None:
            # plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list)
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
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
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
class bivariate_analysis():
    def __init__(self):
        pass
    def run(self):
        self.bivariate_plot()
        # self.xy_map_growth_rate()  ## growth rate, rainfall seasonal distribution use the same  color scale
        # self.xy_map_heat_event()


        pass
    def bivariate_plot(self):
        result_root = rf'E:\Project3\Result\\'
        # print(result_root)

        import xymap
        tif_long_term= result_root+rf'\3mm\annual_growth_rate\trend\\phenology_LAI_mean_trend.tif'
        tif_window=result_root+rf'\3mm\relative_change_growing_season\trend\\phenology_LAI_mean_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outdir=result_root + rf'3mm\\\bivariate_analysis\\'
        T.mk_dir(outdir, force=True)
        outtif=outdir+rf'\\relative_change_growth_rate.tif'
        T.mk_dir(result_root + rf'bivariate_analysis\\')
        tif1=tif_long_term
        tif2=   tif_window

        tif1_label='Relative_change (%/year)'
        tif2_label='growing rate (%)/year'
        min1=-0.3
        max1=0.3
        min2=-0.02
        max2=0.02
        outf=outtif
        upper_right_color = [143, 196, 34],  #
        upper_left_color = [156, 65, 148],  #
        lower_right_color = [29, 46, 97],  #
        lower_left_color = [238, 233, 57],  #
        center_color = [240, 240, 240],  #
        # print(xymap.Bivariate_plot_1().upper_left_color)
        # xymap.Bivariate_plot().plot_bivariate_map(tif1, tif2, tif1_label, tif2_label, min1, max1, min2, max2, outf)
        Biv = xymap.Bivariate_plot_1(upper_right_color =[64, 224, 208],
                                      upper_left_color = [248,222,126],
                                      lower_right_color = [0,100,0],
                                      lower_left_color = [255,0,0],
                                      center_color = [240, 240, 240])
        # Biv.upper_left_color = upper_left_color
        # Biv.upper_right_color = upper_right_color
        # Biv.lower_left_color = lower_left_color
        # Biv.lower_right_color = lower_right_color
        # Biv.center_color = center_color

        # print(Biv.lower_right_color);exit()
        Biv.plot_bivariate(tif1, tif2, tif1_label, tif2_label, min1, max1, min2, max2, outf)
        print(outf)
        plt.show()

    def xy_map_growth_rate(self):


        import xymap


        fdir_CV = rf'E:\Project3\Data\ERA5_daily\dict\trend_analysis_moving_window\\'
        fdir_trend = rf'D:\Project3\Result\trend_analysis\relative_change\OBS_extend\\'
        outdir = rf'D:\Project3\Result\bivariate\\\LAI_trend_vs_CV\\'
        T.mk_dir(outdir,force=True)

        tif_LAI4g_trend = fdir_CV+'LAI4g_CV_trend.tif'
        tif_LAI4g_trend_growth_rate_trend = fdir_trend+'LAI4g_trend.tif'

        outf = join(outdir,f'LAI_trend_vs_CV.tif')
        x_label = 'LAI4g_CV_trend(%/year)'
        y_label = 'LAI4g_trend (%//year)'
        min1 = -1
        max1 = 1
        min2 = -.05
        max2 = .05
        xymap.Bivariate_plot_1(alpha = 255,upper_left_color = (255,202, 202), #
                 upper_right_color = (148, 202, 112), #
                 lower_left_color = (110,0, 0), #
                 lower_right_color = (0, 0, 110), #
                 center_color = (240,240, 240), ).plot_bivariate(
            tif_LAI4g_trend, tif_LAI4g_trend_growth_rate_trend,
            x_label, y_label, min1, max1, min2, max2, outf)
        print(outf)


    def xy_map_heat_event(self): ##

        import xymap

        fdir = rf'D:\Project3\Result\bivariate\\seasonality_rainfall_LAI_CV\\'
        outdir = rf'D:\Project3\Result\bivariate\results\Bivariate_plot\\\tif\\seasonality_rainfall_LAI_CV\\'
        T.mk_dir(outdir,force=True)

        tif_LAI4g_trend = join(fdir,'heat_event_frequency_trend.tif')
        tif_LAI4g_trend_growth_rate_trend = join(fdir,'detrended_annual_LAI4g_CV_trend.tif')

        outf = join(outdir,f'heat_event_frequency_trend.tif')
        x_label = 'heat_event_frequency_trend(unitless/year)'
        y_label = 'CV trend (%/year)'
        min1 = -0.1
        max1 = 0.1
        min2 = -.5
        max2 = .5
        xymap.Bivariate_plot_1(alpha = 255,upper_right_color = (255,202, 202), #
                 upper_left_color = (148, 202, 112), #
                 lower_right_color = (110,0, 0), #
                 lower_left_color = (0, 0, 110), #
                 center_color = (240,240, 240), ).plot_bivariate(
            tif_LAI4g_trend, tif_LAI4g_trend_growth_rate_trend,
            x_label, y_label, min1, max1, min2, max2, outf)
        print(outf)
class multi_regression_window():
    def __init__(self):
        self.this_root = 'E:\Project3\\'
        self.data_root = 'E:/Project3/Data/'
        self.result_root = 'E:/Project3/Result/'

        self.fdirX=self.result_root+rf'\3mm\moving_window_multi_regression\moving_window\window_trend_growing_season\\'
        self.fdir_Y=self.result_root+rf'\3mm\moving_window_multi_regression\moving_window\window_trend_growing_season\\'

        self.xvar_list = ['sum_rainfall','Tmax','VPD']
        self.y_var = ['phenology_LAI_mean_relative_change']
        pass

    def run(self):
        # self.detrend()
        # self.moving_window_extraction()

        self.window = 38-15+1
        outdir = self.result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend\\'
        T.mk_dir(outdir, force=True)

        # # ####step 1 build dataframe
        # for i in range(self.window):
        #
        #     df_i = self.build_df(self.fdirX, self.fdir_Y, self.xvar_list, self.y_var,i)
        #     outf= outdir+rf'\\window{i:02d}.npy'
        #     if os.path.isfile(outf):
        #         continue
        #     print(outf)
        # #
        #     self.cal_multi_regression_beta(df_i,self.xvar_list, outf)  # 修改参数
        # ###step 2 crate individial files
        # self.plt_multi_regression_result(outdir,self.y_var)
#
        # ##step 3 covert to time series

        # self.convert_files_to_time_series(outdir,self.y_var)
        ### step 4 build dataframe using build Dataframe function and then plot here
        # self.plot_moving_window_time_series()
        ## spatial trends of sensitivity
        self.calculate_trend_trend()
        # self.plot_sensitivity_preicipation_trend()
    def detrend(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=result_root + rf'\3mm\moving_window_multi_regression\anomaly_growing_season\\'
        outdir=result_root + rf'\3mm\moving_window_multi_regression\anomaly_growing_season\\detrend\\'
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

        fdir_all = self.result_root + rf'3mm\moving_window_multi_regression\anomaly_growing_season\\detrend\\'

        outdir = self.result_root  + rf'\3mm\moving_window_multi_regression\moving_window\window_detrend_growing_season\\'
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



        variable_list = ['sum_rainfall']



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

    def plot_sensitivity_preicipation_trend(self):
        ### plot heamap of sensitivity of precipitation to LAI as function of  mean precipitaiton and precipitation trends

        dff = rf'D:\Project3\Result\Dataframe\relative_changes\relative_changes.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        # T.print_head_n(df)
        df = df.dropna(subset=['LAI4g_trend'])

        ## x is CRU, y CO2, z is LAI4g
        x = df['GPCC_LAI4g_trend_100mm_unit']
        y = df['GPCC_trend'].tolist()

        z=df['LAI4g_trend'].tolist()
        plt.hist(x, bins=50)
        plt.show()
        # plt.hist(y, bins=50)
        # plt.show()
        # plt.hist(z, bins=50)
        # plt.show()

        x_list = np.linspace(-10, 10, 21)
        y_list = np.linspace(-2, 2, 21)
        df_group1, bins_list_str1 = T.df_bin(df, 'GPCC_LAI4g_trend_100mm_unit',x_list)

        for name1, df_group_i1 in df_group1:
            df_group2, bins_list_str2 = T.df_bin(df_group_i1, 'GPCC_trend', y_list)
            name1_ = name1[0].left
            # print(name1_);exit()
            matrix_i = []
            matrix_ii = []
            x_labels = []
            # print(len(df_group2))
            # print(len(CO2_bin_list)-1)
            # print('---')

            for name2, df_group_i2 in df_group2:
                name2_ = name2[0].left
                x_labels.append(name2_)
                # print(name1,name2)
                # print(len(df_group_i2))
                vals = df_group_i2['LAI4g_trend'].tolist()
                vals = np.array(vals)
                vals[vals < -999] = np.nan
                vals[vals > 999] = np.nan
                val = np.nanmean(vals)


                count = len(df_group_i2)

                x_pos = name1_
                y_pos = name2_
                plt.scatter(x_pos, y_pos, s=100,c=val,cmap='RdBu',vmin=-1,vmax=1,marker='s')
                ### add text of the value
                plt.text(x_pos, y_pos, count, ha='center', va='center', fontsize=6)
                plt.xlabel('Trends in LAI sensitivity to precipitation (%/100mm/year)')
                plt.ylabel('Precipitaiton_trend (%/year)')
                # plt.title('Trends in LAI sensitivity to precipitation (%/mm/year)')

        ## colorbars has text
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('LAI trend (%/year)')
        plt.show()

    pass

    def calculate_trend_trend(self):  ## calculate the trend of trend

    ## here input is the npy file
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=rf'E:\Project3\Result\3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend\npy_time_series\\'
        outdir = rf'E:\Project3\Result\3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend\npy_time_series\\trend\\'

        T.mkdir(outdir)

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
            plt.imshow(arr_trend)
            plt.colorbar()
            plt.show()
            outf = outdir + f.split('.')[0] + '_trend.tif'
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_trend,outf)
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_p_value, outf + '_p_value.tif')
                ## save
            # np.save(outf, trend_dic)
            # np.save(outf+'_p_value', p_value_dic)

            ##tiff

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

class multi_regression():  ###linaer regression for CO2 effects.
    def __init__(self):
        self.this_root = 'E:\Project3\\'
        self.data_root = 'E:/Project3/Data/'
        self.result_root = 'E:/Project3/Result/'

        self.fdirX=self.result_root+rf'\3mm\moving_window_multi_regression\moving_window\window_trend_growing_season\\'
        self.fdir_Y=self.result_root+rf'\3mm\moving_window_multi_regression\moving_window\window_trend_growing_season\\'

        self.xvar_list = ['sum_rainfall','Tmax','VPD']
        self.y_var = ['phenology_LAI_mean_relative_change']
        pass

    def run(self):
        # self.detrend()
        # self.moving_window_extraction()

        self.window = 38-15+1
        outdir = self.result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend\\'
        T.mk_dir(outdir, force=True)

        # # ####step 1 build dataframe
        # for i in range(self.window):
        #
        #     df_i = self.build_df(self.fdirX, self.fdir_Y, self.xvar_list, self.y_var,i)
        #     outf= outdir+rf'\\window{i:02d}.npy'
        #     if os.path.isfile(outf):
        #         continue
        #     print(outf)
        # #
        #     self.cal_multi_regression_beta(df_i,self.xvar_list, outf)  # 修改参数
        # ###step 2 crate individial files
        # self.plt_multi_regression_result(outdir,self.y_var)
#
        # ##step 3 covert to time series

        # self.convert_files_to_time_series(outdir,self.y_var)
        ### step 4 build dataframe using build Dataframe function and then plot here
        # self.plot_moving_window_time_series()
        ## spatial trends of sensitivity
        self.calculate_trend_trend()
        # self.plot_sensitivity_preicipation_trend()

    def moving_window_extraction(self):

        fdir_all = self.result_root + rf'3mm\moving_window_multi_regression\anomaly_growing_season\\detrend\\'

        outdir = self.result_root  + rf'\3mm\moving_window_multi_regression\moving_window\window_detrend_growing_season\\'
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



        variable_list = ['sum_rainfall']



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

    def plot_sensitivity_preicipation_trend(self):
        ### plot heamap of sensitivity of precipitation to LAI as function of  mean precipitaiton and precipitation trends

        dff = rf'D:\Project3\Result\Dataframe\relative_changes\relative_changes.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        # T.print_head_n(df)
        df = df.dropna(subset=['LAI4g_trend'])

        ## x is CRU, y CO2, z is LAI4g
        x = df['GPCC_LAI4g_trend_100mm_unit']
        y = df['GPCC_trend'].tolist()

        z=df['LAI4g_trend'].tolist()
        plt.hist(x, bins=50)
        plt.show()
        # plt.hist(y, bins=50)
        # plt.show()
        # plt.hist(z, bins=50)
        # plt.show()

        x_list = np.linspace(-10, 10, 21)
        y_list = np.linspace(-2, 2, 21)
        df_group1, bins_list_str1 = T.df_bin(df, 'GPCC_LAI4g_trend_100mm_unit',x_list)

        for name1, df_group_i1 in df_group1:
            df_group2, bins_list_str2 = T.df_bin(df_group_i1, 'GPCC_trend', y_list)
            name1_ = name1[0].left
            # print(name1_);exit()
            matrix_i = []
            matrix_ii = []
            x_labels = []
            # print(len(df_group2))
            # print(len(CO2_bin_list)-1)
            # print('---')

            for name2, df_group_i2 in df_group2:
                name2_ = name2[0].left
                x_labels.append(name2_)
                # print(name1,name2)
                # print(len(df_group_i2))
                vals = df_group_i2['LAI4g_trend'].tolist()
                vals = np.array(vals)
                vals[vals < -999] = np.nan
                vals[vals > 999] = np.nan
                val = np.nanmean(vals)


                count = len(df_group_i2)

                x_pos = name1_
                y_pos = name2_
                plt.scatter(x_pos, y_pos, s=100,c=val,cmap='RdBu',vmin=-1,vmax=1,marker='s')
                ### add text of the value
                plt.text(x_pos, y_pos, count, ha='center', va='center', fontsize=6)
                plt.xlabel('Trends in LAI sensitivity to precipitation (%/100mm/year)')
                plt.ylabel('Precipitaiton_trend (%/year)')
                # plt.title('Trends in LAI sensitivity to precipitation (%/mm/year)')

        ## colorbars has text
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('LAI trend (%/year)')
        plt.show()

    pass

    def calculate_trend_trend(self):  ## calculate the trend of trend

    ## here input is the npy file
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir=rf'E:\Project3\Result\3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend\npy_time_series\\'
        outdir = rf'E:\Project3\Result\3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend\npy_time_series\\trend\\'

        T.mkdir(outdir)

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
            plt.imshow(arr_trend)
            plt.colorbar()
            plt.show()
            outf = outdir + f.split('.')[0] + '_trend.tif'
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_trend,outf)
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_p_value, outf + '_p_value.tif')
                ## save
            # np.save(outf, trend_dic)
            # np.save(outf+'_p_value', p_value_dic)

            ##tiff

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
    # Data_processing_2().run()
    # Phenology().run()
    # build_dataframe().run()
    build_moving_window_dataframe().run()
    # CO2_processing().run()
    # greening_analysis().run()
    # multi_regression_window().run()
    # bivariate_analysis().run()

    # visualize_SHAP().run()
    # PLOT_dataframe().run()
    # Plot_Robinson().robinson_template()
    # plt.show()



    pass

if __name__ == '__main__':
    main()