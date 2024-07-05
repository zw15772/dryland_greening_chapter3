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
T=Tools()




this_root = 'D:\Project3\\'
data_root = 'D:/Project3/Data/'
result_root = 'D:/Project3/Result/'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)


class build_dataframe():


    def __init__(self):

        self.this_class_arr = result_root + 'Dataframe\moving_window_multiregression_anomaly\\'


        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'moving_window_multiregression_anomaly.df'



        pass

    def run(self):

        df = self.__gen_df_init(self.dff)


        # df=self.foo2(df)
        df=self.append_attribute(df)

        # df=self.add_multiregression_tiff_to_df(df)
        # df=self.add_multiregression_npy_to_df(df)


        # df=self.add_AI_classfication(df)
        # # #
        # df=self.add_aridity_to_df(df)
        # # # #
        # df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        # df=self.add_landcover_classfication_to_df(df)
        # df=self.add_row(df)
        # df=self.add_maxmium_LC_change(df)
        # df=self.add_continent_to_df(df)
        # df=self.add_lat_lon_to_df(df)
        # df=self.add_soil_texture_to_df(df)
        # df=self.add_trend_to_df(df)
        # df=self.add_max_trend_to_df(df)

        # df=self.add_Ndepostion_to_df(df)
        #

        # df=self.rename_columns(df)

        # df = self.drop_field(df)
        self.show_field(df)

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



    def foo2(self, df):  # 新建trend

        f = result_root + rf'multi_regression\1982_2020\\CO2_LAI4g_1982_2020.tif'
        val_array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)

        # val_array = np.load(f)
        val_array[val_array<-99]=np.nan
        val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
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
    def append_attribute(self, df):

        fdir = result_root + rf'\multi_regression_moving_window\window15_anomaly\npy_time_series\\'
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
                continue


            # array=np.load(fdir+f)
            # dic = DIC_and_TIF().spatial_arr_to_dic(array)
            dic = T.load_npy(fdir + f)
            key_name = f.split('.')[0]
            print(key_name)

            # df[key_name] = df['pix'].map(dic)
            # T.print_head_n(df)
            df = T.add_spatial_dic_to_df(df, dic, key_name)
        return df
        pass

    def add_multiregression_tiff_to_df(self, df):
        fdir = result_root + rf'multi_regression_moving_window\window15_anomaly\trend\\'
        for f in os.listdir(fdir):

            if not f.endswith('.tif'):
                continue



            print(f.split('.')[0])
            # exit()

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
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            fname=f.split('.')[0]
            print(fname)
            # exit()
            df[fname] = val_list
        return df



        pass


    def add_multiregression_npy_to_df(self, df):
        period='2002_2020'
        f = result_root + rf'\multi_regression\anomaly\\{period}\\CO2_LAI4g_{period}.npy'
        val_dic = T.load_npy(f)

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
        fname=f'CO2_LAI4g_{period}'
        print(fname)
        # exit()
        df[fname] = val_list
        return df



        pass

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




    def rename_columns(self, df):
        df = df.rename(columns={'VPD_1982_differences': 'VPD_differences',


                            }

                               )


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
    def add_row(self, df):
        r_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix
            r_list.append(r)
        df['row'] = r_list
        return df
    def add_trend_to_df(self,df):

        fdir=result_root+rf'trend_analysis\anomaly\OBS_extend\\'

        for f in os.listdir(fdir):
            if not 'GPCC' in f:
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
            f_name=f.split('.')[0]
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

    def drop_field(self, df):
        df = df.drop(columns=['GPCC_1982_differences',
                              'LAI4g_1982_differences',
                             ]
                     )
        return df

    def show_field(self, df):
        print(df.columns)
        pass
class summary_multiregression_model_result():
    def run (self):

        df = self.clean_df(self.df)
        # self.plot_multiregression_moving_window()
        # self.plt_histgram_CO2_sensitivity(df)
        # self.plt_histgram_precipitation_sensitivity(df)
        # self.calculate_diff_between_periods_sensitivity()

        # self.heatmap_WUE()

        self.heatmap_VPD_SM_CO2()
        # self.plot_contribution_bar_plot()



        pass

    def __init__(self):
        self.dff = result_root + 'Dataframe\\multi_regression\\multi_regression.df'
        self.df = T.load_df(self.dff)

        pass

    def clean_df(self, df):
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]

        df=df[df['LC_max']<20]
        df=df[df['landcover_classfication']!='Cropland']


        # df=df[df['LAI4g_p_value']<0.05]
        # df=df[df['LAI4g_trend']>0]
        # df=df[df['continent']=='Australia']

        # df = df[df['continent'] == 'Africa']
        # df=df[df['continent']=='Asia']

        # df=df[df['continent']=='South_America']

        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        return df

    def plt_histgram_CO2_sensitivity(self,df):
        self.period_list = ['1982_2001', '2002_2020',]
        color_list = ['r', 'b']
        continent_list = ['Africa', 'Asia', 'Australia', 'South_America',  'North_America','global']

        variable_list= ['CO2_LAI4g',]
        fig = plt.figure(figsize=(15, 10))
        i=1


        for continent in continent_list:
            ax=fig.add_subplot(2, 3, i)


            flag=1
            if continent=='North_America':
                df_continent = df[df['lon'] > -125]
                df_continent = df_continent[df_continent['lon'] < -105]
                df_continent = df_continent[df_continent['lat'] > 0]
                df_continent = df_continent[df_continent['lat'] < 45]
            elif continent in ['Africa', 'Asia', 'Australia', 'South_America']:
                df_continent = df[df['continent'] == continent]

            else:
                df_continent = df
            for period in self.period_list:

                col=f'CO2_LAI4g_{period}'
                vals=df_continent[col].to_list()
                vals=np.array(vals,dtype=float)
                vals=vals*100  #
                  ## convert to %/100ppm

                vals=vals[~np.isnan(vals)]
                ax.hist(vals,bins=100,color=color_list[flag-1],alpha=0.5,label=f'{period} {continent}',density=True,edgecolor=color_list[flag-1],range=(-100,200))


                flag=flag+1
            i=i+1
            ## add line at x=0
            plt.axvline(x=0, color='black', linestyle='--')
            plt.xlabel('LAI sensitivity to CO2 (%/100ppm)')
            plt.ylabel('Frequency(%)')
            plt.title(continent)
            plt.legend(self.period_list)
        plt.show()

    def plot_multiregression_moving_window(self): #### plt average
        ## here I would like to plot average

        df = T.load_df(
            result_root + rf'Dataframe\moving_window_multiregression_anomaly\\moving_window_multiregression_anomaly.df')
        T.print_head_n(df)
        print(len(df))
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 20]
        df = df.dropna(subset=['window00_CO2_LAI4g'])
        # continent_list = ['Global', 'Africa', 'Asia', 'Australia', 'South_America', 'North_America']
        continent_list = ['Global']
        all_continent_dict = {}
        fig=plt.figure()
        i=1
        for continent in continent_list:
            ax = fig.add_subplot(1, 1, i)
            if continent == 'North_America':
                df_continent = df[df['lon'] > -125]
                df_continent = df_continent[df_continent['lon'] < -105]
                df_continent = df_continent[df_continent['lat'] > 0]
                df_continent = df_continent[df_continent['lat'] < 45]
            elif continent == 'Global':
                df_continent = df
            else:
                df_continent = df[df['continent'] == continent]

            vals = df_continent['CO2_LAI4g'].tolist()
            vals = np.array(vals)
            vals=vals*100

            # print(vals)

            vals_nonnan = []
            for val in vals:
                if type(val) == float:  ## only screening
                    continue
                if len(val) == 0:
                    continue
                val[val < -99] = np.nan

                if not len(val) == 39:
                    ## add nan to the end of the list
                    for j in range(1):
                        val = np.append(val, np.nan)
                    # print(val)
                    # print(len(val))

                vals_nonnan.append(list(val))

                # exit()
                # print(type(val))
                # print(len(val))
                # print(vals)

            ###### calculate mean
            vals_mean = np.array(vals_nonnan) ## axis=0, mean of each row  竖着加
            vals_mean = np.nanmean(vals_mean, axis=0)
            val_std = np.nanstd(vals_mean, axis=0)



            # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
            plt.plot(vals_mean, label=continent,color='g')
            plt.fill_between(range(len(vals_mean)), vals_mean - val_std, vals_mean + val_std, alpha=0.3, color='g')

            ax.set_xticks(range(0, 24, 4))
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
            plt.xticks(range(len(year_range_str))[::4], year_range_str[::4], rotation=45, ha='right', fontsize=12)
            plt.xlabel('year', fontsize=12)
            plt.yticks(fontsize=12)

            plt.ylabel('LAI sensitivity to CO2 (%/100ppm)', fontsize=12)
            i=i+1

        # plt.legend()

        plt.show()


    def plt_histgram_precipitation_sensitivity(self,df):
        self.period_list = ['1982_2001', '2002_2020']
        color_list = ['r', 'b']
        continent_list = ['Africa', 'Asia', 'Australia', 'South_America',  'North_America','global']


        fig = plt.figure(figsize=(15, 10))
        i=1


        for continent in continent_list:
            ax=fig.add_subplot(2, 3, i)


            flag=1
            if continent=='North_America':
                df_continent = df[df['lon'] > -125]
                df_continent = df_continent[df_continent['lon'] < -105]
                df_continent = df_continent[df_continent['lat'] > 0]
                df_continent = df_continent[df_continent['lat'] < 45]
            elif continent in ['Africa', 'Asia', 'Australia', 'South_America']:
                df_continent = df[df['continent'] == continent]
            else:
                df_continent = df
            for period in self.period_list:

                col=f'GPCC_LAI4g_{period}'
                vals=df_continent[col].to_list()
                vals=np.array(vals,dtype=float)
                vals=vals*100

                vals=vals[~np.isnan(vals)]
                ax.hist(vals,bins=100,color=color_list[flag-1],alpha=0.5,label=f'{period} {continent}',density=True,edgecolor=color_list[flag-1],range=(-1,2))


                flag=flag+1
            i=i+1
            ## add line at x=0
            plt.axvline(x=0, color='black', linestyle='--')
            plt.xlabel('LAI sensitivity to Precip (m2/m2/100mm)')
            plt.ylabel('Frequency(%)')
            plt.title(continent)
            plt.legend(self.period_list)
        plt.show()


        pass


    def heatmap_WUE(self): ## plot the heatmap of WUE
        dff = result_root + 'Dataframe\\multi_regression\\multi_regression.df'
        df = T.load_df(dff)
        T.print_head_n(df)

        df = df[df['row'] > 120]
        df=df[df['landcover_classfication']!='Cropland']
        # df=df[df['classfication_SM']!='no_sig_no_sig']
        print(len(df))
        ## get the unique landcover_classfications
        # landcover_classfications_list=df['landcover_classfication'].tolist()
        # landcover_classfications_list_unique=T.get_df_unique_val_list(df,'landcover_classfication')
        # print(landcover_classfications_list_unique)
        # exit()
        # df=df[df['landcover_classfication']=='Shrub']
        # print(len(df))
        # plt.hist(df[rf'GPCC_trend'], bins=100)
        # plt.show()
        # # plt.hist(df['GLEAM_SMroot_LAI4g_diff_detrended_diff'], bins=100)
        # # plt.show()
        # plt.hist(df[rf'VPD_trend'], bins=100)
        # plt.show()
        # exit()

        SM_p1_bin_list = np.linspace(-0.5,0.5, 11)
        VPD_p1_bin_list = np.linspace(-0.015, 0.015, 11)
        df_group1, bins_list_str1 = T.df_bin(df, rf'GPCC_trend', SM_p1_bin_list)
        matrix = []
        y_labels = []
        for name1, df_group_i1 in df_group1:
            df_group2, bins_list_str2 = T.df_bin(df_group_i1, rf'VPD_trend', VPD_p1_bin_list)
            name1_ = name1[0].left
            name1_=name1_*10

            matrix_i = []
            x_labels = []

            for name2, df_group_i2 in df_group2:
                name2_ = name2[0].left
                name2_=name2*100
                name2_=round(name2_,2)
                x_labels.append(name2_)
                # print(name1,name2)
                # print(len(df_group_i2))
                val = np.nanmean(df_group_i2[rf'GLEAM_SMroot_LAI4g_diff_detrended_diff'])

                matrix_i.append(val)

                # count=len(df_group_i2)
                # matrix_i.append(count)

                # matrix_i.append(val)
            matrix.append(matrix_i)
            y_labels.append(name1_)
        matrix = np.array(matrix)
        matrix = matrix[::-1, :]
        plt.imshow(matrix, cmap='RdBu', vmin=-3, vmax=3)
        # plt.imshow(matrix,cmap='RdBu')
        SM_p1_bin_list=SM_p1_bin_list*10
        VPD_p1_bin_list=VPD_p1_bin_list*100

        plt.xticks(np.arange(len(SM_p1_bin_list) - 1), x_labels, rotation=45)
        plt.yticks(np.arange(len(VPD_p1_bin_list) - 1), y_labels[::-1])

        plt.xlabel('VPD trend *100 (Kpa/yr)')
        plt.ylabel('Precip trend *10 (mm/month)')
        ## draw 1:1 line
        # plt.plot([20, 0], [0, 20], 'k-', lw=2)
        ## add colorbar unit

        cbar=plt.colorbar()
        cbar.set_label(r'$\Delta$ ($\delta$ LAI/$\delta$ SM) (m2/m2/100mm)')

        plt.show()


    def heatmap_WUE(self): ## plot the heatmap of WUE
        dff = result_root + 'Dataframe\\multi_regression\\multi_regression.df'
        df = T.load_df(dff)
        T.print_head_n(df)

        df = df[df['row'] > 120]
        df=df[df['landcover_classfication']!='Cropland']
        # df=df[df['classfication_SM']!='no_sig_no_sig']
        print(len(df))
        ## get the unique landcover_classfications
        # landcover_classfications_list=df['landcover_classfication'].tolist()
        # landcover_classfications_list_unique=T.get_df_unique_val_list(df,'landcover_classfication')
        # print(landcover_classfications_list_unique)
        # exit()
        # df=df[df['landcover_classfication']=='Shrub']
        # print(len(df))
        # plt.hist(df[rf'GPCC_trend'], bins=100)
        # plt.show()
        # # plt.hist(df['GLEAM_SMroot_LAI4g_diff_detrended_diff'], bins=100)
        # # plt.show()
        # plt.hist(df[rf'VPD_trend'], bins=100)
        # plt.show()
        # exit()

        SM_p1_bin_list = np.linspace(-0.5,0.5, 11)
        VPD_p1_bin_list = np.linspace(-0.015, 0.015, 11)
        df_group1, bins_list_str1 = T.df_bin(df, rf'GPCC_trend', SM_p1_bin_list)
        matrix = []
        y_labels = []
        for name1, df_group_i1 in df_group1:
            df_group2, bins_list_str2 = T.df_bin(df_group_i1, rf'VPD_trend', VPD_p1_bin_list)
            name1_ = name1[0].left
            name1_=name1_*10

            matrix_i = []
            x_labels = []

            for name2, df_group_i2 in df_group2:
                name2_ = name2[0].left
                name2_=name2*100
                name2_=round(name2_,2)
                x_labels.append(name2_)
                # print(name1,name2)
                # print(len(df_group_i2))
                val = np.nanmean(df_group_i2[rf'GLEAM_SMroot_LAI4g_diff_detrended_diff'])

                matrix_i.append(val)

                # count=len(df_group_i2)
                # matrix_i.append(count)

                # matrix_i.append(val)
            matrix.append(matrix_i)
            y_labels.append(name1_)
        matrix = np.array(matrix)
        matrix = matrix[::-1, :]
        plt.imshow(matrix, cmap='RdBu', vmin=-3, vmax=3)
        # plt.imshow(matrix,cmap='RdBu')
        SM_p1_bin_list=SM_p1_bin_list*10
        VPD_p1_bin_list=VPD_p1_bin_list*100

        plt.xticks(np.arange(len(SM_p1_bin_list) - 1), x_labels, rotation=45)
        plt.yticks(np.arange(len(VPD_p1_bin_list) - 1), y_labels[::-1])

        plt.xlabel('VPD trend *100 (Kpa/yr)')
        plt.ylabel('Precip trend *10 (mm/month)')
        ## draw 1:1 line
        # plt.plot([20, 0], [0, 20], 'k-', lw=2)
        ## add colorbar unit

        cbar=plt.colorbar()
        cbar.set_label(r'$\Delta$ ($\delta$ LAI/$\delta$ SM) (m2/m2/100mm)')

        plt.show()



    def heatmap_VPD_SM_CO2(self): ## plot the heatmap of WUE
        T.color_map_choice()
        dff = result_root + 'Dataframe\\multi_regression\\multi_regression.df'
        df = T.load_df(dff)
        T.print_head_n(df)

        df = df[df['row'] > 120]
        df=df[df['landcover_classfication']!='Cropland']
        df=df[df['LC_max']<20]
        df=df[df['MODIS_LUCC']!=12]
        # df=df[df['CRU_p_value']<0.05]


        print(len(df))
        ## get the unique landcover_classfications
        # landcover_classfications_list=df['landcover_classfication'].tolist()
        # landcover_classfications_list_unique=T.get_df_unique_val_list(df,'landcover_classfication')
        # print(landcover_classfications_list_unique)
        # exit()
        # df=df[df['landcover_classfication']=='Shrub']
        # print(len(df))
        ###################3
        plt.hist(df[rf'CRU_trend'], bins=100)
        plt.show()
        # # plt.hist(df['GLEAM_SMroot_LAI4g_diff_detrended_diff'], bins=100)
        # # plt.show()
        # plt.hist(df[rf'VPD_trend'], bins=100)
        # plt.show()
        # exit()

        SM_p1_bin_list = np.linspace(-5, 5, 11)
        VPD_p1_bin_list = np.linspace(-0.002, 0.013, 11)

        # SM_p1_bin_list = np.linspace(-5,7, 9)
        # # SM_p1_bin_list = np.linspace(-5, 5, 11)
        # VPD_p1_bin_list = np.linspace(-0.0018, 0.012, 9)
        df_group1, bins_list_str1 = T.df_bin(df, rf'CRU_trend', SM_p1_bin_list)
        # df_group1, bins_list_str1 = T.df_bin(df, rf'CRU_trend', SM_p1_bin_list)
        matrix = []
        matrix_count=[]
        y_labels = []
        for name1, df_group_i1 in df_group1:
            df_group2, bins_list_str2 = T.df_bin(df_group_i1, rf'VPD_trend', VPD_p1_bin_list)
            name1_ = name1[0].left
            name1_=name1_
            # name1_ = name1_ * 100

            matrix_i = []
            x_labels = []
            matrix_i_count=[]

            for name2, df_group_i2 in df_group2:
                name2_ = name2[0].left
                name2_=name2_*100
                name2_=round(name2_,2)
                x_labels.append(name2_)
                # print(name1,name2)
                # print(len(df_group_i2))
                val = np.nanmean(df_group_i2[rf'CO2_LAI4g_trend'])*100





                count=len(df_group_i2)
                if count<50:

                    matrix_i_count.append('')
                    matrix_i.append(np.nan)
                    continue


                matrix_i_count.append(count)
                matrix_i.append(val)

                # matrix_i.append(val)
            matrix.append(matrix_i)
            matrix_count.append(matrix_i_count)

            y_labels.append(name1_)
        matrix = np.array(matrix)
        matrix = matrix[::-1, :]
        plt.imshow(matrix, cmap='RdBu', vmin=-2, vmax=2)
        # plt.imshow(matrix, cmap='Greens', vmin=-0.3, vmax=0.3)
        ## add number of data points
        matrix_count = np.array(matrix_count)
        matrix_count = matrix_count[::-1, :]
        for i in range(matrix_count.shape[0]):
            for j in range(matrix_count.shape[1]):
                plt.text(j, i, matrix_count[i, j], ha='center', va='center', color='black')



        # plt.imshow(matrix,cmap='RdBu')
        SM_p1_bin_list=SM_p1_bin_list
        VPD_p1_bin_list=VPD_p1_bin_list*100

        # SM_p1_bin_list = SM_p1_bin_list*100
        # VPD_p1_bin_list = VPD_p1_bin_list * 100

        plt.xticks(np.arange(len(SM_p1_bin_list) - 1), x_labels, rotation=45)
        plt.yticks(np.arange(len(VPD_p1_bin_list) - 1), y_labels[::-1])

        plt.xlabel('$\Delta$ VPD *100 (Kpa/yr)')
        # plt.ylabel('$\Delta$ SM *100 (m3/m3/yr)')
        plt.ylabel('$\Delta$ Precip (mm/yr)')
        ## draw 1:1 line
        # plt.plot([20, 0], [0, 20], 'k-', lw=2)
        ## add colorbar unit

        cbar=plt.colorbar()
        cbar.set_label(r'$\Delta$ LAI/ $\Delta$ CO2 (%/100ppm/yr)')

        plt.show()

    def plot_contribution_bar_plot(self):
        df = self.clean_df(self.df)
        dic_result= {}
        val_CRU_contribution=df['CRU_trend_contribution'].to_list()
        val_CRU_contribution=np.array(val_CRU_contribution,dtype=float)
        val_CRU_contribution[val_CRU_contribution<-99]=np.nan
        val_CRU_contribution[val_CRU_contribution>99]=np.nan
        val_VPD_contribution=df['VPD_trend_contribution'].to_list()
        val_VPD_contribution=np.array(val_VPD_contribution,dtype=float)
        val_VPD_contribution[val_VPD_contribution<-99]=np.nan
        val_VPD_contribution[val_VPD_contribution>99]=np.nan
        val_CO2_contribution=df['CO2_trend_contribution'].to_list()
        val_CO2_contribution=np.array(val_CO2_contribution,dtype=float)
        val_CO2_contribution[val_CO2_contribution<-99]=np.nan
        val_CO2_contribution[val_CO2_contribution>99]=np.nan

        dic_result['Precip'] = val_CRU_contribution
        dic_result['CO2'] = val_CO2_contribution
        dic_result['VPD'] = val_VPD_contribution
        print(dic_result)
        ## plot boxplot
        df = pd.DataFrame(dic_result)


        df = df.dropna()

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        sns.boxplot(data=df, ax=ax,showfliers=False)
        plt.xticks(fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.ylabel('Trend in LAI (m2/m2/yr)', fontsize=12)
        plt.show()













        pass
class summary_CV_result():
    def run (self):
        df=self.clean_df(self.df)
        self.plot_CV_trend(df)
        pass
    def __init__(self):
        self.dff = result_root + '\Dataframe\moving_window_CV\\moving_window_CV.df'
        self.df = T.load_df(self.dff)
        pass
    def clean_df(self,df):
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df=df[df['LC_max']<20]
        return df
    def plot_CV_trend(self,df):
        ### plot bin
        LAI_trend=df['LAI4g_trend'].to_list()
        CV_trend=df['LAI4g_CV_trend'].to_list()
        CV_trend=np.array(CV_trend,dtype=float)
        # CV_trend=CV_trend*100
        plt.hist(CV_trend,bins=100)
        plt.hist(LAI_trend,bins=100)
        plt.show()

        LAI_trend_bin_list=np.linspace(-0.02,0.02,21)

        vals_list = []
        err_list = []
        df_group1, bins_list_str1 = T.df_bin(df, 'LAI4g_trend', LAI_trend_bin_list)

        ## plot bin

        for name1, df_group_i1 in df_group1:


            vals=df_group_i1['LAI4g_CV_trend'].to_list()
            vals=np.array(vals,dtype=float)
            vals=vals*100
            val_mean=np.nanmean(vals)
            err,_,_=T.uncertainty_err(vals)
            err_list.append(err)


            print(name1,val_mean)
            vals_list.append(val_mean)
        plt.plot(LAI_trend_bin_list[:-1],vals_list)
        err_list=np.array(err_list)
        plt.fill_between(LAI_trend_bin_list[:-1],vals_list-err_list,vals_list+err_list,alpha=0.5)
        plt.xlabel('LAI trend (m2/m2/yr)')
        plt.ylabel('CV trend (%/yr)')

        plt.show()


class summary_time_series_analysis_result():
    def run(self):

        # self.plot_time_series_GPCC()
        self.plot_time_series_LAI()
        # self.plot_time_series_VPD()
        pass
    def __init__(self):

        pass
    def clean_df(self,df):
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df=df[df['LC_max']<20]
        print(len(df))
        df=df[df['MODIS_LUCC']!=12]
        print(len(df))
        # exit()
        return df
    def plot_time_series_LAI(self,):

        df = T.load_df(result_root + rf'Dataframe\anomaly_LAI\anomaly_LAI_new.df')
        print(len(df))

        df=self.clean_df(df)



        climate_variable_list = ['LAI4g']


        color_list = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'pink', 'grey', 'brown',
                      'cyan', 'magenta', 'olive', 'lime', 'teal', 'aqua']

        fig = plt.figure()
        flag = 1

        for continent in ['global',]:
            ax = fig.add_subplot(2, 3, flag)
            if continent == 'North_America':

                # df_continent = df[df['lon'] > -125]
                # df_continent = df_continent[df_continent['lon'] < -105]
                # df_continent = df_continent[df_continent['lat'] > 0]
                # df_continent = df_continent[df_continent['lat'] < 45]
                df_continent = df
            elif continent in 'global':
                df_continent = df
            else:

                df_continent = df[df['continent'] == continent]

            print(len(df_continent))



            for variable in climate_variable_list:

                column_name = variable
                vals = df_continent[column_name].tolist()

                # print(vals)

                vals_nonnan = []
                for val in vals:
                    if type(val) == float:  ## only screening
                        continue
                    vals_nonnan.append(list(val))
                    if not len(val) == 39:
                        print(val)
                        print(len(val))
                        exit()
                    # print(type(val))
                    # print(len(val))

                ###### calculate mean
                vals_mean = np.array(vals_nonnan)  ## axis=0, mean of each row  竖着加
                vals_mean = np.nanmean(vals_mean, axis=0)
                val_std = np.nanstd(vals_mean, axis=0)

                plt.plot(vals_mean, label=variable, color=color_list[flag])
                # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])


                flag = flag + 1

                ax.set_xticks(range(0, 40, 4))
                ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
                plt.ylim(-0.15, 0.15)


                plt.xlabel('year')

                plt.ylabel(f'{variable},(m2/m2)')
                # plt.legend()

                plt.title(f'{continent}_{variable}')
                plt.grid()

                # plt.legend()
        plt.show()

        pass

    def plot_time_series_GPCC(self,):

        df = T.load_df(result_root + rf'Dataframe\anomaly_LAI\anomaly_LAI_new.df')
        print(len(df))

        df=self.clean_df(df)



        climate_variable_list = ['CRU']


        color_list = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'pink', 'grey', 'brown',
                      'cyan', 'magenta', 'olive', 'lime', 'teal', 'aqua']

        fig = plt.figure()
        flag = 1

        for continent in ['global','Australia', 'Asia', 'Africa', 'South_America', 'North_America']:
            ax = fig.add_subplot(2, 3, flag)
            if continent == 'North_America':

                # df_continent = df[df['lon'] > -125]
                # df_continent = df_continent[df_continent['lon'] < -105]
                # df_continent = df_continent[df_continent['lat'] > 0]
                # df_continent = df_continent[df_continent['lat'] < 45]
                df_continent = df
            elif continent in 'global':
                df_continent = df
            else:

                df_continent = df[df['continent'] == continent]

            print(len(df_continent))



            for variable in climate_variable_list:

                column_name = variable
                vals = df_continent[column_name].tolist()

                # print(vals)

                vals_nonnan = []
                for val in vals:
                    if type(val) == float:  ## only screening
                        continue
                    vals_nonnan.append(list(val))
                    if not len(val) == 39:
                        print(val)
                        print(len(val))
                        exit()
                    # print(type(val))
                    # print(len(val))

                ###### calculate mean
                vals_mean = np.array(vals_nonnan)  ## axis=0, mean of each row  竖着加
                vals_mean = np.nanmean(vals_mean, axis=0)
                val_std = np.nanstd(vals_mean, axis=0)

                plt.plot(vals_mean, label=variable, color=color_list[flag])
                # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])


                flag = flag + 1

                ax.set_xticks(range(0, 40, 4))
                ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
                plt.ylim(-150, 200)


                plt.xlabel('year')

                plt.ylabel(f'Precip (mm)')
                # plt.legend()

                plt.title(f'{continent}_{variable}')
                plt.grid()

                # plt.legend()
        plt.show()

    def plot_time_series_VPD(self,):

        df = T.load_df(result_root + rf'Dataframe\anomaly_LAI\anomaly_LAI_new.df')
        print(len(df))

        df=self.clean_df(df)



        climate_variable_list = ['VPD']


        color_list = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'pink', 'grey', 'brown',
                      'cyan', 'magenta', 'olive', 'lime', 'teal', 'aqua']

        fig = plt.figure()
        flag = 1

        for continent in ['global','Australia', 'Asia', 'Africa', 'South_America', 'North_America']:
            ax = fig.add_subplot(2, 3, flag)
            if continent == 'North_America':

                # df_continent = df[df['lon'] > -125]
                # df_continent = df_continent[df_continent['lon'] < -105]
                # df_continent = df_continent[df_continent['lat'] > 0]
                # df_continent = df_continent[df_continent['lat'] < 45]
                df_continent = df
            elif continent in 'global':
                df_continent = df
            else:

                df_continent = df[df['continent'] == continent]

            print(len(df_continent))



            for variable in climate_variable_list:

                column_name = variable
                vals = df_continent[column_name].tolist()

                # print(vals)

                vals_nonnan = []
                for val in vals:
                    if type(val) == float:  ## only screening
                        continue
                    vals_nonnan.append(list(val))
                    if not len(val) == 39:
                        print(val)
                        print(len(val))
                        exit()
                    # print(type(val))
                    # print(len(val))

                ###### calculate mean
                vals_mean = np.array(vals_nonnan)  ## axis=0, mean of each row  竖着加
                vals_mean = np.nanmean(vals_mean, axis=0)
                val_std = np.nanstd(vals_mean, axis=0)

                plt.plot(vals_mean, label=variable, color=color_list[flag])
                # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])


                flag = flag + 1

                ax.set_xticks(range(0, 40, 4))
                ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
                plt.ylim(-0.2,0.2)


                plt.xlabel('year')

                plt.ylabel(f'VPD (Kpa)')
                # plt.legend()

                plt.title(f'{continent}_{variable}')
                plt.grid()

                # plt.legend()
        plt.show()

class summary_trend_analysis_result():

    def __init__(self):
        self.dff = result_root + rf'Dataframe\anomaly_LAI\\anomaly_LAI_new.df'
        self.df = T.load_df(self.dff)

        pass
    def run(self):
        df=self.clean_df(self.df)
        # self.plot_histgram(df)
        self.heatmap_VPD_SM_LAI(df)
        pass

    def clean_df(self,df):
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df=df[df['LC_max']<20]
        df= df[df['MODIS_LUCC'] != 12]
        return df
    def plot_histgram(self,df):
        vals_first = df['LAI4g_2002_2020_trend'].to_list()
        vals_second = df['LAI4g_1982_2001_trend'].to_list()
        vals_first = np.array(vals_first, dtype=float)
        vals_second = np.array(vals_second, dtype=float)
        vals_first = vals_first
        vals_second = vals_second
        ##plot probability density function
        # plt.hist(vals_first, bins=100, alpha=0.5, label='2002_2020', density=True)
        # plt.hist(vals_second, bins=100, alpha=0.5, label='1982_2001', density=True)

        x_i, y_i = Plot().plot_hist_smooth(vals_first, bins=100)
        plt.plot(x_i, y_i, label='2002_2020', c='r')

        x_i, y_i = Plot().plot_hist_smooth(vals_second, bins=100)
        plt.plot(x_i, y_i, label='1982_2001', c='b')





        plt.xlabel('LAI trend (m2/m2/yr)')
        plt.ylabel('Frequency(%)')
        plt.legend()
        plt.show()




        pass

    def heatmap_VPD_SM_LAI(self,df): ## plot the heatmap of WUE


        print(len(df))
        print(df.columns)
        ## get the unique landcover_classfications
        # landcover_classfications_list=df['landcover_classfication'].tolist()
        # landcover_classfications_list_unique=T.get_df_unique_val_list(df,'landcover_classfication')
        # print(landcover_classfications_list_unique)
        # exit()
        # df=df[df['landcover_classfication']=='Shrub']
        # print(len(df))
        ###################3
        # plt.hist(df[rf'CRU_trend'], bins=100)
        # plt.show()
        # # plt.hist(df['GLEAM_SMroot_LAI4g_diff_detrended_diff'], bins=100)
        # # plt.show()
        # plt.hist(df[rf'VPD_trend'], bins=100)
        # plt.show()
        # exit()

        SM_p1_bin_list = np.linspace(-5,5, 7)
        # SM_p1_bin_list = np.linspace(-0.002, 0.002, 15)
        VPD_p1_bin_list = np.linspace(-0.002, 0.013, 7)
        df_group1, bins_list_str1 = T.df_bin(df, rf'CRU_trend', SM_p1_bin_list)
        matrix = []
        matrix_count=[]
        y_labels = []
        for name1, df_group_i1 in df_group1:
            df_group2, bins_list_str2 = T.df_bin(df_group_i1, rf'VPD_trend', VPD_p1_bin_list)
            name1_ = name1[0].left
            name1_=name1_

            matrix_i = []
            x_labels = []
            matrix_i_count=[]

            for name2, df_group_i2 in df_group2:
                name2_ = name2[0].left
                name2_=name2_*100
                name2_=round(name2_,2)
                x_labels.append(name2_)
                # print(name1,name2)
                # print(len(df_group_i2))
                val = np.nanmean(df_group_i2[rf'LAI4g_trend'])





                count=len(df_group_i2)
                if count<50:

                    matrix_i_count.append('')
                    matrix_i.append(np.nan)
                    continue
                matrix_i_count.append(count)
                matrix_i.append(val)

                # matrix_i.append(val)
            matrix.append(matrix_i)
            matrix_count.append(matrix_i_count)

            y_labels.append(name1_)
        matrix = np.array(matrix)
        matrix = matrix[::-1, :]
        # plt.imshow(matrix, cmap='RdBu', vmin=-0.005, vmax=0.005)
        plt.imshow(matrix, cmap='RdBu', vmin=-0.005, vmax=0.005)
        ## add number of data points
        matrix_count = np.array(matrix_count)
        matrix_count = matrix_count[::-1, :]
        for i in range(matrix_count.shape[0]):
            for j in range(matrix_count.shape[1]):
                plt.text(j, i, matrix_count[i, j], ha='center', va='center', color='black')



        # plt.imshow(matrix,cmap='RdBu')
        # SM_p1_bin_list=SM_p1_bin_list*10
        # VPD_p1_bin_list=VPD_p1_bin_list*100

        SM_p1_bin_list = SM_p1_bin_list
        VPD_p1_bin_list = VPD_p1_bin_list * 100

        plt.xticks(np.arange(len(SM_p1_bin_list) - 1), x_labels, rotation=45)
        plt.yticks(np.arange(len(VPD_p1_bin_list) - 1), y_labels[::-1])

        plt.xlabel('$\Delta$ VPD *100 (Kpa/yr)')
        plt.ylabel('$\Delta$ Precip (mm/yr)')
        # plt.ylabel('$\Delta$ GLEAM_SMroot *100 (m3/m3/yr)')
        ## draw 1:1 line
        # plt.plot([20, 0], [0, 20], 'k-', lw=2)
        ## add colorbar unit

        cbar=plt.colorbar()
        cbar.set_label(r'$\Delta$ LAI (m2/m2/yr)')

        plt.show()



class plot_spatial_map():
    def run(self):
        self.testrobinson()
        pass
    def __init__(self):

        pass

    def testrobinson(self):
        f = rf'D:\Project3\Result\trend_analysis\event\\rainfall_frequency_trend.tif'
        fpath_p_value = rf'D:\Project3\Result\trend_analysis\event\\\\rainfall_frequency_p_value.tif'
        temp_root = result_root + rf'extract_window\extract_detrend_original_window_CV\trend\\'
        # T.mk_dir(temp_root, force=True)



        m, ret = Plot().plot_Robinson(f, vmin=-1, vmax=1, is_discrete=False, colormap_n=11,)
        # m, ret = Plot().plot_Robinson(f, vmin=0, vmax=4, is_discrete=True, colormap_n=6,  )

        Plot().plot_Robinson_significance_scatter(m,fpath_p_value,temp_root,0.05,s=10)

        # fname = 'Precip (mm/yr)'
        # fname='LAI trend (m2/m2/yr)'
        # fname='VPD trend (Kpa/yr)'
        # fname='SM trend (m3/m3/yr)'
        # fname='seasonality precipitation (CV) trend/yr'
        # fname='LAI change by precipitation (m2/m2/yr)'
        # fname='LAI sensitivity to CO2 (%/100ppm/yr)'
        fname='intra Precip CV (%/yr)'
        plt.title(f'{fname}')

        plt.show()

class summary_greening_drying():
    ## add wetting_Drying and extract drying greening pixesl
    def __init__(self):




        pass


    def run(self):
        ## 1. calculate the drying and wetting trend
        ## 2. add df
        ## 3. based on greening trend and drying trend, calculate the percentage of drying and greening pixels for each region
        ## 4. plot the percentage of drying and greening pixels for each region
        ## 5. plot the map which contains drying-greening, drying-browning, wetting-0greening and wetting browning and add to df


        ## step 1
        # self.calculate_drying_wetting_trend()
        # step 2
        # self.add_Precip_trend_label_to_df()
        # step 3
        # self.add_drying_greening_attribute_to_df()
        # step 4
        # self.plot_moisture_greening()  ## bar plot for each continent
        ## step 5
        self.plot_moisture_greening_map()
        pass

    def calculate_drying_wetting_trend(self):

        f_sm = result_root + rf'\anomaly\OBS_extend\\CRU.npy'

        dic = T.load_npy(f_sm)
        result_dic = {}
        result_tif_dic = {}

        for pix in tqdm(dic):
            time_series = dic[pix]
            time_series = np.array(time_series)
            time_series[time_series < -999] = np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            a, b, r, p = T.nan_line_fit(np.arange(len(time_series)), time_series)
            if a > 0 and p < 0.05:
                result_dic[pix] = 'sig_wetting'
                result_tif_dic[pix] = 2
            elif a < 0 and p < 0.05:
                result_dic[pix] = 'sig_drying'
                result_tif_dic[pix] = -2
            elif a > 0 and p > 0.05:
                result_dic[pix] = 'non_sig_wetting'
                result_tif_dic[pix] = 1
            else:
                result_dic[pix] = 'non_sig_drying'
                result_tif_dic[pix] = -1
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(result_tif_dic, data_root + '\\Precip_trend_label_mark.tif')
        T.save_npy(result_dic, data_root + 'Base_data\\Precip_trend_label_mark.npy')

        pass

    ## calculate the wetting and drying trend



    def add_Precip_trend_label_to_df(self):
        df=T.load_df(result_root + rf'Dataframe\\anomaly_LAI\\anomaly_LAI_new.df')


        f = data_root + rf'\\Base_data\Precip_trend_label_mark.npy'


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
        T.save_df(df, result_root + rf'Dataframe\\anomaly_LAI\\anomaly_LAI_new.df')
        T.df_to_excel(df, result_root + rf'Dataframe\\anomaly_LAI\\anomaly_LAI_new')

    def add_drying_greening_attribute_to_df(
            self):  # generate map which contains drying-greening, drying-browning, wetting-0greening and wetting browning
        df = T.load_df(result_root + 'Dataframe\\anomaly_LAI\\anomaly_LAI_new.df')
        df = self.clean_df(df)

        cm = 1 / 2.54
        label_list = []
        for i, row in df.iterrows():
            pix = row['pix']
            wetting_drying = row['wetting_drying_trend']
            val = row['LAI4g_trend']
            p_value = row['LAI4g_p_value']
            if wetting_drying == 'sig_wetting':
                if p_value < 0.05:
                    if val > 0:
                        label_list.append('sig_greening_sig_wetting')
                    else:
                        label_list.append('sig_browning_sig_wetting')
                else:
                    if val > 0:
                        label_list.append('non_sig_greening_sig_wetting')
                    else:
                        label_list.append('non_sig_browning_sig_wetting')
            elif wetting_drying == 'sig_drying':
                if p_value < 0.05:
                    if val > 0:
                        label_list.append('sig_greening_sig_drying')
                    else:
                        label_list.append('sig_browning_sig_drying')
                else:
                    if val > 0:
                        label_list.append('non_sig_greening_sig_drying')
                    else:
                        label_list.append('non_sig_browning_sig_drying')
            else:
                label_list.append(np.nan)

            pass
        df['mositure_greening'] = label_list
        T.print_head_n(df)

        ## save df
        T.save_df(df, result_root + 'Dataframe\\anomaly_LAI\\anomaly_LAI_new.df')
        ## save xlsx
        T.df_to_excel(df, result_root + 'Dataframe\\anomaly_LAI\\anomaly_LAI_new')
        exit()


    def plot_moisture_greening(self):  # calculate the percentage of drying and greening pixels for each region
        df= T.load_df(result_root + 'Dataframe\\anomaly_LAI\\anomaly_LAI_new.df')
        df=self.clean_df(df)

        wetting_drying_list=['non_sig_wetting','sig_wetting','non_sig_drying','sig_drying']

        color_list=['green','lime','orange','red']
        continent_list=['global','Australia', 'Asia', 'Africa', 'South_America', 'North_America']
        cm = 1 / 2.54

        for continent in continent_list:


            df_temp = df[df['continent'] == continent]
            dic={}
            for wetting_drying in wetting_drying_list:
                df_temp2=df_temp[df_temp['wetting_drying_trend']==wetting_drying]
                vals = df_temp2[f'LAI4g_trend'].tolist()
                p_values = df_temp2[f'LAI4g_p_value'].tolist()

                sig_browning = 0
                sig_greening = 0
                non_sig_browning = 0
                non_sig_greening = 0

                for i in range(len(vals)):
                    if p_values[i] < 0.05:
                        if vals[i] < 0:
                            sig_browning = sig_browning + 1
                        else:
                            sig_greening = sig_greening + 1
                    else:
                        if vals[i] < 0:
                            non_sig_browning = non_sig_browning + 1
                        else:
                            non_sig_greening = non_sig_greening + 1
                        ##percentage
                if len(vals) == 0:
                    continue
                sig_browning = sig_browning / len(vals) *100
                sig_greening = sig_greening / len(vals)*100
                non_sig_browning = non_sig_browning / len(vals) *100
                non_sig_greening = non_sig_greening / len(vals)*100
                dic[wetting_drying] = [sig_greening, non_sig_greening, non_sig_browning, sig_browning
                                          ]
            df_new = pd.DataFrame(dic, index=['sig_greening', 'non_sig_greening', 'non_sig_browning', 'sig_browning'])
            df_new_T = df_new.T
            df_new_T.plot.bar(stacked=True, color=color_list, legend=False)
            plt.legend()

            plt.title(f'{continent}')
            plt.ylabel('percentage')
            plt.xticks(rotation=45)
            plt.tight_layout()
        plt.show()


        pass



        ## plot
    def plot_moisture_greening_map(self):
        df = T.load_df(result_root + 'Dataframe\\anomaly_LAI\\anomaly_LAI_new.df')

        ### calculate every cluster
        df=df.dropna()

        ## calculate percentage
        dic={}
        cluster=['sig_greening_sig_wetting', 'sig_browning_sig_wetting', 'non_sig_greening_sig_wetting',
         'non_sig_browning_sig_wetting', 'sig_greening_sig_drying', 'sig_browning_sig_drying',
         'non_sig_greening_sig_drying', 'non_sig_browning_sig_drying']

        cluster=['sig_greening_sig_wetting', 'sig_browning_sig_wetting',
          'sig_greening_sig_drying', 'sig_browning_sig_drying',
         ]

        for label in cluster:
            df_temp=df[df['mositure_greening']==label]
            dic[label]=len(df_temp)/len(df)*100
        print(dic)
        #plt.bar
        plt.bar(dic.keys(),dic.values(),color='grey',alpha=0.75)
        ## xticks size and rotation

        plt.xticks(rotation=45,fontsize=12)
        plt.ylabel('percentage (%)',fontsize=12)
        plt.tight_layout()
        plt.show()
        # exit()



        ####### plot map

        spatica_dic = {}
        dic_label = {'sig_greening_sig_wetting': 1, 'sig_browning_sig_wetting': 2, 'non_sig_greening_sig_wetting': 3,

                        'non_sig_browning_sig_wetting': 4, 'sig_greening_sig_drying': 5, 'sig_browning_sig_drying': 6,

                        'non_sig_greening_sig_drying': 7, 'non_sig_browning_sig_drying': 8, np.nan: 0}
        for i, row in df.iterrows():
            pix = row['pix']
            label = row['mositure_greening']
            spatica_dic[pix] = dic_label[label]

        arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatica_dic)
        plt.imshow(arr_trend)
        plt.colorbar()
        plt.show()
        ## save
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, result_root + 'Dataframe\\anomaly_LAI\\anomaly_trends_map.tif')


        pass

    def plot_climate_variables_based_moisture_greening(self):
        df = T.load_df(result_root + 'Dataframe\\anomaly_LAI\\anomaly_LAI_new.df')
        df = self.clean_df(df)
        climate_variable_list = ['CRU', 'VPD', 'GLEAM_SMroot']
        color_list = ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'pink', 'grey', 'brown',

                        'cyan', 'magenta', 'olive', 'lime', 'teal', 'aqua']
        fig = plt.figure()
        flag = 1


        pass

    def save_df(self, df, outf):
        df.to_pickle(outf)

    def clean_df(self,df):
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df=df[df['LC_max']<20]
        df= df[df['MODIS_LUCC'] != 12]
        return df

    def df_to_excel(self, df, dff, n=1000, random=False):
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))



def main():

    # build_dataframe().run()
    # summary_multiregression_model_result().run()
    # summary_time_series_analysis_result().run()
    plot_spatial_map().run()
    # summary_trend_analysis_result().run()
    # summary_CV_result().run()
    # summary_greening_drying().run()


    pass

if __name__ == '__main__':
    main()
