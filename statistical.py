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

        self.this_class_arr = result_root + '\\Dataframe\\multi_regression\\'


        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'multi_regression.df'



        pass

    def run(self):

        df = self.__gen_df_init(self.dff)


        # df=self.foo2(df)
        # df=self.add_multiregression_to_df(df)




        # df=self.add_AI_classfication(df)
        # # #
        # df=self.add_aridity_to_df(df)
        # # # #
        # df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        # df=self.add_landcover_classfication_to_df(df)
        # df=self.add_row(df)
        # df=self.add_continent_to_df(df)
        # df=self.add_lat_lon_to_df(df)
        # df=self.add_soil_texture_to_df(df)
        df=self.add_trend_to_df(df)

        # df=self.add_Ndepostion_to_df(df)
        #

        # df=self.rename_columns(df)
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

    def add_multiregression_to_df(self, df):
        fdir = result_root + rf'multi_regression\detrended_anomaly\\diff\\'
        for f in os.listdir(fdir):

            if not f.endswith('.tiff'):
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
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            fname=f.split('.')[0]+'_detrended_diff'
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
        df = df.rename(columns={'Tmax_LAI4g_2002_2020': 'tmax_LAI4g_2002_2020',


                            }

                               )


        return df
    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        exit()
        df = df.drop(columns=['GPCP_precip', 'GPCP_precip_pre', 'ENSO_index_DFJ',
                     'ENSO_index_distance', 'GPCC_raw', 'GPCP_precip_pre_raw', 'noy','nhx',
                              'GPCP_precip_pre_raw', 'VPD_raw','tmin_raw','tmax_raw',])
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

class summary_multiregression_model_result():
    def run (self):

        df = self.clean_df(self.df)
        # self.plt_histgram_CO2_sensitivity(df)
        # self.plt_histgram_precipitation_sensitivity(df)
        # self.calculate_diff_between_periods_sensitivity()
        self.heatmap()



        pass

    def __init__(self):
        self.dff = result_root + 'Dataframe\\multi_regression\\multi_regression.df'
        self.df = T.load_df(self.dff)

        pass

    def clean_df(self, df):
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]


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

        variable_list= ['CO2_LAI4g', 'tmax_LAI4g','VPD_LAI4g','GPCC_LAI4g']
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
                vals=vals*100

                vals=vals[~np.isnan(vals)]
                ax.hist(vals,bins=100,color=color_list[flag-1],alpha=0.5,label=f'{period} {continent}',density=True,edgecolor=color_list[flag-1],range=(-1,1))


                flag=flag+1
            i=i+1
            ## add line at x=0
            plt.axvline(x=0, color='black', linestyle='--')
            plt.xlabel('LAI sensitivity to CO2 (m2/m2/100ppm)')
            plt.ylabel('Frequency(%)')
            plt.title(continent)
            plt.legend(self.period_list)
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
    def calculate_diff_between_periods_sensitivity(self):
        tiff_1=result_root+rf'multi_regression\detrended_anomaly\1982_2001\\GLEAM_SMroot_LAI4g_1982_2001.tif'
        tiff_2=result_root+rf'multi_regression\detrended_anomaly\2002_2020\\GLEAM_SMroot_LAI4g_2002_2020.tif'
        outtiff=result_root+rf'multi_regression\detrended_anomaly\\GLEAM_SMroot_LAI4g_diff.tiff'
        array1, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff_1)
        array1 = np.array(array1, dtype=float)
        array1[array1>99]=np.nan
        array1[array1<-99]=np.nan
        array2, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff_2)
        array2 = np.array(array2, dtype=float)
        array2[array2>99]=np.nan
        array2[array2<-99]=np.nan
        arr=array2-array1
        arr=np.array(arr,dtype=float)
        arr[arr>99]=np.nan
        arr[arr<-99]=np.nan
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr,outtiff)


        pass






    def heatmap(self):
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
class summary_CV_result():
    def run (self):
        self.df=self.clean_df(self.df)
        self.plot_CV()
        pass
    def __init__(self):
        self.dff = result_root + 'Dataframe\\multi_regression\\multi_regression.df'
        self.df = T.load_df(self.dff)
        pass
    def clean_df(self,df):
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        return df
    def plot_CV(self):   ## plot CV as funtion of aridity, drier, xx
        pass

def main():

    # build_dataframe().run()
    summary_multiregression_model_result().run()


    pass

if __name__ == '__main__':
    main()
