# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg
from openpyxl.styles.builtins import percent
# from green_driver_trend_contribution import *
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import t

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
result_root = 'D:/Project3/Result/'


class bivariate_analysis():
    def __init__(self):
        pass
    def run(self):
        # self.generate_bivarite_map()
        # self.generate_df()
        # build_dataframe().run()
        # self.generate_three_dimension()
        self.plot_figure1d()

    def generate_bivarite_map(self):  ##

        import xymap
        tif_rainfall = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\trend\detrended_sum_rainfall_CV_trend.tif'
        # tif_CV=  result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'
        tif_sensitivity = result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year_SNU_LAI\npy_time_series\\sum_rainfall_detrend_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outdir = result_root + rf'3mm\\\bivariate_analysis\\'
        T.mk_dir(outdir, force=True)
        outtif = outdir + rf'\\interannual_CVrainfall_beta.tif'
        T.mk_dir(result_root + rf'bivariate_analysis\\')
        tif1 = tif_rainfall
        tif2 = tif_sensitivity

        dic1 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif1)
        dic2 = DIC_and_TIF(pixelsize=0.5).spatial_tif_to_dic(tif2)
        dics = {'interannual_CVrainfall': dic1,
                'beta': dic2}
        df = T.spatial_dics_to_df(dics)
        # print(df)
        df['interannual_CVrainfall_increase'] = df['interannual_CVrainfall'] > 0
        df['beta_increase'] = df['beta'] > 0

        print(df)
        label_list = []
        for i, row in df.iterrows():
            if row['interannual_CVrainfall_increase'] and row['beta_increase']:
                label_list.append(1)
            elif row['interannual_CVrainfall_increase'] and not row['beta_increase']:
                label_list.append(2)
            elif not row['interannual_CVrainfall_increase'] and row['beta_increase']:
                label_list.append(3)
            else:
                label_list.append(4)

        df['label'] = label_list
        result_dic = T.df_to_spatial_dic(df, 'label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(result_dic, outtif)

    pass


    def generate_df(self):
        ##rainfall_trend +sensitivity+ greening
        ftiff=result_root + rf'3mm\bivariate_analysis\\interannual_CVrainfall_beta.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(ftiff)

        dic_beta_CVrainfall=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(array)

        f_CVLAItiff=result_root+rf'3mm\\extract_SNU_LAI_phenology_year\moving_window_extraction\trend\\detrended_SNU_LAI_CV_trend.tif'
        array_CV_LAI,originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_CVLAItiff)
        dic_CV_LAI=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(array_CV_LAI)


        df=T.spatial_dics_to_df({'CVrainfall_beta':dic_beta_CVrainfall,'LAI4g_CV':dic_CV_LAI})

        T.save_df(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend.df')
        T.df_to_excel(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend.xlsx')
        exit()





    def generate_three_dimension(self):
        dff=result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend.df'
        df=T.load_df(dff)
        self.df_clean(df)
        df=df[df['CVrainfall_beta']>=0]



        df["LAI4g_CV_trends"] = df["LAI4g_CV"].apply(lambda x: "increaseCV" if x >= 0 else "decreaseCV")

        category_mapping = {
            ("increaseCV", 1): 1,
            ("increaseCV", 2): 2,
            ("increaseCV", 3): 3,
            ("increaseCV", 4): 4,
            ("decreaseCV", 1): 5,
            ("decreaseCV", 2): 6,
            ("decreaseCV", 3): 7,
            ("decreaseCV", 4): 8,
        }

        # Apply the mapping to create a new column 'new_category'
        df["CV_rainfall_beta_LAI"] = df.apply(
            lambda row: category_mapping[(row["LAI4g_CV_trends"], row["CVrainfall_beta"])],
            axis=1)
        # T.save_df(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend.df')
        # T.df_to_excel(df, result_root + rf'\3mm\bivariate_analysis\Dataframe\\Trend.xlsx')

        # Display the result
        print(df)
        outdir = result_root + rf'\3mm\bivariate_analysis\\'
        outf = outdir + rf'CV_rainfall_beta_LAI.tif'

        spatial_dic = T.df_to_spatial_dic(df, 'CV_rainfall_beta_LAI')
        DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(spatial_dic, outf)
        ##save pdf
        # outf = outdir + rf'\\three_dimension.pdf'
        # plt.savefig(outf)
        # plt.close()

    def plot_figure1d(self):
        tiff_f=rf'D:\Project3\Result\3mm\bivariate_analysis\CV_rainfall_beta_LAI.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff_f)
        dic_label={1:'CVLAI+_CVrainfall+_posbeta',2:'CVLAI+_CVrainfall+_negbeta',3:'CVLAI+_CVrainfall-posbeta',4:'CVLAI+_CVrainfall-negbeta',
                   5:'CVLAI-_CVrainfall+_posbeta',6:'CVLAI-_CVrainfall+_negbeta',7:'CVLAI-_CVrainfall-posbeta',8:'CVLAI-_CVrainfall-negbeta'}
        dic={}



        greening_list=[]
        browning_list=[]
        for i in range(1,9):
            if i<5:
                array_i=array==i
                count = np.sum(array_i)
                greening_list.append(count)
            else:
                array_i = array == i
                count = np.sum(array_i)
                browning_list.append(count)
        greening_sum = np.sum(greening_list)
        browning_sum = np.sum(browning_list)

        ['#e7f4cb','#b7dee3', '#75b1d3', '#2c7bb6', '#d7191c', '#ed6e43','#fdba6e', '#fee8a4',]

        color_list2 = [
             '#75b1d3',

            '#e7f4cb',

            '#fdba6e',
            '#d7191c',

        ]

        color_list1 = [
            '#2c7bb6',


            '#b7dee3',


            '#fee8a4',
            '#ed6e43',
        ]

            ## count the number of pixels
        for i in range(1,9):

            if i<5:
                array_i = array == i
                count = np.sum(array_i)
                dic[i] = count/greening_sum*100
            else:
                array_i = array == i
                count = np.sum(array_i)
                dic[i] = count/browning_sum*100
        pprint(dic)


        dic[5]=-dic[5]
        dic[6]=-dic[6]
        dic[7]=-dic[7]
        dic[8]=-dic[8]

        group_labels = ['CVinterannualrainfall+', 'CVinterannualrainfall-', 'CVinterannualrainfall+', 'CVinterannualrainfall-']
        group_data = [
            [dic[1], dic[2]],
            [dic[3], dic[4]],
            [dic[5], dic[6]],
            [dic[7], dic[8]],
        ]

        # 拆分上下部分


        bottoms = [0 if v[0] >= 0 else v[0] for v in group_data]
        heights1 = [v[0] for v in group_data]
        heights2 = [v[1] for v in group_data]

        x = np.arange(len(group_labels))
        bar_width = 0.5

        fig, ax = plt.subplots(figsize=(12 * centimeter_factor,  8 * centimeter_factor))

        # 第一层

        bars1 = ax.bar(x, heights1, width=bar_width, color=color_list1)

        # 第二层堆叠
        bars2 = ax.bar(x, heights2, width=bar_width, bottom=heights1, color=color_list2)

        # 坐标轴 & 图例
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel('Percentage (%)')

        ax.axhline(0, color='black', linewidth=0.8)
        plt.show()


        # plt.tight_layout()
        # plt.savefig(rf'D:\Project3\Result\3mm\FIGURE\three_dimension_statistics.pdf')


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



class build_dataframe():


    def __init__(self):

        self.this_class_arr = (result_root+rf'\3mm\bivariate_analysis\Dataframe\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + rf'Trend.df'

        pass

    def run(self):


        df = self.__gen_df_init(self.dff)


        df=self.add_aridity_to_df(df)
        df=self.add_dryland_nondryland_to_df(df)
        df=self.add_MODIS_LUCC_to_df(df)
        df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        df=self.add_landcover_classfication_to_df(df)
        df=self.add_maxmium_LC_change(df)
        df=self.add_row(df)
        # df=self.add_continent_to_df(df)
        df=self.add_lat_lon_to_df(df)



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


    def add_detrend_zscore_to_df(self, df):

        fdir=rf'D:\Project3\Result\3mm\NDVI_LAI\LAI4g_predict\average_LAI4g_phenology_year\moving_window_extraction\\'


        for f in os.listdir(fdir):
            variable=f.split('.')[0]
            print(variable)
            if not 'CV' in variable:
                continue




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

                if len(vals)==0:
                    NDVI_list.append(np.nan)
                    continue


                if len(vals)==33 :
                    nan_list=np.array([np.nan]*5)
                    vals=np.append(vals,nan_list)



                v1= vals[year - 0]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)


            df[f'{variable}'] = NDVI_list
        # exit()
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
        tiff = rf'E:\Project3\Data\Base_data\\continent_05.tif'
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
        fdir = data_root + rf'\Base_data\\SoilGrid\SOIL_Grid_05_unify\\weighted_average\\'
        for f in (os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue

            tiff = fdir + f

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
            array = np.array(array, dtype=float)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
            fname=f.split('.')[0]
            print(fname)
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
        fdir=data_root+ rf'\Base_data\SoilGrid\SOIL_Grid_05_unify\weighted_average\\'
        for f in os.listdir(fdir):


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
                # if val > 99:
                #     val_list.append(np.nan)
                #     continue
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
        df = df.rename(columns={'detrended_GIMMS_plus_NDVI_CV_trend': 'GIMMS_plus_NDVI_detrend_CV_trend',
                                'detrended_GIMMS_plus_NDVI_CV_p_value': 'GIMMS_plus_NDVI_detrend_CV_p_value',
                                'detrended_NDVI4g_CV_p_value': 'NDVI4g_detrend_CV_p_value',
                                'detrended_NDVI4g_CV_trend': 'NDVI4g_detrend_CV_trend',
                                'detrended_NDVI_trend': 'NDVI_detrend_trend',
                                'detrended_NDVI_p_value': 'NDVI_detrend_p_value',




                            }

                               )



        return df
    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=[


                              'weighted_avg_GOSIF',

                              'weighted_avg_contribution_GOSIF',



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

        f = data_root+rf'\Base_data\lc_trend\\max_trend.tif'

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

        f = data_root + rf'\Base_data\aridity_index_05\\aridity_index.tif'

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
    def add_dryland_nondryland_to_df(self, df):
        fpath=data_root+rf'\\Base_data\\aridity_index_05\\aridity_index.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue

            val = val_dic[pix]
            if val <= 0.65:
                label = 'Dryland'
            elif 0.65 < val <= 0.8:
                label = 'Sub-humid'
            elif 0.8 < val <= 1.5:
                label = 'Humid'
            elif val > 1.5:
                label = 'Very Humid'
            elif val < -99:
                label = np.nan
            else:
                raise
            val_list.append(label)
        df['Dryland_Humid'] = val_list

        return df




    def show_field(self, df):
        for col in df.columns:
            print(col)
        return df
        pass


def main():

    bivariate_analysis().run()







if __name__ == '__main__':
    main()