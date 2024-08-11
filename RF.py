# coding='utf-8'
import sys
import xgboost as xgb

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
import shap

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

this_root = 'E:\Data\ERA5_precip\ERA5_daily\\'
data_root = 'E:\Data\ERA5_precip\ERA5_daily\\'
result_root = 'E:\Data\ERA5_precip\ERA5_daily\\'
D_025 = DIC_and_TIF(pixelsize=0.25)

result_root_this_script = join(result_root, 'statistic')

class Dataframe_func:

    def __init__(self,df,is_clean_df=True):
        print('add lon lat')
        df = self.add_lon_lat(df)

        print('add GLC2000')
        df = self.add_GLC_landcover_data_to_df(df)
        print('add Aridity Index')
        df = self.add_AI_to_df(df)
        print('add AI_reclass')
        df = self.AI_reclass(df)
        self.df = df

    def add_GLC_landcover_data_to_df(self, df):
        f = join(data_root,'GLC2000/reclass_lc_dic.npy')
        val_dic=T.load_npy(f)
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

    def add_AI_to_df(self, df):
        f = join(data_root, 'Aridity_index/aridity_index_clip.tif')
        spatial_dict = D_China.spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'aridity_index')
        return df

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, D_China)
        return df

    def AI_reclass(self,df):
        AI_class = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='AI_reclass'):
            AI = row['aridity_index']
            if AI < 0.65:
                AI_class.append('Arid')
            elif AI >= 0.65:
                AI_class.append('Humid')
            elif np.isnan(AI):
                AI_class.append(np.nan)
            else:
                print(AI)
                raise ValueError('AI error')
        df['AI_class'] = AI_class
        return df

    def add_koppen(self,df):
        f = join(data_root, 'koppen/koppen_reclass_dic.npy')
        val_dic = T.load_npy(f)
        df = T.add_spatial_dic_to_df(df, val_dic, 'Koppen')
        return df


class Dataframe:

    def __init__(self):

        self.this_class_arr = result_root + rf'Dataframe\RF\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'RF.df'

        pass
    def run(self):
        df=self.__gen_df_init(self.dff)
        # df=self.foo1(df)

        # df=self.add_anomaly_to_df(df)
        df=self.add_LUCC_to_df(df)
        # df=self.add_soil_texture_to_df(df)
        # df=self.add_SOC_to_df(df)
        # df=self.add_rooting_depth_to_df(df)
        # self.add_trend_to_df(df)
        # self.add_MAT_MAP(df)
        # self.add_AI_classfication(df)
        # self.add_SM_trend_label(df)


        T.save_df(df, self.dff)

        self.__df_to_excel(df, self.dff)
        pass

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

        f = result_root + rf'anomaly\OBS\\LAI4g.npy'

        dic = T.load_npy(f)


        year_list=range(1982,2020)
        pix_list=D_025.void_spatial_dic()  #same as DIC_and_TIF().void_spatial_dic(dic)
        pix_list_all = []
        year_list_all = []

        for pix in tqdm(pix_list,total=len(pix_list)):
            if not pix in dic:
                continue
            for year in year_list:
                pix_list_all.append(pix)
                year_list_all.append(year)


        df['pix'] = pix_list_all

        df['year'] = year_list_all


        return df


    def add_anomaly_to_df(self, df):
        fdir = result_root + rf'anomaly\OBS_extend\\'


        for f in os.listdir(fdir):


            variable= f.split('.')[0]
            print(variable)

            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row['year']
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

                v1 = vals[year - 1982]
                    # print(v1,year,len(vals))
                NDVI_list.append(v1)
            df[variable] = NDVI_list

        return df

    def add_LUCC_to_df(self, df):
        fdir = rf'D:\Project3\Data\landcover_composition_DIC\urban\\'


        val_dic = T.load_npy_dir(fdir)

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            year = row['year']
            # pix = row.pix
            pix = row['pix']
            r, c = pix


            if not pix in val_dic:
                NDVI_list.append(np.nan)
                continue


            vals = val_dic[pix]
            # print(len(vals))
            if year<1992:

                NDVI_list.append(np.nan)
                continue

            v1 = vals[year - 1992]
                # print(v1,year,len(vals))
            NDVI_list.append(v1)
        df['urban'] = NDVI_list

        return df


    def add_trend_to_df(self,df):
        fdir=result_root+rf'trend_analysis\anomaly\ALL_ensemble\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue


            val_array = np.load(fdir + f)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
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
                val_list.append(val)
            df[f'{f_name}']=val_list

    def add_soil_texture_to_df(self,df):

        f = rf'D:\Project3\Data\Base_data\HWSD\tif_025\\S_SILT.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        f_name = f.split('.')[0].split('\\')[-1]
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
        df[f'{f_name}'] = val_list
        return df

    def add_SOC_to_df(self,df):

        f = rf'D:\Project3\Data\Base_data\SOC\tif_sum\\SOC_sum.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        f_name = f.split('.')[0].split('\\')[-1]

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
        df[f'{f_name}'] = val_list
        return df



    def add_rooting_depth_to_df(self,df):

        f = rf'D:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\rooting_depth.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        f_name = f.split('.')[0].split('\\')[-1]
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
        df[f'{f_name}'] = val_list
        return df


    def add_MAT_MAP(self,df):

        fdir = result_root + rf'state_variables\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            val_dic=np.load(fdir+f,allow_pickle=True).item()
            # val_array = DIC_and_TIF().dic
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
                val_list.append(val)
            df[f'{f_name}'] = val_list
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
            else:
                label='Sub-Humid'

            val_list.append(label)

        df['AI_classfication'] = val_list
        return df



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

class Bivariate_statistic:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Bivariate_statistic', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe/dataframe.df')
        pass

    def run(self):
        # self.plot_china_shp_pdf()
        # self.copy_df()
        self.xy_map_lag()
        self.plot_xy_map_lag()
        self.xy_map_correlation()
        self.plot_xy_map_correlation()
        pass

    def copy_df(self):
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir,force=True)
        outf = self.dff
        dff_origin = Dataframe().dff
        dff_origin_xlsx = Dataframe().dff + '.xlsx'
        shutil.copy(dff_origin,outf)
        shutil.copy(dff_origin_xlsx,join(outdir,'dataframe.df.xlsx'))
        pass


    def xy_map_lag(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        outdir = join(self.this_class_tif,'xy_map_lag')
        T.mk_dir(outdir)
        tif_lag = join(outdir,'lag.tif')
        tif_lag_trend = join(outdir,'lag_trend.tif')
        max_lag_spatial_dict = T.df_to_spatial_dic(df,'max_lag')
        max_lag_trend_spatial_dict = T.df_to_spatial_dic(df,'max_lag_trend')

        D_China.pix_dic_to_tif(max_lag_spatial_dict,tif_lag)
        D_China.pix_dic_to_tif(max_lag_trend_spatial_dict,tif_lag_trend)

        outf = join(outdir,'xy_map_lag.tif')
        x_label = 'lag'
        y_label = 'lag_trend'
        min1 = 0
        max1 = 6
        min2 = -0.1
        max2 = 0.1
        xymap.Bivariate_plot().plot_bivariate_map(tif_lag, tif_lag_trend, x_label, y_label, min1, max1, min2, max2, outf)
        # T.open_path_and_file(outdir)

    def plot_xy_map_lag(self):
        shp_line_f = global_china_shp
        shp_provinces_f = global_china_shp_provinces
        fpath = join(self.this_class_tif,'xy_map_lag/xy_map_lag.tif')
        arr_deg, originX_deg, originY_deg, pixelWidth_deg, pixelHeight_deg = ToRaster().raster2array(fpath)

        m = Basemap(projection='aea', resolution='i',
                    llcrnrlon=80, llcrnrlat=14, urcrnrlon=140, urcrnrlat=52,
                    lon_0=105, lat_0=0, lat_1=25, lat_2=47)
        proj4string = m.proj4string
        tif_reproj = join(self.this_class_tif,'xy_map_lag/xy_map_lag_reproj.png')
        gdal.Warp(tif_reproj, fpath, dstSRS=proj4string)


    def xy_map_correlation(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        # exit()
        outdir = join(self.this_class_tif,'xy_map_correlation')
        T.mk_dir(outdir)
        tif_correlation = join(outdir,'correlation.tif')
        tif_correlation_trend = join(outdir,'correlation_trend.tif')
        tif_correlation_spatial_dict = T.df_to_spatial_dic(df,'longterm_corr')
        tif_correlation_trend_spatial_dict = T.df_to_spatial_dic(df,'r_trend')

        D_China.pix_dic_to_tif(tif_correlation_spatial_dict,tif_correlation)
        D_China.pix_dic_to_tif(tif_correlation_trend_spatial_dict,tif_correlation_trend)

        outf = join(outdir,'xy_map_correlation.tif')
        x_label = 'correlation'
        y_label = 'correlation_trend'
        min1 = -0.4
        max1 = 0.4
        min2 = -0.03
        max2 = 0.03
        xymap.Bivariate_plot().plot_bivariate_map(tif_correlation, tif_correlation_trend, x_label, y_label, min1, max1, min2, max2, outf)
        # T.open_path_and_file(outdir)

    def plot_xy_map_correlation(self):
        fpath = join(self.this_class_tif,'xy_map_correlation/xy_map_correlation.tif')
        arr_deg, originX_deg, originY_deg, pixelWidth_deg, pixelHeight_deg = ToRaster().raster2array(fpath)

        m = Basemap(projection='aea', resolution='i',
                    llcrnrlon=80, llcrnrlat=14, urcrnrlon=140, urcrnrlat=52,
                    lon_0=105, lat_0=0, lat_1=25, lat_2=47)
        proj4string = m.proj4string
        tif_reproj = join(self.this_class_tif,'xy_map_correlation/xy_map_correlation.png')
        gdal.Warp(tif_reproj, fpath, dstSRS=proj4string)

    def plot_china_shp_pdf(self):
        outf = join(this_root,'conf','China_shp.pdf')
        m = Basemap(projection='aea', resolution='i',
                    llcrnrlon=80, llcrnrlat=14, urcrnrlon=140, urcrnrlat=52,
                    lon_0=105, lat_0=0, lat_1=25, lat_2=47)
        plt.axis('off')
        m.readshapefile(global_china_shp, 'a', drawbounds=True, linewidth=0.5, color='k', zorder=100)
        m.readshapefile(global_china_shp_provinces, 'ooo', drawbounds=True, linewidth=0.3, color='k', zorder=100)
        plt.savefig(outf)
        plt.close()


class Correlation_Lag_statistic:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Correlation_Lag_statistic', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe/dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        self.Aridity_index()
        pass

    def copy_df(self):
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir,force=True)
        outf = self.dff
        dff_origin = Dataframe().dff
        dff_origin_xlsx = Dataframe().dff + '.xlsx'
        shutil.copy(dff_origin,outf)
        shutil.copy(dff_origin_xlsx,join(outdir,'dataframe.df.xlsx'))
        pass

    def Aridity_index(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        col = 'longterm_corr'
        # col = 'max_r'
        # col = 'r_trend'
        AI_vals = df.aridity_index.tolist()
        AI_bins = np.linspace(0,2,21)
        df_group, bins_list_str = T.df_bin(df,'aridity_index',AI_bins)
        x_list = []
        y_list = []
        err_list = []
        for name,df_group_i in df_group:
            left = name[0].left
            vals = df_group_i[col].tolist()
            mean = np.nanmean(vals)
            err = np.nanstd(vals)
            x_list.append(left)
            y_list.append(mean)
            err_list.append(err)
        plt.figure()
        plt.errorbar(x_list,y_list,yerr=err_list)
        plt.title(col)
        plt.show()
        pass

class Trend_statistic:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Trend_statistic', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe/dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        # self.plot_AI()
        # self.PFTs()
        # self.MAT_MAP()
        # self.bar_percentage_AI_trend()
        self.bar_percentage_PFT_trend()
        pass

    def copy_df(self):
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir,force=True)
        outf = self.dff
        dff_origin = Dataframe().dff
        dff_origin_xlsx = Dataframe().dff + '.xlsx'
        shutil.copy(dff_origin,outf)
        shutil.copy(dff_origin_xlsx,join(outdir,'dataframe.df.xlsx'))
        pass

    def trend_bars(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        col_name_list = ['r_trend', 'max_lag_trend']
        pass

    def PFTs(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        lc_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # col_name = 'max_lag_trend'
        # col_name = 'r_trend'
        col_name_list = ['r_trend','max_lag_trend']
        for col_name in col_name_list:
            mean_list = []
            std_list = []
            plt.figure()
            for lc in lc_list:
                df_lc = df[df['landcover_GLC']==lc]
                vals = df_lc[col_name].tolist()
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                mean_list.append(mean)
                std_list.append(std)
            plt.bar(lc_list,mean_list,yerr=std_list)
            plt.ylabel(col_name)
        plt.show()

    def MAT_MAP(self):
        outdir = join(self.this_class_png,'MAT_MAP')
        T.mk_dir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        MAT_bins = np.linspace(-10,20,16)
        MAP_bins = np.linspace(0,2000,21)
        col_name_list = ['r_trend', 'max_lag_trend', 'longterm_corr', 'max_lag']
        col_name_lim_dict = {
            'r_trend':[-0.02,0.02],
            'longterm_corr':[-0.4,0.4],
            'max_lag':[0,6],
            'max_lag_trend':[-0.1,0.1],
        }
        col_name_cmap_dict = {
            'r_trend':'RdBu_r',
            'longterm_corr':'RdBu_r',
            'max_lag':'RdBu',
            'max_lag_trend':'RdBu',
        }
        for col_name in col_name_list:
            plt.figure(figsize=(15.6*centimeter_factor,9.1*centimeter_factor))
            df_group_MAT, bins_list_str_MAT = T.df_bin(df,'MAT',MAT_bins)
            for name_MAT,df_group_MAT_i in df_group_MAT:
                MAT_i = name_MAT[0].left
                df_group_MAP, bins_list_str_MAP = T.df_bin(df_group_MAT_i,'MAP',MAP_bins)
                for name_MAP,df_group_MAP_i in df_group_MAP:
                    MAP_i = name_MAP[0].left
                    vals = df_group_MAP_i[col_name].tolist()
                    if len(vals) == 0:
                        continue
                    mean = np.nanmean(vals)
                    plt.scatter(MAP_i,MAT_i,c=mean,
                                cmap=col_name_cmap_dict[col_name],marker='s',s=100,linewidths=0,
                                vmin=col_name_lim_dict[col_name][0],vmax=col_name_lim_dict[col_name][1])
            plt.colorbar()
            plt.title(col_name)
            outf = join(outdir,col_name+'.pdf')
            plt.savefig(outf)
            plt.close()
            # plt.show()
        T.open_path_and_file(outdir)

    def plot_AI(self):
        outdir = join(self.this_class_tif,'AI')
        outdir_png = join(self.this_class_png,'AI')
        T.mk_dir(outdir)
        T.mk_dir(outdir_png)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        spatial_dict = T.df_to_spatial_dic(df,'aridity_index')
        fpath_tif = join(outdir,'aridity_index.tif')
        D_China.pix_dic_to_tif(spatial_dict,fpath_tif)
        plt.figure(figsize=(8,6))
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        # Blue represents high values, and red represents low values.
        cmap = Tools().cmap_blend(color_list)
        Plot().plot_China_Albers(fpath_tif,global_china_shp,global_china_shp_provinces,
                                 vmin=0.0,vmax=1.3,cmap=cmap,is_discrete=True,colormap_n=14)
        plt.savefig(join(outdir_png,'aridity_index.png'))
        plt.close()
        T.open_path_and_file(outdir_png)
        pass

    def bar_percentage_AI_trend(self):
        outdir = join(self.this_class_png, 'bar_percentage_AI_trend')
        T.mk_dir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        col_name_list = ['r_trend', 'max_lag_trend']
        AI_class_list = ['Arid','Humid']
        for col in col_name_list:
            plt.figure()
            for AI_class in AI_class_list:
                df_AI = df[df['AI_class']==AI_class]
                df_negative = df_AI[df_AI[col] < 0]
                df_positive = df_AI[df_AI[col] >= 0]
                df_negative_significant = df_negative[df_negative[col+'_p'] < 0.05]
                df_positive_significant = df_positive[df_positive[col+'_p'] < 0.05]

                df_negative_ratio = len(df_negative)/len(df_AI) * 100
                df_positive_ratio = len(df_positive)/len(df_AI) * 100

                df_negative_significant_ratio = len(df_negative_significant)/len(df_AI) * 100
                df_positive_significant_ratio = len(df_positive_significant)/len(df_AI) * 100

                df_negative_ratio = - df_negative_ratio
                df_negative_significant_ratio = - df_negative_significant_ratio
                plt.barh(AI_class,df_negative_ratio,color='none',edgecolor='k',zorder=100)
                plt.barh(AI_class,df_negative_significant_ratio,zorder=99,color='r',edgecolor='k')

                plt.barh(AI_class,df_positive_ratio,color='none',edgecolor='k',zorder=100)
                plt.barh(AI_class,df_positive_significant_ratio,zorder=99,color='b',edgecolor='k')
                plt.text(df_positive_significant_ratio + 2, AI_class,
                         f'{df_positive_ratio:.1f} ({df_positive_significant_ratio:.1f})', ha='left', va='center')
                plt.text(df_negative_significant_ratio - 2, AI_class,
                         f'{df_negative_ratio:.1f} ({df_negative_significant_ratio:.1f})', ha='right', va='center')

            plt.title(col)
            outf = join(outdir, col + '.pdf')
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)
        # plt.show()

    def bar_percentage_PFT_trend(self):
        outdir = join(self.this_class_png,'bar_percentage_PFT_trend')
        T.mk_dir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        col_name_list = ['r_trend', 'max_lag_trend']
        PFT_list = T.get_df_unique_val_list(df,'landcover_GLC')
        for col in col_name_list:
            plt.figure()
            for PFT in PFT_list:
                df_PFT = df[df['landcover_GLC']==PFT]
                df_negative = df_PFT[df_PFT[col] < 0]
                df_positive = df_PFT[df_PFT[col] >= 0]
                df_negative_significant = df_negative[df_negative[col+'_p'] < 0.05]
                df_positive_significant = df_positive[df_positive[col+'_p'] < 0.05]

                df_negative_ratio = len(df_negative)/len(df_PFT) * 100
                df_positive_ratio = len(df_positive)/len(df_PFT) * 100

                df_negative_significant_ratio = len(df_negative_significant)/len(df_PFT) * 100
                df_positive_significant_ratio = len(df_positive_significant)/len(df_PFT) * 100

                df_negative_ratio = - df_negative_ratio
                df_negative_significant_ratio = - df_negative_significant_ratio
                plt.barh(PFT,df_negative_ratio,color='none',edgecolor='k',zorder=100)
                plt.barh(PFT,df_negative_significant_ratio,zorder=99,color='r',edgecolor='k')

                plt.barh(PFT,df_positive_ratio,color='none',edgecolor='k',zorder=100)
                plt.barh(PFT,df_positive_significant_ratio,zorder=99,color='b',edgecolor='k')

                # plt.text(df_positive_significant_ratio+2,PFT,'%.1f'%df_positive_ratio,ha='left',va='center')
                plt.text(df_positive_significant_ratio+2,PFT,f'{df_positive_ratio:.1f} ({df_positive_significant_ratio:.1f})',ha='left',va='center')
                plt.text(df_negative_significant_ratio-2,PFT,f'{df_negative_ratio:.1f} ({df_negative_significant_ratio:.1f})',ha='right',va='center')
            plt.title(col)
            outf = join(outdir,col+'.pdf')
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)
        # plt.show()

class Random_Forests:

    def __init__(self):
        self.this_class_arr = data_root + 'RF_pix\\'
        self.this_class_png = data_root + 'SHAP\\png\\'


        self.dff = rf'E:\Data\ERA5_precip\ERA5_daily\RF_pix\Dataframe\\raw_data.df'
        self.variables_list()

        ##----------------------------------

        self.y_variable = 'LAI4g_raw'
        ####################

        self.x_variable_list = self.x_variable_list
        # self.x_variable_range_dict = self.x_variable_range_dict_AUS
    #
    #     pass

    def run(self):
        # self.copy_df()
        df = self.__gen_df_init()

        # self.check_variables_valid_ranges()
        # self.run_important_for_each_pixel()
        # self.run_important_for_each_pixel_for_two_period()
        # self.plot_importance_result_for_each_pixel()
        # self.plot_importance_R2_for_each_pixel()
        self.plot_most_important_factor_for_each_pixel()
        # self.summarized_important_factor_for_each_continent()
        # self.run_permutation_importance()
        # self.plot_importance_result()
        # self.plot_importance_result_R2()
        # self.partial_SHAP(df,self.x_variable_list,self.y_variable_list[0])
        # self.run_partial_dependence_plots()

        # self.plot_run_partial_dependence_plots()


        pass

    def copy_df(self):
        outdir = join(self.this_class_arr,'RF')
        T.mk_dir(outdir,force=True)
        outf = self.dff
        dff_origin =  'D:\Project3\Result\Dataframe\RF\RF.df'
        dff_origin_xlsx = 'D:\Project3\Result\Dataframe\RF\RF.xlsx'
        shutil.copy(dff_origin,outf)
        shutil.copy(dff_origin_xlsx,join(outdir,'RF.df.xlsx'))
        pass

    def add_CV(self,df):
        data_obj_list = [
            Load_Data().CRU_tmp_origin_GS,
            Load_Data().CRU_precip_origin_GS,
            Load_Data().P_ET_diff_origin_GS
        ]
        T.print_head_n(df)
        for data_obj in data_obj_list:
            data_dict,data_name = data_obj()
            spatial_dict_CV = {}
            for pix in data_dict:
                vals = data_dict[pix]
                if T.is_all_nan(vals):
                    continue
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                CV = std/mean
                spatial_dict_CV[pix] = CV
            df = T.add_spatial_dic_to_df(df,spatial_dict_CV,data_name+'_CV')
        return df
        pass

    def add_trend(self,df):
        data_obj_list = [
            Load_Data().CRU_tmp_anomaly_GS,
            Load_Data().CRU_precip_anomaly_GS,
            Load_Data().P_ET_diff_anomaly_GS
        ]
        T.print_head_n(df)
        for data_obj in data_obj_list:
            data_dict, data_name = data_obj()
            spatial_dict_trend = {}
            for pix in tqdm(data_dict,desc=data_name):
                vals = data_dict[pix]
                if T.is_all_nan(vals):
                    continue
                a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
                spatial_dict_trend[pix] = a
            df = T.add_spatial_dic_to_df(df,spatial_dict_trend,data_name+'_trend')
        return df

    def check_variables_valid_ranges(self):
        dff = self.dff
        df = T.load_df(dff)
        plt.figure(figsize=(6,6))
        flag = 1
        x_variable_list = self.x_variable_list
        for x_var in x_variable_list:
            plt.subplot(3,3,flag)
            flag += 1
            vals = df[x_var].tolist()
            vals = np.array(vals)
            vals[vals>1] = np.nan
            vals[vals<0] = np.nan
            plt.hist(vals,bins=100)
            plt.xlabel(x_var)
        plt.tight_layout()
        plt.show()


    def clean_dataframe(self,df):
        x_variable_range_dict = self.x_variable_range_dict
        for x_var in x_variable_range_dict:
            x_var_range = x_variable_range_dict[x_var]
            df = df[df[x_var] >= x_var_range[0]]
            df = df[df[x_var] <= x_var_range[1]]

        return df

    def run_important_for_each_pixel(self):

        dff = self.dff
        df = T.load_df(dff)
        df=self.df_clean(df)
        pix_list = T.get_df_unique_val_list(df,'pix')
        spatial_dict = {}
        for pix in pix_list:
            spatial_dict[pix] = 1
        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.show()
        # ## plot spatial df
        # T.print_head_n(df)


        group_dic = T.df_groupby(df,'pix')
        # spatial_dict = {}
        # for pix in group_dic:
        #     df_pix = group_dic[pix]
        #     spatial_dict[pix] = len(df_pix)
        # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.show()
        ## plot spatial df
        # spatial_dic = T.df_to_spatial_dic(df,'pix')
        # array= DIC_and_TIF().spatial_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()


        outdir= join(self.this_class_arr,'raw_importance_for_each_pixel')
        T.mk_dir(outdir,force=True)

        for y_var in self.y_variable_list:
            importance_spatial_dict = {}

            for pix in tqdm(group_dic):
                df_pix = group_dic[pix]

                ### to extract 1983-2020
                # vals_list=[]
                # name_list=[]
                #
                # for col in self.x_variable_list:
                #     vals = df_pix[col].tolist()
                #     vals=vals[1:]
                #     name=col
                #     vals_list.append(vals)
                #     name_list.append(name)
                # y_vals = df_pix[y_var].tolist()
                # y_vals = y_vals[1:]
                # vals_list.append(y_vals)
                # name_list.append(y_var)
                # dic_new = dict(zip(name_list,vals_list))
                # df_new = pd.DataFrame(dic_new)
                #
                #
                # T.print_head_n(df_new)

                x_variable_list = self.x_variable_list
                ## extract the data[1:]
                df_new = df_pix.dropna(subset=[y_var] + self.x_variable_list, how='any')
                if len(df_new) < 20:
                    continue
                X=df_new[x_variable_list]
                Y=df_new[y_var]
                # T.print_head_n(df_new)

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                                    random_state=1)  # split the data into training and testing
                # clf=xgb.XGBRegressor(objective="reg:squarederror",booster='gbtree',n_estimators=100,
                #                  max_depth=11,eta=0.05,random_state=42,n_jobs=12)
                clf = RandomForestRegressor(n_estimators=20)  # build a random forest model
                clf.fit(X_train, Y_train)  # train the model
                # R2= clf.score(X_test, Y_test)  # calculate the R2
                R = stats.pearsonr(clf.predict(X_test),Y_test)[0]
                R2 = R**2

                importance=clf.feature_importances_
                # print(importance)
                importance_dic=dict(zip(x_variable_list,importance))
                importance_dic['R2']=R2

                # print(importance_dic)
                importance_spatial_dict[pix]=importance_dic
            importance_dateframe = T.dic_to_df(importance_spatial_dict, 'pix')
            T.print_head_n(importance_dateframe)
            outf = join(outdir, f'{y_var}.df')
            T.save_df(importance_dateframe, outf)
            outf_xlsx = outf + '.xlsx'
            T.df_to_excel(importance_dateframe, outf_xlsx)

    def run_important_for_each_pixel_for_two_period(self):  ### run for two_period
        dff = self.dff
        df = T.load_df(dff)
        df_early = df[df['year_range']<17]
        df_late = df[df['year_range']>=17]
        df_early = df_early.dropna(subset=[self.y_variable_list[0]]+self.x_variable_list,how='any')
        df_late = df_late.dropna(subset=[self.y_variable_list[0]]+self.x_variable_list,how='any')
        df_list = [df_early,df_late]
        period_list = ['early','late']
        for df_i in df_list:
            period=period_list[df_list.index(df_i)]
            period='late'


            group_dic = T.df_groupby(df_i,'pix')
            outdir= join(self.this_class_arr,'important_for_each_pixel')
            T.mk_dir(outdir,force=True)

            for y_var in self.y_variable_list:
                importance_spatial_dict = {}

                for pix in tqdm(group_dic):
                    df_pix = group_dic[pix]

                    df_pix=df_pix.dropna(subset=[y_var]+self.x_variable_list,how='any')
                    # T.print_head_n(df_pix)
                    if len(df_pix)<10:
                        continue

                    x_variable_list = self.x_variable_list
                    X=df_pix[x_variable_list]
                    Y=df_pix[y_var]

                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                                        random_state=1)  # split the data into training and testing
                    clf = RandomForestRegressor(n_estimators=20)  # build a random forest model
                    clf.fit(X_train, Y_train)  # train the model
                    R2= clf.score(X_test, Y_test)  # calculate the R2
                    importance=clf.feature_importances_
                    # print(importance)
                    importance_dic=dict(zip(x_variable_list,importance))
                    importance_dic['R2']=R2

                    # print(importance_dic)
                    importance_spatial_dict[pix]=importance_dic
                importance_dateframe = T.dic_to_df(importance_spatial_dict, 'pix')
                T.print_head_n(importance_dateframe)

                ## save to df with two period

                outdf=join(outdir,f'{y_var}_{period}.df')
                T.save_df(importance_dateframe,outdf)
                outf_xlsx = outdf + '.xlsx'
                T.df_to_excel(importance_dateframe, outf_xlsx)


    def plot_importance_result_for_each_pixel(self):
        keys=list(range(len(self.x_variable_list)))
        x_variable_dict=dict(zip(self.x_variable_list, keys))
        print(x_variable_dict)
        # exit()

        fdir = rf'E:\Data\ERA5_precip\ERA5_daily\RF_pix\raw_importance_for_each_pixel\\'
        for f in os.listdir(fdir):

            if not f.endswith('.df'):
                continue
            fpath=join(fdir,f)
            fname=f.split('.')[0]


            df = T.load_df(fpath)

            T.print_head_n(df)
            spatial_dic={}
            sptial_R2_dic={}
            x_variable_list = self.x_variable_list
            for x_var in x_variable_list:


            ## plot individual importance
                for i, row in df.iterrows():
                    pix = row['pix']
                    importance_dic = row.to_dict()
                    # print(importance_dic)

                    spatial_dic[pix] = importance_dic[x_var]
                arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)

                plt.imshow(arr,vmin=0,vmax=0.5,interpolation='nearest',cmap='RdYlGn')

                plt.colorbar()
                plt.title(f'{fname}_{x_var}')
                plt.show()
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr,join(fdir,f'{fname}_{x_var}.tif'))

    def plot_importance_R2_for_each_pixel(self):
        keys=list(range(len(self.x_variable_list)))
        x_variable_dict=dict(zip(self.x_variable_list, keys))
        print(x_variable_dict)
        # exit()

        fdir = rf'E:\Data\ERA5_precip\ERA5_daily\RF_pix\raw_importance_for_each_pixel\\'
        for f in os.listdir(fdir):

            if not f.endswith('.df'):
                continue
            fpath=join(fdir,f)
            fname=f.split('.')[0]


            df = T.load_df(fpath)

            T.print_head_n(df)

            spatial_R2_dic={}




            ## plot individual importance
            for i, row in df.iterrows():
                pix = row['pix']
                importance_dic = row.to_dict()
                # print(importance_dic)

                spatial_R2_dic[pix] = importance_dic['R2']
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_R2_dic)

            plt.imshow(arr,vmin=0,vmax=0.5,interpolation='nearest',cmap='RdYlGn')

            plt.colorbar()
            plt.title(f'{fname}_R2')
            plt.show()
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr,join(fdir,f'{fname}_R2.tif'))




    def plot_most_important_factor_for_each_pixel(self):  ## second important factor/ third important factor
        keys=list(range(len(self.x_variable_list)))
        x_variable_dict=dict(zip(self.x_variable_list, keys))
        print(x_variable_dict)
        # exit()

        fdir = rf'E:\Data\ERA5_precip\ERA5_daily\RF_pix\raw_importance_for_each_pixel\\'
        for f in os.listdir(fdir):

            if not f.endswith('.df'):
                continue
            fpath=join(fdir,f)
            fname=f.split('.')[0]


            df = T.load_df(fpath)

            T.print_head_n(df)
            spatial_dic={}
            sptial_R2_dic={}
            for i, row in df.iterrows():
                pix = row['pix']
                importance_dic = row.to_dict()
                # print(importance_dic)
                x_variable_list = self.x_variable_list
                importance_dici = {}
                for x_var in x_variable_list:
                    importance_dici[x_var] = importance_dic[x_var]
                    # print(importance_dici)
                max_var = max(importance_dici, key=importance_dici.get)
                ## second important factor
                # max_var = sorted(importance_dici, key=importance_dici.get, reverse=True)[1]
                ## third important factor
                # max_var = sorted(importance_dici, key=importance_dici.get, reverse=True)[2]
                max_var_val=x_variable_dict[max_var]
                spatial_dic[pix]=max_var_val

                # print(max_var_val)
                # print(max_var)
                importance_dici['R2'] = importance_dic['R2']
                R2 = importance_dic['R2']
                sptial_R2_dic[pix]=R2
            #### print average R2
            # R2_list = list(sptial_R2_dic.values())
            # R2_mean = np.nanmean(R2_list)
            # print(f'{fname} R2 mean: {R2_mean}')
            # exit()

                ### plot R2
            arrR2 = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(sptial_R2_dic)
            arrR2[arrR2<0]=np.nan
            plt.imshow(arrR2,vmin=0,vmax=0.5,interpolation='nearest',cmap='RdYlGn')
            plt.colorbar()
            plt.title(f'{fname}_R2')
            plt.show()
            outtif_R2 = join(fdir, f'{fname}_R2.tif')

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arrR2, outtif_R2)

            ### plot importance
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)

            plt.imshow(arr,vmin=0,vmax=12,interpolation='nearest')
            plt.colorbar()
            plt.title(fname)
            plt.show()
            outtif=join(fdir,f'{fname}_most.tif')

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr,outtif)

            pass

    def summarized_important_factor_for_each_continent(self):
        dff=rf'D:\Project3\Data\RF_pix\detrended_anomaly_RF\\LAI4g_detrended_anomaly.df'
        df = T.load_df(dff)
        continent_list=['Global','Asia','Australia','North_America','South_America',]
        color_list=['grey','red','blue','green','yellow']
        result_continent_dic={}

        for continent in continent_list:
            if continent == 'Global':
                df_continent = df
            else:

                df_continent = df[df['continent']==continent]


            x_variable_list = self.x_variable_list
            ### count the number of each variable

            VPD_relative_change_detrended_num=0
            GPCC_relative_change_detrended_num=0
            fire_burned_area_num=0
            CV_rainfall_num=0
            average_anomaly_heat_event_num=0

            for i, row in df_continent.iterrows():
                pix = row['pix']
                importance_dic = row.to_dict()
                importance_dici = {}
                for x_var in x_variable_list:


                    importance_dici[x_var] = importance_dic[x_var]

                # max_var = max(importance_dici, key=importance_dici.get)
                ##second
                max_var = sorted(importance_dici, key=importance_dici.get, reverse=True)[2]
                if max_var == 'VPD_relative_change_detrended':
                    VPD_relative_change_detrended_num+=1
                if max_var == 'CV_rainfall':
                    CV_rainfall_num+=1
                if max_var == 'GPCC_relative_change_detrended':

                    GPCC_relative_change_detrended_num+=1
                if max_var == 'fire_burned_area':
                    fire_burned_area_num+=1
                if max_var == 'average_anomaly_heat_event':
                    average_anomaly_heat_event_num+=1
            percentage_VPD_relative_change_detrended = VPD_relative_change_detrended_num/len(df_continent)*100
            percentage_GPCC_relative_change_detrended = GPCC_relative_change_detrended_num/len(df_continent)*100
            percentage_fire_burned_area = fire_burned_area_num/len(df_continent)*100
            percentage_CV_rainfall = CV_rainfall_num/len(df_continent)*100
            percentage_average_anomaly_heat_event = average_anomaly_heat_event_num/len(df_continent)*100
            result_continent_dic[continent] = [percentage_VPD_relative_change_detrended,percentage_GPCC_relative_change_detrended,
            percentage_fire_burned_area,percentage_CV_rainfall,percentage_average_anomaly_heat_event]
        result_continent_df = pd.DataFrame(result_continent_dic,index=['VPD_relative_change_detrended','GPCC_relative_change_detrended',
        'fire_burned_area','CV_rainfall','average_anomaly_heat_event'])
        print(result_continent_df)
        ## plot
        result_continent_df.plot(kind='bar',color=color_list)
        x_label=['VPD_IAV','Precip_IAV', 'fire_burned_area','Rainfall_seasonality','extreme_heat_event_anomaly']
        plt.xticks(np.arange(5),x_label,rotation=0)
        plt.ylabel('Percentage (%)')

        plt.show()
        pass
























    def run_permutation_importance(self): ### run for whole region
        outdir = join(self.this_class_arr, 'permutation_importance')
        T.mk_dir(outdir, force=True)
        y_variable_list = self.y_variable_list
        x_variable_list = self.x_variable_list
        dff = self.dff
        df = T.load_df(dff)
        df=self.df_clean(df)
        T.print_head_n(df)
        #extract region unique values
        # region=df['AI_classfication'].unique().tolist()
        # print(region)
        # exit()

        regions=['Africa','Asia','Australia','Europe','North_America','South_America']
        R2_dic={}
        for region in regions:
            if region=='North_America':

                df_region = df[df['lon'] > -125]
                df_region = df_region[df_region['lon'] < -105]
                df_region = df_region[df_region['lat'] > 0]
                df_region = df_region[df_region['lat'] < 45]
            else:
                df_region = df[df['continent'] == region]


            for y_variable in y_variable_list:
                df_region=df_region.dropna(subset=[y_variable]+x_variable_list,how='any')
                X = df_region[x_variable_list]
                Y = df_region[y_variable]
                variable_list = x_variable_list
                clf, importances_dic, mse, r_model, score, Y_test, y_pred = self._random_forest_train(X, Y,
                                                                                                         variable_list)
                outf = join(outdir, f'{y_variable}_{region}.npy')
                T.save_npy(importances_dic, outf)

                R2_dic[region]=score
                outf_R2 = join(outdir, f'R2_{y_variable}_{region}.npy')
                T.save_npy(R2_dic, outf_R2)

    def run_permutation_importance_for_two_period(self): ### run for two_period
        outdir = join(self.this_class_arr, 'permutation_importance')
        T.mk_dir(outdir, force=True)
        y_variable_list = self.y_variable_list
        x_variable_list = self.x_variable_list
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        ## split df into two periods
        df_early = df[df['year']<2000]
        df_late = df[df['year']>=2000]
        df_early = df_early.dropna(subset=[y_variable_list[0]]+x_variable_list,how='any')
        df_late = df_late.dropna(subset=[y_variable_list[0]]+x_variable_list,how='any')
        X_early = df_early[x_variable_list]
        Y_early = df_early[y_variable_list[0]]
        X_late = df_late[x_variable_list]
        Y_late = df_late[y_variable_list[0]]
        variable_list = x_variable_list
        clf_early, importances_dic_early, mse_early, r_model_early, score_early, Y_test_early, y_pred_early = self._random_forest_train(X_early, Y_early,
                                                                                                        variable_list)
        clf_late, importances_dic_late, mse_late, r_model_late, score_late, Y_test_late, y_pred_late = self._random_forest_train(X_late, Y_late,

                                                                                                        variable_list)
        outf_early = join(outdir, f'{y_variable_list[0]}_early.npy')
        outf_late = join(outdir, f'{y_variable_list[0]}_late.npy')
        T.save_npy(importances_dic_early, outf_early)
        T.save_npy(importances_dic_late, outf_late)
        pass




    def plot_importance_result(self):
        fdir = join(self.this_class_arr,'permutation_importance')
        for f in T.listdir(fdir):
            fname = f.split('.')[0]
            print(fname)

            result_dic=T.load_npy(fdir+'/'+f)
            df = pd.DataFrame(dict(result_dic), index=['imp']).T
            df_sort = df.sort_values(by='imp', ascending=False)
            print(df_sort)
            df_sort.plot(kind='bar')
            plt.title(fname)

            plt.tight_layout()

            plt.show()


        pass

    def plot_importance_result_R2(self):
        fdir = join(self.this_class_arr,'permutation_importance','R2')
        for f in T.listdir(fdir):
            fname = f.split('.')[0]
            print(fname)

            result_dic=T.load_npy(fdir+'/'+f)
            for key in result_dic:
                print(key,result_dic[key])






    def run_partial_dependence_plots(self):
        # fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        dff = self.dff
        outdir = join(self.this_class_arr, 'partial_dependence_plots')
        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list
        for y_variable in self.y_variable_list:
            df = T.load_df(dff)
            df = self.df_clean(df)

            print(df.columns.tolist())
            # exit()
            # df = df[df['AI_class']=='Arid']
            print(len(df))
            # pix_list = T.get_df_unique_val_list(df,'pix')
            # spatial_dict = {}
            # for pix in pix_list:
            #     spatial_dict[pix] = 1
            # arr = D.pix_dic_to_spatial_arr(spatial_dict)
            # plt.imshow(arr)
            # plt.show()
            T.print_head_n(df)
            print('-'*50)
            result_dic = self.partial_dependence_plots(df, x_variable_list, y_variable)
            outf = join(outdir, f'{y_variable}.npy')
            T.save_npy(result_dic, outf)

    def plot_run_partial_dependence_plots(self):
        fdir = join(self.this_class_arr,'partial_dependence_plots')
        outdir = join(self.this_class_png,'partial_dependence_plots')
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):

            fpath = join(fdir,f)

            result_dict = T.load_npy(fpath)
            flag = 1
            plt.figure(figsize=(10,8))
            for key in result_dict:
                result_dict_i = result_dict[key]
                x = result_dict_i['x']
                y = result_dict_i['y']
                y_std = result_dict_i['y_std']
                plt.subplot(2,5,flag)
                flag += 1
                y = SMOOTH().smooth_convolve(y,window_len=5)
                plt.plot(x,y)
                plt.xlabel(key)
                # y_std = y_std / 4
                # plt.fill_between(x,y-y_std,y+y_std,alpha=0.5)
                # plt.legend()
                # plt.ylim(-.5,.5)
                # plt.xlabel(key.replace('_vs_NDVI-anomaly_detrend_','\nsensitivity\n'))
            plt.suptitle(f)
            # plt.tight_layout()
            outpath = join(outdir,f.replace('.npy','.pdf'))
            # plt.savefig(outpath)
            # plt.close()
        plt.show()
        # T.open_path_and_file(outdir)

    def partial_dependence_plots(self,df,x_vars,y_var):
        '''
        :param df: a dataframe
        :param x_vars: a list of x variables
        :param y_var: a y variable
        :return:
        '''
        all_vars = copy.copy(x_vars) # copy the x variables
        all_vars.append(y_var) # add the y variable to the list
        all_vars_df = df[all_vars] # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna() # drop rows with missing values
        X = all_vars_df[x_vars]
        Y = all_vars_df[y_var]
        model, r2 = self.__train_model(X, Y) # train a Random Forests model
        flag = 0
        result_dic = {}
        for var in x_vars:
            flag += 1
            df_PDP = self.__get_PDPvalues(var, X, model) # get the partial dependence plot values
            ppx = df_PDP[var]
            ppy = df_PDP['PDs']
            ppy_std = df_PDP['PDs_std']
            result_dic[var] = {'x':ppx,
                               'y':ppy,
                               'y_std':ppy_std,
                               'r2':r2}
        return result_dic

    def partial_SHAP(self,df,x_vars,y_var):
        '''
        :param df: a dataframe
        :param x_vars: a list of x variables
        :param y_var: a y variable
        :return:
        '''
        ## randomly select 1000 samples
        df = df.sample(n=1006, random_state=1)
        print(len(df))
        all_vars = copy.copy(x_vars) # copy the x variables
        all_vars.append(y_var) # add the y variable to the list
        all_vars_df = df[all_vars] # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna() # drop rows with missing values
        X = all_vars_df[x_vars]
        Y = all_vars_df[y_var]
        model, r2 = self.__train_model(X, Y) # train a Random Forests model
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X)
        print(shap_values)
        print(np.shape(shap_values))

        # shap.bar_plot(shap_values[0], X)
        shap.dependence_plot("Precip_CV", shap_values, X, interaction_index="Precip_CV")
        plt.tight_layout()
        plt.show()

    def _random_forest_train(self, X, Y, variable_list):
        '''
        :param X: a dataframe of x variables
        :param Y: a dataframe of y variable
        :param variable_list: a list of x variables
        :return: details of the random forest model and the importance of each variable
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # split the data into training and testing
        clf = RandomForestRegressor(n_estimators=100, n_jobs=14) # build a random forest model
        clf.fit(X_train, Y_train) # train the model
        result = permutation_importance(clf, X_train, Y_train, scoring=None,
                                        n_repeats=50, random_state=1,
                                        n_jobs=1) # calculate the importance of each variable using permutation importance
        importances = result.importances_mean # get the importance of each variable
        importances_dic = dict(zip(variable_list, importances)) # put the importance of each variable into a dictionary
        labels = []
        importance = []
        for key in variable_list:
            labels.append(key)
            importance.append(importances_dic[key])
        y_pred = clf.predict(X_test) # predict the y variable using the testing data
        r_model = stats.pearsonr(Y_test, y_pred)[0] # calculate the correlation between the predicted y variable and the actual y variable
        mse = sklearn.metrics.mean_squared_error(Y_test, y_pred) # calculate the mean squared error
        score = clf.score(X_test, Y_test) # calculate the R^2
        return clf, importances_dic, mse, r_model, score, Y_test, y_pred

    def plot_rf_result(self):
        fdir = join(self.this_class_arr, 'random_forest')
        outdir = join(self.this_class_png, 'random_forest')
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            result_dict = T.load_npy(fpath)
            importances_dic = result_dict['importances_dic']
            r_model = result_dict['r_model']
            score = result_dict['score']
            title = f'{f}\nR^2={score}, r={r_model}'
            x = importances_dic.keys()
            y = [importances_dic[key] for key in x]
            plt.figure(figsize=(10, 5))
            plt.bar(x, y)
            plt.title(title)
            outf = join(outdir, f'{f}.pdf')
            plt.savefig(outf, dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

    def __train_model(self,X,y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.2) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=7) # build a random forest model
        rf.fit(X_train, y_train) # train the model
        r2 = rf.score(X_test,y_test)
        return rf,r2

    def __get_PDPvalues(self, col_name, data, model, grid_resolution=50):
        '''
        :param col_name: a variable
        :param data: a dataframe of x variables
        :param model: a random forest model
        :param grid_resolution: the number of points in the partial dependence plot
        :return: a dataframe of the partial dependence plot values
        '''
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution) # create a sequence of values
        Y_pdp = []
        Y_pdp_std = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            Y_pdp.append(np.mean(Y_temp))
            Y_pdp_std.append(np.std(Y_temp))
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp, 'PDs_std': Y_pdp_std})

    def variables_list(self):
        self.x_variable_list = [
            'CO2_raw',
            'VPD_raw',
            'CRU_raw'


        ]


        self.y_variable_list = [
            'LAI4g_raw',


        ]
        pass


        self.x_variable_range_dict_Africa = {

            'Noy': [100, 500],

            'Nhx': [50, 400],
            'GMST': [0, 1],

            'VPD': [0.5, 3],
            'GPCC': [0, 125],
            'GPCP_precip_pre': [0, 100],

            'CO2': [350., 450.],

            'average_dry_spell': [0, 20],
            'maximum_dry_spell': [0, 60],
            'CV_rainfall': [0, 6],
            'total_rainfall': [0, 2000],

            'frequency_wet': [0, 50],
            'frequency_dry': [0, 50],
            # 'tmax_CV': [0, 10],
            # 'tmin_CV': [0, 10],

            'frequency_heat_event': [1, 50],
            'average_anomaly_heat_event': [2, 10],

            'silt': [0, 50],
            'rooting_depth': [0, 10],
            'SOC_sum': [0, 1500],
            'ENSO_index_average': [-2, 1],
            'ENSO_index_average_lagged': [-2, 2],
            'ENSO_index_distance': [0.5, 2],
            'ENSO_index_distance_lagged': [0.5, 2],
            'ENSO_index_DFJ': [-2, 2],

            'ENSO_index_binary': [-2, 2],
            'GPCC_trend': [-2, 2],
            'Aridity': [0.2, 0.65],
            'ENSO_index_average_lagged_whole_year': [-1, 1],
            'maximun_heat_spell': [0, 150],
            'average_heat_spell': [0, 30],
            'maximun_cold_spell': [0, 150],
            'average_cold_spell': [0, 30],
        }

        self.x_variable_range_dict_north_america = {

            'ozone': [280, 320],
            'GMST': [0, 1],
            'CO2': [350, 450],
            'Noy': [100, 300],
            'Nhx': [25, 250],

            'VPD': [0, 3],

            'average_dry_spell': [0, 20],
            'maximum_dry_spell': [0, 60],
            'CV_rainfall': [0, 6],
            'total_rainfall': [0, 2000],

            'frequency_wet': [0, 50],
            'frequency_dry': [0, 50],
            'tmax_CV': [0, 10],
            'tmin_CV': [0, 10],

            'frequency_heat_event': [1, 50],
            'average_anomaly_heat_event': [2, 10],

            'GPCC': [0, 100],
            'GPCP_precip_pre': [0, 100],
            'GLEAM_SMroot': [-50, 50],
            # 'tmin': [-10, 10],
            # 'tmax': [-7.5, 4],

            'silt': [0, 60],
            'rooting_depth': [0, 20],
            'SOC_sum': [0, 2000],
            'ENSO_index_average': [-1.5, 1.5],
            'ENSO_index_average_lagged': [-2, 2],
            'ENSO_index_distance': [0.5, 2],
            'ENSO_index_distance_lagged': [0.5, 2],
            'ENSO_index_DFJ': [-2, 2],

            'ENSO_index_binary': [-2, 2],
            'GPCC_trend': [-2, 2],
            'Aridity': [0.2, 0.65],
            'ENSO_index_average_lagged_whole_year': [-1, 1],
            'maximun_heat_spell': [0, 150],
            'average_heat_spell': [0, 30],
            'maximun_cold_spell': [0, 150],
            'average_cold_spell': [0, 30],
        }

        self.x_variable_range_dict_Asia = {

            'Noy': [10, 500],

            'GMST': [0, 1],

            'Nhx': [0, 1000],

            'CO2': [340, 450],

            'VPD': [0.5, 3],
            'average_dry_spell': [0, 25],
            'maximum_dry_spell': [0, 150],
            'CV_rainfall': [0, 6],
            'total_rainfall': [0, 2000],

            'frequency_wet': [0, 50],
            'frequency_dry': [0, 50],

            'frequency_heat_event': [1, 40],
            'average_anomaly_heat_event': [2, 10],
            'average_anomaly_cold_event': [-12, -2],

            'GPCC': [0, 200],
            'GPCP_precip_pre': [0, 100],
            'GLEAM_SMroot': [-50, 50],
            'tmin': [-10, 10],
            'tmax': [-7.5, 4],

            'silt': [0, 50],
            'rooting_depth': [0, 10],
            'ENSO_index_average': [-2, 2],
            'ENSO_index_average_lagged': [-2, 2],
            'ENSO_index_distance': [0.5, 2],
            'ENSO_index_distance_lagged': [0.5, 2],
            'ENSO_index_DFJ': [-2, 2],

            'ENSO_index_binary': [-2, 2],
            'GPCC_trend': [-2, 2],
            'Aridity': [0.2, 0.65],
            'ENSO_index_average_lagged_whole_year': [-1, 1],
            'maximun_heat_spell': [0, 150],
            'average_heat_spell': [0, 30],
            'maximun_cold_spell': [0, 150],
            'average_cold_spell': [0, 40],
        }

        self.x_variable_range_dict_South_America = {

            'Noy': [0, 200],
            'Nhx': [0, 600],
            'GMST': [0, 1],

            'CO2': [340, 450],

            'VPD': [0.5, 3],
            'average_dry_spell': [0, 25],
            'maximum_dry_spell': [0, 75],
            'CV_rainfall': [0, 6],
            'total_rainfall': [0, 2000],

            'frequency_wet': [0, 50],
            'frequency_dry': [0, 50],

            'frequency_heat_event': [1, 40],
            'average_anomaly_heat_event': [2, 10],
            'average_anomaly_cold_event': [-12, -2],

            'GPCC': [0, 200],
            'GPCP_precip_pre': [0, 100],

            'silt': [0, 50],
            'rooting_depth': [0, 20],
            'ENSO_index_average': [-2, 2],
            'ENSO_index_average_lagged': [-2, 2],
            'ENSO_index_distance': [0.5, 2],
            'ENSO_index_distance_lagged': [0.5, 2],
            'ENSO_index_DFJ': [-2, 2],

            'ENSO_index_binary': [-2, 2],
            'GPCC_trend': [-2, 2],
            'Aridity': [0.2, 0.65],
            'ENSO_index_average_lagged_whole_year': [-1, 1],
            'maximun_heat_spell': [0, 150],
            'average_heat_spell': [0, 30],
            'maximun_cold_spell': [0, 150],
            'average_cold_spell': [0, 40],

        }

        self.x_variable_range_dict_AUS = {
            'CO2': [350, 450],
            'GMST': [0, 1],

            'Noy': [20, 150],
            'Nhx': [0, 200],

            'VPD': [0.5, 3],
            'average_dry_spell': [0, 20],
            'maximum_dry_spell': [0, 100],
            'CV_rainfall': [0, 6],
            'total_rainfall': [0, 2000],

            'frequency_wet': [0, 40],
            'frequency_dry': [0, 50],
            'tmax_CV': [0, 10],
            'tmin_CV': [0, 10],
            'frequency_cold_event': [1, 40],
            'average_anomaly_cold_event': [-7, -3],
            'frequency_heat_event': [1, 50],
            'average_anomaly_heat_event': [4, 12],

            'GPCC': [0, 200],
            'GPCP_precip_pre': [0, 100],
            'GLEAM_SMroot': [-50, 50],

            'silt': [0, 50],

            'rooting_depth': [0, 20],
            'SOC_sum': [0, 4000],
            'ENSO_index_average': [-2, 2],

            'GPCC_trend': [-1, 1],
            'Aridity': [0.2, 0.65],

            'maximun_heat_spell': [0, 100],
            'average_heat_spell': [0, 25],
            'maximun_cold_spell': [25, 75],
            'average_cold_spell': [0, 25],

        }

        pass
    def df_clean(self,df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]
        df=df[df['LC_max']<20]


        df = df[df['landcover_classfication'] != 'Cropland']



        return df

    def valid_range_df(self,df):

        print('original len(df):',len(df))
        for var in self.x_variable_list:


            if not var in df.columns:
                print(var,'not in df')
                continue
            min,max = self.x_variable_range_dict[var]
            df = df[(df[var]>=min)&(df[var]<=max)]
        print('filtered len(df):',len(df))
        return df


def main():
    # Dataframe().run()
    # Bivariate_statistic().run()
    # Correlation_Lag_statistic().run()
    # Trend_statistic().run()
    Random_Forests().run()
    pass


if __name__ == '__main__':
    main()