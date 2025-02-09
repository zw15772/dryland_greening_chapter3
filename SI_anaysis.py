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

class greening_analysis():

    def __init__(self):
        self.map_width = 15.3 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor

        pass
    def run(self):


        self.greening_products_basemap_whole_time()
        # self.greening_products_basemap_whole_time_CV()
        # self.greening_products_basemap_whole_time_consistency()
        # self.greening_products_basemap_two_time_periods()
        self.climate_variable_trends()
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
        outf = result_root + rf'\3mm\relative_change_growing_season\TRENDY\\trend_analysis\\greening_products_basemap_whole_time.png'
        plt.savefig(outf, dpi=600)
        plt.close()
        T.open_path_and_file(outf)

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
        self.trend_analysis_simply_linear()
        self.plot_climate_variable_trends()

    def trend_analysis_simply_linear(self):
        from scipy.stats import theilslopes
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root+rf'\3mm\relative_change_growing_season\whole_period\\'
        outdir = result_root + rf'3mm\relative_change_growing_season\whole_period\\\\trend_analysis_simple_linear_0206_lymethod\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):

            if 'detrend' in f:
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

    def plot_climate_variable_trends(self):
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
        fdir=result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\trend\\'
        temp_root = result_root + rf'\3mm\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\\ecosystem_year\\trend_analysis\\temp_root\\'
        T.mk_dir(temp_root)
        # T.open_path_and_file(fdir);exit()

        products = [ 'rainfall_intensity', 'rainfall_frenquency','detrended_sum_rainfall_CV','heat_event_frenquency','rainfall_seasonality_all_year',]

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
        outf = result_root + rf'\3mm\relative_change_growing_season\TRENDY\\trend_analysis\\greening_products_basemap_whole_time.png'
        plt.savefig(outf, dpi=600)
        plt.close()
        T.open_path_and_file(outf)

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
        elif type(cmap) == str:
            cmap = plt.get_cmap(cmap)
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


class Rainfall_product_comparison():
    pass

class TRENDY_trend():
    pass

class TRENDY_CV():
    pass

def main():
    greening_analysis().run()

    pass

if __name__ == '__main__':
    main()