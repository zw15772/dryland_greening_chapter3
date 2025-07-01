# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg

from upload_version_drrland_greening import trend_analysis

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
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lytools import *
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pprint import pprint
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
            outf = result_root + rf'\3mm\relative_change_growing_season\TRENDY\\trend_analysis\\greening_products_basemap_{product}.png'
            plt.savefig(outf, dpi=600)
            plt.close()
            # T.open_path_and_file(outf)

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
        # self.trend_analysis_simply_linear()
        # self.plot_climate_variable_trends()
        self.average_analysis()

    def trend_analysis_simply_linear(self):
        from scipy.stats import theilslopes
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\\'
        outdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\ecosystem_year\relative_trend\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'detrend' in f:
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
                average_value=np.nanmean(time_series)
                try:

                        # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                        slope,b,r,p_value=T.nan_line_fit(np.arange(len(time_series)), time_series)
                        trend_dic[pix] = slope/average_value*100
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

    def average_analysis(self):
        from scipy.stats import theilslopes
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\\'
        outdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\ecosystem_year\average\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if  'detrend' in f:
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

                time_series=dic[pix]['ecosystem_year']
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
                average_value=np.nanmean(time_series)
                try:

                        trend_dic[pix] = np.nanmean(time_series)

                except:
                    continue

            arr_trend = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(trend_dic)
            arr_trend_dryland = arr_trend * array_mask

            # plt.imshow(arr_trend_dryland, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()
            # exit()

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_trend_dryland, outf + '_trend.tif')

            np.save(outf + '_trend', arr_trend_dryland)


    def plot_climate_variable_trends(self):
        ## three products 3 time periods comparison
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np

        vmin_vmax = {
            'NDVI4g': (-0.1, 0.1),

            # 'GOSIF': (-1, 1)
            'rainfall_seasonality_all_year': (-0.2, 0.2),
            'rainfall_intensity': (-0.2, 0.2),
            'heat_event_frenquency': (-1, 1),
            'rainfall_frenquency': (-0.5, 0.5),
            'detrended_sum_rainfall_CV': (-1, 1),

        }

        # Create synthetic data (replace with actual data as needed)
        fdir=result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\relative_trend\\'
        temp_root = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\relative_trend\\temp_root\\'
        T.mk_dir(temp_root)
        # T.open_path_and_file(fdir);exit()

        products = [ 'rainfall_intensity', 'rainfall_frenquency','detrended_sum_rainfall_CV',]
        products = ['heat_event_frenquency', 'rainfall_seasonality_all_year', ]

        # fig, axes = plt.subplots(1, 3, figsize=(self.map_width*2, self.map_height))

        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        # fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.95, bottom=0.08, left=0.05, right=0.95)

        # Loop through products and periods to create subplots

        for i, product in enumerate(products):  # Use 'i' to index the subplot
            # ax = axes[i // 2, i % 2]
            ax=axes[i]


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

            # m.drawcoastlines(linewidth=0.2,color='k',zorder=11)
            # Plot the data using pcolormesh
            vmin, vmax = vmin_vmax[product]


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
            cbar.set_label(f'{product} trend (%/yr)',labelpad=-68)

            tick_labels = np.round(np.linspace(vmin, vmax, 5), 2)
            cbar.set_ticks(tick_labels)
            cbar.set_ticklabels([f"{tick:.2f}" for tick in tick_labels])
            # ax.set_title(product)
            # plt.show()
        outf = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\\climate_variable_trends.png'
        plt.savefig(outf, dpi=600)
        plt.close()
        T.open_path_and_file(outf)

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

    def trend_analysis_plot(self):
        outdir = result_root + rf'\3mm\relative_change_growing_season\TRENDY\trend_analysis\\'
        temp_root = result_root + rf'\3mm\relative_change_growing_season\TRENDY\trend_analysis\\temp_root\\'

        model_list = [  'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai','ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai','LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai', 'SDGVM_S2_lai',
                      'YIBs_S2_Monthly_lai']


        dic_name = {'LAI4g': 'LAI',
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
                    'ORCHIDEE_S2_lai': 'ORCHIDEE',
                    'SDGVM_S2_lai': 'SDGVM',
                    'YIBs_S2_Monthly_lai': 'YIBs',
                    'LPX-Bern_S2_lai': 'LPX-Bern',
                    }





        fdir = result_root + rf'\3mm\relative_change_growing_season\TRENDY\trend_analysis\\'
        count = 1
        for model in model_list:



            fpath= fdir+f'{model}_trend.tif'
            f_p_value = fdir + f'{model}_p_value.tif'


            # ax = plt.subplot(3, 2, count)
            # print(count, f)
            count = count + 1
            # if not count == 9:
            #     continue
            color_list = [
                '#844000',
                '#fc9831',
                '#fffbd4',
                '#86b9d2',
                '#064c6c',
            ]
            my_cmap2 = T.cmap_blend(color_list, n_colors=5)


            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)

            arr = Tools().mask_999999_arr(arr, warning=False)
            arr_m = ma.masked_where(np.isnan(arr), arr)
            lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            plt.figure(figsize=(5, 3))

            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180,
                        resolution='l')


            m = self.plot_sig_scatter(
                m, f_p_value, temp_root, sig_level=0.05, ax=None, linewidths=0.2,
                s=1,
                c='k', marker='x',
                zorder=100, res=4
            )
            plt.title(f'{dic_name[model]}', fontsize=10,font='Arial')

            m.drawcoastlines(linewidth=0.2, color='k', zorder=11)
            ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap=my_cmap2, zorder=-1, vmin=-0.3, vmax=0.3)


            # plt.show()
        # cax = plt.axes([0.5 - 0.15, 0.05, 0.3, 0.02])
        # cb = plt.colorbar(ret, ax=ax, cax=cax, orientation='horizontal')
        # cb.set_label(f'{model}_CV_trend', labelpad=-40)
            outf = outdir + rf'trend_analysis_{model}.png'

            plt.subplots_adjust(hspace=0.055, wspace=0.038)
            #
            plt.savefig(outf, dpi=600)
            plt.close()
            # plt.show()
        T.open_path_and_file(outdir)

    pass

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


class TRENDY_CV():

    def trend_analysis_plot(self):
        outdir = result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\'
        temp_root = result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\temp_root\\'

        model_list = ['LAI4g',  'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai','SDGVM_S2_lai',
                      'YIBs_S2_Monthly_lai']

        dic_name = {'LAI4g': 'LAI',
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
                    'ORCHIDEE_S2_lai': 'ORCHIDEE',
                    'SDGVM_S2_lai': 'SDGVM',
                    'YIBs_S2_Monthly_lai': 'YIBs',
                    'LPX-Bern_S2_lai': 'LPX-Bern',
                    }



        fdir = result_root + rf'3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\'
        count = 1
        for model in model_list:


            fpath= fdir+f'{model}_detrend_CV_trend.tif'
            f_p_value = fdir + f'{model}_detrend_CV_p_value.tif'


            # ax = plt.subplot(3, 2, count)
            # print(count, f)
            count = count + 1
            # if not count == 9:
            #     continue

            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr = Tools().mask_999999_arr(arr, warning=False)
            arr_m = ma.masked_where(np.isnan(arr), arr)
            lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            plt.figure(figsize=(5, 3))

            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180,
                        resolution='l',)

            m = self.plot_sig_scatter(
                m, f_p_value, temp_root, sig_level=0.05, ax=None, linewidths=0.2,
                s=1,
                c='k', marker='x',
                zorder=100, res=4
            )
            plt.title(f'{dic_name[model]}', fontsize=10,font='Arial')

            m.drawcoastlines(linewidth=0.2, color='k', zorder=11)
            ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap='PRGn', zorder=-1, vmin=-1, vmax=1)


            # plt.show()
        # cax = plt.axes([0.5 - 0.15, 0.05, 0.3, 0.02])
        # cb = plt.colorbar(ret, ax=ax, cax=cax, orientation='horizontal')
        # # cb.set_label(f'{model}_CV_trend', labelpad=-40)
            outf = outdir + rf'CV_trend_analysis_plot_{model}.png'

            plt.subplots_adjust(hspace=0.055, wspace=0.038)
            #
            plt.savefig(outf, dpi=600)
            plt.close()
        # plt.show()
        T.open_path_and_file(outdir)
        # exit()
    pass

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
class PLOT_Climate_factors():
    def run(self):
        self.trend_analysis_plot()
        pass
    def trend_analysis_plot(self):
        outdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\Robinson\\'

        T.mk_dir(outdir,True)
        temp_root = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\\temp_root\\'

        model_list = ['pi_average',  'heavy_rainfall_days', 'rainfall_seasonality_all_year',
                      'sum_rainfall', 'VOD_detrend_min', 'fire_ecosystem_year_average',
                      'VPD_max',
                      'Non tree vegetation']

        dic_name = {'pi_average': 'SM-T coupling',
                    'heavy_rainfall_days': 'Heavy rainfall days',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality',
                    'sum_rainfall': 'Total rainfall',
                    'VOD_detrend_min': 'VOD_min',
                    'fire_ecosystem_year_average': 'Fire burn area',
                    'VPD_max': 'VPD_max',

                    'Non tree vegetation': 'Non tree vegetation',
                    }



        fdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\trend2\\'
        count = 1
        for model in model_list:


            fpath= fdir+f'{model}_trend.tif'
            f_p_value = fdir + f'{model}_p_value.tif'


            # ax = plt.subplot(3, 2, count)
            # print(count, f)
            count = count + 1
            # if not count == 9:
            #     continue

            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr = Tools().mask_999999_arr(arr, warning=False)
            arr_m = ma.masked_where(np.isnan(arr), arr)
            lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
            lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
            lon_list, lat_list = np.meshgrid(lon_list, lat_list)
            plt.figure(figsize=(5, 3))

            m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-180, urcrnrlon=180,
                        resolution='l',)

            m = self.plot_sig_scatter(
                m, f_p_value, temp_root, sig_level=0.05, ax=None, linewidths=0.2,
                s=1,
                c='k', marker='x',
                zorder=100, res=4
            )
            plt.title(f'{dic_name[model]}', fontsize=10,font='Arial')

            m.drawcoastlines(linewidth=0.2, color='k', zorder=11)
            dic_vmin={'pi_average': -0.1,
                       'heavy_rainfall_days': -0.3,
                       'rainfall_seasonality_all_year': -0.5,
                       'sum_rainfall': -5,
                      'VOD_detrend_min': -0.005,
                      'fire_ecosystem_year_average': -1,
                      'VPD_max': -0.01,
                      'Non tree vegetation': -0.5


            }

            dic_vmax={'pi_average': 0.1,
                       'heavy_rainfall_days': .3,
                       'rainfall_seasonality_all_year': .5,
                       'sum_rainfall': 5,
                      'VOD_detrend_min': 0.005,
                      'fire_ecosystem_year_average': 1,
                      'VPD_max': 0.01,
                      'Non tree vegetation': 0.5}

            ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap='PRGn', zorder=-1, vmin=dic_vmin[model], vmax=dic_vmax[model])


            # plt.show()
        # cax = plt.axes([0.5 - 0.15, 0.05, 0.3, 0.02])
        # cb = plt.colorbar(ret, ax=ax, cax=cax, orientation='horizontal')
        # # cb.set_label(f'{model}_CV_trend', labelpad=-40)
            outf = outdir + rf'trend_analysis_plot_{model}.png'

            plt.subplots_adjust(hspace=0.055, wspace=0.038)
            #
            plt.savefig(outf, dpi=600)
            plt.close()
        # plt.show()
        T.open_path_and_file(outdir)
        # exit()
    pass

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


class SHAP_CV():


    def __init__(self):

        self.y_variable = 'LAI4g_detrend_CV'

        # self.this_class_png = results_root + 'ERA5\\SHAP\\png\\'
        self.threshold = '3mm'
        self.this_class_png = result_root + rf'\{self.threshold}\\SHAP\\RF_{self.y_variable}\\'
        # self.this_class_png = result_root + rf'\{self.threshold}\CRU_JRA\\SHAP\RF_{self.y_variable}\\'

        T.mk_dir(self.this_class_png, force=True)

        self.dff = rf'E:\Project3\Result\3mm\Dataframe\moving_window_CV\\moving_window_CV_new.df'
        # self.dff = result_root+rf'3mm\Dataframe\moving_window_CV\\moving_window_CV_1mm_3mm.df'
        self.variable_list_rt()
        self.variables_list = ['LAI4g', 'NDVI','CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',
                          'SDGVM_S2_lai',
                          'YIBs_S2_Monthly_lai']

        ##----------------------------------



        ####################

        self.x_variable_list = self.x_variable_list_CRU
        self.x_variable_range_dict = self.x_variable_range_dict_global_CRU

        pass

    def run(self):
        # self.check_df_attributes()

        # self.check_variables_ranges()
        # self.show_colinear()
        # self.check_spatial_plot()
        # self.pdp_shap()
        # self.plot_relative_importance() ## use this
        # self.plot_pdp_shap()
        # self.plot_pdp_shap_density_cloud()  ## use this
        # self.plot_heatmap_ranking()
        # self.plot_pdp_shap_density_cloud_individual()


        # self.spatial_shapely()   ### spatial plot
        # self.variable_contributions()
        # self.max_contributions()
        # self.plot_pdp_shap_normalized()
        # self.pdp_shap_trend()

        pass

    def check_df_attributes(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print(df.columns.tolist())
        print(len(df))
        # exit()
        pass

    def check_variables_ranges(self):

        dff = self.dff
        df = T.load_df(dff)
        df = self.df_clean(df)

        df = self.plot_hist(df)
        df = self.valid_range_df(df)
        # df = self.__select_extreme(df)
        # T.print_head_n(df)
        # exit()

        x_variable_list = self.x_variable_list
        print(len(x_variable_list))
        # exit()
        flag = 1

        for var in x_variable_list:
            print(flag, var)
            vals = df[var].tolist()
            plt.subplot(4, 4, flag)
            flag += 1
            plt.hist(vals, bins=100)
            plt.title(var)
        plt.tight_layout()
        plt.show()


        pass

    def AIC_criteria(self):

        import statsmodels.api as sm

        pass
    def variable_list_rt(self):
        self.x_variable_list = [

            # 'detrended_average_annual_tmax',
            # 'heavy_rainfall_days',

            # 'rainfall_frequency',
            'rainfall_intensity',
            'maxmum_dry_spell',
            'rainfall_seasonality_all_year',
            # 'rainfall_seasonality',
            'detrended_sum_rainfall_interannual_CV',
            # 'CV_rainfall_interseasonal',
            # 'Aridity',
            # 'CO2_gridded',

            'heat_event_frequency',
            'silt',
            # 'rooting_depth',



            # # 'heavy_rainfall_days',


        ]


        self.x_variable_list_CRU = [

            # 'cwdx80_05',
            #
            # 'sand',

            'rainfall_intensity',


            'rainfall_frenquency',

            'rainfall_seasonality_all_year',

            'detrended_sum_rainfall_CV',

            'CO2_interannual_rainfall_interaction',


                'heat_event_frenquency',


            ]
        self.x_variable_range_dict_global = {
            'CO2_ecosystem_year': [350, 410],
            'detrended_average_annual_tmax': [-10, 40],
            'detrended_sum_rainfall_growing_season_CV_ecosystem_year': [0, 70],

            'detrended_sum_rainfall_std': [0, 250],
            'detrended_sum_rainfall': [0, 1000],
            'CV_rainfall_interseasonal': [100, 600],
            'detrended_sum_rainfall_interannual_CV': [0, 70],


            'rainfall_seasonality': [0, 10],  # rainfall_seasonality


            'sum_rainfall': [0, 1500],
            'CO2_gridded': [350, 410],
            'CO2': [350, 410],
            'Aridity': [0, 1],

            'heat_event_frenquency_growing_season': [0, 6],




            'maxmum_dry_spell': [0, 200],  # maxmum_dry_spell
            'rainfall_frequency': [0, 200],  # rainfall_frequency
            'rainfall_intensity': [0, 5],  # rainfall_intensity
            'rainfall_seasonality_all_year': [0, 25],  #
            'heavy_rainfall_days': [0, 50],
            'T_sand': [20, 90],
            'rooting_depth': [0, 30],

        }

        self.x_variable_range_dict_global_CRU = {
            'nitrogen': [0, 500],
            'zroot_cwd80_05': [0, 25000],
            'cwdx80_05': [0, 1000],
            'cec': [0, 400],
            'sand': [0, 900],

            'CO2_interannual_rainfall_interaction': [0, 20000 ],

            'CO2': [350, 410],
            'sum_rainfall': [0, 1500],


            'dry_spell': [0, 20],

            'rainfall_intensity': [0, 25],
            'rainfall_frenquency': [0, 100],
    'rainfall_seasonality_all_year': [15, 80],

            'detrended_sum_rainfall_CV':[0,60],

    'heat_event_frenquency': [0, 3],

            'rainfall_intensity_CPC': [0, 20],
            'rainfall_frenquency_CPC': [0, 100],
            'rainfall_seasonality_all_year_CPC': [10, 80],
            'detrended_sum_rainfall_CV_CPC': [0, 60],

            'rainfall_intensity_MSWEP': [0, 20],
            'rainfall_frenquency_MSWEP': [0, 120],
            'rainfall_seasonality_all_year_MSWEP': [10, 80],
            'detrended_sum_rainfall_CV_MSWEP': [0, 100],

            'rainfall_intensity_ERA5': [0, 20],
            'rainfall_frenquency_ERA5': [0, 150],
            'rainfall_seasonality_all_year_ERA5': [10, 80],
            'detrended_sum_rainfall_CV_ERA5': [0, 100],

            'rainfall_intensity_1mm': [0, 25],

        'rainfall_frenquency_1mm': [0, 140],

        'rainfall_seasonality_all_year_1mm': [10, 80],

        'detrended_sum_rainfall_CV_1mm': [0, 60],

            'rainfall_intensity_5mm': [0, 25],

        'rainfall_frenquency_5mm': [0, 100],

        'rainfall_seasonality_all_year_5mm': [10, 80],

        'detrended_sum_rainfall_CV_5mm': [0, 100],

        }

    def show_colinear(self, ):
        dff = self.dff
        df = T.load_df(dff)
        vars_list = self.x_variable_list
        df = df[vars_list]
        ## add LAI4g_raw
        df['LAI4g_CV_growing_season'] = T.load_df(dff)['LAI4g_CV_growing_season']
        ## plot heat map to show the colinear variables
        import seaborn as sns
        plt.figure(figsize=(10, 10))
        ### x tick label rotate
        plt.xticks(rotation=45)

        sns.heatmap(df.corr(), annot=True, fmt=".2f")
        plt.show()


    def plot_hist(self, df):
        # T.print_head_n(df)
        # exit()
        x_variable_list = self.x_variable_list
        ## combine x and y
        all_list = copy.copy(x_variable_list)
        all_list.append(self.y_variable)
        # print(all_list)
        # exit()
        for var in all_list:
            vals = df[var].tolist()
            vals = np.array(vals)
            # vals[vals<-500] = np.nan
            # vals[vals>500] = np.nan
            # vals = vals[~np.isnan(vals)]
            plt.hist(vals, bins=100)
            plt.title(var)
            plt.show()
        exit()
        return df

    def valid_range_df(self, df):

        print('original len(df):', len(df))
        for var in self.x_variable_list_CRU:

            if not var in df.columns:
                print(var, 'not in df')
                continue
            min, max = self.x_variable_range_dict[var]
            df = df[(df[var] >= min) & (df[var] <= max)]
        print('filtered len(df):', len(df))
        return df

    def df_clean(self, df):
        # T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] >60]
        df = df[df['Aridity'] < 0.65]
        df=df[df['LC_max']<20]
        print(len(df))


        df = df[df['MODIS_LUCC'] != 12]

        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def df_clean_for_consistency(self, df):  ## df clean for three products consistency pixels
        # T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df=df[df['LC_max']<20]
        print(len(df))


        df = df[df['MODIS_LUCC'] != 12]

        # df = df[df['LAI4g_NDVI4g'].isin([1, 4])]
        # df = df[df['LAI4g_NDVI'].isin([1, 4])]
        # df = df[df['LAI4g_GIMMS_NDVI'].isin([1, 4])]
        print(len(df))




        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def check_spatial_plot(self):

        dff = self.dff
        df=T.load_df(dff)
        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        region_arr = DIC_and_TIF(pixelsize=.5).pix_dic_to_spatial_arr(unique_pix_list)
        plt.imshow(region_arr, cmap='jet', vmin=1, vmax=3,interpolation='nearest')
        plt.colorbar()
        plt.show()

    def pdp_shap(self):

        dff = self.dff
        outdir = join(self.this_class_png, 'pdp_shap_CV')

        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list_CRU

        y_variable = self.y_variable
        # plt.hist(T.load_df(dff)[y_variable].tolist(),bins=100)
        # plt.show()
        df = T.load_df(dff)
        # df = self.df_clean(df)
        df=self.df_clean_for_consistency(df)


        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic={}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='jet',interpolation='nearest')
        # plt.colorbar()
        # plt.show()



        T.print_head_n(df)
        print(len(df))
        T.print_head_n(df)
        print('-' * 50)
        ## text select df the first 1000

        # df = df[0:1000]
        # exit()
        # model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        # all_vars_vif = self.discard_vif_vars(df, x_variable_list)
        # all_vars_vif.append('CV_rainfall')
        # print('all_vars_vif:',all_vars_vif)
        # exit()
        # df = self.valid_range_df(df)
        all_vars = copy.copy(x_variable_list)


        all_vars.append(y_variable)  # add the y variable to the list
        all_vars.append('pix')


        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna(subset=x_variable_list, how='any')
        all_vars_df = all_vars_df.dropna(subset=self.y_variable, how='any')

        ## for plot use not training
        ## I want to add CO2 into new df but using all_vars_df to selected from df
        ## so that all_vars_df can be used for future ploting
        # all_vars_df_CO2 = copy.copy(all_vars_df)
        # merged = pd.merge(all_vars_df_CO2, df[["pix", "Aridity"]], on="pix", how="left")
        # T.save_df(merged, join(outdir, 'all_vars_df_aridity.df'))
        # exit()



        ######


        pix_list = all_vars_df['pix'].tolist()
        # print(len(pix_list));exit()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}
        #
        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()


        X = all_vars_df[x_variable_list]

        Y = all_vars_df[y_variable]
        train_data_X_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.X.df')
        train_data_y_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.y.df')

        # exit()

        ## save selected df for future ploting
        T.print_head_n(X)
        # X = X.dropna()
        # print(len(X));exit()



        model, y, y_pred = self.__train_model(X, Y)  # train a Random Forests model
        imp_dict_xgboost = {}
        for i in range(len(x_variable_list)):
            imp_dict_xgboost[x_variable_list[i]] = model.feature_importances_[i]
        #     plt.barh(x_variable_list[i], model.feature_importances_[i])
        # plt.show()
        sorted_imp = sorted(imp_dict_xgboost.items(), key=lambda x: x[1], reverse=True)

        x_ = []
        y_ = []
        for key, value in sorted_imp:
            x_.append(key)
            y_.append(value)
        print(x_)
        plt.figure()
        plt.bar(x_, y_)
        plt.xticks(rotation=45)
        # plt.tight_layout()
        plt.title('RF')
        plt.show()
        # exit()
        # plt.figure()

        ## random sample

        sample_indices = np.random.choice(X.shape[0], 2000, replace=False)
        X_sample = X.iloc[sample_indices]
        explainer = shap.TreeExplainer(model)


        # ### round R2

        # # x_variable_range_dict = self.x_variable_range_dict
        # y_base = explainer.expected_value
        # print('y_base', y_base)
        # print('y_mean', np.mean(y))
        # shap_values = explainer.shap_values(X)
        shap_values = explainer(X_sample)
        outf_shap = join(outdir, self.y_variable + '.shap')
        # ## how to resever X and Y before the shap
        #


        T.save_dict_to_binary(shap_values, outf_shap)
        ## save model

        T.save_dict_to_binary(model, join(outdir, self.y_variable + '.model'))
        # exit()
    def plot_relative_importance(self):  ## bar plot

        ## here plot relative importance of each variable
        x_variable_list = self.x_variable_list


        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': r'Fq$\mathrm{rainfall}$ (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': r'Fq$\mathrm{heat\ event}$ (events/year)',
                    'cwdx80_05': 'S0 (mm)',

                    'sand': 'Sand (g/kg)',

                    }

        inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')
        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)
        total_sum_list = []
        sum_abs_shap_dic = {}
        for i in range(shap_values.values.shape[1]):
            sum_abs_shap_dic[i]=(np.sum(np.abs(shap_values.values[:, i])))

            total_sum_list.append(np.sum(np.abs(shap_values.values[:, i])))
        total_sum_list=np.array(total_sum_list)
        total_sum=np.sum(total_sum_list, axis=0)
        relative_importance={}

        for key in sum_abs_shap_dic.keys():
            relative_importance[key]=sum_abs_shap_dic[key]/total_sum*100

        x_list = []
        y_list = []
        imp_dict = {}
        fig, ax = plt.subplots(figsize=(3, 1.5))
        for key in relative_importance.keys():
            x_list.append(key)
            y_list.append(relative_importance[key])
            imp_dict[key]=relative_importance[key]
        imp_dict_sort = sorted(imp_dict.items(), key=lambda x: x[1])
        x_list_sort = [x_variable_list[x[0]] for x in imp_dict_sort]
        ## use new name from dictionary
        x_list_sort = [name_dic[x] for x in x_list_sort]
        y_list_sort = [x[1] for x in imp_dict_sort]
        # pprint(imp_dict_sort);exit()
        # plt.barh(x_variable_list[::-1], y_list[::-1], color='grey', alpha=0.5)
        ## set color_list
        color_dic = {'Rainfall intensity (mm/events)': 'red',
                     'Fq$\mathrm{rainfall}$ (events/year)': 'blue',
                     'Rainfall seasonality (unitless)': 'green',
                     'CV$_{\mathrm{interannual\ rainfall}}$ (%)': 'orange',
                     'Fq$\mathrm{heat\ event}$ (events/year)': 'purple',
                     'S0 (mm)': 'black',

                     'Sand (g/kg)': 'grey',
                     }
        ax.bar(x_list_sort[::-1], y_list_sort[::-1], color=[color_dic[x] for x in x_list_sort[::-1]], alpha=0.5,edgecolor='black')
        ## fontsize

        print(x_list)

        plt.xticks(fontsize=8,rotation=90)
        plt.yticks(fontsize=8)
        plt.ylabel('Importance (%)', fontsize=8)
        ## add text R2=0.89 in (0.5, 0.5)
        plt.text(4, 20, 'R2=0.94', fontsize=8)
        ## save fig

        plt.savefig(join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.png'), dpi=300, bbox_inches='tight')
        plt.savefig(join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.pdf'), dpi=300, bbox_inches='tight')
        plt.close()





        # plt.tight_layout()
        #
        # plt.show()

        pass

    def plot_pdp_shap(self):
        x_variable_list = self.x_variable_list

        inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')
        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)

        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': r'Fq$_{\mathrm{ rainfall}}$ (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': r'Fq$_{\mathrm{ heat\ event}}$ (events/year)',
                    'cwdx80_05': 'S0 (mm)',

                    'sand': 'Sand (g/kg)',

                    }

        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)

            y_list.append(imp_dict[key])


        flag = 1
        centimeter_factor = 1 / 2.54
        plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))

        for x_var in x_list:

            shap_values_mat = shap_values[:, x_var]

            data_i = shap_values_mat.data
            value_i = shap_values_mat.values

            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # pprint(df_i);exit()
            df_i_random = df_i.sample(n=len(df_i) )
            df_i = df_i_random

            x_variable_range_dict = self.x_variable_range_dict
            ## redefine start, end
            start, end = self.x_variable_range_dict[x_var]



            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            df_i_copy = copy.copy(df_i)
            df_i_copy = df_i_copy[df_i_copy[x_var]>start]
            df_i_copy = df_i_copy[df_i_copy[x_var]<end]
            scatter_x_list = df_i_copy[x_var].tolist()
            scatter_y_list = df_i_copy['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)
            #     err,_,_ = self.uncertainty_err(SM)
            # print(df_i)
            # exit()
            plt.subplot(4, 3, flag)
            plt.scatter(scatter_x_list, scatter_y_list, alpha=0.2, c='gray', marker='.', s=1, zorder=-1)
            # print(data_i[0])
            # exit()
            # interp_model = interpolate.interp1d(x_mean_list, y_mean_list, kind='cubic')
            # y_interp = interp_model(x_mean_list)
            y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
            plt.plot(x_mean_list, y_mean_list, c='red', alpha=1)
            plt.ylabel(r'CV$_{\mathrm{LAI}}$ (%/yr)', fontsize=10)

            # exit()
            # plt.fill_between(x_mean_list, np.array(y_mean_list) - np.array(y_err_list), np.array(y_mean_list) + np.array(y_err_list), alpha=0.3,color='red')
            #### rename x_label remove

            plt.xlabel(name_dic[x_var], fontsize=10)

            flag += 1
            plt.ylim(-5, 5)


        plt.tight_layout()
        plt.show()
        # plt.savefig(outf,dpi=300)
        # plt.close()


    def plot_pdp_shap_density_cloud(self):
        x_variable_list = self.x_variable_list

        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': r'Fq$_{\mathrm{ rainfall}}$ (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': r'Fq$_{\mathrm{ heat\ event}}$ (events/year)',
                    'cwdx80_05': 'S0 (mm)',
                    'sand': 'Sand (g/kg)',

                    }
        inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')


        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)

        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)
            y_list.append(imp_dict[key])

        flag = 1
        centimeter_factor = 1 / 2.54
        # plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        fig, axs = plt.subplots(4, 2,
                                figsize=(20 * centimeter_factor, 18 * centimeter_factor))
        # print(axs);exit()

        axs = axs.flatten()

        for x_var in x_list:
            shap_values_mat = shap_values[:, x_var]
            data_i = shap_values_mat.data
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # pprint(df_i);exit()
            df_i_random = df_i.sample(n=len(df_i) )
            df_i = df_i_random

            ## redefine start, end
            start, end = self.x_variable_range_dict[x_var]

            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            df_i_copy = copy.copy(df_i)
            df_i_copy = df_i_copy[df_i_copy[x_var]>start]
            df_i_copy = df_i_copy[df_i_copy[x_var]<end]
            scatter_x_list = df_i_copy[x_var].tolist()
            scatter_y_list = df_i_copy['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)

            percentiles = [5, 95]
            ## datapoints percentile
            percentile_values = np.percentile(scatter_x_list, percentiles)
            print(percentile_values)

            # plt.subplot(4, 3, flag)
            ax = axs[flag]
            ax.vlines(percentile_values, -5, 5, color='gray', linestyle='--', alpha=1)

            # ax2 = ax.twiny()  # Create a twin x-axis
            # ax2.set_xlim(ax.get_xlim())  # Match the limits with the main axis
            # ax2.set_xticks(percentile_values)  # Set percentile values as ticks
            # ax2.set_xticklabels([f'{p}%' for p in percentiles])  # Label with percentiles


            KDE_plot().plot_scatter(scatter_x_list, scatter_y_list,ax=ax )

            y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
            ax.plot(x_mean_list, y_mean_list, c='red', alpha=1)

            # ax.set_title(name_dic[x_var], fontsize=12)
            ax.set_xlabel(name_dic[x_var], fontsize=10)
            ax.set_ylabel(r'CV$_{\mathrm{LAI}}$ (%/year)', fontsize=10)

            flag += 1
            ax.set_ylim(-5, 5)
        last_subplot = axs[0]

        last_subplot.set_frame_on(False)
        last_subplot.set_xticks([])
        last_subplot.set_yticks([])
        plt.tight_layout()



        plt.savefig(join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '_shap.png'), dpi=300, bbox_inches='tight')
        plt.savefig(join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '_shap.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

        # plt.tight_layout()
        # plt.show()
        # plt.savefig(outf,dpi=300)
        # plt.close()




    def plot_heatmap_ranking(self):
        ##  plot heatmap for the ranking of the x variables

        fdir_all = result_root+rf'\3mm\SHAP\\'

        x_variable_list = self.x_variable_list
        # x_variable_list=['rainfall_intensity','rainfall_frenquency','sand','detrended_sum_rainfall_CV','heat_event_frenquency', 'cwdx80_50','  rainfall_seasonality_all_year']


        dic_result = {'rainfall_intensity':0,
                        'rainfall_frenquency':1,
                        'sand':2,
                        'detrended_sum_rainfall_CV':3,
                        'heat_event_frenquency':5,
                        'cwdx80_05':4,
                        'rainfall_seasonality_all_year':6,}



        all_model_results_list = []
        model_list = [ 'LAI4g','CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai','LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',
                      'SDGVM_S2_lai',
                      'YIBs_S2_Monthly_lai']
        dic_name = {
            'LAI4g': 'Obs',
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
                    'ORCHIDEE_S2_lai': 'ORCHIDEE',
                    'SDGVM_S2_lai': 'SDGVM',
                    'YIBs_S2_Monthly_lai': 'YIBs',
                    'LPX-Bern_S2_lai': 'LPX-Bern',
                    }

        for model in model_list[::-1]:

            fdir = join(fdir_all, rf'RF_{model}_detrend_CV_')


            for fdir_ii in T.listdir(fdir):

                for f in T.listdir(join(fdir, fdir_ii)):

                    if not '.shap.pkl' in f:
                        continue
                    print(f)


                    inf_shap = join(fdir, fdir_ii, f)

                    shap_values = T.load_dict_from_binary(inf_shap)
                    print(shap_values)

                    total_sum_list = []
                    sum_abs_shap_dic = {}


                    for i in range(shap_values.values.shape[1]):

                        sum_abs_shap_dic[i]=(np.sum(np.abs(shap_values.values[:, i])))

                        total_sum_list.append(np.sum(np.abs(shap_values.values[:, i])))
                    total_sum_list=np.array(total_sum_list)
                    total_sum=np.sum(total_sum_list, axis=0)
                    relative_importance={}

                    for key in sum_abs_shap_dic.keys():
                        relative_importance[key]=sum_abs_shap_dic[key]/total_sum*100

                    x_list = []
                    y_list = []
                    imp_dict = {}
                    for key in relative_importance.keys():
                        x_list.append(key)
                        y_list.append(relative_importance[key])
                        imp_dict[key]=relative_importance[key]
                        ### sort by importance and the relative importance largest is 6 and smallest is 0

                    imp_dict_sort = sorted(imp_dict.items(), key=lambda x: x[1])


                    x_list_sort = [x_variable_list[x[0]] for x in imp_dict_sort]

                    x_list_sort_number = [dic_result[x] for x in x_list_sort[::-1]]

                    all_model_results_list.append(x_list_sort_number)
        all_model_results_arr = np.array(all_model_results_list)
        ## plot heatmap
        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': r'Fq$\mathrm{rainfall}$ (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': r'Fq$\mathrm{heat\ event}$ (events/year)',
                    'cwdx80_05': 'S0 (mm)',

                    'sand': 'Sand (g/kg)',

                    }
        x_label = ['Rainfall intensity', r'Fq$_{\mathrm{rainfall}}$', 'Sand',
                   'CV$_{\mathrm{interannual\ rainfall}}$', 'S0',
                   r'Fq$_{\mathrm{heat\ event}}$',  'Rainfall seasonality']
        cell_size = 0.5  # Desired size of each square box (in inches)
        fig_width = cell_size * len(x_list_sort)  # Total figure width
        fig_height = cell_size * len(model_list)  # Total figure height
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # sns.heatmap(all_model_results_arr, annot=True, fmt=".2f",
        #             cmap='GnBu_r', cbar=False, linewidths=0.5, linecolor='white', ax=ax, )
        sns.heatmap(all_model_results_arr,
                    cmap='turbo', cbar=False, linewidths=0.5, linecolor='white', ax=ax, )

        ax.set_yticks(np.arange(all_model_results_arr.shape[0]) + 0.5)  # Center labels
        # ax.set_yticklabels(model_list[::-1], rotation=0, va='center')
        ##get name from dic
        ax.set_yticklabels([dic_name[x] for x in model_list[::-1]], rotation=0, va='center')


        ax.set_xticks(np.arange(all_model_results_arr.shape[1]) + 0.5)  # Center labels
        ax.set_xticklabels(x_label, rotation=90, ha='right')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(result_root + rf'\3mm\SHAP\\Figure\\heatmap_ranking.pdf')
        plt.close()


        #
        #
        # plt.show()




            ## plot heatmap


        pass



    def feature_importances_shap_values(self, shap_values, features):
        '''
        Prints the feature importances based on SHAP values in an ordered way
        shap_values -> The SHAP values calculated from a shap.Explainer object
        features -> The name of the features, on the order presented to the explainer
        '''
        # Calculates the feature importance (mean absolute shap value) for each feature
        importances = []
        # for i in range(len(shap_values)):
        #     importances.append(np.abs(shap_values[i]).mean())
        for i in range(shap_values.values.shape[1]):
            importances.append(np.mean(np.abs(shap_values.values[:, i])))


        # Calculates the normalized version
        # importances_norm = softmax(importances)
        # Organize the importances and columns in a dictionary
        feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        # feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
        # Sorts the dictionary
        feature_importances = {k: v for k, v in
                               sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
        # feature_importances_norm = {k: v for k, v in
        #                             sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
        # Prints the feature importances
        # for k, v in feature_importances.items():
        #     print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

        return feature_importances
        # return feature_importances_norm

    def __select_extreme(self, df):
        df = df[df['T_max'] > 1]
        df = df[df['intensity'] < -2]
        return df

    def __train_model(self, X, y):
        from sklearn.model_selection import train_test_split
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.3)  # split the data into training and testing
        # model = RandomForestRegressor(n_estimators=50, random_state=42,n_jobs=-1,) # build a random forest model
        # rf.fit(X_train, y_train) # train the model
        # r2 = rf.score(X_test,y_test)
        # model = xgb.XGBRegressor(objective="reg:squarederror", booster='gbtree', n_estimators=100,
        #                        max_depth=7, eta=0.1, random_state=42, n_jobs=14,  )
        model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=12)

        model.fit(X_train, y_train)
        # model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X_test)
        # print(len(y_pred))
        # plt.scatter(y_test, y_pred)
        # plt.show()
        r = stats.pearsonr(y_test, y_pred)
        r2 = r[0] ** 2
        print('r2:', r2)
        # exit()

        return model, y, y_pred

    def __train_model_RF(self, X, y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, random_state=1, test_size=0.) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # build a random forest model
        rf.fit(X, y)  # train the model
        coef = rf.feature_importances_
        imp_dict = {}
        for i in range(len(coef)):
            imp_dict[self.x_variable_list[i]] = coef[i]

        return imp_dict

    def benchmark_model(self, y, y_pred):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        plt.scatter(y, y_pred)
        plt.plot([0.6, 1.2], [0.6, 1.2], color='r', linestyle='-', linewidth=2)
        plt.ylabel('Predicted', size=20)
        plt.xlabel('Actual', size=20)
        plt.xlim(0.6, 1.2)
        plt.ylim(0.6, 1.2)
        plt.show()

class SHAP_rainfall_seasonality():

    def __init__(self):


        self.dff = rf'E:\Project3\Result\3mm\CRU_JRA\Dataframe\\rainfall_seasonality_unpack\\rainfall_seasonality_unpack_new.df'
        self.variable_list_rt()

        ##----------------------------------

        self.y_variable = 'CV_intraannual_rainfall'
        self.this_class_png = result_root + rf'3mm\CRU_JRA\rainfall_seasonality_RF\\{self.y_variable}\\'
        T.mk_dir(self.this_class_png, force=True)

        ####################

        self.x_variable_list = self.x_variable_list
        self.x_variable_range_dict = self.x_variable_range_dict_global

        pass

    def run(self):
        # self.check_df_attributes()

        # self.check_variables_ranges()
        self.show_colinear()
        # self.pdp_shap()
        # self.plot_relative_importance()


        pass

    def check_df_attributes(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print(df.columns.tolist())
        print(len(df))
        # exit()
        pass

    def check_variables_ranges(self):

        dff = self.dff
        df = T.load_df(dff)
        df = self.df_clean(df)
        df = self.plot_hist(df)
        df = self.valid_range_df(df)
        # df = self.__select_extreme(df)
        # T.print_head_n(df)
        # exit()

        x_variable_list = self.x_variable_list
        print(len(x_variable_list))
        # exit()
        flag = 1
        for var in x_variable_list:
            print(flag, var)
            vals = df[var].tolist()
            plt.subplot(4, 4, flag)
            flag += 1
            plt.hist(vals, bins=100)
            plt.title(var)
        plt.tight_layout()
        plt.show()

        pass

    def variable_list_rt(self):
        self.x_variable_list = [

          'rainfall_frenquency',
            'rainfall_intensity',
            'heavy_rainfall_days',

            'dry_spell',
            'rainfall_seasonality_all_year',



        ]
        self.x_variable_range_dict_global = {
            'dry_spell': [0, 4],  # _dry_spell
            'rainfall_frenquency': [0, 125],  # rainfall_frequency
            'rainfall_intensity': [0, 30],  # rainfall_intensity
            'heavy_rainfall_days': [0, 80],


            'rainfall_seasonality_all_year': [0, 100],  # rainfall_seasonality_all_year




        }

    def show_colinear(self, ):
        dff = self.dff
        df = T.load_df(dff)
        df=self.df_clean(df)
        vars_list = self.x_variable_list
        df = df[vars_list]
        name_dic = {'rainfall_intensity': 'Rainfall intensity',
                    'rainfall_frenquency': r'Fq$\mathrm{rainfall}$',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality',

                    'dry_spell': 'Dry spell',
                    'heavy_rainfall_days': 'Heavy rainfall days',

                    'CV_intraannual_rainfall': r'CV $\mathrm{intraannual rainfall}$', }
        ## add LAI4g_raw
        df['CV_intraannual_rainfall'] = T.load_df(dff)['CV_intraannual_rainfall']
        ## plot heat map to show the colinear variables
        import seaborn as sns
        cell_size =1 # Desired size of each square box (in inches)
        vars_list.append('CV_intraannual_rainfall')
        fig_width = cell_size * len(vars_list)
        fig_height = cell_size * len(vars_list)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        x_labels = vars_list
        y_labels = vars_list
        vmin, vmax= -1,1

        sns.heatmap(df.corr(), annot=True, fmt=".2f",cbar_kws={'shrink': 0.5},cmap="RdBu", vmin=vmin, vmax=vmax,  ax=ax)
        ax.set_yticks(np.arange(len(x_labels)) + 0.5)
        # ax.set_yticklabels(model_list[::-1], rotation=0, va='center')
        ##get name from dic
        ax.set_yticklabels([name_dic[x] for x in vars_list], rotation=0, va='center')

        ax.set_xticks(np.arange(len(y_labels)) + 0.5)
        ax.set_xticklabels([name_dic[x] for x in vars_list],   rotation=45, ha='center')
        ax.set_aspect('equal')


        plt.tight_layout()


        plt.show()

    def discard_vif_vars(self, df, x_vars_list):
        ##################实时计算#####################
        vars_list_copy = copy.copy(x_vars_list)

        X = df[vars_list_copy]
        X = X.dropna()
        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif.round(1)
        selected_vif_list = []
        for i in range(len(vif)):
            feature = vif['features'][i]
            VIF_val = vif['VIF Factor'][i]
            if VIF_val < 5.:
                selected_vif_list.append(feature)
        return selected_vif_list

        pass

    def plot_hist(self, df):
        # T.print_head_n(df)
        # exit()
        x_variable_list = self.x_variable_list
        # print(x_variable_list)
        # exit()
        for var in x_variable_list:
            vals = df[var].tolist()
            vals = np.array(vals)
            # vals[vals<-500] = np.nan
            # vals[vals>500] = np.nan
            # vals = vals[~np.isnan(vals)]
            plt.hist(vals, bins=100)
            plt.title(var)
            plt.show()
        exit()
        return df

    def valid_range_df(self, df):

        print('original len(df):', len(df))
        for var in self.x_variable_list:

            if not var in df.columns:
                print(var, 'not in df')
                continue
            min, max = self.x_variable_range_dict[var]
            df = df[(df[var] >= min) & (df[var] <= max)]
        print('filtered len(df):', len(df))
        return df

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 20]

        df = df[df['MODIS_LUCC'] != 12]

        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        df = df[df['landcover_classfication'] != 'Cropland']

        return df



    def pdp_shap(self):

        dff = self.dff
        outdir = join(self.this_class_png, 'pdp_shap_rainfall')

        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list

        y_variable = self.y_variable
        # plt.hist(T.load_df(dff)[y_variable].tolist(),bins=100)
        # plt.show()
        df = T.load_df(dff)
        df = self.df_clean(df)
        # df = self.valid_range_df(df)

        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='jet',interpolation='nearest')
        # plt.colorbar()
        # plt.show()

        T.print_head_n(df)
        print(len(df))
        T.print_head_n(df)
        print('-' * 50)
        ## text select df the first 1000

        # df = df[0:1000]
        # exit()
        # model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        # all_vars_vif = self.discard_vif_vars(df, x_variable_list)
        # all_vars_vif.append('CV_rainfall')
        # print('all_vars_vif:',all_vars_vif)
        # exit()
        # df = self.valid_range_df(df)
        all_vars = copy.copy(x_variable_list)

        all_vars.append(y_variable)  # add the y variable to the list
        all_vars.append('pix')

        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna(subset=x_variable_list, how='any')
        all_vars_df = all_vars_df.dropna(subset=self.y_variable, how='any')

        ## for plot use not training
        ## I want to add CO2 into new df but using all_vars_df to selected from df
        ## so that all_vars_df can be used for future ploting
        # all_vars_df_CO2 = copy.copy(all_vars_df)
        # merged = pd.merge(all_vars_df_CO2, df[["pix", "Aridity"]], on="pix", how="left")
        # T.save_df(merged, join(outdir, 'all_vars_df_aridity.df'))
        # exit()

        ######

        pix_list = all_vars_df['pix'].tolist()
        # print(len(pix_list));exit()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}
        #
        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()

        X = all_vars_df[x_variable_list]

        Y = all_vars_df[y_variable]
        train_data_X_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.X.df')
        train_data_y_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.y.df')

        # exit()

        ## save selected df for future ploting
        T.print_head_n(X)
        # X = X.dropna()
        # print(len(X));exit()

        model, y, y_pred = self.__train_model(X, Y)  # train a Random Forests model
        imp_dict_xgboost = {}
        for i in range(len(x_variable_list)):
            imp_dict_xgboost[x_variable_list[i]] = model.feature_importances_[i]
        #     plt.barh(x_variable_list[i], model.feature_importances_[i])
        # plt.show()
        sorted_imp = sorted(imp_dict_xgboost.items(), key=lambda x: x[1], reverse=True)

        x_ = []
        y_ = []
        for key, value in sorted_imp:
            x_.append(key)
            y_.append(value)
        print(x_)
        plt.figure()
        plt.bar(x_, y_)
        plt.xticks(rotation=45)
        # plt.tight_layout()
        plt.title('RF')
        plt.show()
        # exit()
        # plt.figure()

        ## random sample

        sample_indices = np.random.choice(X.shape[0], 2000, replace=False)
        X_sample = X.iloc[sample_indices]
        explainer = shap.TreeExplainer(model)

        # ### round R2

        # # x_variable_range_dict = self.x_variable_range_dict
        # y_base = explainer.expected_value
        # print('y_base', y_base)
        # print('y_mean', np.mean(y))
        # shap_values = explainer.shap_values(X) ##### not use!!!
        shap_values = explainer(X_sample)
        outf_shap = join(outdir, self.y_variable + '.shap')
        # ## how to resever X and Y before the shap
        #

        T.save_dict_to_binary(shap_values, outf_shap)
        ## save model

        T.save_dict_to_binary(model, join(outdir, self.y_variable + '.model'))
        # exit()



    def plot_relative_importance(self):  ## bar plot

        ## here plot relative importance of each variable
        x_variable_list = self.x_variable_list


        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': r'Fq$\mathrm{rainfall}$ (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',

                    'dry_spell': 'Dry spell duration (days)',
                    'heavy_rainfall_days': 'Heavy rainfall days (days)',


                    }

        inf_shap = join(self.this_class_png, 'pdp_shap_rainfall', self.y_variable + '.shap.pkl')
        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)
        total_sum_list = []
        sum_abs_shap_dic = {}
        for i in range(shap_values.values.shape[1]):
            sum_abs_shap_dic[i]=(np.sum(np.abs(shap_values.values[:, i])))

            total_sum_list.append(np.sum(np.abs(shap_values.values[:, i])))
        total_sum_list=np.array(total_sum_list)
        total_sum=np.sum(total_sum_list, axis=0)
        relative_importance={}

        for key in sum_abs_shap_dic.keys():
            relative_importance[key]=sum_abs_shap_dic[key]/total_sum*100

        x_list = []
        y_list = []
        imp_dict = {}
        fig, ax = plt.subplots(figsize=(3, 1.5))
        for key in relative_importance.keys():
            x_list.append(key)
            y_list.append(relative_importance[key])
            imp_dict[key]=relative_importance[key]
        imp_dict_sort = sorted(imp_dict.items(), key=lambda x: x[1])
        x_list_sort = [x_variable_list[x[0]] for x in imp_dict_sort]
        ## use new name from dictionary
        x_list_sort = [name_dic[x] for x in x_list_sort]
        y_list_sort = [x[1] for x in imp_dict_sort]
        # pprint(imp_dict_sort);exit()
        # plt.barh(x_variable_list[::-1], y_list[::-1], color='grey', alpha=0.5)
        ## set color_list
        color_dic = {'Rainfall intensity (mm/events)': 'red',
                     'Fq$\mathrm{rainfall}$ (events/year)': 'blue',
                     'Rainfall seasonality (unitless)': 'green',
                     'Dry spell duration (days)': 'orange',
                     'Heavy rainfall days (days)': 'purple',

                     }
        ax.bar(x_list_sort[::-1], y_list_sort[::-1], color=[color_dic[x] for x in x_list_sort[::-1]], alpha=0.5,edgecolor='black')
        ## fontsize

        print(x_list)

        plt.xticks(fontsize=8,rotation=90)
        plt.yticks(fontsize=8)
        plt.ylabel('Importance (%)', fontsize=8)
        ## add text R2=0.89 in (0.5, 0.5)
        plt.text(3, 50, 'R2=0.98', fontsize=8)
        ## save fig

        plt.savefig(join(self.this_class_png, 'pdp_shap_rainfall', self.y_variable + '.png'), dpi=300, bbox_inches='tight')
        plt.savefig(join(self.this_class_png, 'pdp_shap_rainfall', self.y_variable + '.pdf'), dpi=300, bbox_inches='tight')
        plt.close()





        # plt.tight_layout()
        #
        # plt.show()

        pass



    def feature_importances_shap_values(self, shap_values, features):
        '''
        Prints the feature importances based on SHAP values in an ordered way
        shap_values -> The SHAP values calculated from a shap.Explainer object
        features -> The name of the features, on the order presented to the explainer
        '''
        # Calculates the feature importance (mean absolute shap value) for each feature
        importances = []
        for i in range(shap_values.values.shape[1]):
            importances.append(np.mean(np.abs(shap_values.values[:, i])))
        # Calculates the normalized version
        # importances_norm = softmax(importances)
        # Organize the importances and columns in a dictionary
        feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        # feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
        # Sorts the dictionary
        feature_importances = {k: v for k, v in
                               sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
        # feature_importances_norm = {k: v for k, v in
        #                             sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
        # Prints the feature importances
        # for k, v in feature_importances.items():
        #     print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

        return feature_importances
        # return feature_importances_norm

    def __select_extreme(self, df):
        df = df[df['T_max'] > 1]
        df = df[df['intensity'] < -2]
        return df

    def __train_model(self, X, y):
        from sklearn.model_selection import train_test_split
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.3)  # split the data into training and testing
        # model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=7,) # build a random forest model
        # rf.fit(X_train, y_train) # train the model
        # r2 = rf.score(X_test,y_test)
        model = xgb.XGBRegressor(objective="reg:squarederror", booster='gbtree', n_estimators=100,
                                 max_depth=13, eta=0.05, random_state=42, n_jobs=12)
        # model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=12)
        model.fit(X_train, y_train)
        # model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X_test)
        # print(len(y_pred))
        # plt.scatter(y_test, y_pred)
        # plt.show()
        r = stats.pearsonr(y_test, y_pred)
        r2 = r[0] ** 2
        print('r2:', r2)
        # exit()

        return model, y, y_pred

    def __train_model_RF(self, X, y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, random_state=1, test_size=0.) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=7)  # build a random forest model
        rf.fit(X, y)  # train the model
        coef = rf.feature_importances_
        imp_dict = {}
        for i in range(len(coef)):
            imp_dict[self.x_variable_list[i]] = coef[i]

        return imp_dict

    def benchmark_model(self, y, y_pred):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        plt.scatter(y, y_pred)
        plt.plot([0.6, 1.2], [0.6, 1.2], color='r', linestyle='-', linewidth=2)
        plt.ylabel('Predicted', size=20)
        plt.xlabel('Actual', size=20)
        plt.xlim(0.6, 1.2)
        plt.ylim(0.6, 1.2)
        plt.show()

class SHAP_CO2_interaction():


    def __init__(self):

        self.y_variable = 'LAI4g_detrend_CV'

        # self.this_class_png = results_root + 'ERA5\\SHAP\\png\\'
        self.threshold = '3mm'
        self.this_class_png = result_root + rf'\{self.threshold}\\SHAP\\RF_{self.y_variable}_CO2_interaction\\'
        # self.this_class_png = result_root + rf'\{self.threshold}\CRU_JRA\\SHAP\RF_{self.y_variable}\\'

        T.mk_dir(self.this_class_png, force=True)

        self.dff = rf'E:\Project3\Result\3mm\Dataframe\moving_window_CV\\moving_window_CV_new.df'
        # self.dff = result_root+rf'3mm\Dataframe\moving_window_CV\\moving_window_CV_1mm_3mm.df'
        self.variable_list_rt()
        self.variables_list = ['LAI4g', 'NDVI','CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',
                          'SDGVM_S2_lai',
                          'YIBs_S2_Monthly_lai']

        ##----------------------------------



        ####################

        self.x_variable_list = self.x_variable_list_CRU
        self.x_variable_range_dict = self.x_variable_range_dict_global_CRU

        pass

    def run(self):
        # self.check_df_attributes()

        # self.check_variables_ranges()
        # self.show_colinear()
        # self.check_spatial_plot()
        self.pdp_shap()
        # self.plot_relative_importance() ## use this
        # self.plot_pdp_shap()
        # self.plot_pdp_shap_density_cloud()  ## use this
        # self.plot_heatmap_ranking()
        # self.plot_pdp_shap_density_cloud_individual()


        # self.spatial_shapely()   ### spatial plot
        # self.variable_contributions()
        # self.max_contributions()
        # self.plot_pdp_shap_normalized()
        # self.pdp_shap_trend()

        pass

    def check_df_attributes(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print(df.columns.tolist())
        print(len(df))
        # exit()
        pass

    def check_variables_ranges(self):

        dff = self.dff
        df = T.load_df(dff)
        df = self.df_clean(df)

        df = self.plot_hist(df)
        df = self.valid_range_df(df)
        # df = self.__select_extreme(df)
        # T.print_head_n(df)
        # exit()

        x_variable_list = self.x_variable_list
        print(len(x_variable_list))
        # exit()
        flag = 1

        for var in x_variable_list:
            print(flag, var)
            vals = df[var].tolist()
            plt.subplot(4, 4, flag)
            flag += 1
            plt.hist(vals, bins=100)
            plt.title(var)
        plt.tight_layout()
        plt.show()


        pass

    def AIC_criteria(self):

        import statsmodels.api as sm

        pass
    def variable_list_rt(self):
        self.x_variable_list = [

            # 'detrended_average_annual_tmax',
            # 'heavy_rainfall_days',

            # 'rainfall_frequency',
            'rainfall_intensity',
            'maxmum_dry_spell',
            'rainfall_seasonality_all_year',
            # 'rainfall_seasonality',
            'detrended_sum_rainfall_interannual_CV',
            # 'CV_rainfall_interseasonal',
            # 'Aridity',
            # 'CO2_gridded',

            'heat_event_frequency',
            'silt',
            # 'rooting_depth',



            # # 'heavy_rainfall_days',


        ]


        self.x_variable_list_CRU = [

            # 'cwdx80_05',
            #
            # 'sand',

            'rainfall_intensity',


            'rainfall_frenquency',

            'rainfall_seasonality_all_year',

            'detrended_sum_rainfall_CV',

            'CO2_interannual_rainfall_interaction',


                'heat_event_frenquency',


            ]
        self.x_variable_range_dict_global = {
            'CO2_ecosystem_year': [350, 410],
            'detrended_average_annual_tmax': [-10, 40],
            'detrended_sum_rainfall_growing_season_CV_ecosystem_year': [0, 70],

            'detrended_sum_rainfall_std': [0, 250],
            'detrended_sum_rainfall': [0, 1000],
            'CV_rainfall_interseasonal': [100, 600],
            'detrended_sum_rainfall_interannual_CV': [0, 70],


            'rainfall_seasonality': [0, 10],  # rainfall_seasonality


            'sum_rainfall': [0, 1500],
            'CO2_gridded': [350, 410],
            'CO2': [350, 410],
            'Aridity': [0, 1],

            'heat_event_frenquency_growing_season': [0, 6],




            'maxmum_dry_spell': [0, 200],  # maxmum_dry_spell
            'rainfall_frequency': [0, 200],  # rainfall_frequency
            'rainfall_intensity': [0, 5],  # rainfall_intensity
            'rainfall_seasonality_all_year': [0, 25],  #
            'heavy_rainfall_days': [0, 50],
            'T_sand': [20, 90],
            'rooting_depth': [0, 30],

        }

        self.x_variable_range_dict_global_CRU = {
            'nitrogen': [0, 500],
            'zroot_cwd80_05': [0, 25000],
            'cwdx80_05': [0, 1000],
            'cec': [0, 400],
            'sand': [0, 900],

            'CO2_interannual_rainfall_interaction': [0, 400000 ],

            'CO2': [350, 410],
            'sum_rainfall': [0, 1500],


            'dry_spell': [0, 20],

            'rainfall_intensity': [0, 25],
            'rainfall_frenquency': [0, 100],
    'rainfall_seasonality_all_year': [15, 80],

            'detrended_sum_rainfall_CV':[0,60],

    'heat_event_frenquency': [0, 3],

            'rainfall_intensity_CPC': [0, 20],
            'rainfall_frenquency_CPC': [0, 100],
            'rainfall_seasonality_all_year_CPC': [10, 80],
            'detrended_sum_rainfall_CV_CPC': [0, 60],

            'rainfall_intensity_MSWEP': [0, 20],
            'rainfall_frenquency_MSWEP': [0, 120],
            'rainfall_seasonality_all_year_MSWEP': [10, 80],
            'detrended_sum_rainfall_CV_MSWEP': [0, 100],

            'rainfall_intensity_ERA5': [0, 20],
            'rainfall_frenquency_ERA5': [0, 150],
            'rainfall_seasonality_all_year_ERA5': [10, 80],
            'detrended_sum_rainfall_CV_ERA5': [0, 100],

            'rainfall_intensity_1mm': [0, 25],

        'rainfall_frenquency_1mm': [0, 140],

        'rainfall_seasonality_all_year_1mm': [10, 80],

        'detrended_sum_rainfall_CV_1mm': [0, 60],

            'rainfall_intensity_5mm': [0, 25],

        'rainfall_frenquency_5mm': [0, 100],

        'rainfall_seasonality_all_year_5mm': [10, 80],

        'detrended_sum_rainfall_CV_5mm': [0, 100],

        }

    def show_colinear(self, ):
        dff = self.dff
        df = T.load_df(dff)
        vars_list = self.x_variable_list
        df = df[vars_list]
        ## add LAI4g_raw
        df['LAI4g_CV_growing_season'] = T.load_df(dff)['LAI4g_CV_growing_season']
        ## plot heat map to show the colinear variables
        import seaborn as sns
        plt.figure(figsize=(10, 10))
        ### x tick label rotate
        plt.xticks(rotation=45)

        sns.heatmap(df.corr(), annot=True, fmt=".2f")
        plt.show()


    def plot_hist(self, df):
        # T.print_head_n(df)
        # exit()
        x_variable_list = self.x_variable_list
        ## combine x and y
        all_list = copy.copy(x_variable_list)
        all_list.append(self.y_variable)
        # print(all_list)
        # exit()
        for var in all_list:
            vals = df[var].tolist()
            vals = np.array(vals)
            # vals[vals<-500] = np.nan
            # vals[vals>500] = np.nan
            # vals = vals[~np.isnan(vals)]
            plt.hist(vals, bins=100)
            plt.title(var)
            plt.show()
        exit()
        return df

    def valid_range_df(self, df):

        print('original len(df):', len(df))
        for var in self.x_variable_list_CRU:

            if not var in df.columns:
                print(var, 'not in df')
                continue
            min, max = self.x_variable_range_dict[var]
            df = df[(df[var] >= min) & (df[var] <= max)]
        print('filtered len(df):', len(df))
        return df

    def df_clean(self, df):
        # T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] >60]
        df = df[df['Aridity'] < 0.65]
        df=df[df['LC_max']<20]
        print(len(df))


        df = df[df['MODIS_LUCC'] != 12]

        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def df_clean_for_consistency(self, df):  ## df clean for three products consistency pixels
        # T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df=df[df['LC_max']<20]
        print(len(df))


        df = df[df['MODIS_LUCC'] != 12]

        # df = df[df['LAI4g_NDVI4g'].isin([1, 4])]
        # df = df[df['LAI4g_NDVI'].isin([1, 4])]
        # df = df[df['LAI4g_GIMMS_NDVI'].isin([1, 4])]
        print(len(df))




        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def check_spatial_plot(self):

        dff = self.dff
        df=T.load_df(dff)
        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        region_arr = DIC_and_TIF(pixelsize=.5).pix_dic_to_spatial_arr(unique_pix_list)
        plt.imshow(region_arr, cmap='jet', vmin=1, vmax=3,interpolation='nearest')
        plt.colorbar()
        plt.show()

    def pdp_shap(self):

        dff = self.dff
        outdir = join(self.this_class_png, 'pdp_shap_CV')

        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list_CRU

        y_variable = self.y_variable
        # plt.hist(T.load_df(dff)[y_variable].tolist(),bins=100)
        # plt.show()
        df = T.load_df(dff)
        # df = self.df_clean(df)
        df=self.df_clean_for_consistency(df)


        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic={}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='jet',interpolation='nearest')
        # plt.colorbar()
        # plt.show()



        T.print_head_n(df)
        print(len(df))
        T.print_head_n(df)
        print('-' * 50)
        ## text select df the first 1000

        # df = df[0:1000]
        # exit()
        # model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        # all_vars_vif = self.discard_vif_vars(df, x_variable_list)
        # all_vars_vif.append('CV_rainfall')
        # print('all_vars_vif:',all_vars_vif)
        # exit()
        # df = self.valid_range_df(df)
        all_vars = copy.copy(x_variable_list)


        all_vars.append(y_variable)  # add the y variable to the list
        all_vars.append('pix')


        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna(subset=x_variable_list, how='any')
        all_vars_df = all_vars_df.dropna(subset=self.y_variable, how='any')

        ## for plot use not training
        ## I want to add CO2 into new df but using all_vars_df to selected from df
        ## so that all_vars_df can be used for future ploting
        # all_vars_df_CO2 = copy.copy(all_vars_df)
        # merged = pd.merge(all_vars_df_CO2, df[["pix", "Aridity"]], on="pix", how="left")
        # T.save_df(merged, join(outdir, 'all_vars_df_aridity.df'))
        # exit()



        ######


        pix_list = all_vars_df['pix'].tolist()
        # print(len(pix_list));exit()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}
        #
        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()


        X = all_vars_df[x_variable_list]

        Y = all_vars_df[y_variable]
        train_data_X_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.X.df')
        train_data_y_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.y.df')

        # exit()

        ## save selected df for future ploting
        T.print_head_n(X)
        # X = X.dropna()
        # print(len(X));exit()



        model, y, y_pred = self.__train_model(X, Y)  # train a Random Forests model
        imp_dict_xgboost = {}
        for i in range(len(x_variable_list)):
            imp_dict_xgboost[x_variable_list[i]] = model.feature_importances_[i]
        #     plt.barh(x_variable_list[i], model.feature_importances_[i])
        # plt.show()
        sorted_imp = sorted(imp_dict_xgboost.items(), key=lambda x: x[1], reverse=True)

        x_ = []
        y_ = []
        for key, value in sorted_imp:
            x_.append(key)
            y_.append(value)
        print(x_)
        plt.figure()
        plt.bar(x_, y_)
        plt.xticks(rotation=45)
        # plt.tight_layout()
        plt.title('RF')
        plt.show()
        # exit()
        # plt.figure()

        ## random sample

        sample_indices = np.random.choice(X.shape[0], 2000, replace=False)
        X_sample = X.iloc[sample_indices]
        explainer = shap.TreeExplainer(model)


        # ### round R2

        # # x_variable_range_dict = self.x_variable_range_dict
        # y_base = explainer.expected_value
        # print('y_base', y_base)
        # print('y_mean', np.mean(y))
        # shap_values = explainer.shap_values(X)
        shap_values = explainer(X_sample)
        outf_shap = join(outdir, self.y_variable + '.shap')
        # ## how to resever X and Y before the shap
        #


        T.save_dict_to_binary(shap_values, outf_shap)
        ## save model

        T.save_dict_to_binary(model, join(outdir, self.y_variable + '.model'))
        # exit()
    def plot_relative_importance(self):  ## bar plot

        ## here plot relative importance of each variable
        x_variable_list = self.x_variable_list


        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': r'Fq$\mathrm{rainfall}$ (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': r'Fq$\mathrm{heat\ event}$ (events/year)',
                    'cwdx80_05': 'S0 (mm)',

                    'sand': 'Sand (g/kg)',

                    }

        inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')
        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)
        total_sum_list = []
        sum_abs_shap_dic = {}
        for i in range(shap_values.values.shape[1]):
            sum_abs_shap_dic[i]=(np.sum(np.abs(shap_values.values[:, i])))

            total_sum_list.append(np.sum(np.abs(shap_values.values[:, i])))
        total_sum_list=np.array(total_sum_list)
        total_sum=np.sum(total_sum_list, axis=0)
        relative_importance={}

        for key in sum_abs_shap_dic.keys():
            relative_importance[key]=sum_abs_shap_dic[key]/total_sum*100

        x_list = []
        y_list = []
        imp_dict = {}
        fig, ax = plt.subplots(figsize=(3, 1.5))
        for key in relative_importance.keys():
            x_list.append(key)
            y_list.append(relative_importance[key])
            imp_dict[key]=relative_importance[key]
        imp_dict_sort = sorted(imp_dict.items(), key=lambda x: x[1])
        x_list_sort = [x_variable_list[x[0]] for x in imp_dict_sort]
        ## use new name from dictionary
        x_list_sort = [name_dic[x] for x in x_list_sort]
        y_list_sort = [x[1] for x in imp_dict_sort]
        # pprint(imp_dict_sort);exit()
        # plt.barh(x_variable_list[::-1], y_list[::-1], color='grey', alpha=0.5)
        ## set color_list
        color_dic = {'Rainfall intensity (mm/events)': 'red',
                     'Fq$\mathrm{rainfall}$ (events/year)': 'blue',
                     'Rainfall seasonality (unitless)': 'green',
                     'CV$_{\mathrm{interannual\ rainfall}}$ (%)': 'orange',
                     'Fq$\mathrm{heat\ event}$ (events/year)': 'purple',
                     'S0 (mm)': 'black',

                     'Sand (g/kg)': 'grey',
                     }
        ax.bar(x_list_sort[::-1], y_list_sort[::-1], color=[color_dic[x] for x in x_list_sort[::-1]], alpha=0.5,edgecolor='black')
        ## fontsize

        print(x_list)

        plt.xticks(fontsize=8,rotation=90)
        plt.yticks(fontsize=8)
        plt.ylabel('Importance (%)', fontsize=8)
        ## add text R2=0.89 in (0.5, 0.5)
        plt.text(4, 20, 'R2=0.94', fontsize=8)
        ## save fig

        plt.savefig(join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.png'), dpi=300, bbox_inches='tight')
        plt.savefig(join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.pdf'), dpi=300, bbox_inches='tight')
        plt.close()





        # plt.tight_layout()
        #
        # plt.show()

        pass

    def plot_pdp_shap(self):
        x_variable_list = self.x_variable_list

        inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')
        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)

        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': r'Fq$_{\mathrm{ rainfall}}$ (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': r'Fq$_{\mathrm{ heat\ event}}$ (events/year)',
                    'cwdx80_05': 'S0 (mm)',

                    'sand': 'Sand (g/kg)',

                    }

        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)

            y_list.append(imp_dict[key])


        flag = 1
        centimeter_factor = 1 / 2.54
        plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))

        for x_var in x_list:

            shap_values_mat = shap_values[:, x_var]

            data_i = shap_values_mat.data
            value_i = shap_values_mat.values

            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # pprint(df_i);exit()
            df_i_random = df_i.sample(n=len(df_i) )
            df_i = df_i_random

            x_variable_range_dict = self.x_variable_range_dict
            ## redefine start, end
            start, end = self.x_variable_range_dict[x_var]



            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            df_i_copy = copy.copy(df_i)
            df_i_copy = df_i_copy[df_i_copy[x_var]>start]
            df_i_copy = df_i_copy[df_i_copy[x_var]<end]
            scatter_x_list = df_i_copy[x_var].tolist()
            scatter_y_list = df_i_copy['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)
            #     err,_,_ = self.uncertainty_err(SM)
            # print(df_i)
            # exit()
            plt.subplot(4, 3, flag)
            plt.scatter(scatter_x_list, scatter_y_list, alpha=0.2, c='gray', marker='.', s=1, zorder=-1)
            # print(data_i[0])
            # exit()
            # interp_model = interpolate.interp1d(x_mean_list, y_mean_list, kind='cubic')
            # y_interp = interp_model(x_mean_list)
            y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
            plt.plot(x_mean_list, y_mean_list, c='red', alpha=1)
            plt.ylabel(r'CV$_{\mathrm{LAI}}$ (%/yr)', fontsize=10)

            # exit()
            # plt.fill_between(x_mean_list, np.array(y_mean_list) - np.array(y_err_list), np.array(y_mean_list) + np.array(y_err_list), alpha=0.3,color='red')
            #### rename x_label remove

            plt.xlabel(name_dic[x_var], fontsize=10)

            flag += 1
            plt.ylim(-5, 5)


        plt.tight_layout()
        plt.show()
        # plt.savefig(outf,dpi=300)
        # plt.close()


    def plot_pdp_shap_density_cloud(self):
        x_variable_list = self.x_variable_list

        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': r'Fq$_{\mathrm{ rainfall}}$ (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': r'Fq$_{\mathrm{ heat\ event}}$ (events/year)',
                    'cwdx80_05': 'S0 (mm)',
                    'sand': 'Sand (g/kg)',

                    }
        inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')


        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)

        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)
            y_list.append(imp_dict[key])

        flag = 1
        centimeter_factor = 1 / 2.54
        # plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        fig, axs = plt.subplots(4, 2,
                                figsize=(20 * centimeter_factor, 18 * centimeter_factor))
        # print(axs);exit()

        axs = axs.flatten()

        for x_var in x_list:
            shap_values_mat = shap_values[:, x_var]
            data_i = shap_values_mat.data
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # pprint(df_i);exit()
            df_i_random = df_i.sample(n=len(df_i) )
            df_i = df_i_random

            ## redefine start, end
            start, end = self.x_variable_range_dict[x_var]

            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            df_i_copy = copy.copy(df_i)
            df_i_copy = df_i_copy[df_i_copy[x_var]>start]
            df_i_copy = df_i_copy[df_i_copy[x_var]<end]
            scatter_x_list = df_i_copy[x_var].tolist()
            scatter_y_list = df_i_copy['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)

            percentiles = [5, 95]
            ## datapoints percentile
            percentile_values = np.percentile(scatter_x_list, percentiles)
            print(percentile_values)

            # plt.subplot(4, 3, flag)
            ax = axs[flag]
            ax.vlines(percentile_values, -5, 5, color='gray', linestyle='--', alpha=1)

            # ax2 = ax.twiny()  # Create a twin x-axis
            # ax2.set_xlim(ax.get_xlim())  # Match the limits with the main axis
            # ax2.set_xticks(percentile_values)  # Set percentile values as ticks
            # ax2.set_xticklabels([f'{p}%' for p in percentiles])  # Label with percentiles


            KDE_plot().plot_scatter(scatter_x_list, scatter_y_list,ax=ax )

            y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
            ax.plot(x_mean_list, y_mean_list, c='red', alpha=1)

            # ax.set_title(name_dic[x_var], fontsize=12)
            ax.set_xlabel(name_dic[x_var], fontsize=10)
            ax.set_ylabel(r'CV$_{\mathrm{LAI}}$ (%/year)', fontsize=10)

            flag += 1
            ax.set_ylim(-5, 5)
        last_subplot = axs[0]

        last_subplot.set_frame_on(False)
        last_subplot.set_xticks([])
        last_subplot.set_yticks([])
        plt.tight_layout()



        plt.savefig(join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '_shap.png'), dpi=300, bbox_inches='tight')
        plt.savefig(join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '_shap.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

        # plt.tight_layout()
        # plt.show()
        # plt.savefig(outf,dpi=300)
        # plt.close()




    def plot_heatmap_ranking(self):
        ##  plot heatmap for the ranking of the x variables

        fdir_all = result_root+rf'\3mm\SHAP\\'

        x_variable_list = self.x_variable_list
        # x_variable_list=['rainfall_intensity','rainfall_frenquency','sand','detrended_sum_rainfall_CV','heat_event_frenquency', 'cwdx80_50','  rainfall_seasonality_all_year']


        dic_result = {'rainfall_intensity':0,
                        'rainfall_frenquency':1,
                        'sand':2,
                        'detrended_sum_rainfall_CV':3,
                        'heat_event_frenquency':5,
                        'cwdx80_05':4,
                        'rainfall_seasonality_all_year':6,}



        all_model_results_list = []
        model_list = [ 'LAI4g','CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai','LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',
                      'SDGVM_S2_lai',
                      'YIBs_S2_Monthly_lai']
        dic_name = {
            'LAI4g': 'Obs',
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
                    'ORCHIDEE_S2_lai': 'ORCHIDEE',
                    'SDGVM_S2_lai': 'SDGVM',
                    'YIBs_S2_Monthly_lai': 'YIBs',
                    'LPX-Bern_S2_lai': 'LPX-Bern',
                    }

        for model in model_list[::-1]:

            fdir = join(fdir_all, rf'RF_{model}_detrend_CV_')


            for fdir_ii in T.listdir(fdir):

                for f in T.listdir(join(fdir, fdir_ii)):

                    if not '.shap.pkl' in f:
                        continue
                    print(f)


                    inf_shap = join(fdir, fdir_ii, f)

                    shap_values = T.load_dict_from_binary(inf_shap)
                    print(shap_values)

                    total_sum_list = []
                    sum_abs_shap_dic = {}


                    for i in range(shap_values.values.shape[1]):

                        sum_abs_shap_dic[i]=(np.sum(np.abs(shap_values.values[:, i])))

                        total_sum_list.append(np.sum(np.abs(shap_values.values[:, i])))
                    total_sum_list=np.array(total_sum_list)
                    total_sum=np.sum(total_sum_list, axis=0)
                    relative_importance={}

                    for key in sum_abs_shap_dic.keys():
                        relative_importance[key]=sum_abs_shap_dic[key]/total_sum*100

                    x_list = []
                    y_list = []
                    imp_dict = {}
                    for key in relative_importance.keys():
                        x_list.append(key)
                        y_list.append(relative_importance[key])
                        imp_dict[key]=relative_importance[key]
                        ### sort by importance and the relative importance largest is 6 and smallest is 0

                    imp_dict_sort = sorted(imp_dict.items(), key=lambda x: x[1])


                    x_list_sort = [x_variable_list[x[0]] for x in imp_dict_sort]

                    x_list_sort_number = [dic_result[x] for x in x_list_sort[::-1]]

                    all_model_results_list.append(x_list_sort_number)
        all_model_results_arr = np.array(all_model_results_list)
        ## plot heatmap
        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': r'Fq$\mathrm{rainfall}$ (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': r'Fq$\mathrm{heat\ event}$ (events/year)',
                    'cwdx80_05': 'S0 (mm)',

                    'sand': 'Sand (g/kg)',

                    }
        x_label = ['Rainfall intensity', r'Fq$_{\mathrm{rainfall}}$', 'Sand',
                   'CV$_{\mathrm{interannual\ rainfall}}$', 'S0',
                   r'Fq$_{\mathrm{heat\ event}}$',  'Rainfall seasonality']
        cell_size = 0.5  # Desired size of each square box (in inches)
        fig_width = cell_size * len(x_list_sort)  # Total figure width
        fig_height = cell_size * len(model_list)  # Total figure height
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # sns.heatmap(all_model_results_arr, annot=True, fmt=".2f",
        #             cmap='GnBu_r', cbar=False, linewidths=0.5, linecolor='white', ax=ax, )
        sns.heatmap(all_model_results_arr,
                    cmap='turbo', cbar=False, linewidths=0.5, linecolor='white', ax=ax, )

        ax.set_yticks(np.arange(all_model_results_arr.shape[0]) + 0.5)  # Center labels
        # ax.set_yticklabels(model_list[::-1], rotation=0, va='center')
        ##get name from dic
        ax.set_yticklabels([dic_name[x] for x in model_list[::-1]], rotation=0, va='center')


        ax.set_xticks(np.arange(all_model_results_arr.shape[1]) + 0.5)  # Center labels
        ax.set_xticklabels(x_label, rotation=90, ha='right')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(result_root + rf'\3mm\SHAP\\Figure\\heatmap_ranking.pdf')
        plt.close()


        #
        #
        # plt.show()




            ## plot heatmap


        pass



    def feature_importances_shap_values(self, shap_values, features):
        '''
        Prints the feature importances based on SHAP values in an ordered way
        shap_values -> The SHAP values calculated from a shap.Explainer object
        features -> The name of the features, on the order presented to the explainer
        '''
        # Calculates the feature importance (mean absolute shap value) for each feature
        importances = []
        # for i in range(len(shap_values)):
        #     importances.append(np.abs(shap_values[i]).mean())
        for i in range(shap_values.values.shape[1]):
            importances.append(np.mean(np.abs(shap_values.values[:, i])))


        # Calculates the normalized version
        # importances_norm = softmax(importances)
        # Organize the importances and columns in a dictionary
        feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        # feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
        # Sorts the dictionary
        feature_importances = {k: v for k, v in
                               sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
        # feature_importances_norm = {k: v for k, v in
        #                             sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
        # Prints the feature importances
        # for k, v in feature_importances.items():
        #     print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

        return feature_importances
        # return feature_importances_norm

    def __select_extreme(self, df):
        df = df[df['T_max'] > 1]
        df = df[df['intensity'] < -2]
        return df

    def __train_model(self, X, y):
        from sklearn.model_selection import train_test_split
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.3)  # split the data into training and testing
        # model = RandomForestRegressor(n_estimators=50, random_state=42,n_jobs=-1,) # build a random forest model
        # rf.fit(X_train, y_train) # train the model
        # r2 = rf.score(X_test,y_test)
        # model = xgb.XGBRegressor(objective="reg:squarederror", booster='gbtree', n_estimators=100,
        #                        max_depth=7, eta=0.1, random_state=42, n_jobs=14,  )
        model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=12)

        model.fit(X_train, y_train)
        # model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X_test)
        # print(len(y_pred))
        # plt.scatter(y_test, y_pred)
        # plt.show()
        r = stats.pearsonr(y_test, y_pred)
        r2 = r[0] ** 2
        print('r2:', r2)
        # exit()

        return model, y, y_pred

    def __train_model_RF(self, X, y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, random_state=1, test_size=0.) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # build a random forest model
        rf.fit(X, y)  # train the model
        coef = rf.feature_importances_
        imp_dict = {}
        for i in range(len(coef)):
            imp_dict[self.x_variable_list[i]] = coef[i]

        return imp_dict

    def benchmark_model(self, y, y_pred):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        plt.scatter(y, y_pred)
        plt.plot([0.6, 1.2], [0.6, 1.2], color='r', linestyle='-', linewidth=2)
        plt.ylabel('Predicted', size=20)
        plt.xlabel('Actual', size=20)
        plt.xlim(0.6, 1.2)
        plt.ylim(0.6, 1.2)
        plt.show()
class Bivariate_analysis():
    def __init__(self):
        pass
    def run (self):
        # self.plot_robinson()
        self.statistis()

    def plot_robinson(self):

        fdir_trend = result_root + rf'\3mm\bivariate_analysis\\'
        temp_root = result_root + rf'\3mm\bivariate_analysis\\temp\\'
        outdir = result_root + rf'\3mm\FIGURE\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):

            if not f.endswith('.tif'):
                continue

            fname = f.split('.')[0]


            fpath = rf"D:\Project3\Result\3mm\bivariate_analysis\heat_events_CVinternnaul.tif"
            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            # m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=8, is_discrete=True, colormap_n=8, cmap='RdYlBu_r',)

            color_list1 = [
                '#d01c8b',
                '#f3d705',
                '#3d46e8',
                '#4dac26',

            ]

            my_cmap2 = T.cmap_blend(color_list1, n_colors=5)
            # arr = ToRaster().raster2array(fpath)[0]
            # arr[arr<-999]=np.nan
            # plt.imshow(arr,cmap=my_cmap,vmin=1,vmax=8,interpolation='nearest')
            # plt.colorbar()
            # plt.show()
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=5, is_discrete=True, colormap_n=5,
                                                   cmap=my_cmap2, )

            plt.title(f'{fname}')
            # plt.show()
            outf = outdir + 'heat_events_CVinternnaul.pdf'
            plt.savefig(outf)
            plt.close()
            exit()

    def statistis(self):
        tiff=result_root+rf'\3mm\bivariate_analysis\heat_events_CVinternnaul.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        whole_number_pixels =array[array>-999].size

        dic={}
        for i in range(1,5):
            array[array==i]=i
            count=np.count_nonzero(array==i)
            percentage=count/whole_number_pixels*100
            dic[i]=percentage
        color_list = [
            '#d01c8b',
            '#f3d705',
            '#3d46e8',
            '#4dac26',


        ]

        print(dic)
        label_list=['Inc_Inc','Inc_Dec','Dec_Inc','Dec_Dec']
        plt.bar(dic.keys(),dic.values(),color=color_list)
        plt.xticks(range(1,5),label_list,size=12)
        plt.ylabel('Percentage (%)', size=12)
        # plt.show()
        plt.savefig(result_root + rf'\3mm\FIGURE\heatmap.pdf')

class Trend_CV():
    def __init__(self):
        pass
    def run(self):
        self.greening_CV()
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
    def greening_CV(self):
        dff=result_root+rf'\3mm\Dataframe\Trend_new\Trend.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        df=df[df['LAI4g_p_value']<0.05]
        df=df[df['LAI4g_detrend_CV_p_value']<0.05]
       ## percentile ==90th
        threshold_greening = df['LAI4g_trend'].quantile(0.90)
        threshold_browning=df['LAI4g_trend'].quantile(0.1)


        df_very_greening = df[df['LAI4g_trend'] > threshold_greening]
        df_very_browning= df[df['LAI4g_trend'] < threshold_browning]
        df_middle=df[(df['LAI4g_trend'] > threshold_browning) & (df['LAI4g_trend'] < threshold_greening)]
        ## check CV
        CV_greening_pixels=df_very_greening['LAI4g_detrend_CV_trend']
        CV_browning_pixels=df_very_browning['LAI4g_detrend_CV_trend']
        CV_middle_pixels=df_middle['LAI4g_detrend_CV_trend']

        plt.bar([1,2,3], [np.nanmean(CV_greening_pixels),np.nanmean(CV_browning_pixels),np.nanmean(CV_middle_pixels)],color=['green','red','blue'])
        plt.show()

class calculate_longterm_CV():
    def __init__(self):
        pass
    def run (self):
        self.calculate_long_term_CV()
        pass
    def calculate_long_term_CV(self):
        fpath=result_root+rf'3mm\extract_LAI4g_phenology_year\dryland\detrend_TRENDY\\LAI4g_detrend.npy'
        dic=T.load_npy(fpath)
        result_dic={}
        for pix in dic:
            vals=dic[pix]
            mean=np.nanmean(vals)
            std=np.nanstd(vals)
            CV=std/mean*100
            result_dic[pix]=CV
        outf=result_root+rf'3mm\extract_LAI4g_phenology_year\dryland\detrend_TRENDY\\long_term_LAI4g_detrend_CV.npy'
        np.save(outf,result_dic)

        pass

def main():
    # greening_analysis().run()
    # climate_variables().run()
    # TRENDY_trend().trend_analysis_plot()
    # TRENDY_CV().trend_analysis_plot()
    # PLOT_Climate_factors().run()
    calculate_longterm_CV().run()
    # SHAP_CV().run()
    # SHAP_rainfall_seasonality().run()
    # SHAP_CO2_interaction().run()
    # Bivariate_analysis().run()
    # Trend_CV().run()


    pass

if __name__ == '__main__':
    main()