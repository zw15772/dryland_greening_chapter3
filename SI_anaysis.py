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


        # self.greening_products_basemap_whole_time()
        self.plot_spatial_histgram_period()
        pass
    def greening_products_basemap_whole_time(self):
        ## three products 3 time periods comparison
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np

        # Create synthetic data (replace with actual data as needed)
        fdir=result_root+rf'\3mm\relative_change_growing_season\trend_analysis\\'

        products = ['LAI4g', 'NDVI', 'NIRv']



        # fig, axes = plt.subplots(1, 3, figsize=(self.map_width*2, self.map_height))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.95, bottom=0.08, left=0.05, right=0.95)

        # Loop through products and periods to create subplots

        for i, product in enumerate(products):  # Use 'i' to index the subplot
            ax = axes[i]

            f_trend = fdir + rf'\{product}_trend.tif'
            f_p_value = fdir + rf'\{product}_p_value.tif'

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
            ax.set_title(f'{product}')
        cbar = fig.colorbar(ret, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08)
        cbar.set_label('Trend Value')


        plt.show()

    def greening_products_basemap_two_time_periods(self):
        ## three products 3 time periods comparison
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np

        # Create synthetic data (replace with actual data as needed)
        fdir=result_root+rf'\3mm\relative_change_growing_season\trend_analysis\\'

        products = ['LAI4g', 'NDVI', 'NIRv']
        period_list = ['all' ]

        fig, axes = plt.subplots(1, 3, figsize=(12, 9))
        fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.92, bottom=0.08, left=0.05, right=0.95)

        # Loop through products and periods to create subplots

        for period in period_list:
            for product in products:
                if period == 'all':
                    f_trend = fdir + rf'\{product}_trend.tif'
                    f_p_value = fdir + rf'\{product}_p_value.tif'
                else:
                    f_trend=fdir+rf'\{product}_{period}_trend.tif'
                    f_p_value=fdir+rf'\{product}_{period}_p_value.tif'
                array_trend, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_trend)
                array_p_value, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(f_p_value)
                array_trend[array_trend<-999]=np.nan


                ax = axes[products.index(product), period_list.index(period)]

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
        # period_list=['1983_2001','2002_2020']
        period_list = ['1983_2020', ]
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
            color_list = [ 'green', ]
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


            #

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