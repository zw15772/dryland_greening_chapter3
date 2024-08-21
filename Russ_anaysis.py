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

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)


class PLOT:
    def __init__(self):
        pass

    def __del__(self):
        pass
    def run(self):
        # self.plot_anomaly_LAI_based_on_cluster()
        self.trend_analysis()


    def clean_df(self, df):
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['Aridity'] < 0.65]

        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 0]
        df = df[df['lat'] < 45]

        return df
    def plot_anomaly_LAI_based_on_cluster(self):  ##### plot for 4 clusters

        df = T.load_df(result_root + rf'\Dataframe\growing_season_original\\growing_season_original.df')
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
        variable_list=['GLEAM_SMroot',]
        # variable_list=['CRU','GPCC']

        # variable_list=['LAI4g']
        scenario='S2'
        # variable_list= ['LAI4g',f'CABLE-POP_{scenario}_lai', f'CLASSIC_{scenario}_lai', 'CLM5',  f'IBIS_{scenario}_lai', f'ISAM_{scenario}_lai',
        #      f'ISBA-CTRIP_{scenario}_lai', f'JSBACH_{scenario}_lai', f'JULES_{scenario}_lai',  f'LPJ-GUESS_{scenario}_lai', f'LPX-Bern_{scenario}_lai',
        #      f'ORCHIDEE_{scenario}_lai', f'SDGVM_{scenario}_lai', f'YIBs_{scenario}_Monthly_lai']
        region_unique = T.get_df_unique_val_list(df, 'landcover_classfication')
        print(region_unique)


        for continent in ['Grass', 'Evergreen', 'Shrub', 'Deciduous','Mixed','North_America']:
            ax = fig.add_subplot(2, 3, i)
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
                    val[val<-99]=np.nan

                    if not len(val) == 38:
                        ## add nan to the end of the list
                        for j in range(1):
                            val=np.append(val,np.nan)
                            val[val>9999]=np.nan
                            val[val < -9999] = np.nan
                        # print(val)
                        # print(len(val))


                    vals_nonnan.append(list(val))


                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)
                # vals_mean=vals_mean/pixel_area_sum

                val_std=np.nanstd(vals_mean,axis=0)

                # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
                plt.plot(vals_mean,label=product,color=color_list[variable_list.index(product)],linewidth=linewidth_list[variable_list.index(product)])
                # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])


                # plt.scatter(range(len(vals_mean)),vals_mean)
                # plt.text(0,vals_mean[0],product,fontsize=8)
            i=i+1

            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
            # plt.ylim(-0.2, 0.2)
            # plt.ylim(-1,1)


            # plt.xlabel('year')

            plt.ylabel(f'GLEAM_SMroot(m^3/m^3)')
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


        fdir = result_root+rf'\extract_GS\OBS_LAI_extend\\'
        outdir = result_root + rf'trend_analysis\\western_US\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'LAI4g' in f:
                continue


            outf=outdir+f.split('.')[0]


            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            resolution=0.25
            for pix in tqdm(dic):
                r,c=pix



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

            arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)

            p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            ## extract the region of interest

            # lon_start, lon_end = -125, -105
            # lat_start, lat_end = 0, 45
            # Calculate row (latitude) and column (longitude) indices
            lat_start_index = 180
            lat_end_index = 360

            lon_start_index = 220
            lon_end_index = 300

            arr_trend = arr_trend[lat_start_index:lat_end_index+1, lon_start_index:lon_end_index+1]
            p_value_arr = p_value_arr[lat_start_index:lat_end_index+1, lon_start_index:lon_end_index+1]
            ## plot point using p_value
            plt.imshow(arr_trend, cmap='jet', vmin=-0.005, vmax=0.005)
            plt.colorbar(label='LAI_trend (m2/m2/year)')

            significant_point = np.where(p_value_arr < 0.05)
            plt.scatter(significant_point[1], significant_point[0], s=0.5, c='black',label='p<0.05',marker='*',alpha=0.5)


            ## set y lim
            plt.ylim(lat_start_index, lat_end_index)
            plt.xlim(lon_start_index, lon_end_index)

            plt.title(f.split('.')[0])
            plt.show()
            #
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')
            #
            # np.save(outf + '_trend', arr_trend)
            # np.save(outf + '_p_value', p_value_arr)



def main():
    PLOT().run()

    pass

if __name__ == '__main__':
    main()