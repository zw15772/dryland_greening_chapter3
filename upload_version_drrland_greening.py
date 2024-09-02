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

class growth_rate:
    from scipy import stats, linalg
    def run(self):

        self.calculate_annual_growth_rate()

        # self.plot_growth_rate()
        # self.bar_plot()
    pass

    def calculate_annual_growth_rate(self):
        fdir=result_root + rf'\relative_change\OBS_LAI_extend\\'
        outdir=result_root + rf'growth_rate\\growth_rate_relative_change\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if f.split('.')[0] not in ['LAI4g','CO2','CRU','GPCC','VPD','tmax']:
                continue

            dict=np.load(fdir+f,allow_pickle=True).item()
            growth_rate_dic={}
            for pix in tqdm(dict):
                time_series=dict[pix]
                # print(len(time_series))
                growth_rate_time_series=np.zeros(len(time_series)-1)
                for i in range(len(time_series)-1):
                    if time_series[i]==0:
                        continue
                    growth_rate_time_series[i]=(time_series[i+1]-time_series[i])/time_series[i]*100
                growth_rate_dic[pix]=growth_rate_time_series
            np.save(outdir+f,growth_rate_dic)

        pass


    def plot_growth_rate_npy(self):  ##### plot double y axis
        fdir = r'D:\Project3\Result\growth_rate\growth_rate_trend\\'
        for f in T.listdir(fdir):
            if not 'LAI4g' in f:
                continue
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            vals_list =[]
            for pix in tqdm(spatial_dict):
                vals = spatial_dict[pix]
                vals_list.append(vals)
            vals_mean = np.nanmean(vals_list, axis=0)
            print(vals_mean)
            plt.scatter(range(len(vals_mean)),vals_mean)
            plt.show()
        pass

    def plot_growth_rate(self):  ##### plot double y axis

        df = T.load_df(result_root + rf'\growth_rate\DataFrame\\growth_rate_all_years.df')
        fdir = r'D:\Project3\Result\growth_rate\growth_rate_trend\\'
        print(len(df))
        T.print_head_n(df)
        # exit()

        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['Aridity'] < 0.65]

        df = df[df['MODIS_LUCC'] != 12]

        # print(len(df))
        # exit()
        #

        # create color list with one green and another 14 are grey

        color_list = ['grey'] * 16
        color_list[0] = 'green'
        # color_list[1] = 'red'
        # color_list[2]='blue'
        # color_list=['green','blue','red','orange','aqua','brown','cyan', 'black', 'yellow', 'purple', 'pink', 'grey', 'brown','lime','teal','magenta']
        linewidth_list = [1] * 16
        linewidth_list[0] = 3


        fig = plt.figure()
        fig.set_size_inches(10, 10)

        i = 1


        region_unique = T.get_df_unique_val_list(df, 'AI_classfication')
        print(region_unique)
        region_val_dict = {
            'Arid': 1,
            'Semi-Arid': 2,
            'Sub-Humid': 3,
        }
        region_val = []
        # for i,row in df.iterrows():
        #     region = row['AI_classfication']
        #     val = region_val_dict[region]
        #     region_val.append(val)
        # df['region_val'] = region_val
        # spatial_dict_region = T.df_to_spatial_dic(df, 'region_val')
        # region_arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_region)
        # plt.imshow(region_arr, cmap='jet', vmin=1, vmax=3,interpolation='nearest')
        # plt.colorbar()
        # plt.show()


        for continent in ['Arid', 'Semi-Arid', 'Sub-Humid', 'global']:
            ax = fig.add_subplot(2, 2, i)
            if continent == 'global':
                df_continent = df
            else:

                df_continent = df[df['AI_classfication'] == continent]

            vals_growth_rate_relative_change = df_continent['LAI4g_growth_rate_trend_method2'].tolist()
            # print(vals_growth_rate_relative_change);exit()
            vals_relative_change = df_continent['LAI4g_relative_change'].tolist()

            vals_growth_rate_list = []
            vals_relative_change_list = []
            for val_growth_rate in vals_growth_rate_relative_change:

                if type(val_growth_rate) == float:  ## only screening
                    continue
                if len(val_growth_rate) == 0:
                    continue
                val_growth_rate[val_growth_rate < -99] = np.nan
                val_growth_rate = np.array(val_growth_rate)*100
                # print(val_growth_rate)

                if not len(val_growth_rate) == 39:
                    ## add nan to the end of the list
                    for j in range(1):
                        val_growth_rate = np.append(val_growth_rate, np.nan)
                    # print(val)
                    # print(len(val))

                vals_growth_rate_list.append(list(val_growth_rate))

            for val_relative_change in vals_relative_change:
                if type(val_relative_change) == float:  ## only screening
                    continue
                if len(val_relative_change) == 0:
                    continue
                val_relative_change[val_relative_change < -99] = np.nan

                if not len(val_relative_change) == 39:
                    ## add nan to the end of the list
                    for j in range(1):
                        val_relative_change = np.append(val_relative_change, np.nan)
                    # print(val)
                    # print(len(val))
                vals_relative_change_list.append(list(val_relative_change))



            ###### calculate mean
            vals_mean_growth_rate = np.array(vals_growth_rate_list)
            ## remove inf
            vals_mean_growth_rate[vals_mean_growth_rate >9999]=np.nan
            vals_mean_growth_rate = np.nanmean(vals_mean_growth_rate, axis=0)
            vals_mean_relative_change = np.array(vals_relative_change_list)
            vals_mean_relative_change = np.nanmean(vals_mean_relative_change, axis=0)

            plt.plot(vals_mean_growth_rate, color='red', linewidth=2)
            ## add fiting line
            x = np.arange(1, 39)
            y = vals_mean_growth_rate[1:]
            result_i = stats.linregress(range(1, 39), y)


            plt.plot(x, result_i.intercept + result_i.slope * x, 'r--')



            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), rotation=45)


            ax.set_ylabel('Growth_rate (%)', color='r')
            ax.set_ylim(-10, 10)
            ax2 = ax.twinx()


            ax2.plot(vals_mean_relative_change, color='green', linewidth=2)
            ## set y axis color
            for tl in ax2.get_yticklabels():
                tl.set_color('g')
            for tl in ax.get_yticklabels():
                tl.set_color('r')

            x2 = np.arange(0, 39)
            y2 = vals_mean_relative_change
            result_i2 = stats.linregress(range(0, 39), y2)
            ax2.plot(x2, result_i2.intercept + result_i2.slope * x2, 'g--')


            ax2.set_ylabel('Relative_change (%)', color='g')
            ax2.set_ylim(-10, 10)
            ## add line when y=0
            plt.axhline(y=0, color='grey', linestyle='-',alpha=0.5)

            plt.title(f'{continent}')
            ## add slope and p_value
            ax.text(0.1, 0.9, f'slope:{result_i.slope:.3f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
            ax.text(0.1, 0.8, f'p_value:{result_i.pvalue:.3f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
            ax2.text(0.1, 0.7, f'slope:{result_i2.slope:.3f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='green')
            ax2.text(0.1, 0.6, f'p_value:{result_i2.pvalue:.3f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='green')
            ## add legend
            # ax.legend(['growth_rate', 'growth_rate_fit'], loc='upper left')
            # ax2.legend(['relative_change', 'relative_change_fit'], loc='upper right')

            i = i + 1
        plt.tight_layout()
        plt.show()



class trend_analysis():  ## figure 1
    def __init__(self):
        pass
    def run(self):
        # self.trend_analysis_spatial()
        self.robinson()
        # self.plt_histgram()
    def trend_analysis_spatial(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        fdir = rf'E:\Data\ERA5_daily\dict\rainfall_CV\\'
        outdir =rf'E:\Data\ERA5_daily\dict\rainfall_CV\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
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
                if r<120:
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

            arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            arr_trend_dryland = arr_trend * array_mask
            p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            p_value_arr_dryland = p_value_arr * array_mask


            # plt.imshow(arr_trend_dryland, cmap='jet', vmin=-0.01, vmax=0.01)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend_dryland, outf + '_trend.tif')
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr_dryland, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend_dryland)
            np.save(outf + '_p_value', p_value_arr_dryland)

    def robinson(self):
        fdir=rf'E:\Data\ERA5_daily\dict\\dry_spell\\'
        temp_root=result_root+r'Result_new\trend_anaysis\\robinson\\'
        out_pdf_fdir=result_root+r'Result_new\trend_anaysis\\robinson\\pdf\\'

        T.mk_dir(out_pdf_fdir,force=True)

        variable='rainfall_intensity'
        f_trend=fdir+variable+'_trend.tif'

        f_p_value=fdir+variable+'_p_value.tif'


        m,ret=Plot().plot_Robinson(f_trend, vmin=-1,vmax=1,is_discrete=True,colormap_n=7,)
        self.plot_Robinson_significance_scatter(m, f_p_value,temp_root,0.05,s=5)


        plt.title(f'{variable}_(day/yr)')
        # plt.title('r')
        # plt.show()
        ## save fig pdf
        #save pdf
        plt.savefig(out_pdf_fdir+variable+'.pdf', dpi=300, bbox_inches='tight')
        plt.close()


    def plot_Robinson_significance_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5, s=5,
                                        c='k', marker='.',
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




        pass

class extract_rainfall:
    ## 1) extract rainfall CV
    ## 2) extract rainfall total
    ## 3) extract rainfall frequency
    ## extract dry frequency
    ## 4) extract rainfall intensity
    ## 5) extract rainfall wet spell
    ## 6) extract rainfall dry spell
    def run(self):

        # self.extract_growing_season()
        # self.extract_rainfall_CV_total()
        self.extract_rainfall_CV()
        # self.rainfall_frequency()
        # self.dry_spell()

        # self.check_spatial_map()
        pass

    def define_quantile_threshold(self):
        # 1) extract extreme wet event based on 90th percentile and calculate frequency and total duration
        # 2) extract extreme dry event based on 10th percentile and calculate frequency and total duration
        # 3) extract wet event intensity
        ## 4) extract dry event intensity
        ## extract VPD and calculate the frequency of VPD>2kpa
        fdir=data_root+rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir = data_root+rf'\ERA5\ERA5_daily\dict\\define_quantile_threshold\\'
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            spatial_dic = np.load(fdir+f,allow_pickle=True).item()
            result_dic={}
            for pix in tqdm(spatial_dic):

                vals=spatial_dic[pix]
                vals_flatten=[item for sublist in vals for item in sublist]
                vals_flatten = np.array(vals_flatten)

                if T.is_all_nan(vals_flatten):
                    continue
                # plt.bar(range(len(vals_flatten)),vals_flatten)
                # plt.show()

                val_90th= np.percentile(vals_flatten,90)
                val_10th = np.percentile(vals_flatten, 10)
                val_95th = np.percentile(vals_flatten, 95)
                val_5th = np.percentile(vals_flatten, 5)
                val_99th = np.percentile(vals_flatten, 99)
                val_1st = np.percentile(vals_flatten, 1)
                dic_i={
                    '90th':val_90th,
                    '10th':val_10th,
                    '95th':val_95th,
                    '5th':val_5th,
                    '99th':val_99th,
                    '1st':val_1st
                }
                result_dic[pix]=dic_i
            outf=outdir+f
            np.save(outf,result_dic)

    def extract_extreme_rainfall_event(self):
        ENSO_type = 'La_nina'
        fdir_threshold = data_root+rf'ERA5\ERA5_daily\dict\define_quantile_threshold\\'
        fdir_yearly_all=rf'D:\Project3\Data\ERA5\ERA5_daily\dict\ENSO_year_extraction\\{ENSO_type}\\'
        outdir = data_root+rf'\ERA5\ERA5_daily\dict\\extreme_event_extraction\\{ENSO_type}\\'
        T.mk_dir(outdir,force=True)
        spatial_threshold_dic=T.load_npy_dir(fdir_threshold)
        result_dic = {}
        for f in T.listdir(fdir_yearly_all):
            spatial_dic = T.load_npy(fdir_yearly_all+f)
            for pix in tqdm(spatial_dic):
                if not pix in spatial_threshold_dic:
                    continue
                threshold_dic=spatial_threshold_dic[pix]

                val_90th = threshold_dic['90th']
                print(val_90th)
                val_10th = threshold_dic['10th']
                print(val_10th)
                EI_nino_dic= spatial_dic[pix]
                result_dic_i = {}
                for year_range in EI_nino_dic:

                    extreme_wet_event = []
                    extreme_dry_event = []
                    for val in EI_nino_dic[year_range]:
                        if val > val_90th:
                            extreme_wet_event.append(val)

                    ## calculate the frequency and average intensity of extreme wet event and extreme dry event
                    ## intensity
                    average_intensity_extreme_wet_event = np.nanmean(extreme_wet_event)

                    ## frequency
                    frequency_extreme_wet_event = len(extreme_wet_event)




                    result_dic_i[year_range] = {
                        f'{ENSO_type}_average_intensity_extreme_wet_event':average_intensity_extreme_wet_event,

                        f'{ENSO_type}_frequency_extreme_wet_event':frequency_extreme_wet_event,



                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def extract_rainfall_CV_total(self):  ## extract total and CV of rainfall
        fdir = data_root+rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir_CV = data_root+rf'\ERA5\ERA5_daily\dict\\rainfall_CV_total\\'

        T.mk_dir(outdir_CV,force=True)

        for f in T.listdir(fdir):

            spatial_dic = np.load(fdir+f,allow_pickle=True).item()
            result_dic = {}

            for pix in tqdm(spatial_dic):
                ### ui==if northern hemisphere
                r,c = pix
                vals = spatial_dic[pix]
                vals_flatten = np.array(vals).flatten()


                result_dic_i = {}

                for i in range(38):

                    if 120<r<=240:  # Northern hemisphere
                        ### April to October is growing season

                        vals_growing_season = vals_flatten[i*365+120:(i+1)*365+304]

                    elif 240<r<480:### whole year is growing season

                        vals_growing_season = vals_flatten[i*365:(i+1)*365]


                    else: ## october to April is growing season  Southern hemisphere
                        if i >=37:
                            break


                        vals_growing_season = vals_flatten[i*365+304:(i+1)*365+120]

                    vals_growing_season = np.array(vals_growing_season)
                    if T.is_all_nan(vals_growing_season):
                        continue
                    CV = np.std(vals_growing_season)/np.mean(vals_growing_season)
                    total = np.nansum(vals_growing_season)
                    result_dic_i[i] = {f'CV_rainfall':CV,
                                       }
                result_dic[pix] = result_dic_i

            outf = outdir_CV+f

            np.save(outf,result_dic)

    def extract_rainfall_CV(self):  ## extract CV of rainfall ready for multiregression
        fdir = rf'E:\Data\ERA5_daily\dict\precip_transform\\'
        outdir_CV = rf'E:\Data\\ERA5_daily\dict\\rainfall_CV\\'

        T.mk_dir(outdir_CV,force=True)



        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r,c = pix
            vals = spatial_dic[pix]
            vals_flatten = np.array(vals).flatten()


            CV_list = []

            for i in range(38):

                if 120<r<=240:  # Northern hemisphere
                    ### April to October is growing season

                    vals_growing_season = vals_flatten[i*365+120:(i+1)*365+304]

                elif 240<r<480:### whole year is growing season

                    vals_growing_season = vals_flatten[i*365:(i+1)*365]


                else: ## october to April is growing season  Southern hemisphere
                    if i >=37:
                        break


                    vals_growing_season = vals_flatten[i*365+304:(i+1)*365+120]

                vals_growing_season = np.array(vals_growing_season)
                if T.is_all_nan(vals_growing_season):
                    continue
                CV = np.std(vals_growing_season)/np.mean(vals_growing_season)
                CV_list.append(CV)
            result_dic[pix] = CV_list

        outf = outdir_CV+'CV_rainfall.npy'

        np.save(outf,result_dic)

    def extract_growing_season(self):
        fdir = data_root + rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir = data_root + rf'\ERA5\ERA5_daily\dict\\growing_season_extraction\\'
        T.mk_dir(outdir, force=True)

        for f in T.listdir(fdir):
            spatial_dic = np.load(fdir + f, allow_pickle=True).item()
            result_dic = {}
            for pix in tqdm(spatial_dic):

                spatial_dic = np.load(fdir + f, allow_pickle=True).item()

                r, c = pix
                vals = spatial_dic[pix]
                vals_flatten = np.array(vals).flatten()

                for i in range(38):

                    if 120 < r <= 240:  # Northern hemisphere
                        ### April to October is growing season

                        vals_growing_season = vals_flatten[i * 365 + 120:(i) * 365 + 304]

                    elif 240 < r < 480:  ### whole year is growing season

                        vals_growing_season = vals_flatten[i * 365:(i + 1) * 365]


                    else:  ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break

                        vals_growing_season = vals_flatten[i * 365 + 304:(i + 1) * 365 + 120]

                    vals_growing_season = np.array(vals_growing_season)
                    result_dic[pix] = vals_growing_season


            outf = outdir + f
            np.save(outf, result_dic)

    def rainfall_frequency(self):

        fdir =rf'E:\Data\\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir= rf'D:\Project3\Result\anomaly\OBS\\'
        threshold_f=rf'E:\Data\\\ERA5\ERA5_daily\dict\\define_quantile_threshold\\'
        dic_threshold = T.load_npy_dir(threshold_f)
        T.mk_dir(outdir,force=True)


        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}
        for pix in tqdm(spatial_dic):
            r,c=pix
            if not pix in dic_threshold:
                continue
            vals = spatial_dic[pix]
            vals_flatten = np.array(vals).flatten()
            threshold = dic_threshold[pix]
            threshold_wet = threshold['90th']




            result_dic_i = {}
            frequency_wet_list = []
            for i in range(38):
                if 120<r<=240:  # Northern hemisphere

                    vals_growing_season = vals_flatten[i * 365 + 120:(i) * 365 + 304]
                    ## calculate days >threshold



                elif 240 < r < 480:  ### whole year is growing season

                    vals_growing_season = vals_flatten[i * 365:(i + 1) * 365]

                else:  ## october to April is growing season  Southern hemisphere
                    if i >= 37:
                        break

                    vals_growing_season = vals_flatten[i * 365 + 304:(i + 1) * 365 + 120]
                vals_growing_season = np.array(vals_growing_season)
                if T.is_all_nan(vals_growing_season):
                    continue
                frequency_wet = len(np.where(vals_growing_season > threshold_wet)[0])
                frequency_wet_list.append(frequency_wet)


            result_dic[pix] = frequency_wet_list
            outf = outdir + 'wet_frequency_90th.npy'
            np.save(outf, result_dic)

    def dry_spell(self):

        fdir = data_root + rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir = data_root + rf'\ERA5\ERA5_daily\dict\\dry_spell\\'
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):

            spatial_dic = np.load(fdir + f, allow_pickle=True).item()

            result_dic = {}
            for pix in tqdm(spatial_dic):
                r,c=pix

                vals = spatial_dic[pix]
                vals_flatten = np.array(vals).flatten()

                result_dic_i = {}

                for i in range(38):
                    if 120<r<=240:  # Northern hemisphere

                        vals_growing_season = vals_flatten[i * 365 + 120:(i) * 365 + 304]
                    elif 240 < r < 480:  ### whole year is growing season

                        vals_growing_season = vals_flatten[i * 365:(i + 1) * 365]
                    else:  ## october to April is growing season  Southern hemisphere
                        if i >= 37:
                            break
                        vals_growing_season = vals_flatten[i * 365 + 304:(i + 1) * 365 + 120]
                    vals_growing_season = np.array(vals_growing_season)
                    vals_wet = vals_growing_season.copy()

                    vals_wet[vals_wet >= 1] = np.nan

                    dry_index = np.where(~np.isnan(vals_wet))
                    if len(dry_index[0]) == 0:
                        continue
                    dry_index = np.array(dry_index)
                    dry_index = dry_index.flatten()
                    dry_index_groups = T.group_consecutive_vals(dry_index)

                    # plt.bar(range(len(vals_growing_season)), vals_growing_season)
                    # plt.bar(range(len(vals_growing_season)), vals_wet)
                    # print(dry_index_groups)
                    # plt.show()
                    ## calcuate average wet spell
                    dry_spell = []
                    for group in dry_index_groups:
                        dry_spell.append(len(group))
                    dry_spell = np.array(dry_spell)

                    average_wet_spell = np.nanmean(dry_spell)
                    maxmum_wet_spell = np.nanmax(dry_spell)
                    result_dic_i[i] = {
                        f'average_dry_spell': average_wet_spell,
                        f'maximum_dry_spell': maxmum_wet_spell
                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def peak_rainfall_timing(self):  ## Weighted Mean of the Peak Rainfall Timing
        from scipy.ndimage import gaussian_filter1d
        time = np.arange(0, 365)
        fdir = data_root + rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir = data_root + rf'\ERA5\ERA5_daily\dict\\peak_rainfall_timing\\'
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):

            spatial_dic = np.load(fdir + f, allow_pickle=True).item()

            result_dic = {}
            for pix in tqdm(spatial_dic):
                r,c=pix

                vals = spatial_dic[pix]
                vals_flatten = np.array(vals).flatten()

                result_dic_i = {}

                for i in range(38):
                    if 120<r<=240:  # Northern hemisphere

                        vals_growing_season = vals_flatten[i * 365 + 120:(i) * 365 + 304]
                    elif 240 < r < 480:  ### whole year is growing season

                        vals_growing_season = vals_flatten[i * 365:(i + 1) * 365]
                    else:  ## october to April is growing season  Southern hemisphere
                        if i >= 37:
                            break
                        vals_growing_season = vals_flatten[i * 365 + 304:(i + 1) * 365 + 120]
                    vals_growing_season = np.array(vals_growing_season)

                    if T.is_all_nan(vals_growing_season):
                            continue
                        ## smooth rainfall
                    smoothed_rainfall = gaussian_filter1d(vals_growing_season, sigma=5)
                    peaks = (np.diff(np.sign(np.diff(smoothed_rainfall))) < 0).nonzero()[0] + 1

                    peak_timing = []
                    for peak in peaks:
                        # Consider a window around each peak (e.g., +/- 5 days)
                        window = 5
                        start = max(0, peak - window)
                        end = min(len(time), peak + window + 1)

                        peak_time = self.weighted_mean_timing(time[start:end], smoothed_rainfall[start:end])
                        peak_timing.append(peak_time)

                    ## plot
                        plt.figure(figsize=(10, 6))
                        plt.plot(time, vals_growing_season, label='Original Rainfall')
                        plt.plot(time, smoothed_rainfall, label='Smoothed Rainfall', linewidth=2)
                        plt.scatter(peaks, smoothed_rainfall[peaks], color='red', label='Identified Peaks')
                        plt.scatter(peak_timing, smoothed_rainfall[np.round(peak_timing).astype(int)], color='green',
                                    label='Weighted Mean Timing')

                        plt.xlabel('Day of Year')
                        plt.ylabel('Rainfall')
                        plt.legend()
                        plt.title('Rainfall Peak Timing Using Weighted Mean')
                        plt.show()



                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def weighted_mean_timing(time, rainfall):
        weighted_sum = np.sum(time * rainfall)
        total_weight = np.sum(rainfall)
        return weighted_sum / total_weight



    pass
    def check_spatial_map(self):
        fdir = data_root + rf'\ERA5\ERA5_daily\dict\\dry_spell\\'
        spatial_dic= T.load_npy_dir(fdir)
        key_list = ['average_dry_spell','maximum_dry_spell']

        for key in key_list:
            spatial_dict_num = {}
            spatial_dict_mean = {}

            for pix in spatial_dic:

                annual_dict = spatial_dic[pix]
                if len(annual_dict)==0:
                    continue

                valid_year = 0
                vals_list = []
                for year in annual_dict:
                    dict_i = annual_dict[year]
                    if not key in dict_i:
                        continue
                    val = dict_i[key]
                    vals_list.append(val)

                    valid_year+=1
                vals_mean = np.nanmean(vals_list)
                spatial_dict_num[pix] = valid_year
                spatial_dict_mean[pix] = vals_mean

            arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_mean)
            plt.figure()
            plt.imshow(arr,interpolation='nearest')
            plt.title(key)
            plt.colorbar()
        plt.show()



        #     spatial_dict_test[pix] = np.nanmean(vals['average_dry_spell'])
        # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_test)
        # plt.imshow(arr,interpolation='nearest')
        # # plt.title(key)
        # plt.show()




        pass

class bivariate_analysis():
    def __init__(self):
        pass
    def run(self):
        self.bivariate_plot()

        pass
    def bivariate_plot(self):
        result_root = rf'D:\Project3\Result\\'
        # print(result_root)

        import xymap
        tif_long_term= result_root + rf'multi_regression_moving_window\window15_anomaly_GPCC\trend_analysis\\100mm_unit\\GPCC_LAI4g_trend.tif'
        tif_window=result_root + rf'trend_analysis\relative_change\OBS_extend\\GPCC_trend.tif'
        # print(isfile(tif_CRU_trend))
        # print(isfile(tif_CRU_CV))
        # exit()
        outtif=result_root + rf'bivariate_analysis\\sensitivity_trend_GPCC_trend.tif'
        T.mk_dir(result_root + rf'bivariate_analysis\\')
        tif1=tif_long_term
        tif2=   tif_window

        tif1_label='Trends in LAI sensitivity to precipitation (%/100mm/year)'
        tif2_label='Precip trend (%/year)'
        min1=-5
        max1=5
        min2=-1
        max2=1
        outf=outtif
        upper_left_color = [143, 196, 34],  #
        upper_right_color = [156, 65, 148],  #
        lower_left_color = [29, 46, 97],  #
        lower_right_color = [238, 233, 57],  #
        center_color = [240, 240, 240],  #
        # print(xymap.Bivariate_plot_1().upper_left_color)
        # xymap.Bivariate_plot().plot_bivariate_map(tif1, tif2, tif1_label, tif2_label, min1, max1, min2, max2, outf)
        Biv = xymap.Bivariate_plot_1(upper_left_color = [143, 196, 34],
                                      upper_right_color = [156, 65, 148],
                                      lower_left_color = [29, 46, 97],
                                      lower_right_color = [238, 233, 57],
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


class PLOT_dataframe:
    def __init__(self):
        pass

    def plot_time_series(self):
        pass

    def plt_histgram(self):
        ## plot the histogram of spatial distribution of LAI
        dff = rf'D:\Project3\Result\Dataframe\relative_changes\\relative_changes.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        print(len(df))

        vals = df['LAI4g_trend'].to_list()
        ## drop nan

        vals = np.array(vals)

        # df.dropna(subset=['CRU_trend'],inplace=True)
        vals[vals > 50] = np.nan
        vals[vals < -50] = np.nan
        vals = vals[~np.isnan(vals)]

        # plt.hist(vals, bins=20, alpha=0.5, label='Positive', color='green',rwidth=0.9)
        plt.hist(vals, bins=21, alpha=0.5, color='green', range=(-2, 2))
        ## plt probability density function

        # plt.xlabel('Precipitation')
        # plt.xlim(-2,2)
        # plt.ylabel('Count')
        # plt.title('Histogram of Precipitation')
        #
        plt.show()

        pass

    def plot_LAItrend_vs_LAICV(self):  ## LAI trend vs LAI CV trend for corrlation>0 and corrlation<0
        dff=rf'D:\Project3\Result\Dataframe\relative_changes\\relative_changes.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        T.print_head_n(df)


        threhold_list = [ -1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1]
        threhold_list=np.linspace(-1,1,21)


        correlation_list=['positive_R','negative_R']
        result_dict = {}


        for corr in correlation_list:
            CV_list_mean = []
            CV_list_std = []
            if corr == 'positive_R':
                df_temp = df[df['LAI4g_CV_LAI4g_trend_R'] > 0]
                df_temp = df_temp[df_temp['LAI4g_CV_LAI4g_trend_pvalue'] < 0.05]
            elif corr == 'negative_R':
                df_temp = df[df['LAI4g_CV_LAI4g_trend_R'] < 0]
                df_temp = df_temp[df_temp['LAI4g_CV_LAI4g_trend_pvalue'] < 0.05]


            for i in range(len(threhold_list) - 1):
                threhold1 = threhold_list[i]
                threhold2 = threhold_list[i + 1]

                ## extract values in the threhold


                df_ii = df_temp[(df_temp['LAI4g_trend'] > threhold1) & (df_temp['LAI4g_trend'] < threhold2)]

                CV_val=df_ii['LAI4g_CV_trend'].to_list()


                CV_list_mean.append(np.mean(CV_val))
                CV_list_std.append(np.std(CV_val))

            result_dict[corr] = [CV_list_mean,CV_list_std]

        print(result_dict)

        ## plot
        color_list = ['green','red']
        for i in range(len(correlation_list)):
            corr = correlation_list[i]
            CV_list_mean = result_dict[corr][0]
            CV_list_std = result_dict[corr][1]

            plt.errorbar(threhold_list[1:], CV_list_mean, yerr=CV_list_std, label=corr, color=color_list[i])




        plt.legend()
        plt.xlabel('LAI trend %')
        plt.ylabel('LAI CV trend %')
        plt.show()


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




def main():
    # growth_rate().run()

    trend_analysis().run()

    # PLOT_dataframe().plot_LAItrend_vs_LAICV()


    pass

if __name__ == '__main__':
    main()