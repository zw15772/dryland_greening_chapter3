# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg
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

class Data_processing:

    def __init__(self):

        pass

    def run(self):
        # self.nc_to_tif_time_series_fast()

        # self.resample()
        # self.scale()
        # self.extract_dryland_tiff()
        # self.generate_nan_map()


        # self.tif_to_dic()
        self.extract_phenology_fire_mean()
        # self.interpolation()



        pass



    def nc_to_tif_time_series_fast(self):

        fdir_all=rf'D:\Project3\Data\Fire\\nc\\'
        outdir=rf'D:\Project3\Data\Fire\\\\TIFF\\'
        Tools().mk_dir(outdir,force=True)
        for fdir in tqdm(os.listdir(fdir_all)):
            if 'ZIP' in fdir:
                continue
            for f in os.listdir(fdir_all+fdir):
                if not f.endswith('.nc'):
                    continue

                outdir_name = f.split('-')[0]
                # print(outdir_name);exit()

                yearlist = list(range(1982, 2021))
                fpath = join(fdir_all+fdir,f)
                nc_in = xarray.open_dataset(fpath)
                print(nc_in)
                time_bnds = nc_in['time_bnds']
                for t in range(len(time_bnds)):
                    date = time_bnds[t]['time']
                    date = pd.to_datetime(date.values)
                    date_str = date.strftime('%Y%m%d')
                    date_str = date_str.split()[0]
                    outf = join(outdir, f'{date_str}.tif')
                    array = nc_in['burned_area'][t]
                    array = np.array(array)
                    array[array < 0] = np.nan
                    longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.25, -0.25
                    ToRaster().array2raster(outf, longitude_start, latitude_start,
                                            pixelWidth, pixelHeight, array, ndv=-999999)
                    # exit()


                # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
                try:
                    self.nc_to_tif_template(fdir+f, var_name='burned_area', outdir=outdir, yearlist=yearlist)
                except Exception as e:
                    print(e)
                    continue
    def nc_to_tif_template(self, fname, var_name, outdir, yearlist):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())
            time=ncin.variables['time'][:]

        except:
            raise UserWarning('File not supported: ' + fname)
        # lon,lat = np.nan,np.nan
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = basetime_str.split('since')[0]
        basetime_unit = basetime_unit.strip()
        print(basetime_unit)
        print(basetime_str)
        if basetime_unit == 'days':
            timedelta_unit = 'days'
        elif basetime_unit == 'years':
            timedelta_unit = 'years'
        elif basetime_unit == 'month':
            timedelta_unit = 'month'
        elif basetime_unit == 'months':
            timedelta_unit = 'month'
        elif basetime_unit == 'seconds':
            timedelta_unit = 'seconds'
        elif basetime_unit == 'hours':
            timedelta_unit = 'hours'
        else:
            raise Exception('basetime unit not supported')
        basetime = basetime_str.strip(f'{timedelta_unit} since ')
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
                    except:
                        try:
                            basetime = datetime.datetime.strptime(basetime, '%Y-%m')
                        except:
                            try:
                                basetime_ = basetime.split('T')[0]
                                # print(basetime_)
                                basetime = datetime.datetime.strptime(basetime_, '%Y-%m-%d')
                                # print(basetime)
                            except:

                                raise UserWarning('basetime format not supported')
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in tqdm(range(len(time))):
            if basetime_unit == 'days':
                date = basetime + datetime.timedelta(days=int(time[time_i]))
            elif basetime_unit == 'years':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                date2 = f'{int(base_year + time[time_i])}-01-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'month' or basetime_unit == 'months':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                base_month = basetime.month
                date2 = f'{int(base_year + time[time_i] // 12)}-{int(base_month + time[time_i] % 12)}-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'seconds':
                date = basetime + datetime.timedelta(seconds=int(time[time_i]))
            elif basetime_unit == 'hours':
                date = basetime + datetime.timedelta(hours=int(time[time_i]))
            else:
                raise Exception('basetime unit not supported')
            time_str = time[time_i]
            mon = date.month
            year = date.year
            if year not in yearlist:
                continue
            day = date.day
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = []
            lat_list = []
            value_list = []
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    lon_i = xx[i][j]
                    if lon_i > 180:
                        lon_i -= 360
                    lat_i = yy[i][j]
                    value_i = arr[i][j]
                    lon_list.append(lon_i)
                    lat_list.append(lat_i)
                    value_list.append(value_i)
            DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list, outpath)
    # def unify_tif(self):
    #     testf=data_root+r'monthly_data\\Trendy_TIFF\CLASSIC_S2_lai\\19820801.tif'
    #     outf=data_root+r'monthly_data\\Trendy_TIFF\CLASSIC_S2_lai\\19820801_unify.tif'
    #     array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(testf)
    #     array = np.array(array, dtype=float)
    #     ToRaster().array2raster(outf, -180, 90, 1, -1, array, )



    def resample(self):
        fdir=rf'D:\Project3\Data\Fire\TIFF\\'
        outdir=rf'D:\Project3\Data\Fire\resample\\'
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath=fdir+f
            outf=outdir+f
            dataset = gdal.Open(fpath)

            try:
                gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326',resampleAlg='average')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass
    def scale(self):

        fdir = rf'D:\Project3\Data\Fire\resample\\'
        outdir = rf'D:\Project3\Data\Fire\scale\\'
        Tools().mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)

            array = array * 4

            outf = outdir + f
            ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, array)

    def extract_dryland_tiff(self):
        self.datadir=rf'D:\Project3\Data\\'
        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask05.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan


        fdir_all = rf'D:\Project3\Data\Fire\\'

        for fdir in T.listdir(fdir_all):
            if not 'scale' in fdir:
                continue


            fdir_i = join(fdir_all, fdir)

            outdir_i = join(fdir_all, 'dryland_tiff')

            T.mk_dir(outdir_i)
            for fi in tqdm(T.listdir(fdir_i), desc=fdir):
                if not fi.endswith('.tif'):
                    continue
                fpath = join(fdir_i, fi)
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr[np.isnan(array_mask)] = np.nan
                # plt.imshow(arr)
                # plt.show()
                fname=fi.split('_')[-1].split('.')[0]
                # print(fname);exit()
                # outpath = join(outdir_i, fi)
                outpath = join(outdir_i, fname+'.tif')

                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)

        pass
    def generate_nan_map(self):
        ## 1994 has no data before tiff to dic I need to generate nan map
        ## 720*360
        outdir=rf'D:\Project3\Data\Fire\\fill_nan\\'
        T.mk_dir(outdir,force=True)
        ## year 1994 and month 1 to 12

        for i in range(12):
            i=i+1
            i=f'{i:02d}'
            i=str(i)
            # print(i);exit()
            arr_void=np.zeros((360, 720)) * np.nan
            outf=outdir+'1994'+i+'.tif'
            # print(outf);exit()
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_void,outf)


        pass
    def tif_to_dic(self):

        fdir_all = rf'D:\Project3\Data\Fire\\'
        year_list = list(range(1982, 2021))

        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            if not 'fill_nan' in fdir:
                continue
            outdir=join(fdir_all, 'dic')


            T.mk_dir(outdir, force=True)
            all_array = []  #### so important  it should be go with T.mk_dic

            for f in os.listdir(fdir_all+fdir):
                if not f.endswith('.tif'):
                    continue
                if int(f.split('.')[0][0:4]) not in year_list:
                    continue

                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_all, fdir, f))
                array = np.array(array, dtype=float)


                # array_unify = array[:720][:720,
                #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
                array_unify = array[:360][:360,
                              :720]
                array_unify[array_unify < 0] = np.nan


                #
                #
                # plt.imshow(array_unify)
                # plt.show()
                # array_mask = np.array(array_mask, dtype=float)
                # plt.imshow(array_mask)
                # plt.show()

                array_dryland = array_unify
                # plt.imshow(array_dryland)
                # plt.show()

                all_array.append(array_dryland)

            row = len(all_array[0])
            col = len(all_array[0][0])
            key_list = []
            dic = {}

            for r in tqdm(range(row), desc='构造key'):  # 构造字典的键值，并且字典的键：值初始化
                for c in range(col):
                    dic[(r, c)] = []
                    key_list.append((r, c))
            # print(dic_key_list)

            for r in tqdm(range(row), desc='构造time series'):  # 构造time series
                for c in range(col):
                    for arr in all_array:
                        value = arr[r][c]
                        dic[(r, c)].append(value)
                    # print(dic)
            time_series = []
            flag = 0
            temp_dic = {}
            for key in tqdm(key_list, desc='output...'):  # 存数据
                flag = flag + 1
                time_series = dic[key]
                time_series = np.array(time_series)
                temp_dic[key] = time_series
                if flag % 10000 == 0:
                    # print(flag)
                    np.save(outdir + '\\per_pix_dic_%03d' % (flag / 10000), temp_dic)
                    temp_dic = {}
            np.save(outdir + '\\per_pix_dic_%03d' % 0, temp_dic)

    def extract_phenology_monthly_variables(self):
        fdir = rf'D:\Project3\Data\VODCA_CXKu\VODCA_CXKu\daily_images_VODCA_CXKu\dic\\'

        outdir = rf'D:\Project3\Data\VODCA_CXKu\VODCA_CXKu\daily_images_VODCA_CXKu\\\phenology_year_extraction_dryland\\'

        Tools().mk_dir(outdir, force=True)
        f_phenology = rf'D:\Project3\Data\LAI4g\4GST\\4GST.npy'
        phenology_dic = T.load_npy(f_phenology)
        new_spatial_dic={}
        # for pix in phenology_dic:
        #     val=phenology_dic[pix]['SeasType']
        #     new_spatial_dic[pix]=val
        # spatial_array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(new_spatial_dic)
        # plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
        # plt.show()
        # exit()
        for f in T.listdir(fdir):

            outf = outdir + f
            #
            # if os.path.isfile(outf):
            #     continue
            # print(outf)
            spatial_dict = dict(np.load(fdir + f, allow_pickle=True, encoding='latin1').item())
            dic_DOY={15: 0,
                     30: 0,
                     45: 1,
                     60: 1,
                     75: 2,
                     90: 2,
                     105: 3,
                     120: 3,
                     135: 4,
                     150: 4,
                     165: 5,
                     180: 5,
                     195: 6,
                     210: 6,
                     225: 7,
                     240: 7,
                     255: 8,
                     270: 8,
                     285: 9,
                     300: 9,
                     315: 10,
                     330: 10,
                     345: 11,
                     360: 11,}


            result_dic = {}
            for pix in tqdm(spatial_dict):
                if not pix in phenology_dic:
                    continue

                r, c = pix

                SeasType=phenology_dic[pix]['SeasType']
                if SeasType==2:

                    SOS=phenology_dic[pix]['Onsets']
                    try:
                        SOS=float(SOS)

                    except:
                        continue

                    SOS=int(SOS)
                    SOS_biweekly=dic_DOY[SOS]

                    EOS=phenology_dic[pix]['Offsets']
                    EOS=int(EOS)
                    EOS_biweekly=dic_DOY[EOS]


                    time_series = spatial_dict[pix]


                    time_series = np.array(time_series)
                    time_series_flatten = time_series.flatten()
                    time_series_flatten_extraction = time_series_flatten[(EOS_biweekly + 1):-(12 - EOS_biweekly - 1)]
                    time_series_flatten_extraction_reshape = time_series_flatten_extraction.reshape(-1, 12)
                    non_growing_season_list = []
                    growing_season_list = []
                    for vals in time_series_flatten_extraction_reshape:
                        if T.is_all_nan(vals):
                            continue
                        ## non-growing season +growing season is 365

                        non_growing_season = vals[0:SOS_biweekly]
                        growing_season = vals[SOS_biweekly:]
                        # print(growing_season)
                        non_growing_season_list.append(non_growing_season)
                        growing_season_list.append(growing_season)
                    # print(len(growing_season_list))
                    non_growing_season_list = np.array(non_growing_season_list)
                    growing_season_list = np.array(growing_season_list)


                elif SeasType==3:
                    # SeasClass=phenology_dic[pix]['SeasClss']
                    # ## whole year is growing season
                    # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
                    # print(lat,lon)
                    # print(SeasType)
                    # print(SeasClass)
                    time_series = spatial_dict[pix]
                    time_series = np.array(time_series)
                    time_series_flatten = time_series.flatten()
                    time_series_flatten_extraction = time_series_flatten[12:]
                    time_series_flatten_extraction_reshape = time_series_flatten_extraction.reshape(-1, 12)
                    non_growing_season_list = []
                    growing_season_list = time_series_flatten_extraction_reshape

                else:
                    SeasClss=phenology_dic[pix]['SeasClss']
                    print(SeasType,SeasClss)
                    continue

                result_dic[pix]={'SeasType':SeasType,
                    'non_growing_season':non_growing_season_list,
                              'growing_season':growing_season_list,
                              'ecosystem_year':time_series_flatten_extraction_reshape}

            np.save(outf, result_dic)






    def extract_phenology_fire_mean(self):  ## extract LAI average
        fdir = data_root+rf'\Fire\dic\\'

        outdir_CV = result_root+rf'\3mm\extract_fire_ecosystem_year\\extraction_fire\\'

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            ### annual year

            vals = spatial_dic[pix]['ecosystem_year']
            vals_growing_season = spatial_dic[pix]['growing_season']
            vals_non_growing_season = spatial_dic[pix]['non_growing_season']

            ecosystem_mean_list = []
            growing_season_mean_list = []
            non_growing_season_mean_list = []

            for val in vals:
                if T.is_all_nan(val):
                    continue
                val = np.array(val)
                sum_annaul = np.nanmean(val)
                ecosystem_mean_list.append(sum_annaul)

            for val in vals_growing_season:
                if T.is_all_nan(val):
                    continue
                val = np.array(val)

                sum_growing_season = np.nanmean(val)
                growing_season_mean_list.append(sum_growing_season)

            for val in vals_non_growing_season:
                if T.is_all_nan(val):
                    continue
                val = np.array(val)

                sum_non_growing_season = np.nanmean(val)
                non_growing_season_mean_list.append(sum_non_growing_season)

            result_dic[pix] = {'ecosystem_year': ecosystem_mean_list,
                               'growing_season': growing_season_mean_list,
                               'non_growing_season': non_growing_season_mean_list}

        outf = outdir_CV + 'fire.npy'

        np.save(outf, result_dic)


def main():
    Data_processing().run()

if __name__ == '__main__':
    main()

