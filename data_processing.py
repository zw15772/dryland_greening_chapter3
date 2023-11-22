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

class data_processing():
    def __init__(self):
        pass
    def run(self):
        self.nc_to_tif()
        # self.check_tif_length()
        # self.resample_trendy()
        # self.unify_TIFF()

        # self.trendy_ensemble_calculation()


        # self.tif_to_dic()
        # self.extract_GS()
        # self.extend_GS() ## for SDGVM， it has 37 year GS, to align with other models, we add one more year
        # self.split_data()

    def nc_to_tif(self):

        fdir=data_root+'TRENDY_LAI\S2\\nc\\'

        for f in os.listdir(fdir):


            if f.startswith('.'):
                continue
            outdir = data_root + 'TRENDY_LAI\\S2\\TIFF\\' + f.split('.')[0] + '\\'
            T.mk_dir(outdir, force=True)

            outdir_name = f.split('.')[0]
            print(outdir_name)

            yearlist = list(range(1982, 2021))

            # # check nc variables
            # print(nc.variables.keys())
            # exit()

            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir + f, var_name='lai', outdir=outdir, yearlist=yearlist)
            except Exception as e:
                print(e)
                continue

    def nc_to_tif_template(self, fname, var_name, outdir, yearlist):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())

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
    def trendy_ensemble_calculation(self):  # 将提取的original_dataset average
        fdir_all = data_root + f'TRENDY_LAI\S0\\unify\\'
        outdir = data_root + f'TRENDY_LAI\S0\\Trendy_ensemble\\'
        Tools().mk_dir(outdir, force=True)

        year_list = list(range(1982, 2021))
        month_list = list(range(1, 13))

        for year in tqdm(year_list):
            for month in tqdm(month_list):
                data_list = []
                for fdir in tqdm(os.listdir(fdir_all)):
                    if 'SDGVM' in fdir:
                        continue

                    for f in tqdm(os.listdir(fdir_all + fdir)):


                        if not f.endswith('.tif'):
                            continue
                        if f.startswith('._'):
                            continue

                        data_year = f.split('.')[0][0:4]
                        data_month = f.split('.')[0][4:6]

                        if not int(data_year) == year:
                            continue
                        if not int(data_month) == month:
                            continue
                        arr=ToRaster().raster2array(fdir_all + fdir+'\\'+f)[0]
                        arr_unify = arr[:720][:720,
                                    :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
                        arr_unify = np.array(arr_unify)
                        arr_unify[arr_unify <0] = np.nan
                        arr_unify[arr_unify > 7] = np.nan
                        data_list.append(arr_unify)
                data_list = np.array(data_list)
                print(data_list.shape)
                # print(len(data_list))
                # exit()

                ##define arr_average and calculate arr_average

                arr_average = np.nanmean(data_list, axis=0)
                arr_average = np.array(arr_average)
                arr_average[arr_average <=0] = np.nan
                arr_average[arr_average > 7] = np.nan
                if np.isnan(np.nanmean(arr_average)):
                    continue
                if np.nanmean(arr_average) < 0.:
                    continue
                # plt.imshow(arr_average)
                # plt.title(f'{year}{month}')
                # plt.show()

                # save

                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_average, outdir + '{}{:02d}{:02d}.tif'.format(year, month, 11))



    pass

    def tif_to_dic(self):

        fdir_all = data_root+'TRENDY_LAI\S2\\unify\\'

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan

        year_list = list(range(1982, 2021))

        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            print(fdir)


            outdir = data_root + rf'TRENDY_LAI\S2\\DIC\\{fdir}\\'
            if os.path.isdir(outdir):
                continue
            T.mk_dir(outdir, force=True)

            all_array = []  #### so important

            for f in os.listdir(fdir_all+fdir):
                if not f.endswith('.tif'):
                    continue
                if int(f.split('.')[0][0:4]) not in year_list:  #
                    continue
                if f.startswith('._'):
                    continue
                outf=outdir+f.split('.')[0]+'.npy'

                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir_all+fdir+'\\'+f)
                array = np.array(array, dtype=float)


                array_unify = array[:720][:720,
                              :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

                array_unify[array_unify < -999] = np.nan
                array_unify[array_unify > 7] = np.nan
                # array[array ==0] = np.nan
                array_unify[array_unify < 0] = np.nan  # 当变量是LAI 的时候，<0!!
                # plt.imshow(array)
                # plt.show()
                array_mask = np.array(array_mask, dtype=float)
                # plt.imshow(array_mask)
                # plt.show()
                array_dryland = array_unify * array_mask
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
                    np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                    temp_dic = {}
            np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)


    def extract_GS(self):  ## here using new extraction method: 240<r<480 all year growing season
        fdir_all = data_root + 'TRENDY_LAI\S2\\DIC\\'
        outdir = result_root + f'extract_GS\TRENDY_LAI\S2\\'
        Tools().mk_dir(outdir, force=True)
        date_list=[]

        # print(date_list)
        # exit()

        for year in range(1982, 2021):
            for mon in range(1, 13):
                date_list.append(datetime.datetime(year, mon, 1))
        for fdir in os.listdir(fdir_all):


            spatial_dict = {}
            outf = outdir + fdir.split('.')[0] + '.npy'

            if os.path.isfile(outf):
                continue
            print(outf)

            for f in os.listdir(fdir_all + fdir):

                spatial_dict_i =dict(np.load(fdir_all + fdir+'\\'+f, allow_pickle=True, ).item())
                spatial_dict.update(spatial_dict_i)

            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict):
                r,c=pix
                # if not 240<r<480:
                #     continue

                gs_mon = global_get_gs(pix)
                vals = spatial_dict[pix]
                vals = np.array(vals)
                # vals[vals == 65535] = np.nan
                #
                # vals = np.array(vals)/100

                vals[vals < -999] = np.nan
                vals[vals > 7] = np.nan

                vals[vals<0]=np.nan

                if T.is_all_nan(vals):
                    continue

                vals_dict = dict(zip(date_list, vals))
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)

                        date_list_index.append(i)

                consecutive_ranges = self.group_consecutive_vals(date_list_index)
                date_dict = dict(zip(list(range(len(date_list))), date_list))

                # annual_vals_dict = {}
                annual_gs_list = []

                if len(consecutive_ranges[0])>12:
                    consecutive_ranges=np.reshape(consecutive_ranges,(-1,12))

                for idx in consecutive_ranges:
                    date_gs = [date_dict[i] for i in idx]
                    if not len(date_gs) == len(gs_mon):
                        continue
                    year = date_gs[0].year

                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    mean = np.nanmean(vals_gs)

                    annual_gs_list.append(mean)

                annual_gs_list = np.array(annual_gs_list)

                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            np.save(outf, annual_spatial_dict)

    pass


    def group_consecutive_vals(self, in_list):
        # 连续值分组
        ranges = []
        #### groupby 用法
        ### when in_list=468, how to groupby
        for _, group in groupby(enumerate(in_list), lambda index_item: index_item[0] - index_item[1]):

            group = list(map(itemgetter(1), group))
            # print(group)
            # exit()
            if len(group) > 1:
                ranges.append(list(range(group[0], group[-1] + 1)))
            else:
                ranges.append([group[0]])
        return ranges
    def split_data(self):

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask= DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir=result_root + rf'extract_GS\\'

        outdir=result_root+rf'split_original\\'
        T.mk_dir(outdir,force=True)

        for f in os.listdir(fdir):
            outf=outdir+f.split('.')[0]+'_'
            print(outf)

            dic_i = {}
            dic_ii = {}


            dic = dict(np.load(fdir + f, allow_pickle=True, ).item())


            for pix in tqdm(dic):
                r,c=pix
                if r<480:
                    continue
                if pix not in dic_dryland_mask:
                    continue
                time_series=dic[pix]
                print(len(time_series))
                time_series[time_series < -99] = np.nan
                time_series=np.array(time_series)
                # time_series[time_series >7] = np.nan

                time_series_i=time_series[:19]
                time_series_ii=time_series[19:]
                print(len(time_series_i))
                print(len(time_series_ii))

                plt.plot(time_series_i)
                plt.plot(time_series_ii)

                plt.show()

                dic_i[pix]=time_series_i
                dic_ii[pix]=time_series_ii

            np.save(outf+'1982_2000.npy',dic_i)
            np.save(outf+'2001_2020.npy',dic_ii)
    def extend_GS(self):
        f= result_root + rf'extract_GS\TRENDY_LAI\S2\\SDGVM_S2_lai.npy'
        outf=result_root + rf'extract_GS\\TRENDY_LAI\S2\\SDGVM_S2_lai_new.npy'
        dic = dict(np.load(f, allow_pickle=True, ).item())
        dic_new={}
        for pix in tqdm(dic):
            time_series=dic[pix]
            time_series=np.array(time_series)
            time_series[time_series<-999]=np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            time_series_new=np.append(time_series,np.nan)
            dic_new[pix]=time_series_new
        np.save(outf,dic_new)

    def check_tif_length(self):  ## count the number of tif files in each year
        fdir=data_root+'TRENDY_LAI\S0\TIFF\\'
        for fdir_i in os.listdir(fdir):
            year_dic={}
            flag=0
            for f in os.listdir(fdir+fdir_i):


                if not f.endswith('.tif'):
                    continue
                flag = flag + 1
                year=f.split('.')[0][0:4]
                if year not in year_dic:
                    year_dic[year]=0
                year_dic[year]+=1
            print(f'{fdir_i}:',year_dic)
            print(flag)


    def resample_trendy(self):
        fdir_all = data_root + '\TRENDY_LAI\S2\TIFF\\'
        for fdir in tqdm(os.listdir(fdir_all)):

            outdir = data_root + rf'\TRENDY_LAI\S2\\resample\\{fdir}\\'
            if os.path.isdir(outdir):
                continue


            T.mk_dir(outdir, force=True)
            year = list(range(1982, 2021))
            # print(year)
            # exit()
            for f in tqdm(os.listdir(fdir_all + fdir + '\\'), ):
                if not f.endswith('.tif'):
                    continue

                if f.startswith('._'):
                    continue

                # year_selection=f.split('.')[1].split('_')[1]
                # print(year_selection)
                # if not int(year_selection) in year:  ##一定注意格式
                #     continue
                # fcheck=f.split('.')[0]+f.split('.')[1]+f.split('.')[2]+'.'+f.split('.')[3]
                # if os.path.isfile(outdir+'resample_'+fcheck):  # 文件已经存在的时候跳过
                #     continue
                # date = f[0:4] + f[5:7] + f[8:10] MODIS
                print(f)
                # exit()
                date = f.split('.')[0]
                date_2 = date.split('_')[-1]
                print(date_2)

                # print(date)
                # exit()
                dataset = gdal.Open(fdir_all + fdir + '\\' + f)
                # print(dataset.GetGeoTransform())
                original_x = dataset.GetGeoTransform()[1]
                original_y = dataset.GetGeoTransform()[5]

                # band = dataset.GetRasterBand(1)
                # newRows = dataset.YSize * 2
                # newCols = dataset.XSize * 2
                try:
                    gdal.Warp(outdir + '{}.tif'.format(date_2), dataset, xRes=0.25, yRes=0.25, dstSRS='EPSG:4326')
                # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
                # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
                except Exception as e:
                    pass
    def unify_TIFF(self):
        fdir_all=data_root + '\\TRENDY_LAI\\S2\\resample\\'
        for fdir in tqdm(os.listdir(fdir_all)):
            outdir = data_root + rf'\TRENDY_LAI\S2\\unify\\{fdir}\\'
            T.mk_dir(outdir, force=True)
            for f in tqdm(os.listdir(fdir_all+fdir+'\\')):
                fpath=fdir_all+fdir+'\\'+f
                outpath=outdir+f
                if not f.endswith('.tif'):
                    continue
                if f.startswith('._'):
                    continue
                unify_tiff=DIC_and_TIF().unify_raster1(fpath,outpath,0.25)


class statistic_analysis():
    def __init__(self):
        pass
    def run(self):

        # self.detrend()  ##original
        # self.detrend_zscore_monthly()
        # self.zscore()
        # self.detrend()

        # self.anomaly_GS()
        # self.anomaly_GS_ensemble()
        # self.zscore_GS()

        # self.trend_analysis()
        self.scerios_analysis() ## this method tried to calculate different scenarios



    def detrend(self):

        fdir=result_root + rf'split_original\\'
        outdir=result_root + rf'detrend_original\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            outf=outdir+f.split('.')[0]
            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

            detrend_zscore_dic={}

            for pix in tqdm(dic):
                # r, c= pix
                # if r<480:
                #     continue

                print(len(dic[pix]))
                time_series = dic[pix]
                print(len(time_series))

                time_series=np.array(time_series)
                # plt.plot(time_series)
                # plt.show()

                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.nanmean(time_series) <= 0.:
                    continue


                # mean = np.nanmean(time_series)
                # std=np.nanstd(time_series)
                # if std == 0:
                #     continue
                # delta_time_series = (time_series - mean) / std
                if np.isnan(time_series).any():
                    continue

                detrend_delta_time_series = signal.detrend(time_series)

                # plt.plot(detrend_delta_time_series)
                # plt.show()

                detrend_zscore_dic[pix] = detrend_delta_time_series

            np.save(outf, detrend_zscore_dic)

    def detrend_zscore_monthly(self): #  detrend based on each month


        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir_all = data_root + 'split\\'
        for fdir in os.listdir(fdir_all):


            outdir = result_root + rf'detrend_zscore\\{fdir}\\'
            # if os.path.isdir(outdir):
            #     continue
            Tools().mk_dir(outdir, force=True)

            for f in os.listdir(fdir_all+fdir):

                outf=outdir+f.split('.')[0]
                print(outf)


                dic = dict(np.load(fdir_all+fdir+'\\'+f, allow_pickle=True, ).item())

                detrend_zscore_dic={}


                for pix in tqdm(dic):
                    detrend_delta_time_series_list = []
                    if pix not in dic_dryland_mask:
                        continue

                    print(len(dic[pix]))
                    time_series = dic[pix]
                    print(len(time_series))

                    time_series=np.array(time_series)
                    time_series[time_series < -999] = np.nan


                    if np.isnan(np.nanmean(time_series)):
                        continue
                    if np.nanmean(time_series) <= 0.:
                        continue
                    time_series_reshape=time_series.reshape(-1,12)
                    time_series_reshape_T=time_series_reshape.T
                    for i in range(len(time_series_reshape_T)):
                        time_series_i=time_series_reshape_T[i]

                        mean = np.nanmean(time_series_i)
                        std=np.nanstd(time_series_i)
                        if std == 0:
                            continue

                        delta_time_series = (time_series_i - mean) / std
                        if np.isnan(delta_time_series).any():
                            continue

                        detrend_delta_time_series = signal.detrend(delta_time_series)
                        detrend_delta_time_series_list.append(detrend_delta_time_series)
                    detrend_delta_time_series_array=np.array(detrend_delta_time_series_list)
                    detrend_delta_time_series_array=detrend_delta_time_series_array.T
                    detrend_delta_time_series_result=detrend_delta_time_series_array.flatten()

                    # detrend_delta_time_series_result2=detrend_delta_time_series_array.reshape(-1)   ##flatten and reshape 是一个东西
                    ##plot
                    # plt.plot(detrend_delta_time_series_result1,'r' ,linewidth=0.5, marker='*', markerfacecolor='blue', markersize=1 )
                    # plt.plot(detrend_delta_time_series_result,'b' ,linewidth=1,linestyle='--')
                    # plt.show()

                    detrend_zscore_dic[pix] = detrend_delta_time_series_result

                np.save(outf, detrend_zscore_dic)


    def zscore(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)


        fdir_all=data_root+'split\\'


        for fdir in os.listdir(fdir_all):


            outdir = result_root + rf'zscore\\{fdir}\\'
            # if os.path.isdir(outdir):
            #     continue
            Tools().mk_dir(outdir, force=True)

            for f in os.listdir(fdir_all + fdir):

                outf = outdir + f.split('.')[0]
                print(outf)

                dic = dict(np.load(fdir_all + fdir + '\\' + f, allow_pickle=True, ).item())

                zscore_dic = {}

                for pix in tqdm(dic):
                    delta_time_series_list = []
                    if pix not in dic_dryland_mask:
                        continue

                    # print(len(dic[pix]))
                    time_series = dic[pix]
                    # print(len(time_series))

                    time_series = np.array(time_series)
                    time_series[time_series < -999] = np.nan

                    if np.isnan(np.nanmean(time_series)):
                        continue
                    if np.nanmean(time_series) <= 0.:
                        continue
                    time_series_reshape = time_series.reshape(-1, 12)
                    time_series_reshape_T = time_series_reshape.T
                    for i in range(len(time_series_reshape_T)):
                        time_series_i = time_series_reshape_T[i]

                        mean = np.nanmean(time_series_i)
                        std = np.nanstd(time_series_i)

                        delta_time_series = (time_series_i - mean) / std

                        delta_time_series_list.append(delta_time_series)
                    delta_time_series_array = np.array(delta_time_series_list)
                    delta_time_series_array = delta_time_series_array.T
                    delta_time_series_result = delta_time_series_array.flatten()

                    # detrend_delta_time_series_result2=detrend_delta_time_series_array.reshape(-1)   ##flatten and reshape 是一个东西
                    ##plot
                    # plt.plot(detrend_delta_time_series_result1,'r' ,linewidth=0.5, marker='*', markerfacecolor='blue', markersize=1 )
                    # plt.plot(detrend_delta_time_series_result,'b' ,linewidth=1,linestyle='--')
                    # plt.show()

                    zscore_dic[pix] = delta_time_series_result

                np.save(outf, zscore_dic)

        pass

    def detrend(self):  # detrend based on two period 1982-2000 and 2001-2020

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir_all = result_root + 'zscore\\'
        for fdir in os.listdir(fdir_all):


            outdir = result_root + rf'detrend_zscore_Yang\\{fdir}\\'
            # if os.path.isdir(outdir):
            #     continue
            Tools().mk_dir(outdir, force=True)

            for f in os.listdir(fdir_all + fdir):

                outf = outdir + f.split('.')[0]
                print(outf)

                dic = dict(np.load(fdir_all + fdir + '\\' + f, allow_pickle=True, ).item())

                detrend_zscore_dic = {}

                for pix in tqdm(dic):

                    if pix not in dic_dryland_mask:
                        continue

                    print(len(dic[pix]))
                    time_series = dic[pix]
                    print(len(time_series))

                    time_series = np.array(time_series)
                    time_series[time_series < -999] = np.nan

                    if np.isnan(np.nanmean(time_series)):
                        continue

                    # if np.isnan(time_series).any():
                    #     continue
                    detrend_time_series=T.detrend_vals(time_series)
                    detrend_zscore_dic[pix] = detrend_time_series
                    # plt.plot(detrend_time_series)
                    # plt.show()

                np.save(outf, detrend_zscore_dic)



    def anomaly_GS(self):  ### anomaly GS
        dryland_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(dryland_mask_f)


        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root + 'extract_GS\\TRENDY_LAI\\S0\\'
        outdir = result_root + f'anomaly\\S0\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)

            dic = np.load(fdir + f, allow_pickle=True, ).item()

            anomaly_dic = {}

            for pix in tqdm(dic):
                if pix not in dic_dryland_mask:
                    continue

                classval=dic_dryland_mask[pix]
                if np.isnan(classval):
                    continue

                r, c = pix
                # if not 240 < r < 480:
                #     continue

                time_series = dic[pix]
                print(len(time_series))

                time_series = np.array(time_series)

                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                # plt.plot(time_series)
                # plt.show()

                mean = np.nanmean(time_series)

                delta_time_series = (time_series - mean)


                # plt.plot(delta_time_series)
                # plt.show()

                anomaly_dic[pix] = delta_time_series

            np.save(outf, anomaly_dic)
    def anomaly_GS_ensemble(self):  ### calculate the ensemble mean of anomaly GS

        fdir = result_root + 'anomaly\\\\S3\\'
        outdir = result_root + f'anomaly\\S3\\'
        Tools().mk_dir(outdir, force=True)


        data_list=[]
        for f in os.listdir(fdir):


            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)


            dic = np.load(fdir + f, allow_pickle=True, ).item()
            data_list.append(dic)

        dic_new={}

        for pix in tqdm(dic):
            all_data=[]
            for i in tqdm(range(len(data_list))):
                if pix not in data_list[i]:
                    continue

                # if pix not in dic_new:
                #     dic_new[pix]=[]
                temp=data_list[i][pix]
                all_data.append(temp)
            all_data=np.array(all_data)
            average=np.nanmean(all_data,axis=0)
            dic_new[pix]=average
        np.save(outdir+'Ensemble.npy',dic_new)

        # dic_spatial_plot={}
        # for pix in dic_new:
        #     test=dic_new[pix]
        #     number=len(test)
        #     dic_spatial_plot[pix]=number
        # array_average=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(dic_spatial_plot)
        # plt.imshow(array_average)
        # plt.show()









    def zscore_GS(self):  ### anomaly GS
        dryland_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(dryland_mask_f)


        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root + 'extract_GS\\'
        outdir = result_root + f'zscore\\1982_2020\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):


            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)
            if os.path.isfile(outf):
                continue

            dic = np.load(fdir + f, allow_pickle=True, ).item()

            anomaly_dic = {}

            for pix in tqdm(dic):
                if pix not in dic_dryland_mask:
                    continue

                classval=dic_dryland_mask[pix]
                if np.isnan(classval):
                    continue

                r, c = pix
                # if not 240 < r < 480:
                #     continue

                time_series = dic[pix]
                print(len(time_series))

                time_series = np.array(time_series)

                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                # plt.plot(time_series)
                # plt.show()

                mean = np.nanmean(time_series)
                std=np.nanstd(time_series)
                if std==0:
                    continue

                delta_time_series = (time_series - mean)/std


                # plt.plot(delta_time_series)
                # plt.show()

                anomaly_dic[pix] = delta_time_series

            np.save(outf, anomaly_dic)
    def trend_analysis(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)


        fdir = result_root + f'anomaly\S3\\'
        outdir = result_root + rf'trend_analysis\\anomaly\\S3\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):


            outf=outdir+f.split('.')[0]
            if os.path.isfile(outf+'_trend.npy'):
                continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):

                time_series = dic[pix]

                time_series = np.array(time_series)
                # print(len(time_series))
                # plt.plot(time_series)
                # plt.show()

                time_series[time_series < -99.] = np.nan
                #remove divide by zero error, remove the number all the same
                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.nanmax(time_series) == np.nanmin(time_series):
                    continue


                # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                slope,b,r,p_value=T.nan_line_fit(np.arange(len(time_series)), time_series)
                trend_dic[pix] = slope
                p_value_dic[pix] = p_value

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



        pass
    def scerios_analysis(self):
        scenarios='S3-S2'
        fdir_S0 = result_root + f'trend_analysis\\anomaly\\S2\\'
        fdir_S1=result_root + f'trend_analysis\\anomaly\\S3\\'
        outdir = result_root + rf'trend_analysis\\anomaly\\{scenarios}\\'
        Tools().mk_dir(outdir, force=True)
        for f0 in os.listdir(fdir_S0):
            if not f0.endswith('.npy'):
                continue
            if not 'trend' in f0:
                continue

            fmodel=f0.split('.')[0].split('_')[0]
            array_0=np.load(fdir_S0+f0)
            array_0[array_0<-999]=np.nan
            for f1 in os.listdir(fdir_S1):
                if not f1.endswith('.npy'):
                    continue
                if not 'trend' in f1:
                    continue
                if f1.split('.')[0].split('_')[0]!=fmodel:
                    continue
                array_1=np.load(fdir_S1+f1)
                array_1[array_1<-999]=np.nan
                array_new=array_1-array_0


                outf=outdir+fmodel+f'_{scenarios}.npy'
                print(outf)
                # exit()

                plt.imshow(array_new,cmap='jet',vmin=-0.01,vmax=0.01)
                # plt.colorbar()
                # plt.title(fmodel)
                # plt.show()


                np.save(outf, array_new)
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_new,outdir+fmodel+f'_{scenarios}.tif')

class moving_window():
    def __init__(self):
        pass
    def run(self):
        self.moving_window_extraction()
        pass
    def moving_window_extraction(self):
        variables=['LAI4g','NDVI4g','GPP_CFE','GPP_baseline']
        fdir = data_root + rf'Extraction\\'
        outdir = result_root + rf'\\extract_window\\extract_detrend_original_window\\'
        T.mk_dir(outdir, force=True)
        for variable in variables:

            f=fdir+variable+'.npy'
            outf=outdir+variable+'.npy'
            outf_i = join(outdir, fdir)
            if os.path.isfile(outf_i):
                continue
            dic = T.load_npy(f)
            window = 15

            new_x_extraction_by_window = {}
            for pix in tqdm(dic):

                time_series = dic[pix]
                time_series = np.array(time_series)

                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    print('error')
                    continue
                print((len(time_series)))
                ### if all values are identical, then continue
                if np.nanmax(time_series) == np.nanmin(time_series):
                    continue

                new_x_extraction_by_window[pix] = self.forward_window_extraction(time_series, window)

            T.save_npy(new_x_extraction_by_window, outf)
    def forward_window_extraction(self, x, window):
        # 前窗滤波
        # window = window-1
        # 不改变数据长度

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        # new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        new_x_extraction_by_window=[]
        for i in range(len(x)):
            if i + window >= len(x):
                continue
            else:
                anomaly = []
                x_vals=[]
                for w in range(window):
                    x_val=(x[i + w])
                    x_vals.append(x_val)
                if np.isnan(np.nanmean(x_vals)):
                    continue

                # x_mean=np.nanmean(x_vals)

                # for i in range(len(x_vals)):
                #     if x_vals[0]==None:
                #         continue
                #     x_anomaly=x_vals[i]-x_mean
                #
                #     anomaly.append(x_anomaly)
                new_x_extraction_by_window.append(x_vals)
        return new_x_extraction_by_window
class multi_regression_window():
    def __init__(self):
        self.fdirX=result_root+rf'extract_window\extract_detrend_original_window\15_year_window_1982_2020\X\\'
        self.y_f=result_root+rf'extract_window\extract_detrend_original_window\15_year_window_1982_2020\Y\\LAI4g_clean.npy'

        self.xvar_list = ['Tmax', 'GLEAM_SMroot']
        self.y_var = ['LAI4g_clean']
        pass

    def run(self):

        window = 39-15

        # step 1 build dataframe
        for i in range(window):
            outdir = result_root + rf'multi_regression_moving_window\\window{window}\\'
            df_i = self.build_df(self.fdirX, self.y_f, self.xvar_list, i)

            T.mk_dir(outdir,force=True)
            outf= result_root + rf'multi_regression_moving_window\\window15\\LAI_SMroot_window{i:02d}.npy'
            # if os.path.isfile(outf):
            #     continue
            print(outf)

            self.cal_multi_regression_beta(df_i,self.xvar_list, outf)  # 修改参数
            # self.plt_multi_regression_result(outdir,self.y_var,i)

    def build_df(self, fdir_X, y_f, xvar_list, w):


        df = pd.DataFrame()
        dic_y=T.load_npy(y_f)
        pix_list = []
        y_val_list=[]

        for pix in dic_y:
            if len(dic_y[pix]) == 0:
                continue
            vals = dic_y[pix][w]
            # print(vals)
            # exit()
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            vals = vals
            pix_list.append(pix)
            y_val_list.append(vals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        for xvar in xvar_list:


            x_val_list = []
            x_arr = T.load_npy(fdir_X+xvar+'.npy')
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in x_arr:
                    x_val_list.append([])
                    continue
                vals = x_arr[pix][w]
                vals = np.array(vals)
                if len(vals) == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(vals)

            df[xvar] = x_val_list

        return df



    def __linearfit(self, x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
        for i in range(0, int(N)):
            sx += x[i]
            sy += y[i]
            sxx += x[i] * x[i]
            syy += y[i] * y[i]
            sxy += x[i] * y[i]
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
        b = (sy - a * sx) / N
        r = -(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
        return a, b, r


    def cal_multi_regression_beta(self, df, x_var_list, outf):

        multi_derivative = {}
        multi_pvalue = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']
            y_vals = T.remove_np_nan(y_vals)
            if len(y_vals) == 0:
                continue
            y_vals = np.array(y_vals)
            # y_vals_detrend=signal.detrend(y_vals)


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:
                x_vals = row[x]

                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals = T.interp_nan(x_vals)
                # print(x_vals)
                if x_vals[0] == None:
                    continue
                # x_vals_detrend = signal.detrend(x_vals) #detrend
                df_new[x] = x_vals
                # df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals  # nodetrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1, how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()
            linear_model = LinearRegression()

            linear_model.fit(df_new[x_var_list_valid_new], df_new['y'])
            # coef_ = np.array(linear_model.coef_) / y_mean
            coef_ = np.array(linear_model.coef_)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))
            ## pvalue
            X=df_new[x_var_list_valid_new]
            Y=df_new['y']
            try:
                sse = np.sum((linear_model.predict(X) -Y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])

                se = np.array([
                    np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                    for i in range(sse.shape[0])
                ])

                t = coef_ / se
                p = 2 * (1 - stats.t.cdf(np.abs(t), Y.shape[0] - X.shape[1]))
            except:
                p=np.nan

            multi_derivative[pix] = coef_dic
            multi_pvalue[pix] = p

        T.save_npy(multi_derivative, outf)
        T.save_npy(multi_pvalue, outf.replace('.npy', '_pvalue.npy'))

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir,y_var,w):

        f='D:\Project3\Result\multi_regression_moving_window\window15\\LAI_SMroot_window00.npy'

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:
            # print(pix)
            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in dic:
                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f'{multi_regression_result_dir}\\{var_i}_{y_var}_{w:02d}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr,vmin=-5,vmax=5)

            plt.title(var_i)
            plt.colorbar()

        plt.show()

class multi_regression():
    def __init__(self):
        self.fdirX=result_root+rf'trend_analysis\anomaly\ALL_ensemble\X\\X\\'
        self.fdirY=result_root+rf'trend_analysis\anomaly\ALL_ensemble\\\Y\\'
        self.xvar=['Ensemble_S1-S0_trend','Ensemble_S2-S1_trend','Ensemble_S3-S2_trend']
        self.y_var=['LAI4g']
        self.period=('2001_2020')
        self.multi_regression_result_dir=result_root+rf'multi_regression\\'
        T.mk_dir(self.multi_regression_result_dir,force=True)

        self.multi_regression_result_f = result_root + rf'multi_regression\\LAI_SMroot_{self.period}.npy'
        pass

    def run(self):

        #step 1 build dataframe
        df = self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var,self.period)

        # # # step 2 cal correlation
        self.cal_multi_regression_beta(df, self.xvar)  # 修改参数

        # step 3 plot
        # self.plt_multi_regression_result(self.multi_regression_result_dir,self.y_var[0],self.period)

    def build_df(self,fdir_X,fdir_Y,fx_list,fy,period):

        df = pd.DataFrame()
        filey=fdir_Y+fy[0]+'_trend.npy'
        print(filey)

        # dic_y=T.load_npy(filey)
        array=np.load(filey)
        dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list=[]


        for pix in dic_y:
            vals = dic_y[pix]
            # print(vals)
            # exit()
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            vals = vals
            pix_list.append(pix)
            y_val_list.append(vals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        x_var_list = []
        for xvar in fx_list:

            x_var_list.append(xvar)
            # print(var_name)
            x_val_list = []
            filex=fdir_X+fx[0]+'_trend.npy'

            print(filex)
            exit()
            x_arr = T.load_npy(filex)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in x_arr:
                    x_val_list.append([])
                    continue
                vals = x_arr[pix]
                vals = np.array(vals)
                if len(vals) == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(vals)
            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list

        return df

    def __linearfit(self, x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
        for i in range(0, int(N)):
            sx += x[i]
            sy += y[i]
            sxx += x[i] * x[i]
            syy += y[i] * y[i]
            sxy += x[i] * y[i]
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
        b = (sy - a * sx) / N
        r = -(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
        return a, b, r


    def cal_multi_regression_beta(self, df, x_var_list):


        outf = self.multi_regression_result_f

        multi_derivative = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']
            y_vals = T.remove_np_nan(y_vals)
            if len(y_vals) == 0:
                continue

            y_vals_detrend = signal.detrend(y_vals)
            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:
                x_vals = row[x]

                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals = T.interp_nan(x_vals)
                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue
                x_vals_detrend = signal.detrend(x_vals) #detrend
                # df_new[x] = x_vals
                df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals_detrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1, how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()
            linear_model = LinearRegression()

            linear_model.fit(df_new[x_var_list_valid_new], df_new['y'])
            # coef_ = np.array(linear_model.coef_) / y_mean
            coef_ = np.array(linear_model.coef_)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))
            # print(df_new['y'])
            # exit()
            multi_derivative[pix] = coef_dic
        T.save_npy(multi_derivative, outf)

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir,y_var,period):

        f=self.multi_regression_result_f

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:
            # print(pix)
            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in dic:
                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f'{multi_regression_result_dir}\\{var_i}_{y_var}_{period}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr,vmin=-5,vmax=5)

            plt.title(var_i)
            plt.colorbar()

        plt.show()


class fingerprint():  ##
    def __init__(self):
        self.fdirX=result_root+rf'trend_analysis\anomaly\ALL_ensemble\X\\'
        self.fdirY=result_root+rf'trend_analysis\anomaly\ALL_ensemble\\\Y\\'
        self.xvar=['Ensemble_S1-S0_trend','Ensemble_S2-S1_trend','Ensemble_S3-S2_trend']
        self.y_var=['LAI4g']
        self.period=('2001_2020')
        self.multi_regression_result_dir=result_root+rf'fingerprint\\'
        T.mk_dir(self.multi_regression_result_dir,force=True)

        self.multi_regression_result_f = result_root + rf'fingerprint\\LAI4g_ensemle.npy'
        pass

    def run(self):

        #step 1 build dataframe
        df = self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var,self.period)

        # # # step 2 cal correlation
        self.cal_multi_regression_beta(df, self.xvar)  # 修改参数

        # step 3 plot
        self.plt_multi_regression_result(self.multi_regression_result_dir,self.y_var[0],self.period)

    def build_df(self,fdir_X,fdir_Y,fx_list,fy,period):

        df = pd.DataFrame()
        filey=fdir_Y+fy[0]+'_trend.npy'
        print(filey)

        # dic_y=T.load_npy(filey)
        array=np.load(filey)
        dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list=[]


        for pix in dic_y:
            vals = dic_y[pix]

            vals = np.array(vals)
            vals = vals
            pix_list.append(pix)
            y_val_list.append(vals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        x_var_list = []
        for xvar in fx_list:

            x_var_list.append(xvar)
            # print(var_name)
            x_val_list = []
            filex=fdir_X+xvar+'.npy'

            print(filex)
            # exit()

            arrayx = np.load(filex)
            dic_x = DIC_and_TIF().spatial_arr_to_dic(arrayx)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                vals = dic_x[pix]
                vals = np.array(vals)
                if vals == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(vals)
            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list

        return df

    def __linearfit(self, x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
        for i in range(0, int(N)):
            sx += x[i]
            sy += y[i]
            sxx += x[i] * x[i]
            syy += y[i] * y[i]
            sxy += x[i] * y[i]
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
        b = (sy - a * sx) / N
        r = -(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
        return a, b, r


    def cal_multi_regression_beta(self, df, x_var_list):


        outf = self.multi_regression_result_f

        multi_derivative = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']
            y_vals = T.remove_np_nan(y_vals)
            if len(y_vals) == 0:
                continue

            y_vals_detrend = signal.detrend(y_vals)
            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:
                x_vals = row[x]


                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals = T.interp_nan(x_vals)

                # print(x_vals)
                if x_vals[0] == None:
                    continue
                x_vals_detrend = signal.detrend(x_vals) #detrend
                # df_new[x] = x_vals
                df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals_detrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1, how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()
            linear_model = LinearRegression()

            linear_model.fit(df_new[x_var_list_valid_new], df_new['y'])
            # coef_ = np.array(linear_model.coef_) / y_mean
            coef_ = np.array(linear_model.coef_)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))
            # print(df_new['y'])
            # exit()
            multi_derivative[pix] = coef_dic
        T.save_npy(multi_derivative, outf)

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir,y_var,period):

        f=self.multi_regression_result_f

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:
            # print(pix)
            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in dic:
                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f'{multi_regression_result_dir}\\{var_i}_{y_var}_{period}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr,vmin=-5,vmax=5)

            plt.title(var_i)
            plt.colorbar()

        plt.show()


class selection():
    def __init__(self):
        pass
    def run(self):
        self.select_drying_wetting_trend()
        self.select_drought_event()
        pass
    def select_drying_wetting_trend(self):

        f_sm=data_root+'Extraction\\GLEAM_SMroot.npy'


        dic=T.load_npy(f_sm)
        result_dic={}
        result_tif_dic={}

        for pix in tqdm(dic):
            time_series=dic[pix]
            time_series=np.array(time_series)
            time_series[time_series<-999]=np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            a,b,r,p=T.nan_line_fit(np.arange(len(time_series)),time_series)
            if a>0 and p<0.05:
                result_dic[pix]='sig_wetting'
                result_tif_dic[pix]=2
            elif a<0 and p<0.05:
                result_dic[pix]='sig_drying'
                result_tif_dic[pix]=-2
            elif a>0 and p>0.05:
                result_dic[pix]='non_sig_wetting'
                result_tif_dic[pix]=1
            else:
                result_dic[pix]='non_sig_drying'
                result_tif_dic[pix]=-1
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(result_tif_dic,result_root+'\\SM_trend_label.tif')
        T.save_npy(result_dic,data_root+'Base_data\\GLEAM_SMroot_trend_label_mark.npy')

        pass

class pick_event():
    def __init__(self):
        pass

    def run(self):

        # self.pick_drought_event()
        # self.extract_variables_during_droughts_GS()
        # self.extract_variables_after_droughts_GS() ##### extract variables after droughts mean
        # self.extract_variables_after_droughts_GS_in_nth_year()  ### extract variables after droughts in nth year
        # self.multiregression_based_on_during_droughts()  ###

        # self.rename_variables()
        # self.plot_df()
        self.statistic_variables()
        # self.plt_spatial_df()

    def pick_drought_event(self):

        fdir = result_root+rf'detrend_zscore_Yang\\SPEI3\\'
        outdir = result_root + rf'pick_event_scheme2\\SPEI3\\'
        T.mk_dir(outdir, force=True)

        spatial_dic={}
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            dic_i=T.load_npy(fdir+f)
            spatial_dic.update(dic_i)

            threshold_upper=-2
            threshold_bottom=-3
            threshold_start=-1
            outf=outdir+f.split('.')[0]+f'_({threshold_bottom},{threshold_upper}).df'


            print(outf)
            event_dic={}
            for pix in spatial_dic:
                vals=spatial_dic[pix]

                drought_events_list_extreme, _=self.kernel_find_drought_period(vals,threshold_upper,threshold_bottom,threshold_start)
                if len(drought_events_list_extreme)==0:
                    continue
                event_dic[pix]=drought_events_list_extreme
            df=pd.DataFrame()
            pix_list=[]
            drought_range_list=[]
            for pix in event_dic:
                events_list=event_dic[pix]
                for event in events_list:
                    pix_list.append(pix)
                    drought_range_list.append(event)
            df['pix']=pix_list
            df['drought_range']=drought_range_list
            T.print_head_n(df)
            T.save_df(df,outf)
            self.__df_to_excel(df, outf)


        pass



    def kernel_find_drought_period(self, vals, threshold_upper, threshold_bottom, threshold_start):

        vals = np.array(vals)

        start_of_drought_list = []
        end_of_drought_list = []
        for i in range(len(vals)):
            if i + 1 == len(vals):
                break
            val_left = vals[i]
            vals_right = vals[i + 1]
            if val_left < threshold_start and vals_right > threshold_start:
                end_of_drought_list.append(i + 1)
            if val_left > threshold_start and vals_right < threshold_start:
                start_of_drought_list.append(i)

        drought_events_list = []
        for s in start_of_drought_list:
            for e in end_of_drought_list:
                if e > s:
                    drought_events_list.append((s, e))
                    break

        drought_events_list_extreme = []
        drought_timing_list = []
        for event in drought_events_list:
            s = event[0]
            e = event[1]
            min_index = T.pick_min_indx_from_1darray(vals, list(range(s, e)))
            drought_timing_month = min_index % 12 + 1
            min_val = vals[min_index]
            if min_val <  threshold_upper and min_val > threshold_bottom:
                drought_events_list_extreme.append(event)
                drought_timing_list.append(drought_timing_month)
        return drought_events_list_extreme, drought_timing_list

        pass
    def extract_variables_during_droughts_GS(self):
        fdir_drought = result_root + rf'pick_event_scheme2\\SPEI3\\'
        fdir_variables_all = result_root + rf'detrend_zscore_Yang\\'
        for f in os.listdir(fdir_drought):
            if not f.endswith('.df'):
                continue
            time_range= f.split('_')[0]+'_'+f.split('_')[1]
            threshold=f.split('_')[-1].split('.')[0]
            print(threshold)
            df= T.load_df(fdir_drought + f)
            df_new=pd.DataFrame()
            outdir = result_root + rf'pick_event_scheme2\\extract_variables_during_droughts_GS\\'
            T.mk_dir(outdir, force=True)
            # print(outdir)
            fvariable_list = ['Tmax', 'GLEAM_SMroot', 'LAI4g', 'NDVI4g']

            for fvariable in os.listdir(fdir_variables_all):
                if not fvariable in fvariable_list:
                    continue

                fvariable_path=fdir_variables_all+fvariable+'\\'+f'{time_range}.npy'
                print(fvariable_path)
                var_name=fvariable.split('.')[0]+'_'+threshold+'_'+time_range
                print(var_name)
                # exit()

                data_dict=T.load_npy(fvariable_path)
                # pix_list = T.get_df_unique_val_list(df, 'pix')

                mean_list = []
                for i, row in tqdm(df.iterrows(), total=len(df), ):
                    pix = row['pix']
                    GS = global_get_gs(pix)

                    drought_range = row['drought_range']
                    e,s = drought_range[1],drought_range[0]
                    picked_index = []
                    for idx in range(s,e+1):
                        mon = idx % 12 + 1
                        if not mon in GS:
                            continue

                        picked_index.append(idx)

                    if len(picked_index) == 0:
                        mean_list.append(np.nan)
                        continue
                    if not pix in data_dict:
                        mean_list.append(np.nan)
                        continue
                    vals = data_dict[pix]
                    # print(len(vals))
                    if picked_index[-1] >= len(vals):
                        mean_list.append(np.nan)
                        continue
                    picked_vals = T.pick_vals_from_1darray(vals,picked_index)
                    mean = np.nanmean(picked_vals)
                    if mean == 0:
                        mean_list.append(np.nan)
                        continue
                    mean_list.append(mean)

                df_new['pix'] = df['pix']
                df_new[f'{var_name}'] = mean_list
            T.print_head_n(df_new)
            return df_new

            # T.save_df(df_new, outdir + f'{time_range}_{threshold}.df')
            # self.__df_to_excel(df_new, outdir + f'{time_range}_{threshold}.df')

    def extract_variables_after_droughts_GS(self):
        n_list = [ 1, 2, 3, 4]
        fdir_drought = result_root + rf'pick_event_scheme2\\SPEI3\\'
        fdir_variables_all = result_root + rf'detrend_zscore_Yang\\'
        for f in os.listdir(fdir_drought):
            if not f.endswith('.df'):
                continue
            time_range = f.split('_')[0] + '_' + f.split('_')[1]
            threshold = f.split('_')[-1].split('.')[0]
            print(threshold)
            df = T.load_df(fdir_drought + f)
            df_new = pd.DataFrame()

            outdir = result_root + rf'pick_event_scheme2\\extract_variables_during_droughts_GS\\'
            T.mk_dir(outdir, force=True)
            # print(outdir)
            fvariable_list=['Tmax','GLEAM_SMroot','LAI4g','NDVI4g']

            for fvariable in os.listdir(fdir_variables_all):
                if not fvariable in fvariable_list:
                    continue
                fvariable_path = fdir_variables_all + fvariable + '\\' + f'{time_range}.npy'
                print(fvariable_path)
                var_name = fvariable.split('.')[0] + '_' + threshold + '_' + time_range
                print(var_name)

                data_dict = T.load_npy(fvariable_path)

            ################ add during drought year
                mean_list = []
                for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_0_{var_name}'):
                    pix = row['pix']
                    GS = global_get_gs(pix)
                    if not pix in data_dict:
                        mean_list.append(np.nan)
                        continue
                    vals = data_dict[pix]
                    drought_range = row['drought_range']

                    drought_range_index = list(range(drought_range[0], drought_range[-1] + 1))

                    picked_index = []
                    for idx in drought_range_index:
                        mon = idx % 12 + 1
                        if not mon in GS:
                            continue
                        if idx >= len(vals):
                            picked_index = []
                            break
                        picked_index.append(idx)
                    if len(picked_index) == 0:
                        mean_list.append(np.nan)
                        continue
                    picked_vals = T.pick_vals_from_1darray(vals, picked_index)
                    mean = np.nanmean(picked_vals)
                    if mean == 0:
                        mean_list.append(np.nan)
                        continue
                    mean_list.append(mean)
                df_new[f'{var_name}_post_0_GS'] = mean_list

                ################ add post drought year

                for n in n_list:

                    delta_mon = n*12
                    mean_list = []
                    for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_{n}_{var_name}'):
                        pix = row['pix']
                        GS = global_get_gs(pix)
                        if not pix in data_dict:
                            mean_list.append(np.nan)
                            continue
                        vals = data_dict[pix]
                        drought_range = row['drought_range']
                        end_mon = drought_range[-1]
                        post_drought_range = []
                        for m in range(delta_mon):
                            post_drought_range.append(end_mon + m + 1)
                        picked_index = []
                        for idx in post_drought_range:
                            mon = idx % 12 + 1
                            if not mon in GS:
                                continue
                            if idx >= len(vals):
                                picked_index = []
                                break
                            picked_index.append(idx)
                        if len(picked_index) == 0:
                            mean_list.append(np.nan)
                            continue
                        picked_vals = T.pick_vals_from_1darray(vals, picked_index)
                        mean = np.nanmean(picked_vals)
                        if mean == 0:
                            mean_list.append(np.nan)
                            continue
                        mean_list.append(mean)
                    df_new[f'{var_name}_post_{n}_GS'] = mean_list



            df_new['pix'] = df['pix']

            T.print_head_n(df_new)


            T.save_df(df_new, outdir + f'{time_range}_{threshold}.df')
            self.__df_to_excel(df_new, outdir + f'{time_range}_{threshold}.df')

    def extract_variables_after_droughts_GS_in_nth_year(self):
        n_list = [1, 2, 3, 4]
        fdir_drought = result_root + rf'pick_event_scheme2\\SPEI3\\'
        fdir_variables_all = result_root + rf'detrend_zscore_Yang\\'
        for f in os.listdir(fdir_drought):
            if not f.endswith('.df'):
                continue
            time_range = f.split('_')[0] + '_' + f.split('_')[1]
            threshold = f.split('_')[-1].split('.')[0]
            print(threshold)
            df = T.load_df(fdir_drought + f)
            df_new = pd.DataFrame()

            outdir = result_root + rf'pick_event_scheme2\\extract_variables_after_droughts_GS_in_nth_year\\'
            outf = outdir + f'{time_range}_{threshold}.df'
            if os.path.exists(outf):
                continue
            T.mk_dir(outdir, force=True)
            # print(outdir)
            fvariable_list=['Tmax','GLEAM_SMroot','LAI4g','NDVI4g']

            for fvariable in os.listdir(fdir_variables_all):
                if not fvariable in fvariable_list:
                    continue
                fvariable_path = fdir_variables_all + fvariable + '\\' + f'{time_range}.npy'
                print(fvariable_path)
                var_name = fvariable.split('.')[0] + '_' + threshold + '_' + time_range
                print(var_name)

                data_dict = T.load_npy(fvariable_path)

                ################ add during drought year
                mean_list = []
                for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_0_{var_name}'):
                    pix = row['pix']
                    GS = global_get_gs(pix)
                    if not pix in data_dict:
                        mean_list.append(np.nan)
                        continue
                    vals = data_dict[pix]
                    drought_range = row['drought_range']

                    drought_range_index = list(range(drought_range[0], drought_range[-1] + 1))

                    picked_index = []
                    for idx in drought_range_index:
                        mon = idx % 12 + 1
                        if not mon in GS:
                            continue
                        if idx >= len(vals):
                            picked_index = []
                            break
                        picked_index.append(idx)
                    if len(picked_index) == 0:
                        mean_list.append(np.nan)
                        continue
                    picked_vals = T.pick_vals_from_1darray(vals, picked_index)
                    mean = np.nanmean(picked_vals)
                    if mean == 0:
                        mean_list.append(np.nan)
                        continue
                    mean_list.append(mean)
                df_new[f'{var_name}_post_0_GS'] = mean_list

                ################ add post nth year

                for n in n_list:

                    mean_list = []
                    for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_{n}_{var_name}'):
                        pix = row['pix']
                        GS = global_get_gs(pix)
                        if not pix in data_dict:
                            mean_list.append(np.nan)
                            continue
                        vals = data_dict[pix]
                        drought_range = row['drought_range']
                        end_mon = drought_range[-1]
                        post_drought_range = []
                        assert n>0
                        for m in range((n-1)*12,n*12):

                            post_drought_range.append(end_mon + m + 1)
                        picked_index = []
                        for idx in post_drought_range:
                            mon = idx % 12 + 1
                            if not mon in GS:
                                continue
                            if idx >= len(vals):
                                picked_index = []
                                break
                            picked_index.append(idx)
                        if len(picked_index) == 0:
                            mean_list.append(np.nan)
                            continue
                        picked_vals = T.pick_vals_from_1darray(vals, picked_index)
                        mean = np.nanmean(picked_vals)
                        if mean == 0:
                            mean_list.append(np.nan)
                            continue
                        mean_list.append(mean)
                    df_new[f'{var_name}_post_{n}_GS'] = mean_list


            df_new['pix'] = df['pix']

            T.print_head_n(df_new)


            T.save_df(df_new, outdir + f'{time_range}_{threshold}.df')
            self.__df_to_excel(df_new, outdir + f'{time_range}_{threshold}.df')
    def statistic_variables(self):
        time_range = ['1982_2000', '2001_2020']
        SM_trend_list=['sig_wetting','sig_drying','non_sig_wetting','non_sig_drying']
        # threshold = '(-4,-3)'
        threshold = '(-3,-2)'
        threshold = '(-2,-1)'
        fdir = result_root + rf'pick_event_scheme2\\extract_variables_after_droughts_GS\\'
        variable_list=['GLEAM_SMroot','LAI4g','NDVI4g']


        for time_range in time_range:
            plt.figure(figsize=(10, 5))
            flag = 1
            f_path = fdir + f'{time_range}_{threshold}.df'

            df = T.load_df(f_path)

            n_list = [0, 1, 2, 3, 4]

            for variable in variable_list:
                plt.subplot(1, 3, flag)
                for SM_trend in SM_trend_list:

                    average_list = []
                for n in n_list:
                    vals=df[f'{variable}_{threshold}_{time_range}_post_{n}_GS'].tolist()
                    average=np.nanmean(vals)
                    average_list.append(average)
                plt.bar(n_list,average_list,label=variable)
                flag = flag + 1
                plt.legend()
                plt.ylim(-0.6, 0.3)
                plt.title(f'{variable}_{threshold}_{time_range}')
                plt.xticks(n_list, [f'post_{n}' for n in n_list])
                plt.xticks(rotation=45)
            plt.show()

    # def statistic_variables_back(self):   #### trail
    #     time_range = ['1982_2000', '2001_2020']
    #     SM_trend_list = ['sig_wetting', 'sig_drying', 'non_sig_wetting', 'non_sig_drying']
    #     # threshold = '(-4,-3)'
    #     threshold = '(-3,-2)'
    #     threshold = '(-2,-1)'
    #     fdir = result_root + rf'pick_event_scheme2\\extract_variables_after_droughts_GS\\'
    #     variable_list = ['LAI4g', 'NDVI4g']
    #     for variable in variable_list:
    #         for SM_trend in SM_trend_list:
    #             for period in time_range:
    #                 df=T.load_df(fdir+f'{period}_{threshold}.df')
    #                 n_list = [0, 1, 2, 3, 4]


    def multiregression_based_on_during_droughts(self):   ## LAI/SM
        time_range=['1982_2000','2001_2020']
        threshold='(-4,-3)'
        plt.figure(figsize=(10,5))
        flag=1

        fdir=result_root+rf'pick_event_scheme2\\extract_variables_after_droughts_GS\\'
        for time_range in time_range:
            plt.subplot(1,2,flag)

            f_path=fdir+f'{time_range}_{threshold}.df'

            df=T.load_df(f_path)

            n_list=[0,1,2,3,4]

            bar_list={}

            for n in n_list:
                outf = result_root + rf'multi_regression\\LAI_SMroot_post_{n}_GS.npy'
                df_new = pd.DataFrame()

                # x_var_list=[f'Tmax_{threshold}_{time_range}_post_{n}_GS',f'GLEAM_SMroot_{threshold}_{time_range}_post_{n}_GS']
                # y_vals = df[f'LAI4g_{threshold}_{time_range}_post_{n}_GS']

                x_var_list=[f'LAI4g_{threshold}_{time_range}_post_{n}_GS',f'Tmax_{threshold}_{time_range}_post_{n}_GS']
                y_vals = df[f'GLEAM_SMroot_{threshold}_{time_range}_post_{n}_GS']



                df_new['y'] = y_vals
                for x_var in x_var_list:
                    x_vals=df[x_var]
                    df_new[x_var]=x_vals


                ## remove nan
                df_new = df_new.dropna()

                # T.print_head_n(df_new)

                linear_model = LinearRegression()

                linear_model.fit(df_new[x_var_list], df_new['y'])
                coef_ = np.array(linear_model.coef_)
                coef_dic = dict(zip(x_var_list, coef_))
                bar_list[n]=coef_dic

            SM_corr_list=[]
            for n in bar_list:
                SM_corr=bar_list[n][f'GLEAM_SMroot_{threshold}_{time_range}_post_{n}_GS']
                SM_corr = bar_list[n][f'LAI4g_{threshold}_{time_range}_post_{n}_GS']
                SM_corr_list.append(SM_corr)
            plt.bar(range(len(SM_corr_list)),SM_corr_list)
            plt.xticks(range(len(SM_corr_list)), [f'post_{n}' for n in n_list])
            plt.ylim(0,0.6)
            plt.title(f'delta LAI/delta GLEAM_SMroot_{threshold}_{time_range}')

            flag=flag+1

        plt.show()




            # plt.bar(bar_list[n].keys(),bar_list[n].values(),label=f'post_{n}')
        # plt.legend()
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # plt.show()

















        pass


    def concat_df(self):
        fdir = result_root + rf'pick_event\\extract_variables_during_droughts_GS\\'

        df_list=[]
        for f in os.listdir(fdir):
            if not f.endswith('.df'):
                continue
            df=T.load_df(fdir+f)
            df_list.append(df)
        df=pd.concat(df_list,axis=0)

        T.print_head_n(df)
        T.save_df(df, result_root + rf'pick_event\\extract_variables_during_droughts_GS\\concat_df.df')
        self.__df_to_excel(df, result_root + rf'pick_event\\extract_variables_during_droughts_GS\\concat_df.df')

    # pd.concat([df,df1],axis=1)

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
    def plot_df(self):
        f=result_root+'pick_event\extract_variables_during_droughts_GS\\concat_df.df'
        df=T.load_df(f)
        time_range_list=['1982_2000','2001_2020']
        varname='LAI4g'

        threshold=[-4,-3,-2,-1]

        val_mean_dic = {}
        label_list = []


        for th in threshold:

            for time_range in time_range_list:

                th1=th
                th2=th+1
                if th2>threshold[-1]:
                    continue

                threshold_str=f'({th1},{th2})'

                column_name=f'{varname}_{threshold_str}_{time_range}'

                print(column_name)
                vals=df[column_name]
                vals=T.remove_np_nan(vals)
                vals=np.array(vals)
                val_mean=np.nanmean(vals)
                val_mean_dic[column_name]=val_mean
                label_list.append(join(column_name.split('_')[1],column_name.split('_')[2]))
        plt.bar(range(len(val_mean_dic)),val_mean_dic.values(),tick_label=label_list)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plt_spatial_df(self):


        flag=1
        plt.figure(figsize=(10, 5))

        for n in [0,1,2,3]:
            arr_list = []
            plt.subplot(2, 2, flag)

            for period in ['1982_2000','2001_2020']:

                f=result_root+f'pick_event_scheme2\extract_variables_after_droughts_GS\\{period}_(-2,-1).df'
                f = result_root + f'pick_event_scheme2\extract_variables_after_droughts_GS_in_nth_year\\{period}_(-2,-1).df'
                df=T.load_df(f)
                print(df)
                # col_name=rf'NDVI4g_(-2,-1)_{period}_post_{n}_GS'
                col_name = rf'NDVI4g_(-2,-1)_{period}_post_{n}_GS'

                dic={}
                dic_group=T.df_groupby(df,'pix')
                for pix in dic_group:
                    df_i=dic_group[pix]
                    val=df_i[col_name].tolist()
                    mean=np.nanmean(val)
                    dic[pix]=mean
                arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(dic)
                arr_list.append(arr)
            arr_difference=(arr_list[1]-arr_list[0])
                # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr,result_root+'pick_event\extract_variables_during_droughts_GS\\1982_2000_(-3,-2).tif')
            plt.imshow(arr_difference,vmin=-0.5,vmax=0.5,cmap='RdBu',interpolation='nearest')
            plt.colorbar()
            title=f'NDVI4g_(-2,-1)_post_{n}_GS'
            plt.title(title)
            flag=flag+1
        plt.show()

    def rename_variables(self):
        fdir = result_root + rf'pick_event_scheme2\\extract_variables_after_droughts_GS_in_nth_year_rename\\'
        outdir = result_root + rf'pick_event_scheme2\\extract_variables_after_droughts_GS_in_nth_year_rename2\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not f.endswith('.df'):
                continue
            df=T.load_df(fdir+f)
            rename_dic={}
            for col in df.columns:
                if '_nth' in col:
                    rename_dic[col]=col.replace('_nth','')
            df=df.rename(columns=rename_dic)
            T.print_head_n(df)
            T.save_df(df,outdir+f)
            self.__df_to_excel(df,outdir+f)


def global_get_gs(pix):
    global_northern_hemi_gs = (5, 6, 7, 8, 9, 10)
    global_southern_hemi_gs = (11, 12, 1, 2, 3, 4)
    tropical_gs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    r, c = pix
    if r < 240:
        return global_northern_hemi_gs
    elif 240 <= r < 480:
        return tropical_gs
    elif r >= 480:
        return global_southern_hemi_gs
    else:
        raise ValueError('r not in range')



class build_dataframe():


    def __init__(self):

        self.this_class_arr = result_root + 'Dataframe\\anomaly_trends\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'anomaly_trends.df'

        pass

    def run(self):

        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        df=self.foo2(df)
        # df=self.build_df(df)
        #
        # df=self.append_value(df)
        # df = self.add_detrend_zscore_to_df(df)
        df=self.add_trend_to_df(df)
        # df=self.add_AI_classfication(df)
        # df=self.add_SM_trend_label(df)

        # df = self.add_landcover_data_to_df(df)
        # df=self.add_landcover_classfication_to_df(df)

        # df=self.__rename_dataframe_columns(df)
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

    def __df_to_excel(self, df, dff, n=1000, random=True):
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
    def build_df(self, df):

        fdir = result_root + rf'extract_GS\TRENDY_LAI\S3\\'
        all_dic= {}
        for f in os.listdir(fdir):
            fpath=fdir+f
            if not fpath.endswith('.npy'):
                continue
            dic = T.load_npy(fpath)
            key_name=f.split('.')[0]
            all_dic[key_name]=dic
        # print(all_dic.keys())
        df=T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df


    def append_value(self, df):  ##补齐
        fdir = result_root + rf'extract_GS\TRENDY_LAI\S3\\'
        col_list=[]
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            col_name=f.split('.')[0]
            col_list.append(col_name)

        for col in col_list:
            vals_new=[]

            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'append {col}'):
                pix = row['pix']
                r, c = pix
                vals=row[col]
                if type(vals)==float:
                    vals_new.append(np.nan)
                    continue
                vals=np.array(vals)
                if len(vals)==38:
                    vals=np.append(vals,np.nan)
                vals_new.append(vals)
            df[col]=vals_new

        return df

        pass


    def foo1(self, df):

        f = result_root + rf'anomaly\LAI4g.npy'

        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 1981
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y + 1)
                y = y + 1

        df['pix'] = pix_list

        df['year'] = year
        df['LAI4g'] = change_rate_list
        return df

    def foo2(self, df):  # 新建trend

        f = rf'D:\Project3\Result\trend_analysis\anomaly\ALL_ensemble\\LAI4g_trend.npy'
        val_array = np.load(f)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)

        # exit()

        pix_list = []
        for pix in tqdm(val_dic):
            pix_list.append(pix)
        df['pix'] = pix_list

        return df

    def add_detrend_zscore_to_df(self, df):
        fdir = result_root + rf'extract_GS\TRENDY_LAI\S1\\'

        for f in os.listdir(fdir):

            variable= f.split('.')[0]


            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row['year']
                # pix = row.pix
                pix = row['pix']
                r, c = pix
                if r <480:
                    continue

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
                print(v1,year,len(vals))
                exit()

                NDVI_list.append(v1)
            df[variable] = NDVI_list
        exit()
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


class plot_dataframe():
    def __init__(self):
        scenario='S3'
        self.product_list = [ f'CABLE-POP_{scenario}_lai', f'CLASSIC_{scenario}_lai', 'CLM5',  f'IBIS_{scenario}_lai', f'ISAM_{scenario}_lai',
             f'ISBA-CTRIP_{scenario}_lai', f'JSBACH_{scenario}_lai', f'JULES_{scenario}_lai',  f'LPJ-GUESS_{scenario}_lai', f'LPX-Bern_{scenario}_lai',
             f'ORCHIDEE_{scenario}_lai', f'SDGVM_{scenario}_lai', f'YIBs_{scenario}_Monthly_lai']

        pass
    def run(self):

        # self.plot_annual_zscore_based_region()

        # self.plot_anomaly_trendy()
        # self.plot_anomaly_vegetaton_indices()
        # self.plot_climatic_factors()
        # self.plot_plant_fuctional_types_trend()
        # self.plot_trend_spatial_all()
        self.plot_trend_regional()
        # self.plot_trend()

        # self.plot_trend_spatial()
        self.plot_browning_greening()
        # self.plot_original_data()
        pass


    def plot_annual_zscore_based_region(self):   #based on semi-arid, arid and sub-humid
        df= T.load_df(result_root + 'Dataframe\zscore\zscore.df')

        product_list = ['LAI4g','NDVI4g','GPP_CFE','GPP_baseline']

        fig = plt.figure()
        i = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:

            ax = fig.add_subplot(2, 2, i)

            flag = 0
            color_list=['blue','green','red','orange']

            for variable in product_list:

                colunm_name = variable
                df_region = df[df['AI_classfication'] == region]
                mean_value_yearly, up_list, bottom_list, fit_value_yearly, k_value, p_value = self.calculation_annual_average(df_region, colunm_name)
                xaxis = range(len(mean_value_yearly))
                xaxis = list(xaxis)

                ax.plot(xaxis, mean_value_yearly, label=variable, color=color_list[flag])
                # ax.plot(xaxis, fit_value_yearly, label='k={:0.2f},p={:0.4f}'.format(k_value, p_value), linestyle='--')

                # print(f'{region}_{variable}', 'k={:0.2f},p={:0.4f}'.format(k_value, p_value))
                flag = flag + 1


            plt.legend()
            plt.xlabel('year')
            plt.title(f'{region}')
            # create xticks

            yearlist = list(range(1982, 2021))
            yearlist_str = [int(i) for i in yearlist]
            ax.set_xticks(xaxis[::5])
            ax.set_xticklabels(yearlist_str[::5], rotation=45)


            major_yticks = np.arange(-1.1, 1)
            ax.set_yticks(major_yticks)

            plt.grid(which='major', alpha=0.5)
            plt.tight_layout()
            i = i + 1

        plt.show()



    def plot_anomaly_trendy(self):

        df= T.load_df(result_root + 'Dataframe\\growing_season_original\\growing_season_original.df')

        #create color list with one green and another 14 are grey

        # color_list=['grey']*16
        # # color_list[0]='green'
        # # color_list[1]='black'
        # linewidth_list=[1]*16
        # # linewidth_list[0]=3
        # # linewidth_list[1]=3

        fig = plt.figure()
        i = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]
            ax = fig.add_subplot(1, 3, i)
            for product in self.product_list:
                # if 'SDGVM' in product:
                #     continue
                print(product)
                vals=df_region[product].tolist()
                vals_nonnan=[]
                for val in vals:
                    if type(val)==float: ## only screening
                        continue
                    vals_nonnan.append(val)
                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)

                # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
                plt.plot(vals_mean,label=product)
                plt.scatter(range(len(vals_mean)),vals_mean)
                plt.text(0,vals_mean[0],product,fontsize=8)
            i=i+1

            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), rotation=45)

            plt.xlabel('year')

            plt.ylabel('delta LAI (m3/m3/year)')
            # plt.legend()

            plt.title(region)
        plt.show()
    def plot_anomaly_vegetaton_indices(self):
        vegetation_list=['NDVI4g','LAI4g','GPP_CFE','GPP_baseline']

        df= T.load_df(result_root + 'Dataframe\\zscore\\zscore.df')



        color_list=['red','green','blue','orange']

        fig = plt.figure()
        i = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]
            ax = fig.add_subplot(1, 3, i)
            flag=0
            for variable in vegetation_list:

                print(variable)
                vals=df_region[variable].tolist()
                vals_nonnan=[]
                for val in vals:
                    if type(val)==float: ## only screening
                        continue
                    vals_nonnan.append(val)
                vals_mean=np.nanmean(vals_nonnan,axis=0)  ## axis=0, mean of each row  竖着加
                plt.plot(vals_mean,label=variable,color=color_list[flag],)
                flag=flag+1

            i=i+1

            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), rotation=45)

            plt.xlabel('year')

            plt.ylabel('zscore')

            plt.title(region)
        plt.legend()
        plt.show()
    def plot_climatic_factors(self):
        climatic_factors=['SPEI3',]
        climatic_factors=['GPCC','Precip',]

        df= T.load_df(result_root + 'Dataframe\\extract_GS\\original.df')

        #create color list with one green and another 14 are grey

        color_list=['grey']*16
        color_list[0]='green'
        color_list[1]='black'
        linewidth_list=[1]*16
        linewidth_list[0]=2
        linewidth_list[1]=2

        fig = plt.figure()
        i = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]
            ax = fig.add_subplot(1, 3, i)
            for factor in climatic_factors:
                print(factor)
                vals=df_region[factor].tolist()
                vals_nonnan=[]
                for val in vals:
                    if type(val)==float: ## only screening
                        continue
                    vals_nonnan.append(val)
                vals_mean=np.nanmean(vals_nonnan,axis=0)  ## axis=0, mean of each row  竖着加
                plt.plot(vals_mean,label=factor,color=color_list[climatic_factors.index(factor)],linewidth=linewidth_list[climatic_factors.index(factor)])
                # plt.text(0,vals_mean[0],product,fontsize=8)
            i=i+1

            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), rotation=45)

            plt.xlabel('year')

            plt.ylabel('SM (m3/m3/)')


            plt.title(region)
        plt.legend()
        plt.show()

    def plot_plant_fuctional_types_trend(self):

        df = T.load_df(result_root + 'Dataframe\\anomaly\\anomaly.df')
        product_list= ['LAI4g', 'NDVI4g', 'GPP_CFE', 'GPP_baseline']

        # create color list with one green and another 14 are grey

        color_list = ['red', 'green', 'blue', 'orange']

        fig = plt.figure()
        flag = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]


            ax = fig.add_subplot(1, 3, flag)
            average_dic = {}
            for pft in ['Evergreen','Deciduous','Mixed','Shrub','Grass','Cropland']:
                df_pft=df_region[df_region['landcover_classfication']==pft]

                average_list=[]

                for product in product_list:

                    vals = df_pft[f'{product}_trend_zscore'].tolist()

                    average_val=np.nanmean(vals)
                    average_list.append(average_val)
                average_dic[pft]=average_list

            df_new=pd.DataFrame(average_dic,index=product_list)
            T.print_head_n(df_new)
            df_new=df_new.T
            T.print_head_n(df_new)
            df_new.plot.bar(ax=ax)
            plt.title(region)
            plt.legend()
            plt.ylim(-0.002,0.007)

            flag = flag + 1
            plt.tight_layout()
        plt.show()

    def plot_trend_spatial_all(self):  ## S0, S1, S2, S3 average spatial

        df = T.load_df(result_root + '\\Dataframe\\anomaly_trends\\anomaly_trends.df')
        product_list = ['LAI4g_trend','Ensemble_S0_trend',  'Ensemble_S1_trend', 'Ensemble_S2_trend', 'Ensemble_S3_trend', 'Ensemble_S1-S0_trend', 'Ensemble_S2-S1_trend','Ensemble_S3-S2_trend']
        label_list=['OBS', 'none', 'CO2+Ndep', 'CO2+CLI+Ndep', 'CO2+CLI+LULCC+Nfert+Ndep', 'CO2&Ndep', 'CLIM','LULCC']

        color_list = ['red', 'green', 'blue', 'orange', 'black', 'grey', 'yellow', 'pink', 'purple']


        flag=1
        average_list=[]

        for product in product_list:

            vals = df[f'{product}'].tolist()
            average_val = np.nanmean(vals)
            average_list.append(average_val)
            plt.bar(product,average_val,color=color_list[flag-1],label=label_list[flag-1])
            plt.xticks(range(len(label_list)),label_list,rotation=90)
            plt.ylabel('Trend in LAI (m2/m2/year)')

            # set xticks as xlabel

            plt.tight_layout()
            flag=flag+1
        plt.show()

    def plot_trend_regional(self):  ## S0, S1, S2, S3 average regional

        df = T.load_df(result_root + '\\Dataframe\\anomaly_trends\\anomaly_trends.df')
        product_list = ['LAI4g_trend','Ensemble_S0_trend',  'Ensemble_S1_trend', 'Ensemble_S2_trend', 'Ensemble_S3_trend', 'Ensemble_S1-S0_trend', 'Ensemble_S2-S1_trend','Ensemble_S3-S2_trend']
        label_list=['OBS', 'none', 'CO2+Ndep', 'CO2+CLI+Ndep', 'CO2+CLI+LULCC+Nfert+Ndep', 'CO2&Ndep', 'CLIM','LULCC']

        color_list = ['red', 'green', 'blue', 'orange', 'black', 'grey', 'yellow', 'pink', 'purple']

        fig= plt.figure()
        flag=1
        average_list=[]
        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]
            ax = fig.add_subplot(1, 3, flag)
            average_list = []
            for product in product_list:

                vals = df_region[f'{product}'].tolist()
                average_val = np.nanmean(vals)
                average_list.append(average_val)
            plt.bar(label_list,average_list,color=color_list,label=label_list)
            plt.xticks(range(len(label_list)),label_list,rotation=90)
            plt.ylabel('Trend in LAI (m2/m2/year)')
            plt.title(region)

            # set xticks as xlabel

            plt.tight_layout()
            flag=flag+1
        plt.show()

    def plot_trend(self):   ## slides 22

        df = T.load_df(result_root + '\\Dataframe\\anomaly_trends\\anomaly_trends.df')
        product_list = ['LAI4g_trend', 'Ensemble_S0_trend', 'Ensemble_S1_trend', 'Ensemble_S2_trend',
                        'Ensemble_S3_trend', 'Ensemble_S1-S0_trend', 'Ensemble_S2-S1_trend', 'Ensemble_S3-S2_trend']
        label_list = ['OBS', 'none', 'CO2+Ndep', 'CO2+CLI+Ndep', 'CO2+CLI+LULCC+Nfert+Ndep', 'CO2&Ndep', 'CLIM',
                      'LULCC']

        color_list = ['red', 'green', 'blue', 'orange','black', 'grey', 'yellow', 'pink', 'purple']
        period_list = ['1982_2020']

        fig = plt.figure()
        flag = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]

            ax = fig.add_subplot(1, 3, flag)
            average_dic = {}
            for period in period_list:

                average_list = []

                for product in product_list:
                    # vals = df_region[f'{product}_{period}_trend_zscore'].tolist()
                    vals = df_region[f'{product}_trend_{period}'].tolist()
                    average_val = np.nanmean(vals)
                    average_list.append(average_val)
                average_dic[period] = average_list

            df_new = pd.DataFrame(average_dic, index=product_list)
            T.print_head_n(df_new)
            df_new = df_new.T
            T.print_head_n(df_new)
            df_new.plot.bar(ax=ax)
            plt.title(region)
            # plt.ylabel('trend (unitless)')
            plt.ylabel('trend')

            plt.ylim(-15, 15)

            flag = flag + 1

            plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_trend_spatial(self):   ##for figure slides 23

        df = T.load_df(result_root + 'Dataframe\\anomaly_trends\\anomaly_trends.df')
        product_list = ['LAI4g', 'Ensemble_S0', 'Ensemble_S1', 'Ensemble_S2','Ensemble_S3']



        color_list = ['red', 'green', 'blue', 'orange', 'black', 'grey', 'yellow', 'pink', 'purple']

        fig = plt.figure()
        flag = 1

        for product in product_list:

            ax = fig.add_subplot(3, 3, flag)
            average_dic = {}

            for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
                df_region = df[df['AI_classfication'] == region]

                vals = df_region[f'{product}'].tolist()
                average_val = np.nanmean(vals)

                average_dic[region] = average_val



        for region in average_dic:

            val=average_dic[region]
            #label rotation

            plt.bar(region,val,color=color_list[flag-1],label=region,)
            plt.xticks(rotation=45)

            plt.title(region)
            plt.ylabel('trend (m3/m3/year)')
            # plt.ylabel('trend (kgC/m2/year-2)')
            # plt.ylabel('trend (mm/year)')
            plt.ylim(-0.001, 0.001)

            # plt.ylim(-0.1, 0.1)

            flag=flag+1
        plt.show()



    def plot_browning_greening(self):
        df= T.load_df(result_root + 'Dataframe\\original_trend\\original_trend.df')
        # product_list = ['LAI4g', 'NDVI4g', 'GPP_CFE', 'GPP_baseline']
        color_list=['green','lime','orange','red']

        product='GPP_CFE'
        landcover_list=['Evergreen','Deciduous','Mixed','Shrub','Grass','Cropland']
        period_list=['1982_2000','2001_2020']
        # period_list=['1982_2020']


        period_dic= {}

        for period in period_list:
            region_dic={}

            for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
                dic={}

                df_region = df[df['AI_classfication'] == region]


                for landcover in landcover_list:
                    df_lc=df_region[df_region['landcover_classfication']==landcover]


                    # for product in product_list:


                    vals=df_lc[f'{product}_{period}_trend'].tolist()
                    p_values=df_lc[f'{product}_{period}_p_value'].tolist()


                    sig_browning=0
                    sig_greening=0
                    non_sig_browning=0
                    non_sig_greening=0

                    for i in range(len(vals)):
                        if p_values[i]<0.05:
                            if vals[i]<0:
                                sig_browning=sig_browning+1
                            else:
                                sig_greening=sig_greening+1
                        else:
                            if vals[i]<0:
                                non_sig_browning=non_sig_browning+1
                            else:
                                non_sig_greening=non_sig_greening+1
                            ##percentage
                    if len(vals)==0:
                        continue
                    sig_browning=sig_browning/len(vals)
                    sig_greening=sig_greening/len(vals)
                    non_sig_browning=non_sig_browning/len(vals)
                    non_sig_greening=non_sig_greening/len(vals)


                    dic[landcover]={'sig_greening':sig_greening,'non_sig_greening':non_sig_greening,'non_sig_browning':non_sig_browning,'sig_browning':sig_browning}
                df_new=pd.DataFrame(dic)
                region_dic[region]=df_new
                period_dic[period]=region_dic
                # print(period_dic)

        flag = 1
        fig = plt.figure()

        for period in period_list:
            region_dic=period_dic[period]

            for region in region_dic:
                ax = fig.add_subplot(2, 3, flag)
                df_new=region_dic[region]
                T.print_head_n(df_new)
                df_new_T=df_new.T
                T.print_head_n(df_new_T)
                df_new_T.plot.bar(ax=ax,stacked=True,color=color_list,legend=False)
                plt.title(f'{region}_{period}')
                plt.suptitle(product)
                plt.ylabel('percentage')

                flag = flag + 1
                plt.tight_layout()
        # plt.legend()
        plt.show()
                # exit()










    def plot_original_data(self):
        df = T.load_df(result_root + 'Dataframe\\extract_GS\\growing_season_original.df')
        scenario='S1'
        region_annual_dic = {}
        product_list =  [f'CABLE-POP_{scenario}_lai', f'CLASSIC_{scenario}_lai', 'CLM5', f'DLEM_{scenario}_lai', f'IBIS_{scenario}_lai', f'ISAM_{scenario}_lai',
             f'ISBA-CTRIP_{scenario}_lai', f'JSBACH_{scenario}_lai', f'JULES_{scenario}_lai',  f'LPJ-GUESS_{scenario}_lai', f'LPX-Bern_{scenario}_lai',
             f'ORCHIDEE_{scenario}_lai', f'SDGVM_{scenario}_lai', f'YIBs_{scenario}_Monthly_lai']
        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]

            product_annual_dic = {}
            for product in product_list:

                vals = df_region[product].tolist()
                vals_nonnan = []
                for val in vals:
                    if type(val) == float:
                        continue
                    vals_nonnan.append(val)
                vals_mean = np.nanmean(vals_nonnan, axis=0)
                product_annual_dic[product] = vals_mean
            region_annual_dic[region] = product_annual_dic


        fig = plt.figure()
        flag = 1

        for region in region_annual_dic:
            product_annual_dic = region_annual_dic[region]
            vals_mean_two = []
            product_list = []
            ax = fig.add_subplot(3, 1, flag)
            for product in product_annual_dic:

                vals_mean = product_annual_dic[product]
                vals_mean_two.append(vals_mean)
                product_list.append(product)
                ## plot double y axis
                ##twinsx

            ax.plot(vals_mean_two[0], label=product_list[0], color='red')
            ax2 = ax.twinx()
            ax2.plot(vals_mean_two[1], label=product_list[1], color='blue')
            # yminax=min(vals_mean_two[0])-50
            # ymaxax=max(vals_mean_two[0])+50

            yminax = min(vals_mean_two[0]) - 0.1
            ymaxax = max(vals_mean_two[0]) + 0.1
            yminax2 = min(vals_mean_two[1]) - 0.1
            ymaxax2 = max(vals_mean_two[1]) + 0.1



            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), )
            ax.set_ylim(yminax,ymaxax)
            ax2.set_ylim(yminax2, ymaxax2)
            # ax2.set_ylim(yminax,ymaxax)
            # ax.set_ylabel('GPP_CFE (gC/m2/year)')
            # ax2.set_ylabel('GPP_baseline(gC/m2/year)')
            ax.set_ylabel('LAI4g (m2/m2)')
            ax2.set_ylabel('NDVI4g ')

            plt.title(region)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

            flag = flag + 1

        plt.show()







            ## plot double y axis
        #     ax = fig.add_subplot(3, 1, flag)
        #     df_new.plot(ax=ax, secondary_y=['GPP_CFE'])
        #
        #     flag = flag + 1
        # plt.show()




                ##plot double y axis


        pass



    def calculation_annual_average(self,df,column_name):
        dic = {}
        mean_val = {}
        confidence_value = {}
        std_val = {}
        # year_list = df['year'].to_list()
        # year_list = set(year_list)  # 取唯一
        # year_list = list(year_list)
        # year_list.sort()

        year_list = []
        for i in range(1982, 2021):
            year_list.append(i)
        print(year_list)

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []
            confidence_value[year] = []

        for year in year_list:
            df_pick = df[df['year'] == year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row[column_name]
                dic[year].append(val)
            val_list = np.array(dic[year])
            # val_list[val_list>1000]=np.nan

            n = len(val_list)
            mean_val_i = np.nanmean(val_list)
            std_val_i = np.nanstd(val_list)
            se = stats.sem(val_list)
            h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
            confidence_value[year] = h
            mean_val[year] = mean_val_i
            std_val[year] = std_val_i

        # a, b, r = KDE_plot().linefit(xaxis, val)
        mean_val_list = []  # mean_val_list=下面的mean_value_yearly

        for year in year_list:
            mean_val_list.append(mean_val[year])
        xaxis = range(len(mean_val_list))
        xaxis = list(xaxis)
        print(len(mean_val_list))
        # r, p_value = stats.pearsonr(xaxis, mean_val_list)
        # k_value, b_value = np.polyfit(xaxis, mean_val_list, 1)
        k_value, b_value, r, p_value = T.nan_line_fit(xaxis, mean_val_list)
        print(k_value)

        mean_value_yearly = []
        up_list = []
        bottom_list = []
        fit_value_yearly = []
        p_value_yearly = []

        for year in year_list:
            mean_value_yearly.append(mean_val[year])
            # up_list.append(mean_val[year] + confidence_value[year])
            # bottom_list.append(mean_val[year] - confidence_value[year])
            up_list.append(mean_val[year] + 0.125 * std_val[year])
            bottom_list.append(mean_val[year] - 0.125 * std_val[year])

            fit_value_yearly.append(k_value * (year - year_list[0]) + b_value)



        return mean_value_yearly, up_list, bottom_list, fit_value_yearly, k_value, p_value
        # exit()

class check_data():
    def run (self):
        # self.plot_sptial()
        self.testrobinson()
        # self.plot_time_series()

        pass
    def plot_sptial(self):

        f =  rf'D:\Project3\Result\anomaly\S0\\SDGVM_S0_lai.npy'
        dic=T.load_npy(f)
        # dic = {}
        # for f in os.listdir(fdir):
        #     if not f.endswith(('.npy')):
        #         continue
        #
        #     dic_i=T.load_npy(fdir+f)
        #     dic.update(dic_i)

        len_dic={}
        for pix in dic:
            vals=dic[pix]

            # len_dic[pix]=np.nanmean(vals)
            len_dic[pix] = len(vals)
        arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(len_dic)
        plt.imshow(arr,cmap='RdBu',interpolation='nearest')
        plt.colorbar()
        plt.title('GPP_CFE_1982')
        plt.show()
    def testrobinson(self):

        f = result_root + 'trend_analysis\\anomaly\ALL_ensemble\\Ensemble_S1-S0_trend.tif'
        # f = data_root + rf'split\NDVI4g\2001_2020.npy'
        Plot().plot_Robinson(f, vmin=-0.01,vmax=0.01,is_discrete=True,colormap_n=7)
        plt.title('GPP_baseline(kg/m2/year)')

        plt.show()



    def plot_time_series(self):
        f=rf'D:\Project3\Result\anomaly\S0\\SDGVM_S1_lai.npy'
        # f= result_root+ rf'detrend_zscore_Yang\LAI4g\\1982_2000.npy'
        dic=T.load_npy(f)
        for pix in dic:
            vals=dic[pix]
            vals=np.array(vals)
            # if not len(vals)==19*12:
            #     continue
            # if True in np.isnan(vals):
            #     continue
            # print(len(vals))
            if np.isnan(np.nanmean(vals)):
                continue
            plt.plot(vals)
            plt.show()


class Dataframe_func:

    def run (self):
        fdir = result_root + rf'pick_event_scheme2\extract_variables_after_droughts_GS\\'
        for f in os.listdir(fdir):
            if not f.endswith('.df'):
                continue
            df=T.load_df(fdir+f)
            print('add AI')
            df=self.add_AI_to_df(df)
            print('add AI_reclass')
            df=self.AI_reclass(df)
            print('add SM_trend_label')
            df=self.add_SM_trend_label(df)
            T.save_df(df,fdir+f)
            self.__df_to_excel(df,fdir+f)
        pass


    # def __init__(self,df,is_clean_df=True):
    #
    #     # print('add lon lat')
    #     # df = self.add_lon_lat(df)
    #
    #     # if is_clean_df == True:
    #     #     df = self.clean_df(df)
    #
    #     # print('add landcover')
    #     # df = self.add_GLC_landcover_data_to_df(df)
    #
    #     print('add Aridity Index')
    #     df = self.add_AI_to_df(df)
    #
    #     print('add AI_reclass')
    #     df = self.AI_reclass(df)
    #     print('add SM_trend_label')
    #     df = self.add_SM_trend_label(df)
    #
    #
    #     self.df = df
    #
    # def clean_df(self,df):
    #
    #     df = df[df['lat']>=30]
    #     # df = df[df['landcover_GLC'] != 'Crop']
    #     df = df[df['NDVI_MASK'] == 1]
    #     # df = df[df['ELI_significance'] == 1]
    #     return df

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

    def add_NDVI_mask(self,df):
        # f =rf'C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        f = join(data_root, 'Base_data', 'NDVI_mask.tif')
        print(f)

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
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

    def add_AI_to_df(self, df):
        f = join(data_root, 'Base_data/dryland_AI.tif/dryland.tif')
        spatial_dict = DIC_and_TIF().spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'HI_class')
        return df

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        return df


    def AI_reclass(self,df):
        AI_class = []
        for i,row in df.iterrows():
            AI = row['HI_class']
            if AI < 0.65:
                AI_class.append('Dryland')
            elif AI >= 0.65:
                AI_class.append('Humid')
            elif np.isnan(AI):
                AI_class.append(np.nan)
            else:
                print(AI)
                raise ValueError('AI error')
        df['HI_class'] = AI_class
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





def main():
    data_processing().run()
    # statistic_analysis().run()
    # pick_event().run()
    # selection().run()
    # multi_regression().run()
    # fingerprint().run()
    # moving_window().run()
    # multi_regression_window().run()
    # build_dataframe().run()
    # plot_dataframe().run()
    # check_data().run()
    # Dataframe_func().run()



    pass

if __name__ == '__main__':
    main()