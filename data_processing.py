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
        # self.download_CCI_landcover()
        # self.download_ERA_precip()
        # self.nc_to_tif()
        # self.nc_to_tif_NIRv()
        # self.nc_to_tif_Terraclimate()
        # self.nc_to_tif_TRMM()
        # self.nc_to_tif_LUCC()
        # self.nc_to_tif_inversion_model()
        # self.TRMM_download()

        # self.check_tif_length()
        # self.daily_to_monthly()

        # self.resample_trendy()
        # self.resample_AVHRR_LAI()
        # self.resample_GIMMS3g()
        # self.resample_inversion()
        # self.aggregate_GIMMS3g()
        # self.aggreate_AVHRR_LAI() ## this method is used to aggregate AVHRR LAI to monthly
        # self.unify_TIFF()
        # self.average_temperature()
        # self.scales_Inversion()


        # self.trendy_ensemble_calculation()  ##这个函数不用 因为ensemble original data 会出现最后一年加入数据不全，使得最后一年得知降低


        # self.tif_to_dic()

        # self.extract_GS()
        self.extract_GS_return_monthly_data()
        # self.extract_seasonality()

        # self.extend_GS() ## for SDGVM， it has 37 year GS, to align with other models, we add one more year
        # self.extend_nan()  ##  南北半球的数据不一样，需要后面加nan
        # self.scales_GPP_Trendy()
        # self.split_data()


    def download_CCI_landcover(self):

        import cdsapi


        c = cdsapi.Client()

        for year in range(2011,2012):

            c.retrieve(
                'satellite-land-cover',
                {
                    'year': str(year),

                    # 'version': 'v2.1.1',
                    'version': 'v2.0.7cds',
                    'variable': 'all',
                    'format': 'zip',
                },

                rf'C:\Users\wenzhang1\Desktop\CCI_landcover/CCI_LC_{year:04d}.zip')


    def TRMM_download(self):
        import requests
        from requests.auth import HTTPBasicAuth
        outdir = 'D:\Project3\Data\TRMM\\nc\\'
        T.mk_dir(outdir, force=True)
        urlf='D:\Project3\Data\TRMM\\subset_TRMM_3B42_Daily_7_20240208_194343_.txt'
        with open(urlf) as f:
            lines=f.readlines()
            for line in tqdm(lines):
                line=line.strip()
                # print(line)
                url=line.split(' ')[-1]
                # print(url)
                username = "leeyang1991@gmail.com"
                password = "Asdfasdf911007"

                # Send a GET request to the URL with basic authentication
                response = requests.get(url, auth=HTTPBasicAuth(username, password))
                print(response.status_code)



                content=response.content
                # print(content)
                outf=outdir+url.split('/')[-1]
                # print(outf)
                if os.path.isfile(outf):
                    continue
                with open(outf,'wb') as f:
                    f.write(content)
                # print('done')
                # exit()




    def download_ERA_precip(self):
        outdir='D:\Project3\Data\ERA5\\nc\\'
        T.mk_dir(outdir,force=True)

        import cdsapi

        c = cdsapi.Client()

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'grib',
                'variable': 'total_precipitation',
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'year': [
                    '1982', '1983', '1984',
                    '1985', '1986', '1987',
                ],
            },
            'D:\Project3\Data\ERA5\\nc\\total_precipitation.nc')

        pass




        pass
    def nc_to_tif(self):

        # fdir=data_root+f'\GPP\\NIRvGPP\\nc\\'
        fdir=rf'D:\Project3\Data\TRMM\nc\\'
        outdir=rf'D:\Project3\Data\TRMM\\TIFF\\'
        Tools().mk_dir(outdir,force=True)
        for f in os.listdir(fdir):


            outdir_name = f.split('.')[0]
            print(outdir_name)

            yearlist = list(range(1982, 2021))


            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir+f, var_name='precipitation', outdir=outdir, yearlist=yearlist)
            except Exception as e:
                print(e)
                continue

    def nc_to_tif_NIRv(self):

        fdir = rf'C:\Users\wenzhang1\Desktop\Terra\nc\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\Terra\TIFF\\\\'
        Tools().mk_dir(outdir, True)
        for f in os.listdir(fdir):
            if not f.endswith('.nc'):
                continue
            if not 'tmin' in f:
                continue
            outf=outdir+f.split('.')[0]+'.tif'
            if os.path.isfile(outf):
                continue

            nc = Dataset(fdir + f, 'r')

            print(nc)
            print(nc.variables.keys())
            # t = nc['time']
            # print(t)
            # start_year = int(t.units.split(' ')[-1].split('-')[0])

            # basetime = datetime.datetime(start_year, 1, 1)  # 告诉起始时间
            lat_list = nc['lat']
            lon_list = nc['lon']
            # lat_list=lat_list[::-1]  #取反
            print(lat_list[:])
            print(lon_list[:])

            origin_x = lon_list[0]  # 要为负数-180
            origin_y = lat_list[0]  # 要为正数90
            pix_width = lon_list[1] - lon_list[0]  # 经度0.5
            pix_height = lat_list[1] - lat_list[0]  # 纬度-0.5
            print(origin_x)
            print(origin_y)
            print(pix_width)
            print(pix_height)
            # SIF_arr_list = nc['SIF']
            SPEI_arr_list = nc['tmin']
            print(SPEI_arr_list.shape)
            print(SPEI_arr_list[0])
            # plt.imshow(SPEI_arr_list[5])
            # # plt.imshow(SPEI_arr_list[::])
            # plt.show()

            year=int(f.split('.')[-3][0:4])
            month=int(f.split('.')[-3][4:6])

            fname = '{}{:02d}.tif'.format(year, month)
            print(fname)
            newRasterfn = outdir + fname
            print(newRasterfn)
            longitude_start = origin_x
            latitude_start = origin_y
            pixelWidth = pix_width
            pixelHeight = pix_height
            # array = val
            # array=SPEI_arr_list[::-1]
            array = SPEI_arr_list

            array = np.array(array)
            # method 2
            array = array.T
            array=array* 0.001  ### GPP need scale factor
            array[array < 0] = np.nan

            # plt.imshow(array)
            # plt.colorbar()
            # plt.show()
            ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, )

    def nc_to_tif_TRMM(self):  #### nc4 file
        import netCDF4


        fdir = rf'D:\Project3\Data\TRMM\nc\\'
        outdir = rf'D:\Project3\Data\TRMM\TIFF\\\\'
        Tools().mk_dir(outdir, True)
        for f in os.listdir(fdir):
            if not f.endswith('.nc4'):
                continue

            outf=outdir+f.split('.')[0]+'.tif'
            if os.path.isfile(outf):
                continue

            ### read nc.4 file
            nc = netCDF4. Dataset(fdir + f, 'r')

            print(nc)
            print(nc.variables.keys())
            # t = nc['time']
            # print(t)
            # start_year = int(t.units.split(' ')[-1].split('-')[0])

            # basetime = datetime.datetime(start_year, 1, 1)  # 告诉起始时间
            lat_list = nc['lat']
            lon_list = nc['lon']
            # lat_list=lat_list[::-1]  #取反
            print(lat_list[:])
            print(lon_list[:])

            origin_x = lon_list[0]  # 要为负数-180
            origin_y = lat_list[0]  # 要为正数90
            pix_width = lon_list[1] - lon_list[0]  # 经度0.5
            pix_height = lat_list[1] - lat_list[0]  # 纬度-0.5
            print(origin_x)
            print(origin_y)
            print(pix_width)
            print(pix_height)
            # SIF_arr_list = nc['SIF']
            SPEI_arr_list = nc['precipitation']
            print(SPEI_arr_list.shape)
            print(SPEI_arr_list[0])
            # plt.imshow(SPEI_arr_list[5])
            # # plt.imshow(SPEI_arr_list[::])
            # plt.show()

            year=int(f.split('.')[-3][0:4])
            month=int(f.split('.')[-3][4:6])

            fname = '{}{:02d}.tif'.format(year, month)
            print(fname)
            newRasterfn = outdir + fname
            print(newRasterfn)
            longitude_start = origin_x
            latitude_start = origin_y
            pixelWidth = pix_width
            pixelHeight = pix_height
            # array = val
            # array=SPEI_arr_list[::-1]
            array = SPEI_arr_list

            array = np.array(array)
            # method 2
            array = array.T

               ### mm/h to mm/year
            array=array* 365*24*3600
            array[array < 0] = np.nan

            plt.imshow(array)
            plt.colorbar()
            plt.show()
            ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, )


    def nc_to_tif_Terraclimate(self):


        fdir = rf'C:\Users\wenzhang1\Desktop\Terra\nc\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\Terra\TIFF\\vpd\\'
        Tools().mk_dir(outdir, True)
        for f in os.listdir(fdir):
            if not f.endswith('.nc'):
                continue
            if not 'vpd' in f:
                continue


            nc = Dataset(fdir + f, 'r')

            print(nc)
            print(nc.variables.keys())

            lat_list = nc['lat']
            lon_list = nc['lon']
            # lat_list=lat_list[::-1]  #取反
            print(lat_list[:])
            print(lon_list[:])

            origin_x = lon_list[0]  # 要为负数-180
            origin_y = lat_list[0]  # 要为正数90
            pix_width = lon_list[1] - lon_list[0]  # 经度0.5
            pix_height = lat_list[1] - lat_list[0]  # 纬度-0.5
            print(origin_x)
            print(origin_y)
            print(pix_width)
            print(pix_height)

            SPEI_arr_list = nc['vpd']
            print(SPEI_arr_list.shape)
            print(SPEI_arr_list[0])

            longitude_start = origin_x
            latitude_start = origin_y
            pixelWidth = pix_width
            pixelHeight = pix_height

            for ii in range(len(SPEI_arr_list)):
                arr=SPEI_arr_list[ii]
                year=int(f.split('.')[0].split('_')[-1])

                month= (ii+1)
                ##format month
                if month<10:
                    month='0'+str(month)
                else:
                    month=str(month)
                fname = '{}{}.tif'.format(year, month)



                print(fname)
                newRasterfn = outdir + fname
                print(newRasterfn)
                arr = np.array(arr)
                arr[arr<-100]=np.nan
                # method 2
                # array = array.T

                # plt.imshow(arr)
                # plt.colorbar()
                # plt.show()
                ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr, )

    def nc_to_tif_inversion_model(self):
        fdir = data_root + rf'Inversion\Carbontracker\nc\\'
        outdir = data_root + rf'\\Inversion\Carbontracker\\TIFF\\'
        Tools().mk_dir(outdir, True)

        for f in os.listdir(fdir):
            if not f.endswith('.nc'):
                continue
            nc = Dataset(fdir+f)

            print(nc)
            print(nc.variables.keys())
            t = nc['time']
            # print(t.units)
            # exit()
            # for ti in range(len(t)):
            #     print(t[ti])
            # # exit()
            # start_year = int(t.units.split(' ')[-1].split('-')[0])

            start_year = int(t.units.split(' ')[2].split('-')[0])   # 文件里面所定义的时间


            basetime = datetime.datetime(start_year, 1, 1)  # basetime 时间原点。 之前就是负数，之后就是正数 所有的delta 都是和他比
            lat_list = nc['latitude']
            lon_list = nc['longitude']
            lat_list=lat_list[::-1]  #取反
            print(lat_list[:])
            print(lon_list[:])

            origin_x = lon_list[0]  # 要为负数-180
            origin_y = lat_list[0]  # 要为正数90
            pix_width = lon_list[1] - lon_list[0]  # 经度0.5
            pix_height = lat_list[1] - lat_list[0]  # 纬度-0.5
            print(origin_x)
            print(origin_y)
            print(pix_width)
            print(pix_height)
            # SIF_arr_list = nc['SIF']
            SPEI_arr_list = nc['bio_flux_opt']
            print(SPEI_arr_list.shape)
            print(SPEI_arr_list[0])


            for i in range(len(SPEI_arr_list)):
                date_delta_i = t[i]
                print(date_delta_i)
                date_delta_i = datetime.timedelta(int(date_delta_i))  ##unit day
                # date_delta_i = datetime.timedelta(seconds=int(date_delta_i))##unit second

                date_i = basetime + date_delta_i
                print(date_i)
                # sleep()
                # if date_i.split('-')[0] not in date_list :
                #     continue
                # print(date_i)
                year = date_i.year
                month = date_i.month
                date = date_i.day
                fname = '{}{:02d}{:02d}.tif'.format(year, month, date)
                print(fname)
                newRasterfn = outdir + fname
                print(newRasterfn)
                longitude_start = origin_x
                latitude_start = origin_y

                pixelWidth = pix_width
                pixelHeight = pix_height
                # array = val
                array = SPEI_arr_list[i]
                array = np.array(array)
                array=array[::-1]
                # method 2
                #     array = array.T
                array[array < -100] = np.nan
                # plt.imshow(array)
                # plt.colorbar()
                # plt.show()
                ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, )

    def nc_to_tif_LUCC(self):

        fdir = data_root+'Base_data\\'
        year_list=list(range(1982,2015))


        for f in os.listdir(fdir):

            if f.startswith('.'):
                continue
            if not 'states' in f:
                continue

            nc=Dataset(fdir+f,'r')
            print(nc.variables.keys())
            outdir = data_root + 'Base_data\\c3ann\\'
            T.mk_dir(outdir, force=True)


            # check nc variables
            print(nc.variables.keys())
            ##check time_range
            time=nc.variables['time'][:]
            print(time)
            # exit()



            try:
                self.nc_to_tif_template(fdir + f, var_name='c3ann',outdir=outdir , yearlist=year_list)
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
        fdir_all = data_root + rf'GPP\S3\unify\\'
        outdir = data_root + rf'GPP\S3\monthly\\'
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
                        # arr_unify[arr_unify > 7] = np.nan
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

        fdir_all = data_root + rf'GIMMS3g\\'

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan

        year_list = list(range(1982, 2021))


        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            if not 'monthly_aggragate_GIMMS3g' in fdir:
                continue



            outdir = data_root + rf'GIMMS3g_DIC\\'
            if os.path.isdir(outdir):
                pass

            T.mk_dir(outdir, force=True)
            all_array = []  #### so important  it should be go with T.mk_dic

            for f in os.listdir(fdir_all+ fdir):
                if not f.endswith('.tif'):
                    continue
                if int(f.split('.')[0][0:4]) not in year_list:
                    continue


                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir_all + fdir + '\\' + f)
                array = np.array(array, dtype=float)


                array_unify = array[:720][:720,
                              :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

                array_unify[array_unify < -999] = np.nan
                # array_unify[array_unify > 7] = np.nan
                # array[array ==0] = np.nan

                # array_unify[array_unify <= 0] = np.nan

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

        fdir_all = data_root + rf'GIMMS3g\\'
        outdir = result_root + f'extract_GS\\OBS_LAI\\'
        Tools().mk_dir(outdir, force=True)
        date_list=[]

        # print(date_list)
        # exit()

        for year in range(1982, 2021):
            for mon in range(1, 13):
                date_list.append(datetime.datetime(year, mon, 1))

        for fdir in os.listdir(fdir_all):
            if not 'GIMMS3g_DIC' in fdir:
                continue


            spatial_dict = {}
            outf = outdir + fdir.split('.')[0] + '.npy'

            if os.path.isfile(outf):
                continue
            print(outf)


            for f in os.listdir(fdir_all + fdir):

                spatial_dict_i =dict(np.load(fdir_all + fdir + '\\' + f, allow_pickle=True, ).item())
                spatial_dict.update(spatial_dict_i)

            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict):
                r,c=pix


                gs_mon = global_get_gs(pix)

                vals = spatial_dict[pix]
                vals = np.array(vals)
                # vals[vals == 65535] = np.nan
                #
                # vals = np.array(vals)/100

                vals[vals < -999] = np.nan
                # vals[vals > 7] = np.nan

                # vals[vals<0]=np.nan

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

    def extract_GS_return_monthly_data(self):  ## extract growing season but return monthly data

        fdir_all = data_root + rf'monthly_data\\'
        outdir = result_root + f'extract_GS_return_monthly_data\\OBS_LAI\\'
        Tools().mk_dir(outdir, force=True)
        date_list=[]

        # print(date_list)
        # exit()

        for year in range(1982, 2021):
            for mon in range(1, 13):
                date_list.append(datetime.datetime(year, mon, 1))

        for fdir in os.listdir(fdir_all):
            if not fdir in ['LAI4g']:
                continue


            outf = outdir + fdir.split('.')[0] + '.npy'
            print(outf)

            spatial_dict = T.load_npy_dir(fdir_all + fdir)

            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict):
                r,c=pix


                gs_mon = global_get_gs(pix)

                vals = spatial_dict[pix]
                vals = np.array(vals)
                # vals[vals == 65535] = np.nan
                #
                # vals = np.array(vals)/100

                vals[vals < -999] = np.nan
                # vals[vals > 7] = np.nan

                # vals[vals<0]=np.nan

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
                    ## return monthly data
                    annual_gs_list.append(vals_gs)


                annual_gs_list = np.array(annual_gs_list)

                # if T.is_all_nan(annual_gs_list):
                #     continue
                annual_spatial_dict[pix] = annual_gs_list

            np.save(outf, annual_spatial_dict)

        pass

    def extract_seasonality(self):  ## here using new extraction method: 240<r<480 all year growing season

        fdir_all = data_root + rf'monthly_data\\'
        outdir=result_root + f'extract_GS\\OBS_LAI_seasonality\\'
        Tools().mk_dir(outdir, force=True)
        seasons_list=['spring','summer','autumn','winter']
        date_list=[]

        for year in range(1982, 2015):
            for mon in range(1, 13):
                date_list.append(datetime.datetime(year, mon, 1))


        # print(date_list)
        # exit()

        for season in seasons_list:
            outdir_season=outdir+season+'\\'
            Tools().mk_dir(outdir_season,force=True)

            for fdir in os.listdir(fdir_all):
                if  'AVHRR' in fdir:
                    continue


                outf = outdir_season + fdir.split('.')[0] + '.npy'

                if os.path.isfile(outf):
                    continue
                print(outf)

                for fdir_ii in os.listdir(fdir_all + fdir):
                    if not 'DIC' in fdir_ii:
                        continue

                    spatial_dict=T.load_npy_dir(fdir_all + fdir + '\\' + fdir_ii)

                    annual_spatial_dict = {}
                    for pix in tqdm(spatial_dict):
                        r,c=pix

                        picked_mon=global_get_seasonality(pix,season)


                        vals = spatial_dict[pix]
                        vals = np.array(vals)
                        # vals[vals >20000] = np.nan
                        # #
                        # vals = np.array(vals)/100

                        vals[vals < -999] = np.nan
                        # vals[vals > 7] = np.nan
                        #
                        # vals[vals<0]=np.nan

                        if T.is_all_nan(vals):
                            continue

                        vals_dict = dict(zip(date_list, vals))
                        date_list_gs = []
                        date_list_index = []
                        for i, date in enumerate(date_list):
                            mon = date.month
                            if mon in picked_mon:
                                date_list_gs.append(date)

                                date_list_index.append(i)

                        consecutive_ranges = self.group_consecutive_vals(date_list_index)
                        date_dict = dict(zip(list(range(len(date_list))), date_list))

                        # annual_vals_dict = {}
                        annual_gs_list = []

                        if len(consecutive_ranges[0]) > 12:
                            consecutive_ranges = np.reshape(consecutive_ranges, (-1, 12))

                        for idx in consecutive_ranges:
                            date_gs = [date_dict[i] for i in idx]
                            if not len(date_gs) == len(picked_mon):
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

        fdir=result_root + rf'extract_GS\OBS_LAI_extend\\'

        outdir=result_root+rf'split_original\\'
        T.mk_dir(outdir,force=True)

        for f in os.listdir(fdir):
            if not f.split('.')[0] in ['Tempmean','tmin','tmax','VPD']:
                continue

            outf=outdir+f.split('.')[0]+'_'
            print(outf)

            dic_i = {}
            dic_ii = {}


            dic = dict(np.load(fdir + f, allow_pickle=True, ).item())


            for pix in tqdm(dic):
                r,c=pix
                # if r<480:
                #     continue
                if pix not in dic_dryland_mask:
                    continue
                time_series=dic[pix]
                print(len(time_series))
                time_series[time_series < -99] = np.nan
                time_series=np.array(time_series)
                # time_series[time_series >7] = np.nan

                time_series_i=time_series[:20]
                time_series_ii=time_series[20:]
                print(len(time_series_i))
                print(len(time_series_ii))

                # plt.plot(time_series_i)
                # plt.plot(time_series_ii)
                #
                # plt.show()

                dic_i[pix]=time_series_i
                dic_ii[pix]=time_series_ii

            np.save(outf+'1982_2001.npy',dic_i)
            np.save(outf+'2002_2020.npy',dic_ii)

    def extend_nan(self):
        fdir= result_root + rf'extract_GS\OBS_LAI\\'
        outdir=result_root + rf'extract_GS\\OBS_LAI_extend\\'
        T.mk_dir(outdir,force=True)
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'GIMMS3g' in f:
                continue


            outf=outdir+f.split('.')[0]+'.npy'
            if os.path.isfile(outf):
                continue
            dic = dict(np.load(fdir +f, allow_pickle=True, ).item())
            dic_new={}

            for pix in tqdm(dic):
                time_series=dic[pix]

                time_series=np.array(time_series)
                time_series[time_series<-999]=np.nan
                if np.isnan(np.nanmean(time_series)):
                    continue
                if len(time_series)<37:
                    time_series_new=np.append(time_series,np.nan)
                    dic_new[pix]=time_series_new
                else:
                    dic_new[pix]=time_series
            np.save(outf,dic_new)
    def extend_GS(self):

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan

        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        f= result_root + rf'\extract_GS\OBS_LAI\Tempmean.npy'
        outf=result_root + rf'extract_GS\\OBS_LAI_extend\Tempmean.npy'
        dic = dict(np.load(f, allow_pickle=True, ).item())
        dic_new={}
        for pix in tqdm(dic):


            time_series=dic[pix]


            time_series=np.array(time_series)
            time_series[time_series<-999]=np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            if len(time_series)<39:
                continue
            time_series_new=np.append(time_series,np.nan)
            dic_new[pix]=time_series_new
        np.save(outf,dic_new)

    def check_tif_length(self):  ## count the number of tif files in each year
        fdir=rf'C:\Users\wenzhang1\Desktop\Terra\TIFF\\'
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
    def daily_to_monthly(self):
        fdir=data_root+'Inversion\TIFF\\'
        outdir=data_root+'Inversion\monthly_TIFF\\'
        Tools().mk_dir(outdir,force=True)
        year_list=list(range(1982,2021))
        month_list=list(range(1,13))


        for year in year_list:


            for month in month_list:
                month_list_data = []
                month=format(month,'02d')

                for f in os.listdir(fdir):
                    if not f.endswith('.tif'):
                        continue
                    ## year =year and month =month
                    # print(f.split('.')[0][0:4])
                    # print(f.split('.')[0][4:6])
                    # print(type(f.split('.')[0][0:4]))

                    if not (f.split('.')[0][0:4])==str(year):
                        continue
                    if not (f.split('.')[0][4:6])==str(month):
                        continue


                    print(f)
                    array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
                    array=np.array(array)
                    array[array<-999]=np.nan
                    array[array>10000]=np.nan
                    ## GPP need scale factor need to multiply 10^15 and divide areas
                    array=np.array(array)*10 ** 15
                    month_list_data.append(array)

                month_list_data=np.nanmean(month_list_data,axis=0)
                print(month_list_data.shape)
                # plt.imshow(month_list_data)
                # plt.colorbar()
                # plt.show()
                fname='{}{}.tif'.format(year,month)
                ToRaster().array2raster(outdir+fname,originX,originY,pixelWidth,pixelHeight,month_list_data,)







        pass


    def resample_trendy(self):
        fdir_all = data_root + 'GPP\\NIRvGPP\TIFF\\'
        for fdir in tqdm(os.listdir(fdir_all)):

            outdir = data_root + rf'\GPP\\NIRvGPP\\resample\\\\{fdir}\\'
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
    def resample_AVHRR_LAI(self):
        fdir_all = rf'C:\Users\wenzhang1\Desktop\Terra\TIFF\vpd\\'

        outdir = rf'C:\Users\wenzhang1\Desktop\Terra\resample\\vpd\\'


        T.mk_dir(outdir, force=True)
        year = list(range(1982, 2021))
        # print(year)
        # exit()
        for f in tqdm(os.listdir(fdir_all)):
            if not f.endswith('.tif'):
                continue
            outf=outdir+f
            if os.path.isfile(outf):
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


            # print(date)
            # exit()
            dataset = gdal.Open(fdir_all  + f)
            # print(dataset.GetGeoTransform())
            original_x = dataset.GetGeoTransform()[1]
            original_y = dataset.GetGeoTransform()[5]

            # band = dataset.GetRasterBand(1)
            # newRows = dataset.YSize * 2
            # newCols = dataset.XSize * 2
            try:
                gdal.Warp(outdir + '{}.tif'.format(date), dataset, xRes=0.25, yRes=0.25, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass

    def resample_GIMMS3g(self):
        fdir_all = rf'D:\Project3\Data\BU-GIMMS3gV1-LAI-1981-2018\BU-GIMMS3gV1-LAI-1981-2018\\'

        outdir = rf'D:\Project3\Data\resample_GIMMS3g\\'


        T.mk_dir(outdir, force=True)
        year = list(range(1982, 2021))
        # print(year)
        # exit()
        for f in tqdm(os.listdir(fdir_all)):
            if not f.endswith('.tif'):
                continue

            fname=f.split('.')[1][:7]
            print(fname)

            outf=outdir+fname+'.tif'
            if os.path.isfile(outf):
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
            date = f.split('.')[3]

            #
            # print(date)
            # exit()
            dataset = gdal.Open(fdir_all  + f)
            # print(dataset.GetGeoTransform())
            original_x = dataset.GetGeoTransform()[1]
            original_y = dataset.GetGeoTransform()[5]

            # band = dataset.GetRasterBand(1)
            # newRows = dataset.YSize * 2
            # newCols = dataset.XSize * 2
            try:
                gdal.Warp(outdir + '{}.tif'.format(date), dataset, xRes=0.25, yRes=0.25, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass
    def resample_inversion(self):
        fdir_all = data_root + 'Inversion\Carbontracker\TIFF\\'

        outdir = data_root + rf'\\Inversion\\Carbontracker\\resample\\'

        T.mk_dir(outdir, force=True)
        year = list(range(1982, 2021))
        # print(year)
        # exit()
        for f in tqdm(os.listdir(fdir_all)):
            if not f.endswith('.tif'):
                continue

            if f.startswith('._'):
                continue


            print(f)
            # exit()
            date = f.split('.')[0]



            ToRaster().resample_reproj(fdir_all+f,outdir+f,0.25)
    def aggregate_GIMMS3g(self):  # aggregate biweekly data to monthly

        fdir_all = data_root + rf'GIMMS3g\\resample_GIMMS3g\\'
        outdir = data_root + rf'GIMMS3g\monthly_aggragate_GIMMS3g\\'
        Tools().mk_dir(outdir, force=True)

        year_list = list(range(1982, 2019))
        month_list=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

        dic_month={'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,
                      'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}


        for year in tqdm(year_list):
            for month in tqdm(month_list):

                data_list = []
                for f in tqdm(os.listdir(fdir_all)):
                    if not f.endswith('.tif'):
                        continue
                    month_int=dic_month[month]
                    data_year = f.split('.')[0][0:4]
                    data_month = f.split('.')[0][4:7]
                    if not int(data_year) == year:
                        continue
                    if not str(data_month) == month:
                        continue


                    arr = ToRaster().raster2array(fdir_all + f)[0]

                    arr_unify = arr[:720][:720,
                                :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
                    arr_unify = np.array(arr_unify)
                    arr_unify[arr_unify == 65535] = np.nan
                    arr_unify[arr_unify < 0] = np.nan
                    arr_unify[arr_unify > 7] = np.nan
                    # 当变量是LAI 的时候，<0!!
                    data_list.append(arr_unify)
                data_list = np.array(data_list)
                print(data_list.shape)
                # print(len(data_list))
                # exit()

                ##define arr_average and calculate arr_average

                arr_average = np.nanmax(data_list, axis=0)
                arr_average = np.array(arr_average)
                arr_average[arr_average <= 0] = np.nan
                arr_average[arr_average > 7] = np.nan
                if np.isnan(np.nanmean(arr_average)):
                    continue
                if np.nanmean(arr_average) < 0.:
                    continue
                # plt.imshow(arr_average)
                # plt.title(f'{year}{month}')
                # plt.show()

                # save

                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_average, outdir + '{}{:02d}.tif'.format(year, month_int))

    def aggreate_AVHRR_LAI(self):  # aggregate biweekly data to monthly
        fdir_all = data_root + rf'AVHRR_LAI\resample\\'
        outdir = data_root + f'AVHRR_LAI\monthly_aggragate\\'
        Tools().mk_dir(outdir, force=True)

        year_list = list(range(1982, 2021))
        month_list = list(range(1, 13))

        for year in tqdm(year_list):
            for month in tqdm(month_list):

                data_list = []
                for f in tqdm(os.listdir(fdir_all)):
                    if not f.endswith('.tif'):
                        continue

                    data_year = f.split('.')[0][0:4]
                    data_month = f.split('.')[0][4:6]

                    if not int(data_year) == year:
                        continue
                    if not int(data_month) == int(month):
                        continue
                    arr=ToRaster().raster2array(fdir_all +f)[0]
                    arr=arr/1000 ###
                    arr_unify = arr[:720][:720,
                                :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
                    arr_unify = np.array(arr_unify)
                    arr_unify[arr_unify == 65535] = np.nan
                    arr_unify[arr_unify <0] = np.nan
                    arr_unify[arr_unify > 7] = np.nan
                      # 当变量是LAI 的时候，<0!!
                    data_list.append(arr_unify)
                data_list = np.array(data_list)
                print(data_list.shape)
                # print(len(data_list))
                # exit()

                ##define arr_average and calculate arr_average

                arr_average = np.nanmax(data_list, axis=0)
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

                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_average, outdir + '{}{:02d}.tif'.format(year, month))
    def unify_TIFF(self):
        fdir_all=rf'C:\Users\wenzhang1\Desktop\Terra\resample\tmax\\'
        outdir=rf'C:\Users\wenzhang1\Desktop\Terra\unify\tmax\\'
        Tools().mk_dir(outdir,force=True)

        for f in tqdm(os.listdir(fdir_all)):
            fpath=fdir_all+f
            outpath=outdir+f
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            unify_tiff=DIC_and_TIF().unify_raster1(fpath,outpath,0.25)

    def average_temperature(self):  ### calculate the average temperature of each year
        fdir_tmax=rf'C:\Users\wenzhang1\Desktop\Terra\resample\tmax\\'
        fdir_tmin=rf'C:\Users\wenzhang1\Desktop\Terra\resample\tmin\\'
        outdir=rf'C:\Users\wenzhang1\Desktop\Terra\resample\average\\'
        Tools().mk_dir(outdir,force=True)

        for f in tqdm(os.listdir(fdir_tmax)):
            fpath_tmax=fdir_tmax+f
            fpath_tmin=fdir_tmin+f
            outpath=outdir+f
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            arr_tmax=ToRaster().raster2array(fpath_tmax)[0]
            arr_tmin=ToRaster().raster2array(fpath_tmin)[0]
            arr_average=(arr_tmax+arr_tmin)/2
            arr_average[arr_average<-999]=np.nan
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_average,outpath)

    def scales_GPP_Trendy(self):
        fdir_all = result_root+rf'extract_GS\TRENDY_GPP\S3\Extend\\'



        outdir = result_root + rf'extract_GS\TRENDY_GPP\S3\scales\\'
        Tools().mk_dir(outdir, force=True)

        for f in tqdm(os.listdir(fdir_all)):
            dic=np.load(fdir_all  +f, allow_pickle=True, ).item()
            outf=outdir+'\\'+f.split('.')[0]+'.npy'
            print(outf)
            dic_new={}
            for pix in tqdm(dic):
                time_series=dic[pix]
                time_series=np.array(time_series)
                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.nanmean(time_series) < 0.:
                    continue
                time_series=time_series*60*60*24*365*1000
                dic_new[pix]=time_series
            np.save(outf,dic_new)

    def scales_Inversion(self):
        fdir = data_root + rf'Inversion\Carbontracker\unify\\'

        outdir = data_root + rf'Inversion\Carbontracker\scales\\'
        Tools().mk_dir(outdir, force=True)

        for f in tqdm(os.listdir(fdir)):
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)

            # outf=outdir+'f.split('.')[0][0:4]'+'.tif'
            print(outf)
            array[array < -999] = np.nan
            if np.isnan(np.nanmean(array)):
                continue

            array=array*12*60*60*24*365   #origianl unit is mol/m2/s, now is gC/m2/yr
            #1 mol=12g
            ToRaster().array2raster(outdir+f,originX,originY,pixelWidth,pixelHeight,array)



    pass

class Phenology():

    def __init__(self):

        self.datadir_all = data_root

    def run(self):
        self.hants()
        # self.hants_trendy()
        # self.check_hants()
        # self.per_pixel_annual()
        # self.annual_phenology()
        # self.compose_annual_phenology()
        # self.data_clean()
        # self.average_phenology()
        # self.pick_daily_phenology()
        # self.pick_monthly_phenology()
        # self.plot_phenology()

    def hants(self):

        outdir=self.datadir_all+'LAI//Hants_annually_smooth//'
        T.mkdir(outdir)
        fdir=self.datadir_all+'LAI\MCD_15A3H_DIC\\'
        spatial_dic=T.load_npy_dir(fdir)
        tif_dir=self.datadir_all+'LAI/MCD_15A3H_TIFF/'
        date_list=[]
        for f in os.listdir(tif_dir):
            if f.endswith('.tif'):
                date=f.split('.')[0]
                # y,m,d=date.split('_')
                y=date[0:4]
                m=date[4:6]
                d=date[6:8]
                y=int(y)
                m=int(m)
                d=int(d)
                date_obj=datetime.datetime(y,m,d)
                date_list.append(date_obj)

        hants365={}

        for pix in tqdm(spatial_dic,desc='hants'):
            r,c=pix
            if r>120:
                continue
            vals=spatial_dic[pix]

            if T.is_all_nan(vals):
                continue
            try:
                # plt.plot(vals)
                # plt.show()

                results=HANTS().hants_interpolate( values_list=vals, dates_list=date_list, valid_range=[0.001,10],nan_value=0)
            except:
                continue


            hants365[pix]=results
        np.save(outdir+'hants365.npy',hants365)

            # for year in results:
            #     result=results[year]
            #     plt.plot(result)
            #     plt.title(str(year))
            #     plt.show()
    def hants_trendy(self):


        fdir_all=self.datadir_all+'LAI/DIC/'

        tif_dir_all = self.datadir_all + rf'\LAI\Trendy_Yang\\'


        for fdir in os.listdir(fdir_all):
            if not 'Trendy' in fdir:
                continue

            date_list = []

            print(fdir)
            outdir = self.datadir_all + rf'LAI/Hants_annually_smooth/{fdir}/'
            if isdir(outdir):
                continue
            T.mkdir(outdir, force=True)

            tif_dir=tif_dir_all+fdir+'/'
            for f_tiff in os.listdir(tif_dir):
                if f_tiff.endswith('.tif'):
                    date=f_tiff.split('.')[0]

                    y=int(date[:4])
                    m=int(date[4:6])
                    d=int(date[6:8])
                    # format = '%Y%m%d'
                    if y<2003:
                        continue
                    date_obj = datetime.datetime.strptime(date, '%Y%m%d')
                    # date_obj = datetime.datetime.strptime(date, '%Y%m')
                    # date_obj=datetime.datetime(y,m,d)
                    date_list.append(date_obj)
            # print(len(date_list))

            hants365={}

            spatial_dic = T.load_npy_dir(fdir_all + fdir+ '/')

            for pix in tqdm(spatial_dic,desc='hants'):
                r,c=pix
                if r>120:
                    continue
                vals=spatial_dic[pix]

                if T.is_all_nan(vals):
                    continue
                try:
                    # plt.plot(vals)
                    # plt.show()
                    results=HANTS().hants_interpolate( values_list=vals, dates_list=date_list, valid_range=[0.001,10],nan_value=0)
                except:
                    continue

                hants365[pix]=results
            np.save(outdir+'hants365.npy',hants365)

                # for year in results:
                #     result=results[year]
                #     plt.plot(result)
                #     plt.title(str(year))
                #     plt.show()

    def per_pixel_annual(self):
        fdir_all=data_root+'\LAI\Hants_annually_smooth\\'

        for fdir in os.listdir(fdir_all):
            if isdir(fdir_all+fdir):
                continue

            outdir = data_root + '\LAI\\\per_pix_annual\\' + fdir + '\\'
            Tools().mk_dir(outdir, force=True)

            for f in tqdm (os.listdir(fdir_all+fdir)):
                if f.endswith('.npy'):
                    hants365_dic=np.load(fdir_all+fdir+'/'+f,allow_pickle=True).item()


            for y in range(2003, 2023):
                outf = outdir + '{}.npy'.format(y)
                result_dic = {}
                for pix in hants365_dic:
                    r,c=pix
                    if r>120:
                        continue
                    result = hants365_dic[pix]
                    # plt.plot(result)
                    # plt.show()
                    for year in result:
                        if year != y:
                            continue
                        result_i = result[year]

                        if np.isnan(np.nanmean(result_i)):
                            continue
                        if np.nanmean(result_i) == 0:
                            continue

                        result_dic[pix] = result_i

                np.save(outf, result_dic)

    def check_hants(self):

        hants365=np.load(rf'D:\Greening\Data\FLUXNET_2015\screening_sites_hants\\hants365_DE-Obe.npy',allow_pickle=True).item()
        for pix in hants365:
            result=hants365[pix]
            print(len(result))
            # exit()

            for year in result:

                result_i=result[year]
                print(len(result_i))

                plt.plot(result_i)
                plt.title(pix)
                plt.show()
        # for pix in hants365:
        #     result=hants365[pix]
        #
        #     print(len(result))
        #
        #     plt.plot(result)
        #     plt.title(pix)
        #     plt.show()


    def annual_phenology(self, threshold_i=0.2, ):
        fdir_all = data_root+'\LAI\\\per_pix_annual\\'
        for fdir in os.listdir(fdir_all):
            if not fdir=='MCD':
                continue

            out_dir =data_root+rf'\LAI\phenology\\annual_phenology\\{fdir}\\'
            T.mkdir(out_dir, force=True)

            for f in T.listdir(fdir_all + fdir):
                year = int(f.split('.')[0])

                outf_i = join(out_dir, f'{year}.df')
                hants_smooth_f = join(fdir_all, fdir, f)
                hants_dic = T.load_npy(hants_smooth_f)
                result_dic = {}
                for pix in tqdm(hants_dic, desc=str(year)):
                    r,c=pix
                    if r>120:
                        continue

                    vals = hants_dic[pix]
                    # plt.plot(vals)
                    # plt.show()
                    result = self.pick_phenology(vals, threshold_i)
                    result_dic[pix] = result
                df = T.dic_to_df(result_dic, 'pix')
                T.save_df(df, outf_i)
                T.df_to_excel(df, outf_i)
                np.save(outf_i, result_dic)


    def pick_phenology(self, vals, threshold_i):

        peak = np.nanargmax(vals)
        # if peak == 0 or peak == (len(vals) - 1):
        #     return {}
        # plt.plot(vals)
        # plt.show()
        # print(peak)
        # print(np.max(vals))
        # test=vals[peak]
        # print(test)

        if peak == 0 or peak == (len(vals) - 1):
            return {}
        try:
            early_start = self.__search_left(vals, peak, threshold_i)
            late_end = self.__search_right(vals, peak, threshold_i)
        except:
            early_start = 60
            late_end = 130
            print(vals)
            plt.plot(vals)
            plt.show()
        # method 1
        # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak)
        # method 2
        early_end, late_start = self.__median_early_late(vals, early_start, late_end, peak)

        early_period = early_end - early_start
        peak_period = late_start - early_end
        late_period = late_end - late_start
        dormant_period = 365 - (late_end - early_start)

        result = {
            'early_length': early_period,
            'mid_length': peak_period,
            'late_length': late_period,
            'dormant_length': dormant_period,
            'early_start': early_start,
            'early_start_mon': self.__day_to_month(early_start),

            'early_end': early_end,
            'early_end_mon': self.__day_to_month(early_end),

            'peak': peak,
            'peak_mon': self.__day_to_month(peak),

            'late_start': late_start,
            'late_start_mon': self.__day_to_month(late_start),

            'late_end': late_end,
            'late_end_mon': self.__day_to_month(late_end),
        }
        return result
        pass

    def __search_left(self, vals, maxind, threshold_i):
        left_vals = vals[:maxind]
        left_min = np.nanmin(left_vals)
        max_v = vals[maxind]
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind - step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __search_right(self, vals, maxind, threshold_i):
        right_vals = vals[maxind:]
        right_min = np.nanmin(right_vals)
        max_v = vals[maxind]
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __median_early_late(self, vals, sos, eos, peak):
        # 2 使用sos-peak peak-eos中位数作为sos和eos的结束和开始

        median_left = int((peak - sos) / 2.)
        median_right = int((eos - peak) / 2)
        max_ind = median_left + sos
        min_ind = median_right + peak
        return max_ind, min_ind

    def __day_to_month(self, doy):
        base = datetime.datetime(2000, 1, 1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day
        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month

    def compose_annual_phenology(self, ):
        fdir_all = data_root+rf'LAI\phenology\annual_phenology\\'
        for fdir in os.listdir(fdir_all):
            if not fdir.startswith('MCD'):
                continue

            outdir = data_root + f'LAI/phenology/compose_annual_phenology/{fdir}/'
            T.mkdir(outdir, force=True)
            outf = join(outdir, 'phenology_dataframe.df')

            all_result_dic = {}
            pix_list_all = []
            col_list = None
            for f in os.listdir(fdir_all + fdir):
                f = join(fdir_all, fdir, f)
                if not f.endswith('.df'):
                    continue
                df = T.load_df(f)
                pix_list = T.get_df_unique_val_list(df, 'pix')
                pix_list_all.append(pix_list)
                col_list = df.columns
            all_pix = []
            for pix_list in pix_list_all:
                for pix in pix_list:
                    all_pix.append(pix)
            pix_list = T.drop_repeat_val_from_list(all_pix)

            col_list = col_list.to_list()
            col_list.remove('pix')
            for pix in pix_list:
                dic_i = {}
                for col in col_list:
                    dic_i[col] = {}
                all_result_dic[pix] = dic_i
            # print(len(T.listdir(f_dir)))
            # exit()
            for f in tqdm(T.listdir(fdir_all + fdir)):
                if not f.endswith('.df'):
                    continue
                year = int(f.split('.')[0])
                df = T.load_df(join(fdir_all, fdir, f))
                dic = T.df_to_dic(df, 'pix')
                for pix in dic:
                    for col in dic[pix]:
                        if col == 'pix':
                            continue
                        all_result_dic[pix][col][year] = dic[pix][col]
            df_all = T.dic_to_df(all_result_dic, 'pix')
            T.save_df(df_all, outf)
            T.df_to_excel(df_all, outf)
            np.save(outf, all_result_dic)

    def data_clean(self,):  # 盖帽法

        f_dir = data_root + rf'LAI/phenology/compose_annual_phenology/MODIS//'
        outdir = data_root + rf'LAI/phenology/compose_annual_phenology_clean/MODIS//'
        T.mkdir(outdir, force=True)
        outf = join(outdir, 'phenology_dataframe.df')
        all_result_dic = {}
        pix_list_all = []

        for f in T.listdir(f_dir):
            if not f.endswith('.df'):
                continue
            df = T.load_df(join(f_dir, f))
            columns = df.columns
            column_list = []
            for col in columns:
                if col == 'pix':
                    continue
                column_list.append(col)
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                lon, lat = DIC_and_TIF().pix_to_lon_lat(pix)
                # if lat > 70:
                #     continue
                # address = Tools().lonlat_to_address(lon, lat)
                # print(address)
                for col in column_list:
                    dic_i = row[col]

                    dic_clean = {}
                    # print(dic_i)
                    # values_list=dic_i.values()
                    # values_list=list(values_list)
                    series = pd.Series(dic_i)
                    cap_series = self.cap(series)
                    # print(series)
                    # plt.plot(series)
                    # # plt.title(address)
                    # plt.plot(cap_series)
                    #
                    # plt.show()
                    for year in dic_i:
                        dic_clean[year] = cap_series[year]
                    if pix not in all_result_dic:
                        all_result_dic[pix] = {}

                    all_result_dic[pix][col] = dic_clean
            # convert to df
            df_all = T.dic_to_df(all_result_dic, 'pix')
            #save
            T.save_df(df_all, outf)
            T.df_to_excel(df_all, outf)



    def cap(self, x, quantile=(0.05, 0.95)):

        """盖帽法处理异常值
        Args：
            x：pd.Series列，连续变量
            quantile：指定盖帽法的上下分位数范围
        """

        # 生成分位数
        Q01, Q99 = x.quantile(quantile).values.tolist()

        # 替换异常值为指定的分位数
        if Q01 > x.min():
            x = x.copy()
            x.loc[x < Q01] = Q01

        if Q99 < x.max():
            x = x.copy()
            x.loc[x > Q99] = Q99

        return (x)

    def average_phenology(self, ):   #将多年物候期平均
        fdir_all = data_root + f'LAI/phenology/compose_annual_phenology/'
        for fdir in os.listdir(fdir_all):
            if not 'MCD' in fdir:
                continue

            outdir = data_root + f'LAI/phenology/average_phenology/{fdir}/'
            T.mkdir(outdir, force=True)
            outf = join(outdir, 'phenology_dataframe.df')

            all_result_dic = {}

            for f in T.listdir(fdir_all + fdir):
                if not f.endswith('.df'):
                    continue
                df = T.load_df(join(fdir_all,fdir, f))
                columns = df.columns
                column_list = []
                for col in columns:
                    if col == 'pix':
                        continue
                    column_list.append(col)

                pix_list = T.get_df_unique_val_list(df, 'pix')

                ########################################build dic##############################################################
                for pix in pix_list:
                    dic_i = {}
                    for col in column_list:
                        dic_i[col] = {}
                    all_result_dic[pix] = dic_i

                for i, row in tqdm(df.iterrows(), total=len(df)):
                    pix = row['pix']
                    lon, lat = DIC_and_TIF().pix_to_lon_lat(pix)
                    # address=Tools().lonlat_to_address(lon,lat)
                    # print(address)
                    for col in column_list:
                        dic_i = row[col]
                        # print(dic_i)
                        values = dic_i.values()
                        values = list(values)
                        value_mean = np.mean(values)
                        value_ = round(value_mean, 0)
                        value_std = np.std(values)
                        all_result_dic[pix][col] = value_

            df_all = T.dic_to_df(all_result_dic, 'pix')
            T.save_df(df_all, outf)
            T.df_to_excel(df_all, outf)
            np.save(outf, all_result_dic)

    def pick_daily_phenology(self):  # 转换格式 for example: early [100,150], peak [150,200], late [200,300]
        fdir_all = data_root + f'LAI/phenology/average_phenology/'
        for fdir in os.listdir(fdir_all):
            if not fdir.endswith('MCD'):
                continue
            outdir = data_root + f'LAI/phenology/pick_daily_phenology/{fdir}/'
            T.mkdir(outdir, force=True)
            outf = join(outdir, 'pick_daily_phenology.df')

            phenology_df = T.load_df(
                f'{fdir_all}/{fdir}/phenology_dataframe.df')

            early_dic = {}
            peak_dic = {}
            late_dic = {}
            all_result_dic = {}

            for i, row in tqdm(phenology_df.iterrows(), total=len(phenology_df)):
                pix = row['pix']
                all_result_dic[pix] = {}
                early_start = row['early_start']
                early_end = row['early_end']
                peak_start = row['early_end']
                peak_end = row['late_start']
                late_start = row['late_start']
                late_end = row['late_end']
                early_period = np.arange(int(early_start), int(early_end), 1)
                # print(early_period)
                peak_period = np.arange(int(early_end), int(late_start), 1)
                # print(peak_period)
                late_period = np.arange(int(late_start), int(late_end), 1)
                # print(late_period)
                all_result_dic[pix]['early'] = early_period
                all_result_dic[pix]['peak'] = peak_period
                all_result_dic[pix]['late'] = late_period
                all_result_dic[pix]['early_peak'] = np.concatenate((early_period, peak_period))
                all_result_dic[pix]['early_peak_late'] = np.concatenate((early_period, peak_period, late_period))
                # print(all_result_dic[pix])


            df_all = T.dic_to_df(all_result_dic, 'pix')
            T.save_df(df_all, outf)
            T.df_to_excel(df_all, outf)
            np.save(outf, all_result_dic)

    def pick_monthly_phenology(self):  # 转换格式 for example: early [100,150], peak [150,200], late [200,300]
        fdir_all = data_root + f'LAI/phenology/average_phenology/'
        for fdir in os.listdir(fdir_all):
            if not fdir.endswith('MCD'):
                continue
            outdir = data_root + f'LAI/phenology/pick_monthly_phenology/{fdir}/'
            T.mkdir(outdir, force=True)
            outf = join(outdir, 'pick_monthly_phenology.df')

            phenology_df = T.load_df(
                f'{fdir_all}/{fdir}/phenology_dataframe.df')

            early_dic = {}
            peak_dic = {}
            late_dic = {}
            all_result_dic = {}

            for i, row in tqdm(phenology_df.iterrows(), total=len(phenology_df)):
                pix = row['pix']
                all_result_dic[pix] = {}

                early_start = row['early_start_mon']
                early_end = row['early_end_mon']

                late_start = row['late_start_mon']
                late_end = row['late_end_mon']
                early_period= np.arange(int(early_start), int(early_end))
                early_peak_period = list(range(int(early_start), int(late_start)))
                peak_period=list(range(int(early_end), int(late_start)))
                late_period = list(range(int(late_start), int(late_end)+1))
                print(early_peak_period)
                print(peak_period)
                print(early_period)
                print(late_period)
                print('-------------------')
                # exit()
                all_result_dic[pix]['early_peak'] = early_peak_period
                all_result_dic[pix]['early'] = early_period
                all_result_dic[pix]['peak'] = peak_period
                all_result_dic[pix]['late'] = late_period

                # print(all_result_dic[pix])


            df_all = T.dic_to_df(all_result_dic, 'pix')
            T.save_df(df_all, outf)
            T.df_to_excel(df_all, outf)
            np.save(outf, all_result_dic)



    def plot_phenology(self):
        # df_f=data_root+r'MODIS_LAI_MOD15A2H\phenology\annual_phenology\\2015.df'
        df_f=data_root+r'Trendy\phenology\annual_phenology\MODIS\\2021.df'

        df=T.load_df(df_f)
        colunms='early_start'
        spatial_dic=T.df_to_spatial_dic(df,colunms)
        arr=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr,vmin=100,vmax=160,cmap='jet')
        plt.colorbar()
        plt.show()





class Check_plot():

    def run(self):
        self.foo()

    def foo(self):

        # f='/Volumes/SSD_sumsang/project_greening/Result/detrend/extraction_during_late_growing_season_static/during_late_CSIF_par/per_pix_dic_008.npy'
        f = rf'D:\Greening\Result\zscore\LAI\late\MCD.npy'
        # f = rf'D:\Greening\Data\Trendy\DIC\\CABLE-POP_S2_lai\per_pix_dic_014.npy'
        # f='/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_anomaly_window/1982-2015_during_early/during_early_CO2.npy'

        dic = dict(np.load(f, allow_pickle=True, encoding='latin1').item())
        # ///////check 字典是否不缺值////
        for pix in tqdm(dic, desc='interpolate'):
            r, c = pix
            # china_r=list(range(150,150))
            # china_c=list(range(550,620))

            # china_r = list(range(140, 570))
            # china_c = list(range(550, 620))
            # if r not in china_r:
            #     continue
            # if c not in china_c:
            #     continue
            print(len(dic[pix]))
            # exit()
            if len(dic[pix]) == 0:
                continue

            time_series = dic[pix]
            print(time_series)
            if len(time_series) == 0:
                continue
            # print(time_series)
            # time_series_reshape=time_series.reshape(12,-1)
            # time_series=np.array(time_series)
            plt.plot(time_series)
            # plt.imshow(time_series,aspect='auto')
            # plt.imshow(time_series)
            # plt.title(str(pix))
            plt.show()
            # spatial_dic[pix]=len(time_series)
        #     spatial_dic[pix] = time_series[0]
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.title(str(pix))
        # plt.show()

    pass

class statistic_analysis():
    def __init__(self):
        pass
    def run(self):

        # self.detrend()  ##original
        self.detrend_zscore_monthly()
        # self.zscore()
        # self.detrend()
        # self.LAI_baseline()

        # self.anomaly_GS()
        # self.growth_rate_GS()
        # self.anomaly_GS_ensemble()
        # self.zscore_GS()

        # self.trend_analysis()
        # self.trend_analysis_landcover_composition()
        # self.LUCC_LAI_correlation()
        # self.calculate_asymmetry_response()

        # self.scerios_analysis() ## this method tried to calculate different scenarios




    def detrend(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir=result_root + rf'split_anomaly\Y\\'
        outdir=result_root + rf'detrend_split_anomaly\\Y\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            outf=outdir+f.split('.')[0]
            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

            detrend_zscore_dic={}

            for pix in tqdm(dic):
                dryland_values=dic_dryland_mask[pix]
                if np.isnan(dryland_values):
                    continue
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
                # if np.nanmean(time_series) <= 0.:
                #     continue


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

    def LAI_baseline(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir=result_root + rf'split_original\Y\\'
        outdir=result_root + rf'residual_method\\baseline\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'LAI' in f:
                continue
            outf=outdir+f.split('.')[0]
            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

            baseline_dic={}

            for pix in tqdm(dic):
                dryland_values=dic_dryland_mask[pix]
                if np.isnan(dryland_values):
                    continue
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

                baseline=np.nanmean(time_series[0:3])


                baseline_dic[pix] = baseline

            np.save(outf, baseline_dic)

    def detrend_zscore_monthly(self): #  detrend based on each month

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir_all = data_root + 'monthly_data\\'
        for fdir in os.listdir(fdir_all):
            if not 'LAI4g' in fdir:
                continue

            outdir = result_root + rf'detrend_zscore_monthly\\'
            # if os.path.isdir(outdir):
            #     continue
            Tools().mk_dir(outdir, force=True)
            outf=outdir+fdir.split('.')[0]

            dic=T.load_npy_dir(fdir_all+fdir+'\\')

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

                    delta_time_series = (time_series_i - mean)/std
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




    def anomaly_GS(self):  ### anomaly GS
        dryland_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(dryland_mask_f)

        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root + 'extract_GS\OBS_LAI_extend\\'
        outdir = result_root + f'anomaly\\OBS_extend\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)
            if os.path.isfile(outf):
                continue

            dic = np.load(fdir + f, allow_pickle=True, ).item()

            anomaly_dic = {}

            for pix in tqdm(dic):


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

    def growth_rate_GS(self):  ### anomaly GS
        dryland_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(dryland_mask_f)

        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root + 'extract_GS\OBS_LAI_extend\\'
        outdir = result_root + f'growth_rate\\OBS_extend\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'LAI4g' in f:
                continue

            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)

            dic = np.load(fdir + f, allow_pickle=True, ).item()

            growth_rate_dic = {}

            for pix in tqdm(dic):
                growth_rate_time_series=np.zeros(len(dic[pix])-1)

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





                ##growth_rate= (y2-y1)

                for i in range(len(time_series)-1):
                    growth_rate_time_series[i]=time_series[i+1]-time_series[i]

                a,b,r,p=T.nan_line_fit(np.arange(len(time_series)), time_series)
                a1,b1,r1,p1=T.nan_line_fit(np.arange(len(growth_rate_time_series)), growth_rate_time_series)
                print('original', a, b, r, p)
                if p > 0.0001:
                    continue
                if np.isnan(p):
                    continue
                print('growth_rate', a1, b1, r1, p1)

                plt.plot(time_series)
                print(time_series)




                plt.twinx()

                plt.plot(growth_rate_time_series,'r')
                print(growth_rate_time_series)
                plt.show()

                growth_rate_dic[pix] = growth_rate_time_series

            np.save(outf, growth_rate_dic)
    def anomaly_GS_ensemble(self):  ### calculate the ensemble mean of anomaly GS

        fdir = result_root + 'anomaly\TRENDY_GPP\S3\\'
        outdir = result_root + f'anomaly\\TRENDY_GPP\S3\\'
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
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        # crop_mask[crop_mask == 16] = 0
        # crop_mask[crop_mask == 17] = 0
        # crop_mask[crop_mask == 18] = 0



        fdir = result_root + rf'asymmetry_response\\'
        outdir = result_root + rf'trend_analysis\\asymmetry_response\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):


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
                landcover_value=crop_mask[pix]
                if landcover_value==16 or landcover_value==17 or landcover_value==18:
                    continue


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

    def trend_analysis_landcover_composition(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)


        fdir_all = data_root + f'landcover_composition_DIC\\'
        outdir = result_root + rf'trend_analysis\\landcover_composition\\'
        Tools().mk_dir(outdir, force=True)
        for fdir in os.listdir(fdir_all):
            if not 'urban' in fdir:
                continue

            outf=outdir+fdir
            print(outf)


            dic=T.load_npy_dir(fdir_all+fdir)


            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):

                time_series = dic[pix][10:]

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

    def LUCC_LAI_correlation(self):
        LAI_f=result_root + rf'split_original\Y\\LAI4g_2002_2020.npy'
        LUCC_fdir=data_root + rf'landcover_composition_DIC\\'
        outdir=result_root + rf'LUCC_LAI_correlation\\TIFF\\'
        Tools().mk_dir(outdir, force=True)
        LAI_dic = np.load(LAI_f, allow_pickle=True).item()


        for f_LUCC in os.listdir(LUCC_fdir):
            dic=T.load_npy_dir(LUCC_fdir+f_LUCC)
            spatial_dic={}
            for pix in tqdm(dic):
                if pix not in LAI_dic:
                    continue
                LC_vals=dic[pix]
                if T.is_all_nan(LC_vals):
                    continue


                LC_vals=np.array(LC_vals)
                LC_vals=LC_vals[10:]   ##landcover composition starts from 2002
                LAI_vals=LAI_dic[pix]
                if T.is_all_nan(LAI_vals):
                    continue
                LAI_vals=np.array(LAI_vals)
                ###correlation
                # r,p=T.nan_correlation(LAI_vals,LC_vals)
                try:
                    a, b ,r, p = T.nan_line_fit( LC_vals,LAI_vals,)
                except:
                    continue
                if np.isnan(r):
                    continue

                # spatial_dic[pix]={'r':r,'p':p}
                spatial_dic[pix]={'a':a,'b':b,'r':r,'p':p}
            df=T.dic_to_df(spatial_dic,'pix')
            # T.print_head_n(df)
            dic_a=T.df_to_spatial_dic(df,'a')

            dic_r=T.df_to_spatial_dic(df,'r')
            dic_p=T.df_to_spatial_dic(df,'p')

            outfile_r=outdir+f_LUCC.split('.')[0]+'_r.tif'
            outfile_p=outdir+f_LUCC.split('.')[0]+'_p.tif'
            outfile_a=outdir+f_LUCC.split('.')[0]+'_a.tif'
            DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_a,outfile_a)
            DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_r,outfile_r)
            DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_p,outfile_p)














        pass

    def bivariate_analysis(self):
        growthrate_trend=result_root+'growth_rate\OBS_extend\\LAI4g.npy'
        growthrate_pvalue=result_root+'growth_rate\OBS_extend\\LAI4g_p_value.npy'
        trend_trend=result_root+'trend_analysis\\original\\LAI4g_trend.npy'
        trend_pvalue=result_root+'trend_analysis\\original\\LAI4g_p_value.npy'

        growthrate_trend_dic=np.load(growthrate_trend,allow_pickle=True).item()
        growthrate_pvalue_dic=np.load(growthrate_pvalue,allow_pickle=True).item()

        trend_dic=np.load(trend_trend,allow_pickle=True).item()
        trend_pvalue_dic=np.load(trend_pvalue,allow_pickle=True).item()

        growthrate_list=[]
        trend_list=[]
        result_dic={}
        for pix in growthrate_trend_dic:
            trend_value=trend_dic[pix]
            trend_pvalue=trend_pvalue_dic[pix]
            growthrate_value=growthrate_trend_dic[pix]
            growthrate_pvalue=growthrate_pvalue_dic[pix]

            ###  trend siginificant positive and growthrate siginificant positive  accelerate greening
            # trend siginificant positive but growthrate siginificant negative     slow down greening
            # trend siginificant negative but growthrate siginificant positive     accelerate browning
            # trend siginificant negative but growthrate siginificant negative     slow down browning
            if trend_pvalue<0.05 and growthrate_pvalue<0.05:
                if trend_value>0 and growthrate_value>0:
                    result_dic[pix]='PP'
                elif trend_value>0 and growthrate_value<0:
                    result_dic[pix]='PN'
                elif trend_value<0 and growthrate_value>0:
                    result_dic[pix]='NP'
                elif trend_value<0 and growthrate_value<0:
                    result_dic[pix]='NN'
                else:
                    result_dic[pix]='NA'
            else:
                result_dic[pix]='NA'
        ##calculte the percentage of each category
        PP=0
        PN=0
        NP=0
        NN=0
        NA=0
        for pix in result_dic:
            if result_dic[pix]=='PP':
                PP+=1
            elif result_dic[pix]=='PN':
                PN+=1
            elif result_dic[pix]=='NP':
                NP+=1
            elif result_dic[pix]=='NN':
                NN+=1
            elif result_dic[pix]=='NA':
                NA+=1
        ##spatial plot
        arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic)
        plt.imshow(arr)
        plt.colorbar()
        plt.show()




        pass

    def calculate_PCI(self): ## PCI is the index to measure precipitation concentration in the growing season
        fdir=result_root + rf'split_original\Y\\'
        outdir=result_root + rf'PCI\\'
        Tools().mk_dir(outdir, force=True)
        ## 1) extract growing season 2) calculate PCI base on daily precipitation


        for f in os.listdir(fdir):
            if not 'precip' in f:
                continue
            outf=outdir+f.split('.')[0]
            print(outf)
            dic = dict(np.load(fdir+f, allow_pickle=True, encoding='latin1').item())
            PCI_dic={}
            for pix in tqdm(dic):
                time_series=dic[pix]
                time_series=np.array(time_series)
                time_series[time_series<-999]=np.nan
                if np.isnan(np.nanmean(time_series)):
                    continue
                PCI=self.PCI_index(time_series)
                PCI_dic[pix]=PCI
            np.save(outf, PCI_dic)

        pass

    def PCI_index(self,time_series):  ## calculate PCI index based  Oliver(1980)所定义 PCI指数
        sum=0
        for i in range(len(time_series)):
            val=time_series[i]^2
            sum+=val
        PCI=sum/((sum)^2)

        return PCI

    def calculate_asymmetry_response(self): ##
            fdir=result_root + rf'detrend_anomaly\\'
            outdir=result_root + rf'asymmetry_response\\'
            Tools().mk_dir(outdir, force=True)
            ## 1) extract growing season 2) calculate PCI base on daily precipitation
            for f in os.listdir(fdir):
                if not 'LAI4g' in f:
                    continue
                outf=outdir+f.split('.')[0]
                print(outf)
                dic = dict(np.load(fdir+f, allow_pickle=True, encoding='latin1').item())
                asymmetry_response_dic={}

                for pix in tqdm(dic):
                    asymmetry_response_list=[]
                    time_series=dic[pix]
                    time_series=np.array(time_series)
                    time_series[time_series<-999]=np.nan
                    if np.isnan(np.nanmean(time_series)):
                        continue
                    for window in range(0,24):

                        if window+5>len(time_series):
                            continue
                        asymmetry_response=self.asymmetry_response(time_series[window:window+5])

                        asymmetry_response_list.append(asymmetry_response)
                    asymmetry_response_dic[pix]=asymmetry_response_list
                np.save(outf, asymmetry_response_dic)


    def asymmetry_response(self,time_series):  ##
        positive_asymmetry = (np.max(time_series) - np.nanmean(time_series)) / abs(np.nanmean(time_series))
        negative_asymmetry = (np.nanmean(time_series) - np.min(time_series)) / abs(np.nanmean(time_series))
        asymmetry_response = positive_asymmetry - negative_asymmetry
        return asymmetry_response

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




class ResponseFunction:  # figure 5 in paper
    def __init__(self):

        # This class is used to calculate the structural equation model
        self.this_class_arr = result_root + rf'Dataframe\growing_season_original\\'
        self.dff = self.this_class_arr + 'growing_season_original.df'
        self.outdir = result_root + 'response_function/'
        T.mkdir(self.outdir, force=True)
        pass

    def run(self):
        # self.build_df()
        df, dff = self.__load_df()

        df_clean = self.clean_df(df)

        # self.plot_response_func(df_clean)
        self.plot_response_curve()
        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df, dff
        # return df_early,dff

    def clean_df(self, df):
        df = df[df['row'] > 120]

        # df = df[df['HI_class'] == 'Humid']
        # df = df[df['HI_class'] == 'Dryland']


        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def plot_response_func(self, df):
        T.print_head_n(df, 10)
        z_val_name_list = ['MCD', 'Trendy_ensemble', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLM5',
                           'IBIS_S2_lai',
                           'ISAM_S2_LAI', 'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                           'LPX-Bern_S2_lai', 'DLEM_S2_lai',
                           'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', ]

        z_val_name_list = ['LAI4g','Ensemble_S2','Ensemble_S1','Ensemble_S3' ]

        cm = 1 / 2.54
        plt.figure(figsize=(15 * cm, 7 * cm))

        for z_val_name in z_val_name_list:
            df_temp = df[df[f'{z_val_name}_p_value'] < 0.1]


            x_var = 'SPEI3_trend'
            y_var = 'VPD_trend'
            z_var = f'{z_val_name}_trend'

            x_bins = np.arange(-0.01, 0.01, 0.002)
            y_bins = np.arange(-0.01, 0.01, 0.002)

            matrix = []
            for i in range(len(y_bins)):
                if i == len(y_bins) - 1:
                    continue

                y_left = y_bins[i]
                y_right = y_bins[i + 1]

                matrix_i = []
                for j in range(len(x_bins)):
                    if j == len(x_bins) - 1:
                        continue
                    x_left = x_bins[j]
                    x_right = x_bins[j + 1]

                    df_temp_i = df_temp[df_temp[x_var] >= x_left]
                    df_temp_i = df_temp_i[df_temp_i[x_var] < x_right]
                    df_temp_i = df_temp_i[df_temp_i[y_var] >= y_left]
                    df_temp_i = df_temp_i[df_temp_i[y_var] < y_right]
                    mean = np.nanmean(df_temp_i[z_var].tolist())
                    matrix_i.append(mean)
                matrix.append(matrix_i)
            matrix = np.array(matrix)
            matrix = matrix[::-1, :]  # reverse

            plt.imshow(matrix, cmap='RdBu', interpolation='nearest')
            plt.colorbar()
            x_bins=np.round(x_bins,3)
            y_bins=np.round(y_bins,3)
            plt.xticks(np.arange(len(x_bins) - 1), x_bins[0:-1], rotation=45)
            plt.yticks(np.arange(len(y_bins) - 1), y_bins[0:-1])


            plt.title(f'{z_val_name}')
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.show()
            # plt.savefig(self.outdir + f'{region}_{z_val_name}.pdf', dpi=300)
            # plt.close()
    def plot_response_curve(self):
        df, dff = self.__load_df()
        df = self.clean_df(df)
        T.print_head_n(df, 10)

        x_name='GLEAM_SMroot'
        y_name='LAI4g'
        xvals=df[x_name].tolist()
        yvals=df[y_name].tolist()
        xvals_Clean=[]
        yvals_Clean=[]
        for i in range(len(xvals)):
            x=xvals[i]
            y=yvals[i]
            if type(x)==float:
                continue
            if type(y)==float:
                continue
            xvals_Clean.append(x)
            yvals_Clean.append(y)
        xvals=np.array(xvals_Clean)
        yvals=np.array(yvals_Clean)
        x_vals_flatten=xvals.flatten()
        y_vals_flatten=yvals.flatten()
        x_vals_flatten[x_vals_flatten>0.3]=np.nan
        y_vals_flatten[y_vals_flatten>2]=np.nan
        KDE_plot().plot_scatter(x_vals_flatten,y_vals_flatten,s=2)
        plt.ylim(0,2)
        plt.xlim(0,0.3)
        plt.show()



        pass


class bivariate_analysis():
    def __init__(self):
        pass
    def run(self):
        self.long_term_trend_moving_window_trend()

        pass
    def long_term_trend_moving_window_trend(self):
        import xymap

        tif_long_term_trend=result_root+'trend_analysis\\original\\OBS\\LAI4g_trend.tif'
        tif_moving_window_trend=result_root+rf'\extract_window\extract_original_window_trend_trend\15\\LAI4g_trend.tif'

        vmean_long_term=-0.01
        vmax_long_term=0.01
        vmean_moving_window=-0.001
        vmax_moving_window=0.001
        outdir=result_root+'bivariate_analysis\\'
        Tools().mk_dir(outdir,force=True)
        tif1=tif_long_term_trend
        tif2=tif_moving_window_trend
        x_label='Long-term trend'
        y_label='Moving window trend'
        min1=vmean_long_term
        max1=vmax_long_term
        min2=vmean_moving_window
        max2=vmax_moving_window
        outf=outdir+'LAI4g_trend.tif'

        xymap.Bivariate_plot().plot_bivariate_map(tif1, tif2, x_label, y_label, min1, max1, min2, max2, outf,n=(5,5), legend_title='')





        pass

class CCI_LC_preprocess():

    def __init__(self):
        self.datadir = rf'C:\Users\wenzhang1\Desktop\CCI_landcover\\'
        pass

    def run(self):
        # self.lccs_class_count()
        # self.lc_ratio_025_individal()

        # self.flags()  #not used
        self.composition()


        pass

    def flags(self):

        flags="no_data cropland_rainfed cropland_rainfed_herbaceous_cover cropland_rainfed_tree_or_shrub_cover cropland_irrigated mosaic_cropland mosaic_natural_vegetation tree_broadleaved_evergreen_closed_to_open tree_broadleaved_deciduous_closed_to_open tree_broadleaved_deciduous_closed tree_broadleaved_deciduous_open tree_needleleaved_evergreen_closed_to_open tree_needleleaved_evergreen_closed tree_needleleaved_evergreen_open tree_needleleaved_deciduous_closed_to_open tree_needleleaved_deciduous_closed tree_needleleaved_deciduous_open tree_mixed mosaic_tree_and_shrub mosaic_herbaceous shrubland shrubland_evergreen shrubland_deciduous grassland lichens_and_mosses sparse_vegetation sparse_tree sparse_shrub sparse_herbaceous tree_cover_flooded_fresh_or_brakish_water tree_cover_flooded_saline_water shrub_or_herbaceous_cover_flooded urban bare_areas bare_areas_consolidated bare_areas_unconsolidated water snow_and_ice";

        flags=flags.split(' ')
        values='0UB, 10UB, 11UB, 12UB, 20UB, 30UB, 40UB, 50UB, 60UB, 61UB, 62UB, 70UB, 71UB, 72UB, 80UB, 81UB, 82UB, 90UB, 100UB, 110UB, 120UB, 121UB, 122UB, 130UB, 140UB, 150UB, 151UB, 152UB, 153UB, 160UB, 170UB, 180UB, 190UB, 200UB, 201UB, 202UB, 210UB, 220UB'
        values=values.split(',')
        values=[i.replace('UB','') for i in values]
        values=[int(i) for i in values]
        flags_dic={}
        for i in range(len(flags)):
            flags_dic[values[i]]=flags[i]


        for value in flags_dic:
            print(value,flags_dic[value])
        pass
        return

    def lccs_class_count(self):
        nc_dir_all = join(self.datadir, 'nc')
        outdir = join(self.datadir, 'lccs_class_count')
        T.mk_dir(outdir)

        for fdir in T.listdir(nc_dir_all):

            for f in T.listdir(join(nc_dir_all, fdir)):
                if not f.endswith('.nc'):
                    continue
                print(f)

                nc_path = join(nc_dir_all, fdir, f)
                # nc_path = '/Volumes/NVME4T/hotdrought_CMIP/data/CCI_LC/nc/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1996-v2.0.7cds.nc'
                array = self.nc_to_array(nc_path)
                outf = join(outdir, f.replace('.nc', '.npy'))
                self.data_transform(array, outf)
        pass

    def nc_to_array(self, nc_path):
        nc = nc_path
        ncin = Dataset(nc, 'r')
        # print(ncin.variables)
        # for var in ncin.variables:
            # print(var)
        array = ncin.variables['lccs_class'][0][::]
        return array



    def data_transform(self, array, outf):
        # 不可并行，内存不足
        array = np.array(array, dtype=np.int16)
        row = len(array)
        col = len(array[0])
        window_height = 0.25 / 180 * row
        window_width = 0.25 / 360 * col
        window_height = int(window_height)
        window_width = int(window_width)

        # moving window
        spatial_dict = {}
        for i in tqdm(range(0, row, window_height)):
            for j in range(0, col, window_width):
                pix = (i, j)
                array_i = []
                for k in range(i, i + window_height):
                    for l in range(j, j + window_width):
                        array_i.append(array[k][l])
                array_i = np.array(array_i)
                array_i = np.array(array_i, dtype=np.int16)
                # print(np.std(array_i))
                # count every value number
                # if np.std(array_i) == 0:
                #     continue
                dic = {}
                for k in array_i:
                    if not k in dic:
                        dic[k] = 1
                    else:
                        dic[k] += 1
                dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
                spatial_dict[pix] = dic
        T.save_npy(spatial_dict, outf)

    def group_lc(self):
        group_lc = {
            10: 'crop',
            11: 'crop',
            12: 'crop',
            20: 'crop',
            30: 'crop',

            40: 'EBF',

            46: 'DBF',
            50: 'DBF',
            54: 'DBF',

            55: 'NEF',
            56: 'NEF',
            60: 'NEF',

            61: 'DNF',
            62: 'DNF',
            66: 'DNF',

            70: 'MIX',

            76: 'shrubland',
            80: 'shrubland',
            81: 'shrubland',

            82: 'grassland',

            190: 'urban'


        }
        return group_lc



    def lc_ratio_025_individal(self):  ## 这个生成的每一类lc在0.25度的比例，
        fdir = join(self.datadir, 'lccs_class_count')
        outdir = join(self.datadir, 'lc_ratio_025_individal')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):

            year=f.split('-')[-2]
            print(year)


            fpath = join(fdir, f)
            outdir_year=join(outdir,year)
            self.kernel_lc_ratio_025_individal(fpath,outdir_year)


        pass

    def kernel_lc_ratio_025_individal(self, fpath,outdir):
        spatial_dict = T.load_npy(fpath)
        T.mk_dir(outdir, force=True)

        spatial_dict_group = {}
        lc_type_list = []
        spatial_dict_ratio_all = {}
        pix_dict = {}
        for pix in tqdm(spatial_dict):
            count_dict = spatial_dict[pix]
            spatial_dict_ratio_i = {}
            i, j = pix
            i = i / 90
            j = j / 90
            i = int(i)
            j = int(j)
            pix = (i, j)
            pix_dict[pix] = 1
            for count_dict_i in count_dict:
                lc_type = count_dict_i[0]
                lc_count = count_dict_i[1]
                if not lc_type in lc_type_list:
                    lc_type_list.append(lc_type)
                ratio_i = lc_count / 8100. * 100
                spatial_dict_ratio_i[lc_type] = ratio_i
            # print(spatial_dict_ratio_i)
            spatial_dict_ratio_all[pix] = spatial_dict_ratio_i
        df = T.dic_to_df(spatial_dict_ratio_all, 'pix')
        T.print_head_n(df)
        lc_type_list.sort()
        for lc in lc_type_list:
            print(lc)
            spatial_dict_i = T.df_to_spatial_dic(df, lc)
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict_i)
            outf = join(outdir, f'{lc}.tif')
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, outf)
    def composition_flag(self):
        fdir = join(self.datadir, 'lc_ratio_025_individal')
        composition_lc_dic={
                'crop':(10,11,12,20,30),
                'EBF':[50],
                'DBF':(62,61,60),
                'DNF':(80,81,82),
                'ENF':(70,71,72),
            'grassland':[130],

            'shrubs':(120,121,122),
            'urban':[190]


        }
        reversed_lc_dic={}
        for lc in composition_lc_dic:
            lc_list=composition_lc_dic[lc]
            print(lc_list)
            for lc_i in lc_list:
                reversed_lc_dic[lc_i]=lc
        print(reversed_lc_dic)
        return reversed_lc_dic, composition_lc_dic

        pass

    def composition(self):
        fdir = join(self.datadir, 'lc_ratio_025_individal')
        outdir = join(self.datadir, 'composition')
        T.mk_dir(outdir,force=True)
        reversed_lc_dic, composition_lc_dic=self.composition_flag()
        # print(reversed_lc_dic)
        for year in T.listdir(fdir):


            for lc in composition_lc_dic:


                flag_list=composition_lc_dic[lc]
                fpath_list=[]
                for flag in flag_list:
                    f=f'{flag}.tif'
                    fpath=join(fdir,year,f)
                    fpath_list.append(fpath)


                print(fpath_list)
                outdiri=join(outdir,lc)
                T.mk_dir(outdiri,force=True)
                outf=join(outdiri,f'{year}.tif')
                Pre_Process().compose_tif_list(fpath_list,outf,method='sum')
                array, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(outf)
                array[array<=0]=np.nan
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(array,outf)

        pass


class calculating_variables:  ###

    def run(self):
        self.calculate_CV()
        # self.convert_tiff_to_npy()
        # self.create_CO2_dic()

        pass

    def calculate_CV(self):
        fdir = result_root + rf'extract_GS\OBS_extend\\'
        outdir = result_root + rf'state_variables\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):


            variable = f.split('.')[0]
            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)
            for pix in tqdm(val_dic, desc=variable):
                vals = val_dic[pix]
                if T.is_all_nan(vals):
                    continue
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                CV = std / mean
                val_dic[pix] = CV
            outf = outdir + rf'\{variable}_CV.npy'
            np.save(outf, val_dic)

        pass

    def convert_tiff_to_npy(self):
        fdir = result_root + 'state_variables\\'
        outdir = result_root + rf'\\state_variables\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            arr = ToRaster().raster2array(fdir + f)[0]
            dic=DIC_and_TIF().spatial_arr_to_dic(arr)
            outf = outdir + f.split('.')[0] + '.npy'

            np.save(outf, dic)
        pass
    def create_CO2_dic(self):
        CO2_f=data_root+rf'\Base_data\\monthly_in_situ_co2_mlo.xlsx'
        df=pd.read_excel(CO2_f,sheet_name='Sheet1')
        print(df)
        print(df.columns)

        for col in df.columns:
            new_col=col.replace(' ','')
            df.rename(columns={col:new_col},inplace=True)
            if new_col=='CO2filled[ppm]':
                df[new_col]=df[new_col].astype(float)
            else:
                df[new_col]=df[new_col].astype(int)

        df=df[['Yr','Mn','CO2filled[ppm]']]
        df=df[df['Yr']>=1982]
        df=df[df['Yr']<=2020]
        year_list=df['Yr'].unique()
        average_CO2_list=[]
        for i in range(len(df)):
            for year in year_list:
                if df.iloc[i]['Yr']==year:
                    average_CO2=df[df['Yr']==year]['CO2filled[ppm]'].mean()
                    average_CO2_list.append(average_CO2)
        df['average_CO2']=average_CO2_list
        ### CO2 dic
        CO2_list=[]
        for yr in year_list:

            for i in range(len(df)):
                if df.iloc[i]['Yr']==yr:
                    CO2_value=df.iloc[i]['average_CO2']

            CO2_list.append(CO2_value)

        ########create spatial dic with CO2 dic

        dic=DIC_and_TIF(pixelsize=0.25).void_spatial_dic()

        for pix in dic:

            dic[pix]=CO2_list


        outf=result_root+rf'\extract_GS\OBS_LAI\CO2.npy'
        np.save(outf,dic)



class moving_window():
    def __init__(self):
        pass
    def run(self):
        self.moving_window_extraction()
        # self.moving_window_extraction_for_LAI()
        # self.moving_window_trend_anaysis()
        # self.moving_window_average_anaysis()
        # self.produce_trend_for_each_slides()
        # self.calculate_trend_trend()
        # self.convert_trend_trend_to_tif()

        # self.plot_moving_window_time_series_area()
        # self.calculate_browning_greening_average_trend()
        # self.plot_moving_window_time_series()
        pass
    def moving_window_extraction(self):

        fdir = result_root + rf'extract_GS\OBS_LAI_extend\\'
        outdir = result_root + rf'\\extract_window\\extract_detrend_anomaly_window\\15\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if 'CO2' in f:
                continue


            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)
            if os.path.isfile(outf):
                continue

            dic = T.load_npy(fdir + f)
            window = 15

            new_x_extraction_by_window = {}
            for pix in tqdm(dic):

                time_series = dic[pix]
                time_series = np.array(time_series)

                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    print('error')
                    continue
                # print((len(time_series)))
                ### if all values are identical, then continue
                if np.nanmax(time_series) == np.nanmin(time_series):
                    continue

                new_x_extraction_by_window[pix] = self.forward_window_extraction_detrend_anomaly(time_series, window)

            T.save_npy(new_x_extraction_by_window, outf)

    def moving_window_extraction_for_LAI(self):  ## for LAI, GPP only
        variable_list=['LAI4g','GPP_CFE','GPP_baseline']

        fdir = result_root + rf'extract_GS\OBS_LAI_extend\\'
        outdir = result_root + rf'\\extract_window\\extract_original_window\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            fname=f.split('.')[0]
            print(fname)

            if not fname in variable_list:
                continue

            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)
            if os.path.isfile(outf):
                continue

            dic = T.load_npy(fdir + f)
            window = 15

            new_x_extraction_by_window = {}
            for pix in tqdm(dic):

                time_series = dic[pix]
                time_series = np.array(time_series)

                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    print('error')
                    continue
                # print((len(time_series)))
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

    def forward_window_extraction_detrend_anomaly(self, x, window):
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
        new_x_extraction_by_window = []
        for i in range(len(x)):
            if i + window >= len(x):
                continue
            else:
                anomaly = []

                x_vals = []
                for w in range(window):
                    x_val = (x[i + w])
                    x_vals.append(x_val)
                if np.isnan(np.nanmean(x_vals)):
                    continue

                x_mean=np.nanmean(x_vals)

                for i in range(len(x_vals)):
                    if x_vals[0]==None:
                        continue
                    x_anomaly=x_vals[i]-x_mean

                    anomaly.append(x_anomaly)
                if np.isnan(anomaly).any():
                    continue
                detrend_anomaly=signal.detrend(anomaly)


                new_x_extraction_by_window.append(detrend_anomaly)
        return new_x_extraction_by_window

    def moving_window_trend_anaysis(self):
        window_size=15
        fdir = result_root + rf'extract_window\\extract_original_window\\{window_size}\\'
        outdir = result_root + rf'\\extract_window\\extract_original_window_trend\\{window_size}\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):

            dic = T.load_npy(fdir + f)
            slides = 39-window_size
            outf = outdir + f.split('.')[0] + f'.npy'
            print(outf)
            # if os.path.isfile(outf):
            #     continue

            new_x_extraction_by_window = {}
            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):
                trend_list = []
                p_value_list = []

                time_series_all = dic[pix]
                if len(time_series_all)<24:
                    continue
                time_series_all = np.array(time_series_all)
                for ss in range(slides):
                    if np.isnan(np.nanmean(time_series_all)):
                        print('error')
                        continue
                    # print((len(time_series)))
                    ### if all values are identical, then continue
                    time_series=time_series_all[ss]
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    print(len(time_series))
                    slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series)), time_series)
                    trend_list.append(slope)
                    p_value_list.append(p_value)
                trend_dic[pix]=trend_list
                p_value_dic[pix]=p_value_list
                ## save
            np.save(outf, trend_dic)
            np.save(outf+'_p_value', p_value_dic)
            ##tiff
            # arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            #
            # p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')


            exit()

    def moving_window_average_anaysis(self):
        window_size = 15
        fdir = result_root + rf'extract_window\\extract_original_window\\{window_size}\\'
        outdir = result_root + rf'\\extract_window\\extract_original_window_average\\{window_size}\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):

            dic = T.load_npy(fdir + f)
            slides = 39 - window_size
            outf = outdir + f.split('.')[0] + f'.npy'
            print(outf)
            if os.path.isfile(outf):
                continue

            new_x_extraction_by_window = {}
            trend_dic = {}


            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                time_series_all = np.array(time_series_all)
                for ss in range(slides):
                    if np.isnan(np.nanmean(time_series_all)):
                        print('error')
                        continue

                    ### if all values are identical, then continue
                    if len(time_series_all)<24:
                        continue
                    time_series = time_series_all[ss]
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    print(len(time_series))
                    ##average
                    average=np.nanmean(time_series)

                    trend_list.append(average)

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)

            ##tiff
            # arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            #
            # p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')


    def produce_trend_for_each_slides(self):  ## 从上一个函数生成的一个像素24个trend, 变成 一个trend 一张图
        fdir=rf'D:\Project3\Result\extract_window\extract_original_window_trend\15\\'
        dryland_mask=join(data_root,'Base_data','dryland_mask.tif')
        dic_dryland=DIC_and_TIF().spatial_tif_to_dic(dryland_mask)


        for f in os.listdir(fdir):

            dic=T.load_npy(fdir+f)
            result_dic={}

            for slide in range(1,25):
                for pix in dic:
                    dryland_val=dic_dryland[pix]

                    vals=dic[pix]
                    vals=np.array(vals)
                    vals=vals*dryland_val
                    vals=np.array(vals)
                    if len(vals)!=24:
                        continue
                    result_dic[pix]=vals[slide-1]
                DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(result_dic,fdir+f.split('.')[0]+f'_{slide}.tif')


        pass
    def calculate_trend_trend(self):  ## calculate the trend of trend
        fdir = result_root + rf'extract_window\\extract_original_window_trend\\15\\GPCC\\'
        outdir = result_root + rf'\\extract_window\\extract_original_window_trend_trend\\15\\'
        T.mk_dir(outdir, force=True)
        dryland_mask=join(data_root,'Base_data','dryland_mask.tif')
        dic_dryland=DIC_and_TIF().spatial_tif_to_dic(dryland_mask)

        for f in os.listdir(fdir):
            if not 'npy' in f:
                continue
            if 'p_value' in f:
                continue


            dic = T.load_npy(fdir + f)

            outf = outdir + f.split('.')[0] + f'.npy'
            print(outf)



            trend_dic={}
            p_value_dic={}

            for pix in tqdm(dic):

                time_series_all = dic[pix]
                dryland_value=dic_dryland[pix]
                if np.isnan(dryland_value):
                    continue
                time_series_all = np.array(time_series_all)

                if len(time_series_all) < 24:
                    continue

                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series_all)), time_series_all)

                trend_dic[pix]=slope
                p_value_dic[pix]=p_value
                ## save
            np.save(outf, trend_dic)
            np.save(outf+'_p_value', p_value_dic)

            ##tiff

    def convert_trend_trend_to_tif(self):
        fdir = result_root + rf'extract_window\\extract_original_window_trend_trend\\15\\'
        outdir = result_root + rf'\\extract_window\\extract_original_window_trend_trend\\15\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):

            if 'tif' in f:
                continue

            dic=T.load_npy(fdir+f)

            arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(dic)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outdir + f.split('.')[0] + '.tif')









    def plot_moving_window_time_series_area(self): ## plot the time series of moving window and calculate the area of greening and browning

        f = data_root + rf'\Base_data\\glc_025\\glc2000_025.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)


        fdir = result_root + rf'extract_window\\extract_original_window_trend\\15\\'

        dic_trend=T.load_npy(fdir+'LAI4g.npy')
        dic_p_value=T.load_npy(fdir+'LAI4g.npy_p_value.npy')

        area_dic={}
        for ss in range(39-15):
            print(ss)

            greening_area=0
            browning_area=0
            no_change_area=0

            for pix in tqdm(dic_trend):
                landcover=val_dic[pix]
                if landcover==16:

                    continue

                # print(len(dic_trend[pix]))
                if len(dic_trend[pix])<24:
                    continue
                trend=dic_trend[pix][ss]
                p_value=dic_p_value[pix][ss]
                if trend>0 and p_value<0.1:
                    greening_area+=1
                elif trend<0 and p_value<0.1:
                    browning_area+=1
                else:
                    no_change_area+=1
                greening_area_percent=greening_area/(greening_area+browning_area+no_change_area)
                browning_area_percent=browning_area/(greening_area+browning_area+no_change_area)
                no_change_area_percent=no_change_area/(greening_area+browning_area+no_change_area)


            area_dic[ss]=[greening_area_percent,browning_area_percent,no_change_area_percent]
        df=pd.DataFrame(area_dic)


        df=df.T
        ##plot
        color_list=['green','red','grey']
        df.plot(kind='bar',stacked=True,color=color_list,legend=False)
        plt.legend(['Greening','Browning','No change'],loc='upper left',bbox_to_anchor=(1.0, 1.0))
        plt.ylabel('percentage')
        plt.xlabel('moving window')
        plt.xticks(np.arange(0, 24, 1))
        plt.title('Area of greening and browning')
        plt.show()
        exit()

    def calculate_browning_greening_average_trend(self): ## each winwow, greening or browning pixels average trend
        f = data_root + rf'\Base_data\\glc_025\\glc2000_025.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        fdir = result_root + rf'extract_window\\extract_original_window_trend\\15\\'
        outdir = result_root + rf'\\extract_window\\extract_original_window_trend\\15\\'
        T.mk_dir(outdir, force=True)
        dic_trend=T.load_npy(fdir+'LAI4g.npy')
        dic_p_value=T.load_npy(fdir+'LAI4g.npy_p_value.npy')

        area_dic = {}
        for ss in range(39 - 15):
            print(ss)

            greening_value = []
            browning_value = []
            no_change_value = []
            all_value=[]
            non_sig_greening_value=[]
            non_sig_browning_value=[]
            for pix in tqdm(dic_trend):
                landcover = val_dic[pix]
                if landcover == 16:
                    continue
                # print(len(dic_trend[pix]))
                if len(dic_trend[pix]) < 24:
                    continue
                trend = dic_trend[pix][ss]
                p_value = dic_p_value[pix][ss]

                if p_value<0.1:
                    if trend>0:
                        value=trend
                        greening_value.append(value)
                    elif trend<0:
                        value=trend
                        browning_value.append(value)
                    else:
                        raise
                else:
                    if trend>0:
                        value=trend
                        non_sig_greening_value.append(value)
                    elif trend<0:
                        value=trend
                        non_sig_browning_value.append(value)
                    else:
                        continue

                    value=trend
                    no_change_value.append(value)
                all_value.append(value)


            greening_value_average = np.nanmean(greening_value)
            browning_value_average = np.nanmean(browning_value)
            no_change_value_average = np.nanmean(no_change_value)
            non_sig_greening_value_average=np.nanmean(non_sig_greening_value)
            non_sig_browning_value_average=np.nanmean(non_sig_browning_value)
            all_value_average=np.nanmean(all_value)
            area_dic[ss] = [greening_value_average, browning_value_average, no_change_value_average,all_value_average,non_sig_greening_value_average,non_sig_browning_value_average]

        df = pd.DataFrame(area_dic)
        df = df.T
        ##plot
        color_list = ['green', 'red', 'grey','black','cyan','orange']
        df.plot(kind='bar', stacked=False, color=color_list, legend=False)
        plt.legend(['Greening', 'Browning', 'No change','all_value','non-sig-greening','non-sig-browning'], loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.ylabel('LAI(m3/m3/year/year)')
        plt.xlabel('moving window')
        plt.xticks(np.arange(0, 24, 1))
        #### set line
        plt.axhline(y=-0.02, color='black', linestyle='--', linewidth=0.5)
        plt.axhline(y=0.02, color='black', linestyle='--', linewidth=0.5)
        plt.title('')
        plt.show()
        exit()

    def plot_moving_window_time_series(self): ### each winwow, greening or browning pixels average original
        fdir_trend = result_root + rf'extract_window\\extract_original_window_trend\\15\\GPCC\\'


        dic_trend = T.load_npy(fdir_trend + 'GPCC.npy')


        area_dic = {}
        for ss in range(39 - 15):
            print(ss)

            trend_value_list = []

            for pix in tqdm(dic_trend):
                # print(len(dic_trend[pix]))
                if len(dic_trend[pix]) < 24:
                    continue
                trend_value = dic_trend[pix][ss]

                trend_value_list.append(trend_value)
            trend_value_average = np.nanmean(trend_value_list)
            area_dic[ss] = [trend_value_average]
        df_new = pd.DataFrame(area_dic)
        df_new = df_new.T
        ##plot
        color_list = ['black']
        df_new.plot( color=color_list, legend=False)
        plt.legend(['trend'], loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.ylabel('precipitaton(mm/year/year)')
        plt.xlabel('moving window')
        plt.xticks(np.arange(0, 24, 1))
        plt.show()






        ##


        ####


        df_new = pd.DataFrame(area_dic)
        df_new = df_new.T
        ##plot
        color_list = ['black']
        df_new.plot( color=color_list, legend=False)
        plt.legend(['trend'], loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.ylabel('precipitaton(mm/year/year)')
        plt.xlabel('moving window')
        plt.xticks(np.arange(0, 24, 1))
        plt.show()








class multi_regression_window():
    def __init__(self):
        self.fdirX=result_root+rf'extract_window\extract_detrend_anomaly_window\\15\\'
        self.fdir_Y=result_root+rf'extract_window\extract_detrend_anomaly_window\\15\\'

        self.xvar_list = ['Tempmean','GLEAM_SMroot','VPD']
        self.y_var = ['LAI4g']
        pass

    def run(self):

        self.window = 39-15
        outdir = result_root + rf'multi_regression_moving_window\\window15\\'
        T.mk_dir(outdir, force=True)

        ### step 1 build dataframe
        for i in range(self.window):

            df_i = self.build_df(self.fdirX, self.fdir_Y, self.xvar_list, self.y_var,i)


            outf= outdir+rf'\\window{i:02d}.npy'
            if os.path.isfile(outf):
                continue
            print(outf)

            self.cal_multi_regression_beta(df_i,self.xvar_list, outf)  # 修改参数
        # step 2 crate individial files
        # self.plt_multi_regression_result(outdir,self.y_var)

        # step 3 covert to time series

        # self.convert_files_to_time_series(outdir,self.y_var)
        ### step`
        self.plot_moving_window_time_series()

    def build_df(self, fdir_X, fdir_Y, xvar_list,y_var,w):


        df = pd.DataFrame()
        dic_y=T.load_npy(fdir_Y+y_var[0]+'.npy')
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
                # print(len(x_arr[pix]))
                if len(x_arr[pix]) < self.window:
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

    def plt_multi_regression_result(self, multi_regression_result_dir,y_var):
        fdir = multi_regression_result_dir+'\\'+'npy\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if 'pvalue' in f:
                continue
            print(f)

            w=f.split('\\')[-1].split('.')[0][-2:]


            w=int(w)

            dic = T.load_npy(fdir+f)
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
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f'{multi_regression_result_dir}\\TIFF\\{var_i}_{y_var[0]}_{w:02d}.tif')
                std = np.nanstd(arr)
                mean = np.nanmean(arr)
                vmin = mean - std
                vmax = mean + std
            #     plt.figure()
            #     # arr[arr > 0.1] = 1
            #     plt.imshow(arr,vmin=-5,vmax=5)
            #
            #     plt.title(var_i)
            #     plt.colorbar()
            #
            # plt.show()
    def convert_files_to_time_series(self, multi_regression_result_dir,y_var):
        dryland_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(dryland_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = multi_regression_result_dir+'\\'+'TIFF\\'


        variable_list= ['CO2', 'GLEAM_SMroot', 'Tmax', 'VPD']

        for variable in variable_list:
            array_list = []

            for f in os.listdir(fdir):

                if not variable in f:
                    continue
                if not f.endswith('.tif'):
                    continue
                if 'pvalue' in f:
                    continue
                print(f)

                array= ToRaster().raster2array(fdir+f)[0]
                ## all array concate

                array_list.append(array)
            array_list=np.array(array_list)

            ## array_list to dic
            dic=DIC_and_TIF(pixelsize=0.25).void_spatial_dic()
            for pix in dic:
                r, c = pix
                classval = dic_dryland_mask[pix]
                if np.isnan(classval):
                    continue


                dic[pix]=array_list[:,r,c]
                if np.nanmean(dic[pix])<=5:
                    continue
                # print(len(dic[pix]))
                # exit()
            outdir=multi_regression_result_dir+'\\'+'npy_time_series\\'
            print(outdir)
            # exit()
            T.mk_dir(outdir,force=True)
            outf=outdir+rf'\\{variable}_{y_var[0]}.npy'
            np.save(outf,dic)

        pass

    def plot_moving_window_time_series(self):
        df= T.load_df(result_root + rf'Dataframe\moving_window_15\moving_window_15.df')
        variable_list=['CO2', 'GLEAM_SMroot', 'Tmax', 'VPD']

        df=df.dropna()

        fig = plt.figure()
        i = 1
        variable='Tmax'


        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]
            ax = fig.add_subplot(1, 3, i)

            vals = df_region[f'{variable}_LAI4g'].tolist()
            vals_nonnan = []

            for val in vals:
                if type(val) == float:  ## only screening
                    continue
                if np.isnan(np.nanmean(val)):
                    continue
                if np.nanmean(val) <=-99:
                    continue

                vals_nonnan.append(val)
            ###### calculate mean
            vals_mean = np.array(vals_nonnan)  ## axis=0, mean of each row  竖着加
            vals_mean = np.nanmean(vals_mean, axis=0)
            vals_mean = vals_mean.tolist()
            plt.plot(vals_mean, label=variable)

            i = i + 1

            plt.xlabel('year')

            plt.ylabel(f'{variable}_LAI4g')
            # plt.legend()

            plt.title(region)
        plt.show()

    def plot_sensitivity_as_function_of_SM(self):  ## plot sensitivity for each drying bin

        pass

class residual_method():
    def __init__(self):
        self.period='2002_2020'
        self.x_var_list_climate = ['Tmax', 'GLEAM_SMroot', 'VPD']

        self.x_var_list_all = ['Tmax', 'GLEAM_SMroot', 'VPD','CO2']

        self.residual_result_dir_detrend_climate_model_results=result_root+rf'residual_method\\\\{self.period}\\detrend_climate_model_results\\'
        T.mk_dir(self.residual_result_dir_detrend_climate_model_results,force=True)

        self.multi_regression_result=result_root+rf'residual_method\\{self.period}\\multiregression\\Tmax\\'
        T.mk_dir(self.multi_regression_result,force=True)

        self.multi_regression_result_tiff = result_root + rf'residual_method\\{self.period}\\multiregression\\'
        T.mk_dir(self.multi_regression_result_tiff, force=True)


        pass

    def run(self):

        #step 1 build detrended climatic dataframe model
        df = self.build_df(self.x_var_list_climate,self.period)  ##climate 是detrend
        self.cal_multi_regression_beta(df, self.x_var_list_climate)

        # # step 2 put real data and calculate residual as well as sensitivity


        df_real = self.build_df_real(self.x_var_list_climate, self.period)  ##climate 是real data
        self.residual_calculate(df_real, self.x_var_list_climate)


        # step 3 plot
        self.plt_multi_regression_result(self.multi_regression_result,)

        ##check pixel
        # self.check_pixel()


    def build_df(self,x_var_list,period):

        df = pd.DataFrame()
        fdir_X = result_root + rf'split_anomaly\X\\'
        fdir_Y = result_root + rf'split_anomaly\Y\\'


        y_variable_list=['LAI4g']

        # filey=fdir_Y+y_variable_list[0]+'.npy'   ##all
        filey=fdir_Y+y_variable_list[0]+f'_{period}.npy'
        print(filey)

        dic_y=T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list=[]


        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue
            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
            if yvals[0] == None:
                continue

            yvals_detrend=signal.detrend(yvals)
            if np.isnan(np.nanmean(yvals)):
                continue

            yvals_detrend=np.array(yvals_detrend)
            yvals_detrend_scale=yvals_detrend

            pix_list.append(pix)
            y_val_list.append(yvals_detrend_scale)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x


        for xvar in x_var_list:


            # print(var_name)
            x_val_list = []
            # filex=fdir_X+xvar+'.npy'
            filex=fdir_X+xvar+f'_{period}.npy'


            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
            dic_x = T.load_npy(filex)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                xvals = dic_x[pix]
                xvals = np.array(xvals)
                if len(xvals) == 0:
                    x_val_list.append([])
                    continue

                xvals = T.interp_nan(xvals)
                if xvals[0] == None:
                    x_val_list.append([])
                    continue
                xvals_detrend=signal.detrend(xvals)
                if np.isnan(np.nanmean(xvals)):
                    continue


                xvals_detrend=np.array(xvals_detrend)
                xvals_detrend_scale=xvals_detrend
                x_val_list.append(xvals_detrend_scale)


            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list
        T.print_head_n(df)


        return df

    def build_df_real(self,x_var_name_list,period):

        fdir_X = result_root + rf'split_anomaly\X\\'
        fdir_Y = result_root + rf'split_anomaly\Y\\'

        y_variable_list=['LAI4g']

        df = pd.DataFrame()

        # filey=fdir_Y+y_variable_list[0]+'.npy'
        # print(filey)
        filey=fdir_Y+y_variable_list[0]+f'_{period}.npy'


        dic_y=T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list=[]


        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue
            #####processing yvals
            yvals = T.interp_nan(yvals)
            if yvals[0] == None:
                continue

            yvals = np.array(yvals)

            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x


        for xvar in x_var_name_list:


            # print(var_name)
            x_val_list = []
            # filex=fdir_X+xvar+'.npy'
            filex=fdir_X+xvar+f'_{period}.npy'

            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
            dic_x = T.load_npy(filex)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                xvals = dic_x[pix]
                xvals= np.array(xvals)
                xvals = np.array(xvals)
                xvals= T.interp_nan(xvals)
                if xvals[0] == None:
                    x_val_list.append([])
                    continue
                if len(xvals) == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(xvals)
            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list
        T.print_head_n(df)


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


    def cal_multi_regression_beta(self, df,x_var_list_climate):

        import joblib

        outdir = self.residual_result_dir_detrend_climate_model_results


        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            r,c=pix

            # if not r== 480:
            #     continue
            # if not c== 447:
            #     continue
            # print(r, c)

            y_vals = row['y']


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list_climate:

                x_vals = row[x]


                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue


                df_new[x] = x_vals

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals

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
            ##coef
            # coef=np.array(linear_model.coef_)
            # coef_dic=dict(zip(x_var_list_valid_new,coef))
            # print(coef_dic)
            # exit()
            ##save model

            outf = outdir + rf'\\{pix}.pkl'
            joblib.dump(linear_model, outf)

        return

    pass

    def residual_calculate(self,df,period):
        import joblib

        outdirdetrend_climate_model_results=self.residual_result_dir_detrend_climate_model_results
        outdirmultiregression=self.multi_regression_result

        ##load y_baseline file
        f_y_baseline = result_root + rf'\\residual_method\{period}\baseline\\LAI4g_{period}.npy'
        dic_baseline = T.load_npy(f_y_baseline)

        #### repredict LAI using model and real climate data and calculate residual
        ##load model
        residual_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            ##load model
            model_f=outdirdetrend_climate_model_results+rf'\\{pix}.pkl'
            if not os.path.isfile(model_f):
                residual_list.append([])
                continue

            model_f=joblib.load(model_f)

            y_val=row['y']

            y_val=np.array(y_val)

            x_val_list=[]

            ##### build x using model coefs
            ##get regression name list
            model_f_name=model_f.feature_names_in_

            for x in model_f_name:
                x_val=row[x]

                x_val_list.append(x_val)

            x_val_list=np.array(x_val_list)
            x_val_list=x_val_list.T

            y_pred=model_f.predict(x_val_list)

            if np.nanmean(y_pred)==0:
                residual_list.append([])
                continue

            residual=y_val-y_pred
            #######len residual should be equal to len(df )

            if len(residual)!=len(y_val):
                residual_list.append([])
                continue

            if np.isnan(np.nanmean(residual)):
                residual_list.append([])
                continue
            if np.nanmean(residual)<=-99:
                residual_list.append([])
                continue
            residual_list.append(residual)
        df['residual']=residual_list

        df=df.dropna()
        # T.print_head_n(df)

        #####calculate partial derivative with multi-regression
        multi_derivative = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row.pix
            if not pix in dic_baseline:
                continue
            baseline=dic_baseline[pix]

            y_vals = row['residual']


            #  calculate partial derivative with multi-regression and real climate data and CO2
            df_new = pd.DataFrame()

            x_var_list_valid = []

            for x in self.x_var_list_all:  ###climate and CO2

                x_vals = row[x]

                if len(x_vals) == 0:
                    continue


                if np.isnan(np.nanmean(x_vals)):
                    continue

                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue

                df_new[x] = x_vals


                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals/baseline

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
            ##coef

            coef_ = np.array(linear_model.coef_)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))

            multi_derivative[pix] = coef_dic
        outf=outdirmultiregression+rf'\\residual.npy'

        T.save_npy(multi_derivative, outf)

        pass

    def calculate_climatic_contribution(self, df, period):  ## calculate climatic contribution
        import joblib

        outdirdetrend_climate_model_results = self.residual_result_dir_detrend_climate_model_results
        outdirmultiregression = self.multi_regression_result

        f_y_baseline = result_root + rf'\\residual_method\{period}\baseline\\LAI4g_{period}.npy'
        dic_baseline = T.load_npy(f_y_baseline)

        #### repredict LAI using model and real climate data and calculate residual
        ##load model
        y_predict_list = []
        multi_derivative = {}
        spatial_dic_result = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            baseline=dic_baseline[pix]
            ##load model

            model_f = outdirdetrend_climate_model_results + rf'\\{pix}.pkl'
            if not os.path.isfile(model_f):
                y_predict_list.append([])
                continue

            model_f = joblib.load(model_f)


            x_val_list = []

            ##### build x using model coefs
            ##get regression name list
            model_f_name = model_f.feature_names_in_
            # print(model_f_name)

            for x in model_f_name:
                x_val = row[x]

                x_val_list.append(x_val)

            x_val_list = np.array(x_val_list)
            x_val_list = x_val_list.T

            y_pred = model_f.predict(x_val_list) ## y_climatic

            ####


            spatial_dic_result[pix] = y_pred/baseline

            ### calculate climatic contribution
            liner_model_climatic = LinearRegression()
            liner_model_climatic.fit(x_val_list, y_pred)
            coef_ = np.array(liner_model_climatic.coef_)
            coef_dic = dict(zip(model_f_name, coef_))
            # print(coef_dic)
            # exit()


            multi_derivative[pix] = coef_dic
        outf = outdirmultiregression + rf'\\climatic_contribution.npy'

        T.save_npy(multi_derivative, outf)

    def plt_multi_regression_result(self, multi_regression_result):


        f=multi_regression_result+rf'\\residual.npy'
        # f=multi_regression_result+rf'\\climatic_contribution.npy'

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
            # outdir=self.multi_regression_result_tiff
            outdir=multi_regression_result+rf'\\TIFF_non_climatic\\'
            T.mk_dir(outdir,force=True)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f'{outdir}\\{var_i}.tif')
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

    def check_pixel(self):
        fdir = result_root + rf'residual_method\detrend_climate_model_results\\'
        spatial_dic = {}

        for f in os.listdir(fdir):
            pix=f.split('.')[0]
            print(pix)
            pix=pix.replace('(','')
            pix=pix.replace(')','')
            pix=pix.split(',')
            pix=(int(pix[0]),int(pix[1]))
            print(pix)
            spatial_dic[pix]=1
        arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.show()


class Contribution_Lixin():  ## earth future method
    def __init__(self):
        self.period='2002_2020'


        self.x_var_list_all = ['Tempmean', 'SPEI3', 'VPD','CO2']

        self.residual_result_dir_detrend_climate_model_results=result_root+rf'contribution_Lixin\\\\{self.period}\\all_model_results\\'
        T.mk_dir(self.residual_result_dir_detrend_climate_model_results,force=True)

        self.multi_regression_result=result_root+rf'contribution_Lixin\\{self.period}\\multiregression\\'
        T.mk_dir(self.multi_regression_result,force=True)

        self.multi_regression_result_tiff = result_root + rf'contribution_Lixin\\{self.period}\\multiregression\\'
        T.mk_dir(self.multi_regression_result_tiff, force=True)


        pass

    def run(self):

        #step 1 build detrended climatic dataframe model
        df_all = self.build_df(self.x_var_list_all,self.period)  ##climate 是detrend
        self.cal_multi_regression_beta(df_all, self.x_var_list_all)

        # # step 2 put real data and calculate residual as well as sensitivity
        fixed_var_list=['Tempmean','SPEI3','VPD','CO2']
        fixed_var_list=['Tempmean']
        for fixed_var in fixed_var_list:
            df_fixed = self.build_df_fixed(self.x_var_list_all,self.period,fixed_var=fixed_var)
            self.residual_calculate(df_all,df_fixed, fixed_var=fixed_var)

            self.calculate_fix_factor_contribution(df_real=df_all, df_fixed= df_fixed, fixed_var='Tempmean')

            # step 3 plot
            self.plt_multi_regression_result(self.multi_regression_result,fixed_var='Tempmean')

            ##check pixel
            # self.check_pixel()


    def build_df(self,x_var_list,period):

        df = pd.DataFrame()
        fdir_X = result_root + rf'split_anomaly\X\\'
        fdir_Y = result_root + rf'split_anomaly\Y\\'


        y_variable_list=['LAI4g']

        # filey=fdir_Y+y_variable_list[0]+'.npy'   ##all
        filey=fdir_Y+y_variable_list[0]+f'_{period}.npy'
        print(filey)

        dic_y=T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list=[]


        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue
            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
            if yvals[0] == None:
                continue


            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x


        for xvar in x_var_list:


            # print(var_name)
            x_val_list = []
            # filex=fdir_X+xvar+'.npy'
            filex=fdir_X+xvar+f'_{period}.npy'


            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
            dic_x = T.load_npy(filex)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                xvals = dic_x[pix]
                xvals = np.array(xvals)
                if len(xvals) == 0:
                    x_val_list.append([])
                    continue

                xvals = T.interp_nan(xvals)
                if xvals[0] == None:
                    x_val_list.append([])
                    continue


                x_val_list.append(xvals)


            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list
        T.print_head_n(df)


        return df

    def build_df_fixed(self,x_var_name_list,period, fixed_var='CO2'):

        fdir_X = result_root + rf'split_anomaly\X\\'
        fdir_Y = result_root + rf'split_anomaly\Y\\'

        y_variable_list=['LAI4g']

        df = pd.DataFrame()

        # filey=fdir_Y+y_variable_list[0]+'.npy'
        # print(filey)
        filey=fdir_Y+y_variable_list[0]+f'_{period}.npy'


        dic_y=T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list=[]


        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue
            #####processing yvals
            yvals = T.interp_nan(yvals)
            if yvals[0] == None:
                continue

            yvals = np.array(yvals)

            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x


        for xvar in x_var_name_list:


            # print(var_name)
            x_val_list = []
            # filex=fdir_X+xvar+'.npy'
            filex=fdir_X+xvar+f'_{period}.npy'

            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
            dic_x = T.load_npy(filex)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                if xvar==fixed_var:
                    vals=dic_x[pix][0]*np.ones(len(dic_x[pix]))
                    x_val_list.append(vals)
                    # print(vals)
                    continue
                xvals = dic_x[pix]
                if len(xvals) == 0:
                    x_val_list.append([])
                    continue
                if np.isnan(np.nanmean(xvals)):
                    x_val_list.append([])
                    continue
                if np.nanmean(xvals)<=-99:
                    x_val_list.append([])
                    continue
                if len(xvals) == 0:
                    x_val_list.append([])
                    continue
                xvals= np.array(xvals)
                xvals = np.array(xvals)
                xvals= T.interp_nan(xvals)
                if xvals[0] == None:
                    x_val_list.append([])
                    continue
                if len(xvals) == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(xvals)

            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list
        T.print_head_n(df)

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


    def cal_multi_regression_beta(self, df,x_var_list_climate):

        import joblib

        outdir = self.residual_result_dir_detrend_climate_model_results


        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            r,c=pix

            # if not r== 480:
            #     continue
            # if not c== 447:
            #     continue
            # print(r, c)

            y_vals = row['y']


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list_climate:

                x_vals = row[x]


                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue


                df_new[x] = x_vals

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals

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
            ##coef
            # coef=np.array(linear_model.coef_)
            # coef_dic=dict(zip(x_var_list_valid_new,coef))
            # print(coef_dic)
            # exit()
            ##save model

            outf = outdir + rf'\\{pix}.pkl'
            joblib.dump(linear_model, outf)

        return

    pass

    def residual_calculate(self,df_fixed,df_real,fixed_var):
        import joblib

        outdirdetrend_climate_model_results=self.residual_result_dir_detrend_climate_model_results
        outdirmultiregression=self.multi_regression_result+rf'\\{fixed_var}\\'
        T.mk_dir(outdirmultiregression,force=True)


        #### repredict LAI using model and real climate data and calculate residual
        ##load model
        residual_list = []
        for i, row in tqdm(df_fixed.iterrows(), total=len(df_fixed)):
            pix = row.pix
            ##load model
            model_f=outdirdetrend_climate_model_results+rf'\\{pix}.pkl'
            if not os.path.isfile(model_f):
                residual_list.append([])
                continue

            model_f=joblib.load(model_f)

            y_val=row['y']

            y_val=np.array(y_val)

            x_val_list=[]

            ##### build x using model coefs
            ##get regression name list
            model_f_name=model_f.feature_names_in_

            for x in model_f_name:
                x_val=row[x]

                x_val_list.append(x_val)

            x_val_list=np.array(x_val_list)
            x_val_list=x_val_list.T

            y_pred=model_f.predict(x_val_list)

            if np.nanmean(y_pred)==0:
                residual_list.append([])
                continue

            residual=y_val-y_pred
            #######len residual should be equal to len(df )

            if len(residual)!=len(y_val):
                residual_list.append([])
                continue

            if np.isnan(np.nanmean(residual)):
                residual_list.append([])
                continue
            if np.nanmean(residual)<=-99:
                residual_list.append([])
                continue
            residual_list.append(residual)
        df_fixed['residual']=residual_list

        df=df_fixed.dropna()
        ###save df please
        T.print_head_n(df)
        outf=outdirmultiregression+rf'\\residual.npy'
        T.save_df(df, outf)
        T.df_to_excel(df, outf.replace('.npy','.xlsx'))
        # T.print_head_n(df)

    def calculate_fix_factor_contribution(self,df_real,df_fixed, fixed_var):  ## calculate climatic contribution
        df=T.load_df(self.multi_regression_result+rf'\\{fixed_var}\\residual.npy')

        #####calculate partial derivative with multi-regression

        multi_derivative = {}

        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row.pix

            y_vals = row['residual']

            #  calculate partial derivative with multi-regression and real climate data and CO2
            df_new = pd.DataFrame()

            ### extract x_vals from df_real

            x_vals_real = df_real.loc[df_real['pix'] == pix, f'{fixed_var}'].values[0]
            x_vals_fixed=df_fixed.loc[df_fixed['pix'] == pix, f'{fixed_var}'].values[0]
            x_vals_real = np.array(x_vals_real)
            x_vals_fixed = np.array(x_vals_fixed)

            x_vals_delta=x_vals_real-x_vals_fixed


            if len(x_vals_delta) == 0:
                continue

            if np.isnan(np.nanmean(x_vals_delta)):

                continue
            if len(x_vals_delta) != len(y_vals):
                continue

            x_vals_delta=x_vals_delta.T  ### 1d array

            df_new[fixed_var] = x_vals_delta
            df_new['y'] = y_vals


            # T.print_head_n(df_new)

            df_new = df_new.dropna(axis=1, how='all')

            df_new = df_new.dropna()
            if len(df_new)==0:
                continue

            linear_model = LinearRegression()
            xi=df_new[fixed_var].values.reshape(-1,1)
            yi=df_new['y'].values.reshape(-1,1)
            linear_model.fit(xi, yi)
            ##coef
            beta = linear_model.coef_
            # print(beta)
            # coef_dic = dict(zip(x_var_list_valid_new, coef_))

            ##coef

            multi_derivative[pix] = beta[0][0]

        outf=self.multi_regression_result+rf'\\{fixed_var}_contribution.npy'

        T.save_npy(multi_derivative, outf)

        pass


    def plt_multi_regression_result(self, multi_regression_result,fixed_var):


        f=multi_regression_result+rf'\\{fixed_var}_contribution.npy'
        # f=multi_regression_result+rf'\\climatic_contribution.npy'

        dic = T.load_npy(f)

        spatial_dic = {}
        for pix in dic:
            # print(pix)
            vals = dic[pix]
            spatial_dic[pix] = vals


        arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            # outdir=self.multi_regression_result_tiff
        outdir=multi_regression_result+rf'\\'
        T.mk_dir(outdir,force=True)
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f'{outdir}\\{fixed_var}.tif')
        std = np.nanstd(arr)
        mean = np.nanmean(arr)
        vmin = mean - std
        vmax = mean + std
        plt.figure()
        # arr[arr > 0.1] = 1
        plt.imshow(arr,vmin=-5,vmax=5)


        plt.colorbar()

        plt.show()

    def check_pixel(self):
        fdir = result_root + rf'residual_method\detrend_climate_model_results\\'
        spatial_dic = {}

        for f in os.listdir(fdir):
            pix=f.split('.')[0]
            print(pix)
            pix=pix.replace('(','')
            pix=pix.replace(')','')
            pix=pix.split(',')
            pix=(int(pix[0]),int(pix[1]))
            print(pix)
            spatial_dic[pix]=1
        arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.show()




class quadratic_regression():  ### repeat Matthew's method
    def __init__(self):
        self.fdirX=data_root+rf'monthly_data\Precip\\'
        self.fdirY=data_root+rf'monthly_data\LAI4g\\'


        self.period=('1982_2020')

        self.y_var = ['LAI4g']
        self.xvar = ['Precip']


        self.multi_regression_result_dir=result_root+rf'quadratic_regression\\{self.period}\\'
        T.mk_dir(self.multi_regression_result_dir,force=True)

        self.multi_regression_result_f = result_root + rf'quadratic_regression\\\\{self.period}\\{self.y_var[0]}_no_validation.npy'
        pass

    def run(self):

        #step 1 build dataframe
        df = self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var,self.period)
        T.print_head_n(df)
        # exit()
        #
        # # # # step 2 cal correlation
        self.cal_quadratic_regression(df, self.xvar)  # 修改参数

        # step 3 plot
        # self.plt_quadratic_regression_result(self.multi_regression_result_dir,self.y_var[0],self.period)

        # self.plot_proportation_response()


    def build_df(self,fdir_X,fdir_Y,fx_list,fy,period):

        df = pd.DataFrame()
        # filey=fdir_Y+fy[0]+'.npy'
        # print(filey)
        # dic_y = T.load_npy(fdir_Y)
        dic_y=T.load_npy_dir(fdir_Y)


        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue
            ## detrend

            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
            if yvals[0] == None:
                continue

            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        for xvar in fx_list:

            # print(var_name)
            x_val_list = []
            # filex=fdir_X+xvar+'.npy'
            # filex = fdir_X + xvar + f'_{period}.npy'

            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
            # dic_x = T.load_npy(filex)
            dic_x = T.load_npy_dir(fdir_X)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                xvals = dic_x[pix]
                xvals = np.array(xvals)
                if len(xvals) == 0:
                    x_val_list.append([])
                    continue

                xvals = T.interp_nan(xvals)
                if xvals[0] == None:
                    x_val_list.append([])
                    continue

                x_val_list.append(xvals)

            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list
        T.print_head_n(df)

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


    def cal_quadratic_regression(self, df, x_var_list):

        import statsmodels.api as sm
        import joblib

        outf = self.multi_regression_result_f

        result_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']

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


                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue
                x_vals_detrend = signal.detrend(x_vals) #detrend

                df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue
            df_new['y'] = y_vals_detrend  # detrend


            df_new = df_new.dropna(axis=1, how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()

            ####### linear model

            ## seperate samples
            df_new = df_new.sample(frac=1).reset_index(drop=True)
            df_new_train = df_new.iloc[:int(len(df_new) * 0.7)]
            df_new_valid = df_new.iloc[int(len(df_new) * 0.7):]

            lm = LinearRegression()
            lm.fit(df_new_train[x_var_list_valid_new], df_new_train['y'])
            params = np.append(lm.intercept_, lm.coef_)


            predictions_lm = lm.predict(df_new_valid[x_var_list_valid_new])

            p_values = self.calculate_pvalues(df_new_valid, predictions_lm, x_var_list_valid_new, params)
            lm_aic_intercept_slope = self.aic(df_new_valid['y'], predictions_lm, len(x_var_list_valid_new)+1 )
            print('lm_aic_intercept_slope:', lm_aic_intercept_slope)
            plt.plot(df_new_valid['y'], predictions_lm, '*')
            R_squared_lm = lm.score(df_new_valid[x_var_list_valid_new], df_new_valid['y'])

            #### add quadratic term

            X_quadratic = sm.add_constant(pd.DataFrame({'precipitation': df_new['Precip'],
                                                        'precipitation_squared': df_new[ 'Precip'] ** 2}))
            ### seperate samples and used the same samples as linear model
            X_quadratic_train = X_quadratic.iloc[:int(len(df_new) * 0.7)]
            X_quadratic_valid = X_quadratic.iloc[int(len(df_new) * 0.7):]
            ## df y also need to be the same samples as linear model
            df_y_train = df_new['y'].iloc[:int(len(df_new) * 0.7)]
            df_y_valid = df_new['y'].iloc[int(len(df_new) * 0.7):]


            model_quadratic = sm.OLS(df_y_train, X_quadratic_train).fit()
            predictions_quadratic = model_quadratic.predict(X_quadratic_valid)

            quadratic_aic_intercept_slope=self.aic(df_y_valid, predictions_quadratic, len(x_var_list_valid_new)+1 )
            print('quadratic_aic_intercept_slope:', quadratic_aic_intercept_slope)
            R_squared_quadratic = model_quadratic.rsquared
            print(R_squared_lm, R_squared_quadratic)

            # plt.plot(df_new_valid['y'], predictions_quadratic, 'o')
            ### show R_squared
            # plt.title(f'{pix} R_squared_lm:{R_squared_lm} R_squared_quadratic:{R_squared_quadratic}')
            # plt.show()

            ### filter if p_values[1] > 0.05 and coefficient > 0

            if p_values[1] > 0.1 or lm.coef_[0] < 0:
                print("Not influenced by precipitation during that season.")
                dic = {'intercept': lm.intercept_, 'slope': lm.coef_[0], 'label': 'not influenced'}
            else:
                # Compare AIC values to select the best model
                if lm_aic_intercept_slope< quadratic_aic_intercept_slope:
                    print("Symmetric response to precipitation (Linear model)")
                    dic={'intercept':lm.intercept_,'slope':lm.coef_[0],'label':'symmetric'}
                else:
                    print("Asymmetric response to precipitation:")
                    # Further classify as negative or positive asymmetric
                    if model_quadratic.params['precipitation_squared'] < 0:
                        print("Negative asymmetric relationship")
                        dic={'intercept':model_quadratic.params['const'],'slope':model_quadratic.params['precipitation'],'label':'negative asymmetric'}
                    else:
                        print("Positive asymmetric relationship")
                        dic={'intercept':model_quadratic.params['const'],'slope':model_quadratic.params['precipitation'],'label':'positive asymmetric'}
                ## save
            outf = self.multi_regression_result_f
            result_dic[pix] = dic
        T.save_npy(result_dic, outf)



    pass


    def cal_quadratic_regression_bin(self, df, x_var_list):

        import statsmodels.api as sm
        import joblib

        outf = self.multi_regression_result_f

        result_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']

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


                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue
                x_vals_detrend = signal.detrend(x_vals) #detrend

                df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue
            df_new['y'] = y_vals_detrend  # detrend


            df_new = df_new.dropna(axis=1, how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()

            ####### linear model

            ## seperate samples
            df_new = df_new.sample(frac=1).reset_index(drop=True)
            df_new_train = df_new.iloc[:int(len(df_new) * 0.7)]
            df_new_valid = df_new.iloc[int(len(df_new) * 0.7):]

            lm = LinearRegression()
            lm.fit(df_new_train[x_var_list_valid_new], df_new_train['y'])
            params = np.append(lm.intercept_, lm.coef_)


            predictions_lm = lm.predict(df_new_valid[x_var_list_valid_new])

            p_values = self.calculate_pvalues(df_new_valid, predictions_lm, x_var_list_valid_new, params)
            lm_aic_intercept_slope = self.aic(df_new_valid['y'], predictions_lm, len(x_var_list_valid_new)+1 )
            print('lm_aic_intercept_slope:', lm_aic_intercept_slope)
            # plt.plot(df_new_valid['y'], predictions_lm, '*')
            # R_squared_lm = lm.score(df_new_valid[x_var_list_valid_new], df_new_valid['y'])

            #### add quadratic term

            X_quadratic = sm.add_constant(pd.DataFrame({'precipitation': df_new['Precip'],
                                                        'precipitation_squared': df_new[ 'Precip'] ** 2}))
            ### seperate samples and used the same samples as linear model
            X_quadratic_train = X_quadratic.iloc[:int(len(df_new) * 0.7)]
            X_quadratic_valid = X_quadratic.iloc[int(len(df_new) * 0.7):]
            ## df y also need to be the same samples as linear model
            df_y_train = df_new['y'].iloc[:int(len(df_new) * 0.7)]
            df_y_valid = df_new['y'].iloc[int(len(df_new) * 0.7):]


            model_quadratic = sm.OLS(df_y_train, X_quadratic_train).fit()
            predictions_quadratic = model_quadratic.predict(X_quadratic_valid)

            quadratic_aic_intercept_slope=self.aic(df_y_valid, predictions_quadratic, len(x_var_list_valid_new)+1 )
            print('quadratic_aic_intercept_slope:', quadratic_aic_intercept_slope)
            R_squared_quadratic = model_quadratic.rsquared
            # print(R_squared_lm, R_squared_quadratic)

            # plt.plot(df_new_valid['y'], predictions_quadratic, 'o')
            ### show R_squared
            # plt.title(f'{pix} R_squared_lm:{R_squared_lm} R_squared_quadratic:{R_squared_quadratic}')
            # plt.show()

            ### filter if p_values[1] > 0.05 and coefficient > 0

            if p_values[1] > 0.1 or lm.coef_[0] < 0:
                print("Not influenced by precipitation during that season.")
                dic = {'intercept': lm.intercept_, 'slope': lm.coef_[0], 'label': 'not influenced'}
            else:
                # Compare AIC values to select the best model
                if lm_aic_intercept_slope< quadratic_aic_intercept_slope:
                    print("Symmetric response to precipitation (Linear model)")
                    dic={'intercept':lm.intercept_,'slope':lm.coef_[0],'label':'symmetric'}
                else:
                    print("Asymmetric response to precipitation:")
                    # Further classify as negative or positive asymmetric
                    if model_quadratic.params['precipitation_squared'] < 0:
                        print("Negative asymmetric relationship")
                        dic={'intercept':model_quadratic.params['const'],'slope':model_quadratic.params['precipitation'],'label':'negative asymmetric'}
                    else:
                        print("Positive asymmetric relationship")
                        dic={'intercept':model_quadratic.params['const'],'slope':model_quadratic.params['precipitation'],'label':'positive asymmetric'}
                ## save
            outf = self.multi_regression_result_f
            result_dic[pix] = dic
        T.save_npy(result_dic, outf)



    pass


    def cal_quadratic_regression_no_validation(self, df, x_var_list):

        import statsmodels.api as sm
        import joblib

        outf = self.multi_regression_result_f

        result_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']

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


                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue
                x_vals_detrend = signal.detrend(x_vals) #detrend

                df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue
            df_new['y'] = y_vals_detrend  # detrend


            df_new = df_new.dropna(axis=1, how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()

            ####### linear model


            lm = LinearRegression()
            lm.fit(df_new[x_var_list_valid_new], df_new['y'])
            params = np.append(lm.intercept_, lm.coef_)


            predictions_lm = lm.predict(df_new[x_var_list_valid_new])

            p_values = self.calculate_pvalues(df_new, predictions_lm, x_var_list_valid_new, params)
            lm_aic_intercept_slope = self.aic(df_new['y'], predictions_lm, len(x_var_list_valid_new)+1 )
            print('lm_aic_intercept_slope:', lm_aic_intercept_slope)
            plt.plot(df_new['y'], predictions_lm, '*')
            R_squared_lm = lm.score(df_new[x_var_list_valid_new], df_new['y'])

            #### add quadratic term

            X_quadratic = sm.add_constant(pd.DataFrame({'precipitation': df_new['GPCC'],
                                                        'precipitation_squared': df_new[ 'GPCC'] ** 2}))

            model_quadratic = sm.OLS(df_new, X_quadratic).fit()
            predictions_quadratic = model_quadratic.predict(X_quadratic)

            quadratic_aic_intercept_slope=self.aic(df_new, predictions_quadratic, len(x_var_list_valid_new)+1 )
            print('quadratic_aic_intercept_slope:', quadratic_aic_intercept_slope)
            R_squared_quadratic = model_quadratic.rsquared
            print(R_squared_lm, R_squared_quadratic)

            # plt.plot(df_new_valid['y'], predictions_quadratic, 'o')
            ### show R_squared
            # plt.title(f'{pix} R_squared_lm:{R_squared_lm} R_squared_quadratic:{R_squared_quadratic}')
            # plt.show()

            ### filter if p_values[1] > 0.05 and coefficient > 0

            if p_values[1] > 0.1 or lm.coef_[0] < 0:
                print("Not influenced by precipitation during that season.")
                dic = {'intercept': lm.intercept_, 'slope': lm.coef_[0], 'label': 'not influenced'}
            else:
                # Compare AIC values to select the best model
                if lm_aic_intercept_slope< quadratic_aic_intercept_slope:
                    print("Symmetric response to precipitation (Linear model)")
                    dic={'intercept':lm.intercept_,'slope':lm.coef_[0],'label':'symmetric'}
                else:
                    print("Asymmetric response to precipitation:")
                    # Further classify as negative or positive asymmetric
                    if model_quadratic.params['precipitation_squared'] < 0:
                        print("Negative asymmetric relationship")
                        dic={'intercept':model_quadratic.params['const'],'slope':model_quadratic.params['precipitation'],'label':'negative asymmetric'}
                    else:
                        print("Positive asymmetric relationship")
                        dic={'intercept':model_quadratic.params['const'],'slope':model_quadratic.params['precipitation'],'label':'positive asymmetric'}
                ## save
            outf = self.multi_regression_result_f
            result_dic[pix] = dic
        T.save_npy(result_dic, outf)



    pass

    def aic(self, y, y_pred, k):

        ### AIC is the Akaike Information Criterion
        ### AIC calculation
        ### k is the number of parameters in the model
        ### L is the likelihood of the model
        ### AIC is used to compare the relative quality of statistical models for a given set of data
        resid = y - y_pred.ravel()
        sse = sum(resid ** 2)

        AIC = 2 * k - 2 * np.log(sse)

        return AIC

    def calculate_pvalues(self, df_new, predictions, x_var_list_valid_new, params):
        # newX = pd.DataFrame({"Constant": np.ones(len(df_new['y']))}).join(pd.DataFrame(df_new[x_var_list_valid_new]))
        # MSE = (sum((df_new['y'] - predictions) ** 2)) / (len(newX) - len(newX.columns))

        # Note if you don't want to use a DataFrame replace the two lines above with
        newX = np.append(np.ones((len(df_new['y']), 1)), df_new[x_var_list_valid_new], axis=1)
        MSE = (sum((df_new['y'] - predictions) ** 2)) / (len(newX) - len(newX[0]))

        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b

        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]
        print(p_values[1])
        return p_values

    def plt_quadratic_regression_result(self, multi_regression_result_dir,y_var,period):
        label_dic={'symmetric':1,'negative asymmetric':2,'positive asymmetric':3,'not influenced':4}


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
                ## only select the label

                vals = dic[pix]['label']
                spatial_dic[pix] = label_dic[vals]


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

    def plot_proportation_response(self):
        f=rf'D:\Project3\Result\Dataframe\\asymetry_response\\asymetry_response.df'
        df=T.load_df(f)
        print(len(df))
        df=df.dropna()
        ### no crop land
        df = df[df['landcover_classfication'] != 'Cropland']
        print(len(df))

        dic_label={1:'symmetric',2:'negative asymmetric',3:'positive asymmetric',4:'not influenced'}
        label_list=[1,2,3,4]
        category_list=[]
        for label_i in label_list:
            df_label=df[df['label_LAI4g_1982_2020']==label_i]

            proportion=len(df_label)/len(df)
            print(label_i,proportion)
            category_list.append(proportion)
        plt.bar(label_list,category_list)



        plt.show()





        pass
class quadratic_regression1:
    def __init__(self):
        pass

    def run(self):
        self.cal_opt_temp(0.1)
        # self.spatial_corr()
        # self.cal_anomaly()
        pass

    def spatial_corr(self):
        # # T_f = rf'D:\Project3\Result\detrend_zscore_monthly\\Precip.npy'
        # T_f = rf'D:\Project3\Result\extract_GS_return_monthly_data\OBS_LAI\\GPCC.npy'
        # # NDVI_f = rf'D:\Project3\Result\detrend_zscore_monthly\\LAI4g.npy'
        # NDVI_f = rf'D:\Project3\Result\extract_GS_return_monthly_data\OBS_LAI\\LAI4g.npy'
        #
        # ndvi_dic = T.load_npy(NDVI_f)
        #
        # temp_dic = T.load_npy(T_f)

        # T_dir = rf'D:\Project3\Data\monthly_data\Precip_anomaly\\'
        T_dir =  rf'D:\Project3\Data\monthly_data\GLEAM_SMroot_anomaly\\'
        NDVI_dir = rf'D:\Project3\Data\monthly_data\LAI4g_anomaly\\'

        ndvi_dic = T.load_npy_dir(NDVI_dir)
        temp_dic = T.load_npy_dir(T_dir)

        spatial_dict = {}
        for pix in tqdm(ndvi_dic):
            if not pix in temp_dic:
                continue
            ndvi = ndvi_dic[pix]
            ndvi = ndvi.flatten()

            temp = temp_dic[pix]
            temp = temp.flatten()
            temp = np.array(temp)
            if len(ndvi) != len(temp):
                continue
            r,p = T.nan_correlation(ndvi,temp)
            spatial_dict[pix] = r
        arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,vmin=-.5,vmax=.5,cmap='RdBu_r',interpolation='nearest')
        plt.colorbar()
        plt.show()

        pass

    def cal_anomaly(self):
        # T_dir = rf'D:\Project3\Data\monthly_data\Precip\\'
        # T_dir_anomaly = rf'D:\Project3\Data\monthly_data\Precip_anomaly\\'
        #
        # Pre_Process().cal_anomaly(T_dir,T_dir_anomaly)
        #
        # NDVI_dir = rf'D:\Project3\Data\monthly_data\LAI4g\\'
        # NDVI_dir_anomaly = rf'D:\Project3\Data\monthly_data\LAI4g_anomaly\\'
        # Pre_Process().cal_anomaly(NDVI_dir,NDVI_dir_anomaly)

        # sm_dir = rf'D:\Project3\Data\monthly_data\GLEAM_SMroot\DIC\\'
        # sm_dir_anomaly = rf'D:\Project3\Data\monthly_data\GLEAM_SMroot_anomaly\\'
        # Pre_Process().cal_anomaly(sm_dir,sm_dir_anomaly)

        pass

    def cal_opt_temp(self, step):


        # T_dir = rf'D:\Project3\Data\monthly_data\Precip\\'
        # T_f = rf'D:\Project3\Result\extract_GS_return_monthly_data\OBS_LAI\\GPCC.npy'
        # T_f = rf'D:\Project3\Result\detrend_zscore_monthly\\Precip.npy'
        T_f = rf'D:\Project3\Data\monthly_data\GLEAM_SMroot_anomaly\\per_pix_dic_000.npy'
        # NDVI_dir = join(data_root,'NDVI4g/per_pix/1982-2020')
        # vege_name = 'NDVI4g'
        # NDVI_dir = rf'D:\Project3\Data\monthly_data\LAI4g\\'
        # NDVI_f = rf'D:\Project3\Result\extract_GS_return_monthly_data\OBS_LAI\\LAI4g.npy'
        NDVI_f = rf'D:\Project3\Result\detrend_zscore_monthly\\LAI4g.npy'

        # outdir = join(self.this_class_arr, f'optimal_temperature', f'{vege_name}_step_{step}_celsius')
        # T.mk_dir(outdir, force=True)
        # outdir_i = join(outdir,f'{vege_name}_step_{step}_celsius.tif')


        # ndvi_dic = T.load_npy_dir(NDVI_dir)
        ndvi_dic = T.load_npy(NDVI_f)
        # temp_dic = T.load_npy_dir(T_dir)
        temp_dic = T.load_npy(T_f)
        optimal_temp_dic = {}
        for pix in tqdm(temp_dic):
            if not pix in ndvi_dic:
                continue
            lon, lat = DIC_and_TIF(pixelsize=0.25).pix_to_lon_lat(pix)
            if lat < 0:
                continue
            if lat > 40:
                continue
            ndvi = ndvi_dic[pix]
            ndvi = ndvi.flatten()
            temp = temp_dic[pix]
            temp = temp.flatten()
            temp = np.array(temp)
            try:
                r,p = T.nan_correlation(ndvi,temp)
            except:
                continue
            if r < 0.5:
                continue
            print(r,p)
            # temp = np.array(temp) - 273.15  # Kelvin to Celsius
            df = pd.DataFrame()
            df['ndvi'] = ndvi
            df['temp'] = temp
            df['month'] = list(range(1, 13)) * int(len(temp) / 12)
            season_dict = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON',
                            10: 'SON', 11: 'SON', 12: 'DJF'}
            df['season'] = df['month'].map(season_dict)
            if len(df) == 0:
                continue
            for season in ['MAM', 'JJA', 'SON','DJF']:
                df_season = df[df['season'] == season]
                # print(df)
                # exit()
                # df = df[df['ndvi'] > 0]
                # df = df[df['ndvi'] < 6000]

                # df = df.dropna()

                # print(df)
                # exit()
                # color_list = T.gen_colors(12)
                max_t = max(df_season['temp'])
                min_t = int(min(df_season['temp']))

                plt.scatter(df_season['temp'], df_season['ndvi'], s=20, alpha=0.8,zorder=-1,c=df_season['month'],cmap='jet')
                plt.colorbar()

                plt.title(f'{pix} {lon:0.2f},{lat:0.2f}')
                t_bins = np.arange(start=min_t, stop=max_t, step=step)
                df_group, bins_list_str = T.df_bin(df_season, 'temp', t_bins)
                quantial_90_list = []
                x_list = []
                for name, df_group_i in df_group:
                    vals = df_group_i['ndvi'].tolist()
                    if len(vals) <= 0:
                        continue
                    # quantile_90 = np.nanquantile(vals, 0.9)
                    quantile_90 = np.nanquantile(vals, 0.5)
                    # quantile_90 = np.nanquantile(vals, 0.1)
                    # quantile_90 = np.nanmean(vals)
                    # print(quantile_90)
                    # print(vals)
                    # exit()
                    left = name[0].left
                    x_list.append(left)
                    quantial_90_list.append(quantile_90)
                    # plt.scatter([left] * len(vals), vals, s=20)
                # multi regression to find the optimal temperature
                # parabola fitting
                x = np.array(x_list)
                y = np.array(quantial_90_list)
                if len(x) < 3:
                    continue
                if len(y) < 3:
                    continue
                a_lin,b_lin,r_lin,p_lin = T.nan_line_fit(x,y)
                try:
                    a, b, c = self.nan_parabola_fit(x, y)
                    y_fit = a * x ** 2 + b * x + c
                    y_fit_lin = a_lin * x + b_lin
                    # print(y_fit)
                    plt.plot(x, y_fit, 'r--', lw=2, zorder=99)
                    plt.plot(x, y_fit_lin, 'k', lw=2)
                    plt.text(0.5, 0.5, f'k_lin={a_lin:.2f}', transform=plt.gca().transAxes)
                    plt.text(0.5, 0.4, f'a_quad={a:.2f}', transform=plt.gca().transAxes)
                    plt.show()

                    # exit()
                    T_opt = x[np.argmax(y_fit)]
                    optimal_temp_dic[pix] = T_opt
                except:
                    continue




    def plot_test_cal_opt_temp(self, step):
        dff = join(Dataframe_SM().this_class_arr, 'dataframe/-0.5.df')
        df_global = T.load_df(dff)
        df_global = df_global[df_global['AI_class'] == 'Arid']
        pix_list = T.get_df_unique_val_list(df_global, 'pix')

        # step = 1  # Celsius
        outdir = join(self.this_class_arr, f'optimal_temperature')
        outf = join(outdir, f'step_{step}_celsius')
        T.mk_dir(outdir)

        temp_dic, _ = Load_Data().ERA_Tair_origin()
        ndvi_dic, _ = Load_Data().NDVI4g_origin()
        # ndvi_dic,_ = Load_Data().LT_Baseline_NT_origin()

        optimal_temp_dic = {}
        for pix in tqdm(pix_list):
            ndvi = ndvi_dic[pix]
            temp = temp_dic[pix]
            temp = np.array(temp) - 273.15  # Kelvin to Celsius
            df = pd.DataFrame()
            df['ndvi'] = ndvi
            df['temp'] = temp
            df = df[df['ndvi'] > 0.1]
            df = df.dropna()
            max_t = max(df['temp'])
            min_t = int(min(df['temp']))
            t_bins = np.arange(start=min_t, stop=max_t, step=step)
            df_group, bins_list_str = T.df_bin(df, 'temp', t_bins)
            # ndvi_list = []
            # box_list = []
            color_list = T.gen_colors(len(df_group))
            color_list = color_list[::-1]
            flag = 0
            quantial_90_list = []
            x_list = []
            for name, df_group_i in df_group:
                vals = df_group_i['ndvi'].tolist()
                quantile_90 = np.nanquantile(vals, 0.9)
                left = name[0].left
                x_list.append(left)
                plt.scatter([left] * len(vals), vals, s=20, color=color_list[flag])
                flag += 1
                quantial_90_list.append(quantile_90)
                # box_list.append(vals)
                # mean = np.nanmean(vals)
                # ndvi_list.append(mean)
            # multi regression to find the optimal temperature
            # parabola fitting
            x = np.array(x_list)
            y = np.array(quantial_90_list)
            # a,b,c = np.polyfit(x,y,2)
            a, b, c = self.nan_parabola_fit(x, y)
            # plot abc
            # y = ax^2 + bx + c
            y_fit = a * x ** 2 + b * x + c
            plt.plot(x, y_fit, 'k--', lw=2)
            opt_T = x[np.argmax(y_fit)]
            plt.scatter([opt_T], [np.max(y_fit)], s=200, marker='*', color='r', zorder=99)
            print(len(y_fit))
            print(len(quantial_90_list))
            a_, b_, r_, p_ = T.nan_line_fit(y_fit, quantial_90_list)
            r2 = r_ ** 2
            print(r2)
            # exit()

            plt.plot(x_list, quantial_90_list, c='k', lw=2)
            plt.title(f'a={a:.3f},b={b:.3f},c={c:.3f}')
            print(t_bins)
            # # plt.plot(t_bins[:-1],ndvi_list)
            # plt.boxplot(box_list,positions=t_bins[:-1],showfliers=False)
            plt.show()

            # exit()
        #     t_mean_list = []
        #     ndvi_mean_list = []
        #     for i in range(len(t_bins)):
        #         if i + 1 >= len(t_bins):
        #             continue
        #         df_t = df[df['temp']>t_bins[i]]
        #         df_t = df_t[df_t['temp']<t_bins[i+1]]
        #         t_mean = df_t['temp'].mean()
        #         # t_mean = t_bins[i]
        #         ndvi_mean = df_t['ndvi'].mean()
        #         t_mean_list.append(t_mean)
        #         ndvi_mean_list.append(ndvi_mean)
        #
        #     indx_list = list(range(len(ndvi_mean_list)))
        #     max_indx = T.pick_max_indx_from_1darray(ndvi_mean_list,indx_list)
        #     if max_indx > 999:
        #         optimal_temp = np.nan
        #     else:
        #         optimal_temp = t_mean_list[max_indx]
        #     optimal_temp_dic[pix] = optimal_temp
        # T.save_npy(optimal_temp_dic,outf)

    def nan_parabola_fit(self, val1_list, val2_list):
        if not len(val1_list) == len(val2_list):
            raise UserWarning('val1_list and val2_list must have the same length')
        val1_list_new = []
        val2_list_new = []
        for i in range(len(val1_list)):
            val1 = val1_list[i]
            val2 = val2_list[i]
            if np.isnan(val1):
                continue
            if np.isnan(val2):
                continue
            val1_list_new.append(val1)
            val2_list_new.append(val2)
        a, b, c = np.polyfit(val1_list_new, val2_list_new, 2)

        return a, b, c


class Seasonal_sensitivity:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Seasonal_sensitivity',rf'D:\Project3\Result\\',mode=2)
        # exit()
        pass

    def run(self):
        # self.cal_seasonal_sensitivity()
        # self.plot_seasonal_sensitivity()
        # self.cal_seasonal_trend()
        self.plot_seasonal_trend()
        pass

    def cal_seasonal_sensitivity(self):
        outdir = join(self.this_class_arr, 'seasonal_sensitivity')
        T.mk_dir(outdir, force=True)
        SM_dir = 'D:\Project3\Data\monthly_data\GLEAM_SMroot_anomaly\\'
        LAI_dir = 'D:\Project3\Data\monthly_data\LAI4g_anomaly\\'
        SM_dic = T.load_npy_dir(SM_dir)
        LAI_dic = T.load_npy_dir(LAI_dir)
        result_dict = {}
        for pix in tqdm(SM_dic):
            SM = SM_dic[pix]
            if not pix in LAI_dic:
                continue
            LAI = LAI_dic[pix]
            if np.nanstd(SM) == 0:
                continue
            if np.nanstd(LAI) == 0:
                continue
            df = pd.DataFrame()
            df['SM'] = SM
            df['LAI'] = LAI
            df['month'] = list(range(1, 13)) * int(len(SM) / 12)
            season_dict = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON',
                            10: 'SON', 11: 'SON', 12: 'DJF'}
            df['season'] = df['month'].map(season_dict)
            df = df.dropna()
            season_list = ['MAM', 'JJA', 'SON', 'DJF']
            result_dict_i = {}
            for season in season_list:
                df_season = df[df['season'] == season]
                SM = df_season['SM'].to_list()
                LAI = df_season['LAI'].to_list()
                if np.nanstd(SM) == 0:
                    continue
                if np.nanstd(LAI) == 0:
                    continue
                try:
                    r, p = T.nan_correlation(SM, LAI)
                except:
                    r, p = np.nan,np.nan
                result_dict_i[f'{season}_r'] = r
                result_dict_i[f'{season}_p'] = p

            result_dict[pix] = result_dict_i
        df_result = T.dic_to_df(result_dict)
        T.print_head_n(df_result)
        outf = join(outdir, 'seasonal_sensitivity.df')
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)

    def plot_seasonal_sensitivity(self):
        dff = join(self.this_class_arr, 'seasonal_sensitivity', 'seasonal_sensitivity.df')
        df = T.load_df(dff)
        df['pix'] = df['__key__'].to_list()
        season_list = ['MAM', 'JJA', 'SON', 'DJF']
        for season in season_list:
            col_name = f'{season}_r'
            spatial_dict = T.df_to_spatial_dic(df, col_name)
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict)
            plt.figure()
            plt.imshow(arr, vmin=-.7, vmax=.7, cmap='RdBu_r', interpolation='nearest')
            plt.title(col_name)
            plt.colorbar()
        plt.show()
        pass
    def cal_seasonal_trend(self):
        outdir = join(self.this_class_arr, 'seasonal_trend')
        T.mk_dir(outdir, force=True)
        product = 'SM'
        product='Precip'
        # product = 'LAI'
        # SM_dir = 'D:\Project3\Data\monthly_data\GLEAM_SMroot_anomaly\\'
        # SM_dir = 'D:\Project3\Data\monthly_data\LAI4g_anomaly\\'
        SM_dir = f'D:\Project3\Data\monthly_data\\\GPCC\\dic\\'

        SM_dic = T.load_npy_dir(SM_dir)
        # LAI_dic = T.load_npy_dir(LAI_dir)
        result_dict = {}
        for pix in tqdm(SM_dic):
            SM = SM_dic[pix]
            # if not pix in LAI_dic:
            #     continue
            # LAI = LAI_dic[pix]
            if np.nanstd(SM) == 0:
                continue
            # if np.nanstd(LAI) == 0:
            #     continue
            df = pd.DataFrame()
            df['SM'] = SM
            # df['LAI'] = LAI
            df['month'] = list(range(1, 13)) * int(len(SM) / 12)
            season_dict = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON',
                            10: 'SON', 11: 'SON', 12: 'DJF'}
            df['season'] = df['month'].map(season_dict)
            # df = df.dropna()
            season_list = ['MAM', 'JJA', 'SON', 'DJF']
            result_dict_i = {}
            for season in season_list:
                df_season = df[df['season'] == season]
                SM = df_season['SM'].to_list()
                # LAI = df_season['LAI'].to_list()
                if np.nanstd(SM) == 0:
                    continue
                SM = np.array(SM)
                SM_reshape = SM.reshape(-1, 3)
                SM_mean = np.nanmean(SM_reshape, axis=1)
                # print(len(SM_mean))
                # plt.imshow(SM_reshape)
                # plt.show()
                # exit()
                # if np.nanstd(LAI) == 0:
                #     continue
                try:
                    a,b,r,p = T.nan_line_fit(list(range(len(SM_mean))),SM_mean)
                except:
                    a,b,r,p = np.nan,np.nan,np.nan,np.nan
                result_dict_i[f'{season}_a'] = a
                result_dict_i[f'{season}_p'] = p

            result_dict[pix] = result_dict_i
        df_result = T.dic_to_df(result_dict,'pix')
        T.print_head_n(df_result)
        outf = join(outdir, f'{product}.df')
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)


    def cal_monthly_trend(self):
        outdir = join(self.this_class_arr, 'monthly_trend')
        T.mk_dir(outdir, force=True)
        product = 'SM'
        product='Precip'
        # product = 'LAI'
        # SM_dir = 'D:\Project3\Data\monthly_data\GLEAM_SMroot_anomaly\\'
        # SM_dir = 'D:\Project3\Data\monthly_data\LAI4g_anomaly\\'
        SM_dir = f'D:\Project3\Data\monthly_data\\\GPCC\\dic\\'

        SM_dic = T.load_npy_dir(SM_dir)
        # LAI_dic = T.load_npy_dir(LAI_dir)
        result_dict = {}
        for pix in tqdm(SM_dic):
            SM = SM_dic[pix]
            # if not pix in LAI_dic:
            #     continue
            # LAI = LAI_dic[pix]
            if np.nanstd(SM) == 0:
                continue
            # if np.nanstd(LAI) == 0:
            #     continue
            df = pd.DataFrame()
            df['SM'] = SM
            # df['LAI'] = LAI
            df['month'] = list(range(1, 13)) * int(len(SM) / 12)

            # df = df.dropna()

            try:
                a,b,r,p = T.nan_line_fit(list(range(len(SM)),SM)
            except:
                a,b,r,p = np.nan,np.nan,np.nan,np.nan
            result_dict_i[f'{season}_a'] = a
            result_dict_i[f'{season}_p'] = p

            result_dict[pix] = result_dict_i
        df_result = T.dic_to_df(result_dict,'pix')
        T.print_head_n(df_result)
        outf = join(outdir, f'{product}.df')
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)

    def plot_seasonal_trend(self):
        dff = join(self.this_class_arr, 'seasonal_trend', 'LAI.df')
        # dff = join(self.this_class_arr, 'seasonal_trend', 'SM.df')
        # dff = join(self.this_class_arr, 'seasonal_trend', 'Precip.df')
        df = T.load_df(dff)
        # df['pix'] = df['__key__'].to_list()
        season_list = ['MAM', 'JJA', 'SON', 'DJF']
        for season in season_list:
            col_name = f'{season}_a'
            spatial_dict = T.df_to_spatial_dic(df, col_name)
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict)
            plt.figure()
            plt.imshow(arr, vmin=-0.1, vmax=0.1, cmap='RdBu', interpolation='nearest')
            plt.title(col_name)
            plt.colorbar()
        plt.show()
        pass

    def plot_monthly_trend(self):
        dff = join(self.this_class_arr, 'seasonal_trend', 'LAI.df')
        # dff = join(self.this_class_arr, 'seasonal_trend', 'SM.df')
        # dff = join(self.this_class_arr, 'seasonal_trend', 'Precip.df')
        df = T.load_df(dff)
        # df['pix'] = df['__key__'].to_list()
        season_list = ['MAM', 'JJA', 'SON', 'DJF']
        for season in season_list:
            col_name = f'{season}_a'
            spatial_dict = T.df_to_spatial_dic(df, col_name)
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict)
            plt.figure()
            plt.imshow(arr, vmin=-0.1, vmax=0.1, cmap='RdBu', interpolation='nearest')
            plt.title(col_name)
            plt.colorbar()
        plt.show()
        pass

class multi_regression():
    def __init__(self):
        self.fdirX=result_root+rf'anomaly\OBS_extend\\'
        self.fdirY=result_root+rf'anomaly\OBS_extend\\'


        self.period=('1982_2020')
        # self.y_var = [f'LAI4g_{self.period}']
        # self.xvar = [f'Tmax_{self.period}', f'GPCC_{self.period}', f'CO2_{self.period}', f'VPD_{self.period}']
        self.y_var = ['LAI4g']
        self.xvar = ['tmax', 'tmin','Tempmean','GLEAM_SMroot', 'VPD']


        self.multi_regression_result_dir=result_root+rf'multi_regression\\{self.period}\\'
        T.mk_dir(self.multi_regression_result_dir,force=True)

        self.multi_regression_result_f = result_root + rf'multi_regression\\\\{self.period}\\{self.y_var[0]}.npy'
        pass

    def run(self):

        #step 1 build dataframe
        df = self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var,self.period)
        # df=self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var)
        #
        # # # # step 2 cal correlation
        self.cal_multi_regression_beta(df, self.xvar)  # 修改参数

        # step 3 plot
        self.plt_multi_regression_result(self.multi_regression_result_dir,self.y_var[0],self.period)


    def build_df(self,fdir_X,fdir_Y,fx_list,fy,period):

        df = pd.DataFrame()

        filey=fdir_Y+fy[0]+'.npy'
        print(filey)

        dic_y = T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue
            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
            if yvals[0] == None:
                continue

            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        for xvar in fx_list:

            # print(var_name)
            x_val_list = []
            filex=fdir_X+xvar+'.npy'
            # filex = fdir_X + xvar + f'_{period}.npy'

            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
            dic_x = T.load_npy(filex)
            for i, row in tqdm(df.iterrows(), total=len(df), desc=xvar):
                pix = row.pix
                if not pix in dic_x:
                    x_val_list.append([])
                    continue
                xvals = dic_x[pix]
                xvals = np.array(xvals)
                if len(xvals) == 0:
                    x_val_list.append([])
                    continue

                xvals = T.interp_nan(xvals)
                if xvals[0] == None:
                    x_val_list.append([])
                    continue

                x_val_list.append(xvals)

            # x_val_list = np.array(x_val_list)
            df[xvar] = x_val_list
        T.print_head_n(df)

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
        import joblib


        outf = self.multi_regression_result_f

        multi_derivative = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']
            # y_vals = T.remove_np_nan(y_vals)
            # y_vals = T.interp_nan(y_vals)
            if len(y_vals) == 0:
                continue

            # y_vals_detrend = signal.detrend(y_vals)
            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:

                x_vals = row[x]

                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                # x_vals = T.interp_nan(x_vals)
                # if len(y_vals) == 18:
                #     x_vals = x_vals[:-1]


                if len(x_vals) != len(y_vals):
                    continue
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
            ##save model
            # joblib.dump(linear_model, outf.replace('.npy', f'_{pix}.pkl'))
            ##load model
            # linear_model = joblib.load(outf.replace('.npy', f'_{pix}.pkl'))
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

def global_get_seasonality(pix,season):
    global_northern_hemi_spring = (3,4,5)
    global_northern_hemi_summer = (6,7,8)
    global_northern_hemi_autumn = (9,10,11)
    global_northern_hemi_winter=(12,1,2)
    global_southern_hemi_spring = (9,10,11)
    global_southern_hemi_summer = (12,1,2)
    global_southern_hemi_autumn = (3,4,5)
    global_southern_hemi_winter = (6,7,8)
    northern_season_month_dic={'spring':global_northern_hemi_spring,'summer':global_northern_hemi_summer,'autumn':global_northern_hemi_autumn,'winter':global_northern_hemi_winter}
    southern_season_month_dic={'spring':global_southern_hemi_spring,'summer':global_southern_hemi_summer,'autumn':global_southern_hemi_autumn,'winter':global_southern_hemi_winter}

    tropical_gs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    r, c = pix
    if r < 240:
        return northern_season_month_dic[season]
    elif 240 <= r < 480:
        return tropical_gs
    elif r >= 480:
        return southern_season_month_dic[season]
    else:
        raise ValueError('r not in range')

class build_dataframe():


    def __init__(self):

        self.this_class_arr = result_root + rf'Dataframe\anomaly_LAI\\'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'anomaly_right.df'

        pass

    def run(self):

        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        # df=self.foo2(df)
        # df=self.add_multiregression_to_df(df)
        # df=self.build_df(df)
        # df=self.build_df_monthly(df)
        df=self.append_attributes(df)  ## 加属性
        # df=self.append_cluster(df)  ## 加属性
        # df=self.append_value(df)
        # df = self.add_detrend_zscore_to_df(df)
        # df=self.add_lc_composition_to_df(df)
        # df=self.add_trend_to_df(df)
        # df=self.add_AI_classfication(df)
        # df=self.add_SM_trend_label(df)
        # df=self.add_aridity_to_df(df)
        # #
        # df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        # df=self.add_landcover_classfication_to_df(df)
        # df=self.add_row(df)
        # df=self.add_soil_texture_to_df(df)
        # df=self.add_precipitation_CV_to_df(df)

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
    def build_df(self, df):

        fdir = rf'D:\Project3\Result\extract_window\extract_original_window_average\15\\'
        all_dic= {}
        for f in os.listdir(fdir):
            fname= f.split('.')[0]

            fpath=fdir+f

            dic = T.load_npy(fpath)
            key_name=fname
            all_dic[key_name]=dic
        # print(all_dic.keys())
        df=T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df

    def build_df_monthly(self, df):


        fdir_all = rf'D:\Project3\Data\monthly_data\\'
        all_dic= {}
        for fdir in os.listdir(fdir_all):
            for fdir_ii in os.listdir(fdir_all+fdir):
                if not 'DIC' in fdir_ii:
                    continue
                dic=T.load_npy_dir(fdir_all+fdir+'\\'+fdir_ii)

                key_name=fdir
                all_dic[key_name] = dic
                    # print(all_dic.keys())
            df = T.spatial_dics_to_df(all_dic)
            T.print_head_n(df)
        return df



    def append_attributes(self, df):  ## add attributes
        fdir = result_root+rf'anomaly\OBS_extend\\'
        for f in tqdm(os.listdir(fdir)):
            if not 'GPCC' in f:
                continue

            # array=np.load(fdir+f)
            # dic = DIC_and_TIF().spatial_arr_to_dic(array)
            dic=T.load_npy(fdir+f)
            key_name = f.split('.')[0]
            print(key_name)



            # df[key_name] = df['pix'].map(dic)
            # T.print_head_n(df)
            df=T.add_spatial_dic_to_df(df,dic,key_name)
        return df


    def append_cluster(self, df):  ## add attributes
        dic_label = {'sig_greening_sig_wetting': 1, 'sig_browning_sig_wetting': 2, 'non_sig_greening_sig_wetting': 3,

                     'non_sig_browning_sig_wetting': 4, 'sig_greening_sig_drying': 5, 'sig_browning_sig_drying': 6,

                     'non_sig_greening_sig_drying': 7, 'non_sig_browning_sig_drying': 8, np.nan: 0}

        #### reverse
        dic_label = {v: k for k, v in dic_label.items()}


        fdir = result_root+rf'Dataframe\anomaly_trends\\'
        for f in os.listdir(fdir):
            if not f.endswith('tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)

            # array=np.load(fdir+f)
            dic = DIC_and_TIF().spatial_arr_to_dic(array)

            key_name='label'
            for k in dic:
                if dic[k] <-99:
                    continue
                dic[k]=dic_label[dic[k]]

            df=T.add_spatial_dic_to_df(df,dic,key_name)

        return df






    def append_value(self, df):  ##补齐
        fdir = result_root + rf'anomaly\OBS_extend\\'
        col_list=[]
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'GIMMS3g' in f:
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
                if len(vals)==34:
                    for i in range(5):
                        vals=np.append(vals,np.nan)
                    # print(len(vals))
                elif len(vals)==37:
                    for i in range(2):
                        vals=np.append(vals,np.nan)
                    print(len(vals))

                vals_new.append(vals)

                # exit()
            df[col]=vals_new

        return df

        pass


    def foo1(self, df):

        fdir = data_root + rf'landcover_composition_DIC\crop\\'


        dic = T.load_npy_dir(fdir)


        pix_list = []
        change_rate_list = []
        year = []

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 1991
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y + 1)
                y = y + 1

        df['pix'] = pix_list

        # df['year'] = year
        df['year'] = year
        df['crop'] = change_rate_list
        return df

    def foo2(self, df):  # 新建trend

        f = result_root + rf'quadratic_regression\1982_2020\\label_LAI4g_1982_2020.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        # val_array = np.load(f)
        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)

        # exit()

        pix_list = []
        for pix in tqdm(val_dic):
            pix_list.append(pix)
        df['pix'] = pix_list

        return df
    def add_multiregression_to_df(self, df):
        fdir = result_root + rf'multi_regression\1982_2020\\'
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if not 'CO2' in f:
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
                val_list.append(val)
            df[f.split('.')[0]] = val_list
        return df



        pass

    def add_detrend_zscore_to_df(self, df):
        fdir = result_root + rf'extract_window\extract_original_window_trend\15\\'

        for f in os.listdir(fdir):

            variable= f.split('.')[0]


            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row['window']
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
                v1 = vals[year - 1]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)
            df[variable] = NDVI_list
        # exit()
        return df

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
    def add_soil_texture_to_df(self, df):
        tiff=rf'D:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'Rooting_Depth'
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
        fdir='D:\Project3\Result\state_variables\\'
        for f in os.listdir(fdir):

            val_dic = T.load_npy(fdir + f)
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
            df[f_name] = val_list
        return df

        pass
    def rename_columns(self, df):
        df = df.rename(columns={'GLEAM_SMroot_trend_label_mark': 'wetting_drying_trend'})
        return df
    def drop_field_df(self, df):
        df = df.drop(columns=['label_LAI4g_1982_2020'])
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

        fdir=result_root+ rf'quadratic_regression\1982_2020\\'

        for f in os.listdir(fdir):

            if not f.endswith('.tif'):
                continue
            if not 'label_LAI4g_1982_2020.tif' in f:
                continue
            variable=(f.split('.')[0].split('_')[0])


            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
            array = np.array(array, dtype=float)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

            # val_array = np.load(fdir + f)

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
        scenario='S2'
        self.product_list = ['LAI4g', 'GIMMS3g', f'CABLE-POP_{scenario}_lai', f'CLASSIC_{scenario}_lai', 'CLM5',  f'IBIS_{scenario}_lai', f'ISAM_{scenario}_lai',
             f'ISBA-CTRIP_{scenario}_lai', f'JSBACH_{scenario}_lai', f'JULES_{scenario}_lai',  f'LPJ-GUESS_{scenario}_lai', f'LPX-Bern_{scenario}_lai',
             f'ORCHIDEE_{scenario}_lai', f'SDGVM_{scenario}_lai', f'YIBs_{scenario}_Monthly_lai']

        # self.product_list = ['GPP_baseline', 'GPP_CFE','Ensemble', f'CABLE-POP_{scenario}_gpp', f'CLASSIC_{scenario}_gpp', 'CLM5',  f'IBIS_{scenario}_gpp', f'ISAM_{scenario}_gpp',
        #      f'ISBA-CTRIP_{scenario}_gpp', f'JSBACH_{scenario}_gpp', f'JULES_{scenario}_gpp',   f'LPX-Bern_{scenario}_gpp',
        #      f'ORCHIDEE_{scenario}_gpp', f'SDGVM_{scenario}_gpp', f'YIBs_{scenario}_Monthly_gpp']

        # self.product_list = [ 'GPP_NIRv','GPP_baseline', 'GPP_CFE',]
        self.product_list = ['GPCC','LAI4g', 'GIMMS_AVHRR_LAI', 'GIMMS3g'  ]


        pass
    def run(self):


        # self.plot_annual_zscore_based_region()

        # self.plot_anomaly_trendy_LAI()
        self.plot_anomaly_LAI_based_on_cluster()
        # self.plot_asymetrical_response_based_on_cluster()
        # self.plot_anomaly_trendy_GPP()
        # self.plot_anomaly_vegetaton_indices()
        # self.plot_climatic_factors()
        # self.plot_plant_fuctional_types_trend()
        # self.plot_trend_spatial_all()
        # self.plot_trend_regional()
        # self.plot_trend()
        # self.plot_anomaly_bar()
        # self.plot_bin_sensitivity_for_each_bin()
        # self.plot_drying_greening()
        # self.plot_drying_greening_map()
        self.plot_cluster_variables_trend()
        # self.plot_cluster_LAI_response_to_variables()
        # self.plot_drying_wetting_areas()
        # self.plot_browning_greening_areas()


        # self.plot_trend_spatial()
        # self.plot_browning_greening()
        # self.plot_original_data()
        # self.plot_category()
        # self.plot_landcover_classfication_yearly()  ##
        # self.plot_multiregression()
        # self.plot_multiregression_boxplot()
        # self.plot_Yao_Zhang_method()
        pass
    def clean_df(self,df):
        df=df[df['landcover_classfication']!='Cropland']



        return df



    def plot_annual_zscore_based_region(self):   #based on semi-arid, arid and sub-humid
        df= T.load_df(result_root + rf'Dataframe\anomaly_LAI\anomaly.df')
        print(len(df))
        df=df[df['landcover_classfication']!='Cropland']
        print(len(df))


        product_list = ['LAI4g','GIMMS3g','GIMMS_AVHRR_LAI']

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


    def plot_anomaly_trendy_LAI(self):

        df= T.load_df(result_root + 'Dataframe\\anomaly_LAI\\anomaly_right.df')
        print(len(df))
        df=self.clean_df(df)
        print(len(df))
        # exit()

        #create color list with one green and another 14 are grey

        color_list=['grey']*16
        color_list[0]='green'
        color_list[1] = 'red'
        color_list[2]='blue'
        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=3
        linewidth_list[2]=3

        fig = plt.figure()
        i = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]
            ax = fig.add_subplot(1, 3, i)
            for product in self.product_list:

                print(product)
                vals=df_region[product].tolist()
                vals_nonnan=[]
                for val in vals:
                    if type(val)==float: ## only screening
                        continue
                    # print(len(val))
                    vals_nonnan.append(val)
                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)
                val_std=np.nanstd(vals_mean,axis=0)

                plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
                # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])


                # plt.scatter(range(len(vals_mean)),vals_mean)
                # plt.text(0,vals_mean[0],product,fontsize=8)
            i=i+1

            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
            # plt.ylim(-0.2, 0.2)
            plt.ylim(-0.2,0.2)


            plt.xlabel('year')

            plt.ylabel('anomaly LAI (m2/m2/year)')
            # plt.legend()

            plt.title(region)
        plt.legend()
        plt.show()

    def plot_anomaly_LAI_based_on_cluster(self):  ##### plot for 4 clusters

        df= T.load_df(result_root + 'Dataframe\\anomaly_LAI\\anomaly_right.df')
        print(len(df))
        df=self.clean_df(df)
        print(len(df))
        T.print_head_n(df)

        #create color list with one green and another 14 are grey

        color_list=['grey']*16
        color_list[0]='green'
        color_list[1] = 'red'
        color_list[2]='blue'
        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=3
        linewidth_list[2]=3

        fig = plt.figure()
        i = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:

            df_region = df[df['AI_classfication'] == region]
            ax = fig.add_subplot(1,3, i)
            for product in self.product_list:
                if not 'GPCC' in product:
                    continue

                print(product)
                vals=df_region[product].tolist()

                vals_nonnan=[]
                for val in vals:
                    if type(val)==float: ## only screening
                        continue
                    # print(len(val))
                    vals_nonnan.append(val)
                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)
                val_std=np.nanstd(vals_mean,axis=0)

                plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
                # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])


                # plt.scatter(range(len(vals_mean)),vals_mean)
                # plt.text(0,vals_mean[0],product,fontsize=8)
            i=i+1

            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
            # plt.ylim(-0.2, 0.2)
            plt.ylim(-10,10)


            plt.xlabel('year')

            plt.ylabel('anomaly LAI (m2/m2/year)')
            # plt.legend()

            plt.title(region)
        plt.legend()
        plt.show()

    def plot_asymetrical_response_based_on_cluster(self):  ##### plot for 4 clusters

        df = T.load_df(result_root + 'Dataframe\\anomaly_LAI\\anomaly_right.df')
        print(len(df))
        df = self.clean_df(df)
        print(len(df))
        T.print_head_n(df)

        # create color list with one green and another 14 are grey

        color_list = ['grey'] * 16
        color_list[0] = 'green'
        color_list[1] = 'red'
        color_list[2] = 'blue'
        linewidth_list = [1] * 16
        linewidth_list[0] = 3
        linewidth_list[1] = 3
        linewidth_list[2] = 3

        fig = plt.figure()
        i = 1

        for region in ['sig_greening_sig_drying', 'sig_greening_sig_wetting', 'sig_browning_sig_drying',
                       'sig_browning_sig_wetting', ]:

            df_region = df[df['label'] == region]
            ax = fig.add_subplot(2, 2, i)

            vals = df_region['LAI4g_window5'].tolist()
            vals_nonnan = []
            for val in vals:
                if type(val) == float:  ## only screening
                    continue
                # print(len(val))
                vals_nonnan.append(val)
            ###### calculate mean
            vals_mean = np.array(vals_nonnan)  ## axis=0, mean of each row  竖着加
            vals_mean = np.nanmean(vals_mean, axis=0)
            val_std = np.nanstd(vals_mean, axis=0)

            plt.plot(vals_mean, label='LAI4g', color=color_list[0], linewidth=linewidth_list[0])
            # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])

            # plt.scatter(range(len(vals_mean)),vals_mean)
            # plt.text(0,vals_mean[0],product,fontsize=8)
            i = i + 1

            # ax.set_xticks(range(0, 40, 4))
            # ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
            # # plt.ylim(-0.2, 0.2)
            plt.ylim(-20, 20)

            plt.xlabel('year')

            plt.ylabel('anomaly LAI (m2/m2/year)')
            # plt.legend()

            plt.title(region)
        plt.legend()
        plt.show()



    def plot_anomaly_trendy_GPP(self):

        df = T.load_df(result_root + 'Dataframe\\anomaly_GPP\\anomaly.df')
        print(len(df))
        df = self.clean_df(df)
        # print(len(df))
        # exit()

        # create color list with one green and another 14 are grey

        color_list = ['grey'] * 16
        color_list[0] = 'green'
        color_list[1] = 'red'
        color_list[2] = 'black'
        linewidth_list = [1] * 16
        linewidth_list[0] = 3
        linewidth_list[1] = 3
        linewidth_list[2] = 3
        linewidth_list[3] = 3

        fig = plt.figure()
        i = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]
            ax = fig.add_subplot(1, 3, i)
            for product in self.product_list:
                # if 'SDGVM' in product:
                #     continue
                print(product)
                vals = df_region[product].tolist()
                vals_nonnan = []
                for val in vals:
                    if type(val) == float:  ## only screening
                        continue
                    vals_nonnan.append(val)
                ###### calculate mean
                vals_mean = np.array(vals_nonnan)  ## axis=0, mean of each row  竖着加
                vals_mean = np.nanmean(vals_mean, axis=0)

                plt.plot(vals_mean, label=product, color=color_list[self.product_list.index(product)],
                         linewidth=linewidth_list[self.product_list.index(product)])

                # plt.scatter(range(len(vals_mean)),vals_mean)
                # plt.text(0,vals_mean[0],product,fontsize=8)
            i = i + 1

            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
            plt.ylim(-120, 150)

            plt.xlabel('year')

            plt.ylabel('delta GPP (gC/m2/year)')
            plt.legend()

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

    # def plot_trend(self):   ## slides 22
    #
    #     df = T.load_df(result_root + '\\Dataframe\\anomaly_trends\\anomaly_trends.df')
    #     product_list = ['LAI4g_trend', 'Ensemble_S0_trend', 'Ensemble_S1_trend', 'Ensemble_S2_trend',
    #                     'Ensemble_S3_trend', 'Ensemble_S1-S0_trend', 'Ensemble_S2-S1_trend', 'Ensemble_S3-S2_trend']
    #     label_list = ['OBS', 'none', 'CO2+Ndep', 'CO2+CLI+Ndep', 'CO2+CLI+LULCC+Nfert+Ndep', 'CO2&Ndep', 'CLIM',
    #                   'LULCC']
    #
    #     color_list = ['red', 'green', 'blue', 'orange','black', 'grey', 'yellow', 'pink', 'purple']
    #     period_list = ['1982_2020']
    #
    #     fig = plt.figure()
    #     flag = 1
    #
    #     for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
    #         df_region = df[df['AI_classfication'] == region]
    #
    #         ax = fig.add_subplot(1, 3, flag)
    #         average_dic = {}
    #         for period in period_list:
    #
    #             average_list = []
    #
    #             for product in product_list:
    #                 # vals = df_region[f'{product}_{period}_trend_zscore'].tolist()
    #                 vals = df_region[f'{product}_trend_{period}'].tolist()
    #                 average_val = np.nanmean(vals)
    #                 average_list.append(average_val)
    #             average_dic[period] = average_list
    #
    #         df_new = pd.DataFrame(average_dic, index=product_list)
    #         T.print_head_n(df_new)
    #         df_new = df_new.T
    #         T.print_head_n(df_new)
    #         df_new.plot.bar(ax=ax)
    #         plt.title(region)
    #         # plt.ylabel('trend (unitless)')
    #         plt.ylabel('trend')
    #
    #         plt.ylim(-15, 15)
    #
    #         flag = flag + 1
    #
    #         plt.tight_layout()
    #     plt.legend()
    #     plt.show()

    def plot_anomaly_bar(self):
        df = T.load_df(result_root + '\Dataframe\RF\\RF.df')
        # df = self.clean_df(df)

        T.print_head_n(df)
        all_year_list = [1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991,
                         1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001,
                         2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                         2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        ENSO_year_list = [1982, 1983, 1997, 1998, 2015, 2016, 1992, 1992, 1987, 1988]
        LaNina_year_list = [1988, 1989, 1998, 1999, 2000, 2007, 2008, 2011, 2010]
        great_depression_year_list = [2007, 2008, 2009]

        for year in LaNina_year_list:
            df = df[df['year'] != year]


        color_list = ['red', 'green', 'blue', 'orange', 'black', 'grey', 'yellow', 'pink', 'purple']


        fig = plt.figure()
        flag = 1

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]


            ax = fig.add_subplot(1, 3, flag)
            average_dic = {}
            average_list=[]
            ##calculate trend
            for variable in ['LAI4g']:
                vals = df_region[f'{variable}_trend'].tolist()
                average_val = np.nanmean(vals)
                average_list.append(average_val)




    def plot_bin_sensitivity_for_each_bin(self): ## plot sensitivity for each drying bin
        df= T.load_df(result_root + 'Dataframe\\multiregression\\multiregression.df')
        # df=self.clean_df(df)
        # print(df)

        T.print_head_n(df, 10)


        regions = ['Arid', 'Semi-Arid', 'Sub-Humid']
        cm = 1 / 2.54

        for region in regions:
            plt.figure(figsize=(15 * cm, 7 * cm))

            df_temp = df[df['AI_classfication'] == region]
            x_var_list = ['GLEAM_SMroot_1982_2001_trend', 'GLEAM_SMroot_2002_2020_trend',]
            p_values_list = ['GLEAM_SMroot_1982_2001_p_value', 'GLEAM_SMroot_2002_2020_p_value',]

            y_var_list = ['CO2_1982_2001_LAI4g_1982_2001', 'CO2_2002_2020_LAI4g_2002_2020',]

            x_bin_range = [-0.002, 0.002]

            for x_var in x_var_list:
                p_values=p_values_list[x_var_list.index(x_var)]
                df_temp=df_temp[df_temp[p_values]<0.1]


                period = x_var.split('_')[2]
                for y_var in y_var_list:

                    if not period in y_var:
                        continue

                    print(x_var, y_var)
                    matrix = []
                    for x_bin in np.arange(x_bin_range[0], x_bin_range[1], 0.0001):

                        df_temp2 = df_temp[(df_temp[x_var] >= x_bin) & (df_temp[x_var] < x_bin + 0.0001)]

                        if len(df_temp2) == 0:
                            continue
                        y_mean = np.nanmean(df_temp2[y_var])

                        matrix.append([x_bin, y_mean])
                    matrix = np.array(matrix)
                    # print(matrix)
                    plt.plot(matrix[:, 0], matrix[:, 1], label=y_var)
            plt.title(region)
            plt.xlabel('SM trend (m3/m3/year)', )
            ## xticks
            plt.xticks(np.arange(x_bin_range[0], x_bin_range[1]+0.001, 0.001), rotation=45)
            plt.ylabel('LAI trend (m2/m2/year)')
            plt.tight_layout()

            plt.legend()

            plt.show()


            # plt.savefig(self.outdir + f'{region}_{z_val_name}.pdf', dpi=300)
            # plt.close()

        pass



    def plot_drying_greening(self):  # calculate the percentage of drying and greening pixels for each region
        df= T.load_df(result_root + 'Dataframe\\anomaly_trends\\anomaly_trends.df')
        df=self.clean_df(df)

        wetting_drying_list=['non_sig_wetting','sig_wetting','non_sig_drying','sig_drying']

        color_list=['green','lime','orange','red']
        regions=['Arid','Semi-Arid','Sub-Humid']
        cm = 1 / 2.54

        for region in regions:


            df_temp = df[df['AI_classfication'] == region]
            dic={}
            for wetting_drying in wetting_drying_list:
                df_temp2=df_temp[df_temp['wetting_drying_trend']==wetting_drying]
                vals = df_temp2[f'LAI4g_trend'].tolist()
                p_values = df_temp2[f'LAI4g_p_value'].tolist()

                sig_browning = 0
                sig_greening = 0
                non_sig_browning = 0
                non_sig_greening = 0

                for i in range(len(vals)):
                    if p_values[i] < 0.05:
                        if vals[i] < 0:
                            sig_browning = sig_browning + 1
                        else:
                            sig_greening = sig_greening + 1
                    else:
                        if vals[i] < 0:
                            non_sig_browning = non_sig_browning + 1
                        else:
                            non_sig_greening = non_sig_greening + 1
                        ##percentage
                if len(vals) == 0:
                    continue
                sig_browning = sig_browning / len(vals)
                sig_greening = sig_greening / len(vals)
                non_sig_browning = non_sig_browning / len(vals)
                non_sig_greening = non_sig_greening / len(vals)
                dic[wetting_drying] = [sig_greening, non_sig_greening, non_sig_browning, sig_browning
                                          ]
            df_new = pd.DataFrame(dic, index=['sig_greening', 'non_sig_greening', 'non_sig_browning', 'sig_browning'])
            df_new_T = df_new.T
            df_new_T.plot.bar(stacked=True, color=color_list, legend=False)
            plt.legend()

            plt.title(f'{region}')
            plt.ylabel('percentage')
            plt.xticks(rotation=45)
            plt.tight_layout()
        plt.show()


        pass

    def plot_drying_greening_map(self):  # generate map which contains drying-greening, drying-browning, wetting-0greening and wetting browning
        df= T.load_df(result_root + 'Dataframe\\anomaly_LAI\\anomaly_right.df')
        df=self.clean_df(df)


        cm = 1 / 2.54
        label_list=[]
        for i , row in df.iterrows():
            pix=row['pix']
            wetting_drying=row['wetting_drying_trend']
            val=row['LAI4g_trend']
            p_value=row['LAI4g_p_value']
            if wetting_drying=='sig_wetting':
                if p_value<0.05:
                    if val>0:
                        label_list.append('sig_greening_sig_wetting')
                    else:
                        label_list.append('sig_browning_sig_wetting')
                else:
                    if val>0:
                        label_list.append('non_sig_greening_sig_wetting')
                    else:
                        label_list.append('non_sig_browning_sig_wetting')
            elif wetting_drying=='sig_drying':
                if p_value<0.05:
                    if val>0:
                        label_list.append('sig_greening_sig_drying')
                    else:
                        label_list.append('sig_browning_sig_drying')
                else:
                    if val>0:
                        label_list.append('non_sig_greening_sig_drying')
                    else:
                        label_list.append('non_sig_browning_sig_drying')
            else:
                label_list.append(np.nan)


            pass
        df['label']=label_list
        T.print_head_n(df)

        ## save df
        T.save_df(df,result_root + 'Dataframe\\anomaly_LAI\\anomaly_right.df')
        ## save xlsx
        T.df_to_excel(df, result_root + 'Dataframe\\anomaly_LAI\\anomaly_right.xlsx')
        exit()

        ### calculate every cluster
        df=df.dropna()

        ## calculate percentage
        dic={}
        cluster=['sig_greening_sig_wetting', 'sig_browning_sig_wetting', 'non_sig_greening_sig_wetting',
         'non_sig_browning_sig_wetting', 'sig_greening_sig_drying', 'sig_browning_sig_drying',
         'non_sig_greening_sig_drying', 'non_sig_browning_sig_drying']

        cluster=['sig_greening_sig_wetting', 'sig_browning_sig_wetting',
          'sig_greening_sig_drying', 'sig_browning_sig_drying',
         ]

        for label in cluster:
            df_temp=df[df['label']==label]
            dic[label]=len(df_temp)/len(df)
        print(dic)
        ##plt.bar
        # plt.bar(dic.keys(),dic.values())
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()
        # exit()




        ####### plot map

        spatica_dic = {}
        dic_label = {'sig_greening_sig_wetting': 1, 'sig_browning_sig_wetting': 2, 'non_sig_greening_sig_wetting': 3,

                        'non_sig_browning_sig_wetting': 4, 'sig_greening_sig_drying': 5, 'sig_browning_sig_drying': 6,

                        'non_sig_greening_sig_drying': 7, 'non_sig_browning_sig_drying': 8, np.nan: 0}
        for i, row in df.iterrows():
            pix = row['pix']
            label = row['label']
            spatica_dic[pix] = dic_label[label]

        arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatica_dic)
        plt.imshow(arr_trend)
        plt.colorbar()
        plt.show()
        ## save
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, result_root + 'Dataframe\\anomaly_trends\\anomaly_trends_map.tif')


        pass

    def plot_cluster_variables_trend(self):
        df= T.load_df(result_root + 'Dataframe\\anomaly_LAI\\anomaly_right.df')
        df=self.clean_df(df)
        variable_list=['VPD_trend','tmin_trend','tmax_trend','Tempmean_trend',
                                'GLEAM_SMroot_trend','GLEAM_SMsurf_trend','LAI4g_trend']

        wetting_drying_greening_browning_list=['sig_greening_sig_drying',
                                                'sig_greening_sig_wetting']
        T.print_head_n(df)
        for variable in variable_list:
            for label in wetting_drying_greening_browning_list:
                df_temp=df[df['label']==label]
                x_list=df_temp[variable].tolist()
                plt.hist(x_list,bins=20,label=label,alpha=0.5)
                plt.xticks(rotation=45)

                plt.tight_layout()
                plt.title(variable)
            plt.legend()
            plt.show()
        exit()




        pass
    def plot_cluster_LAI_response_to_variables(self):  # four cluster as function of soil texture, preciptation CV
        df= T.load_df(result_root + 'Dataframe\\anomaly_trends\\anomaly_trends.df')
        df=self.clean_df(df)
        wetting_drying_greening_browning_list=['sig_greening_sig_drying','sig_browning_sig_drying',
                                               'sig_browning_sig_wetting','sig_greening_sig_wetting']

        T.print_head_n(df)

        ## get label column unique value
        label_list = df['label'].unique()
        print(label_list)
        color_list=['red','green','blue','orange']

        ## how to cluster for label four cluster


        cm = 1 / 2.54
        label_list=[]
        for label in wetting_drying_greening_browning_list:
            df_temp=df[df['label']==label]
            ## extract soil texture and preciptation CV
            x_list=df_temp['Rooting_Depth'].tolist()


            LAI_trend_list=df_temp['LAI4g_trend'].tolist()

            print(LAI_trend_list)

            plt.scatter(x_list,LAI_trend_list,label=label, color=color_list[wetting_drying_greening_browning_list.index(label)])
            # plt.xlabel('soil texture')
            plt.xlabel('Rooting_Depth')
            plt.ylabel('LAI trend')
            plt.legend()
        plt.show()
        exit()

            # preciptation_CV_list=df_temp['precip_CV'].tolist()






        df['label']=label_list
        T.print_head_n(df)

        ## save df
        T.save_df(df,result_root + 'Dataframe\\anomaly_trends\\anomaly_trends.df')
        df.to_excel(result_root + 'Dataframe\\anomaly_trends\\anomaly_trends.xlsx')
        exit()








    def plot_drying_wetting_areas(self):  # calculate the percentage of drying and greening pixels for each region
        df= T.load_df(result_root + 'Dataframe\\split_multiregression\\split_multiregression.df')
        # df=self.clean_df(df)
        # print(df)
        period_list=['1982_2001','2002_2020']
        wetting_drying_list=['non_sig_wetting','sig_wetting','non_sig_drying','sig_drying']
        ## calculate area
        region_dic={}
        for region in ['Arid','Semi-Arid','Sub-Humid']:
            dic_period={}
            df_temp = df[df['AI_classfication'] == region]
            for period in period_list:
                dic = {}
                sig_wetting_area = 0
                non_sig_wetting_area = 0
                sig_drying_area = 0
                non_sig_drying_area = 0
                val_trend_list=df_temp[f'LAI4g_{period}_trend'].tolist()
                p_values = df_temp[f'LAI4g_{period}_p_value'].tolist()
                ## calculate each classficaiton area
                for i in range(len(val_trend_list)):
                    if p_values[i]<0.05:
                        if val_trend_list[i]<0:
                            sig_drying_area=sig_drying_area+1
                        else:
                            sig_wetting_area=sig_wetting_area+1
                    else:
                        if val_trend_list[i]<0:
                            non_sig_drying_area=non_sig_drying_area+1
                        else:
                            non_sig_wetting_area=non_sig_wetting_area+1

                dic['sig_wetting_area']=sig_wetting_area/len(val_trend_list)
                dic['non_sig_wetting_area'] = non_sig_wetting_area/len(val_trend_list)
                dic['sig_drying_area'] = sig_drying_area/len(val_trend_list)
                dic['non_sig_drying_area'] = non_sig_drying_area/len(val_trend_list)
                dic_period[period]=dic
            region_dic[region]=dic_period
        print(region_dic)
        # exit()
        color_list=['green','lime','orange','red']
        regions=['Arid','Semi-Arid','Sub-Humid']
        cm = 1 / 2.54
        ## plot area and two periods are in one figure
        fig = plt.figure()
        flag=1
        for region in regions:
            ax = fig.add_subplot(1, 3, flag)
            dic_period=region_dic[region]
            df_new_2=pd.DataFrame(dic_period)

            df_new_2.plot.bar(ax=ax,color=color_list,legend=False)
            plt.title(region)
            plt.ylabel('percentage')
            plt.xticks(rotation=45,ha='right')
            plt.tight_layout()

            flag=flag+1
        plt.legend()
        plt.show()
        # exit()

    def plot_browning_greening_areas(self):  # calculate the percentage of drying and greening pixels for each region
        df= T.load_df(result_root + 'Dataframe\\split_multiregression\\split_multiregression.df')
        # df=self.clean_df(df)
        # print(df)
        period_list=['1982_2001','2002_2020']

        ## calculate area
        region_dic={}
        for region in ['Arid','Semi-Arid','Sub-Humid']:
            dic_period={}
            df_temp = df[df['AI_classfication'] == region]
            for period in period_list:
                dic = {}
                sig_greening_area = 0
                non_sig_greening_area = 0
                sig_browning_area = 0
                non_sig_browning_area = 0
                val_trend_list=df_temp[f'LAI4g_{period}_trend'].tolist()
                p_values = df_temp[f'LAI4g_{period}_p_value'].tolist()
                ## calculate each classficaiton area
                for i in range(len(val_trend_list)):
                    if p_values[i]<0.05:
                        if val_trend_list[i]<0:
                            sig_browning_area=sig_browning_area+1
                        else:
                            sig_greening_area=sig_greening_area+1
                    else:
                        if val_trend_list[i]<0:
                            non_sig_browning_area=non_sig_browning_area+1
                        else:
                            non_sig_greening_area=non_sig_greening_area+1

                dic['sig_greening_area']=sig_greening_area/len(val_trend_list)
                dic['non_sig_greening_area'] = non_sig_greening_area/len(val_trend_list)
                dic['sig_browning_area'] = sig_browning_area/len(val_trend_list)
                dic['non_sig_browning_area'] = non_sig_browning_area/len(val_trend_list)

                dic_period[period]=dic
            region_dic[region]=dic_period
        print(region_dic)
        # exit()
        color_list=['green','lime','orange','red']
        regions=['Arid','Semi-Arid','Sub-Humid']
        cm = 1 / 2.54
        ## plot area and two periods are in one figure
        fig = plt.figure()
        flag=1
        for region in regions:
            ax = fig.add_subplot(1, 3, flag)
            dic_period=region_dic[region]
            df_new_2=pd.DataFrame(dic_period)

            df_new_2.plot.bar(ax=ax,color=color_list,legend=False)
            plt.title(region)
            plt.ylabel('percentage')
            plt.xticks(rotation=45,ha='right')
            plt.tight_layout()

            flag=flag+1
        plt.legend()
        plt.show()
        # exit()





















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
        df= T.load_df(result_root + 'Dataframe\\split_multiregression\\split_multiregression.df')
        # product_list = ['LAI4g', 'NDVI4g', 'GPP_CFE', 'GPP_baseline']
        color_list=['green','lime','orange','red']

        product='LAI4g'
        landcover_list=['Arid','Semi-Arid','Sub-Humid']
        period_list=['1982_2001','2002_2020']
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

    def plot_category(self):  ### plot the category of each pixel
        df= T.load_df(result_root + 'Dataframe\\growth_rate_trend_category\\growth_rate_trend_category.df')


        result_dic = {}
        df = df[df['landcover_classfication'] != 'Cropland']
        print(len(df))

        PN = 0
        NP = 0
        PP = 0
        NN = 0
        NA = 0

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            trend = row['LAI4g_trend']
            p_value = row['LAI4g_p_value']
            growthrate = row['LAI4g_trend_growth_rate']
            growthrate_p_value = row['LAI4g_p_value_growth_rate']


            ###  trend siginificant positive and growthrate siginificant positive  accelerate greening
            # trend siginificant positive but growthrate siginificant negative     slow down greening
            # trend siginificant negative but growthrate siginificant positive     accelerate browning
            # trend siginificant negative but growthrate siginificant negative     slow down browning
            if p_value < 0.1 and growthrate_p_value < 0.1:
                if trend > 0 and growthrate > 0:
                    result_dic[pix] = 'PP'
                    PP = PP + 1
                elif trend > 0 and growthrate < 0:
                    result_dic[pix] = 'PN'
                    PN = PN + 1
                elif trend < 0 and growthrate > 0:
                    result_dic[pix] = 'NP'
                    NP = NP + 1
                elif trend < 0 and growthrate < 0:
                    result_dic[pix] = 'NN'
                    NN = NN + 1
                else:
                    result_dic[pix] = 'NA'
                    NA = NA + 1
            else:
                result_dic[pix] = 'NA'
                NA = NA + 1


        ##calculte the percentage of each category
        total=PN+NP+PP+NN+NA
        print(total)

        PN=PN/total
        NP=NP/total
        PP=PP/total
        NN=NN/total
        NA=NA/total
        print(PN,NP,PP,NN,NA)
        df_new=pd.DataFrame({'PN':PN,'NP':NP,'PP':PP,'NN':NN,'NA':NA},index=['percentage'])
        df_new_T=df_new.T
        df_new_T.plot.bar()
        plt.show()
        exit()
        ##spatial plot
        ##convert dic to arr
        ###PP=1,PN=2,NN=3,NP=4,NA=5

        result_dic2={}

        for pix in result_dic:
            val=result_dic[pix]
            if val=='PP':
                result_dic2[pix]=1
            elif val=='PN':
                result_dic2[pix]=2
            elif val=='NN':
                result_dic2[pix]=3
            elif val=='NP':
                result_dic2[pix]=4
            elif val=='NA':
                result_dic2[pix]=5


        arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic2)
        plt.imshow(arr)
        plt.colorbar()
        plt.show()

    def plot_landcover_classfication_yearly(self): ## plot the landcover classfication yearly based on greening trend
        df= T.load_df(result_root + 'Dataframe\LUCC_change\\LUCC_change.df')
        ##sifting trend> 0 and significant positive
        df = df.dropna(subset=['LAI4g_2002_2020_trend'])
        df=df[df['LAI4g_2002_2020_trend']<0]
        df=df[df['LAI4g_2002_2020_p_value']<0.1]

        landcover_list=['urban',]

        vals_dic={}
        for landcover in landcover_list:
            vals_list = []

            vals=df[landcover].tolist()
            vals_nonnan=[]
            for val in vals:
                if type(val)==float:
                    continue
                vals_nonnan.append(val)
            vals_mean=np.nanmean(vals_nonnan,axis=0)
            vals_list.append(vals_mean)
            vals_dic[landcover]=vals_mean

        ##plot stacked bar
        df_new=pd.DataFrame(vals_dic)

        df_new.plot.bar(stacked=True)
        plt.show()
        exit()

    def plot_multiregression(self):
        df= T.load_df(result_root + 'Dataframe\\residual_method\\residual_method.df')
        landcover_list=['Evergreen','Deciduous','Grass','Shrub','Cropland']
        color_list=['green','lime','orange','red','blue','black','grey']


        fig = plt.figure()

        flag = 1

        dic_period={}

        for period in ['1982_2001','2002_2020']:
            # df_greening = df[df[f'LAI4g_{period}_trend'] > 0]
            # df_browning = df[df[f'LAI4g_{period}_trend'] < 0]
            df_sig = df[df[f'LAI4g_{period}_p_value'] < 0.1]
            # df_browning = df_browning[df_browning[f'LAI4g_{period}_p_value'] < 0.1]

            dic_landcover={}
            ax = fig.add_subplot(2, 1, flag)

            for landcover in landcover_list:

                df_temp = df_sig[df_sig['landcover_classfication'] == landcover]

                dic_variable={}
                for variable in ['CO2',]:

                    vals=df_temp[f'{variable}_{period}'].tolist()
                    vals=np.array(vals)
                    vals[vals<-99]=np.nan

                    # p_values=df_temp[f'{variable}_{period}_LAI4g_{period}_p_value'].tolist()

                    # mean_val=np.nanmean(vals)
                    dic_variable[variable]=vals

                dic_landcover[landcover]=dic_variable

            df_new=pd.DataFrame(dic_landcover)
            # df_new_T=df_new.T
            ##plot boxplot
            column=df_new.columns
            ##


            df_new.boxplot(column=column,ax=ax)
            plt.show()




            # df_new_T.plot.bar(ax=ax,color=color_list[:len(columns)],legend=False)

    def plot_multiregression_boxplot(self):
        df = T.load_df(result_root + 'Dataframe\\residual_method\\residual_method.df')
        landcover_list = ['Evergreen', 'Deciduous', 'Grass', 'Shrub', 'Cropland']
        region_list=['Arid','Semi-Arid','Sub-Humid']
        color_list = ['green', 'lime', 'orange', 'red', 'blue', 'black', 'grey']
        T.print_head_n(df)


        fig = plt.figure()


        for variable in ['CO2']:
            label_list=[]
            boxplot_list=[]


            for landcover in region_list:
                # df_lc=df[df['landcover_classfication']==landcover]
                df_lc = df[df['AI_classfication'] == landcover]



                for period in ['1982_2001', '2002_2020']:
                    df_lc = df_lc[df_lc[f'LAI4g_{period}_trend'] > 0]
                    print(len(df_lc))
                    # df_browning = df[df[f'LAI4g_{period}_trend'] < 0]
                    df_sig = df_lc[df_lc[f'LAI4g_{period}_p_value'] < 0.05]
                    print(len(df_sig))

                    vals = df_sig[f'{variable}_{period}'].tolist()
                    vals = np.array(vals)
                    vals[vals < -99] = np.nan
                    ##remover nan
                    vals=vals[~np.isnan(vals)]

                    boxplot_list.append(vals)
                    label_list.append(f'{landcover}_{period}')
            plt.boxplot(boxplot_list,labels=label_list,showfliers=False,showmeans=True,vert=False,whis=1.5)
            plt.show()






    def plot_Yao_Zhang_method(self,):  ##replicateYao's NC paper aridity, SM trend,greening trend
        df= T.load_df(result_root + 'Dataframe\multiregression\\multiregression.df')


        z_values_list=self.product_list
        ### plot heatmap of aridity, SM trend, greening trend
        df=df[df['landcover_classfication']!='Cropland']
        aridity_bin=np.arange(0,0.8,0.1)
        SM_bin=np.arange(-0.006,0.006,0.001)


        flag=1
        ##plt figure
        fig = plt.figure()
        for z_value in tqdm(z_values_list):
            # df=df[df[f'{z_value}_p_value']<0.1]
            matrix = []
            z_value=z_value+'_trend'
            ax = fig.add_subplot(5, 4, flag)
            flag=flag+1



            for i in range(len(SM_bin)):
                if i == len(SM_bin) - 1:
                    continue

                y_left = SM_bin[i]
                y_right = SM_bin[i + 1]

                matrix_i = []
                for j in range(len(aridity_bin)):
                    if j == len(aridity_bin) - 1:
                        continue
                    x_left = aridity_bin[j]
                    x_right = aridity_bin[j + 1]

                    df_temp_i = df[df['GLEAM_SMroot_trend'] >= y_left]
                    df_temp_i = df_temp_i[df_temp_i['GLEAM_SMroot_trend'] < y_right]
                    df_temp_i = df_temp_i[df_temp_i['Aridity'] >= x_left]
                    df_temp_i = df_temp_i[df_temp_i['Aridity'] < x_right]
                    mean=np.nanmean(df_temp_i[z_value].tolist())

                    matrix_i.append(mean)
                matrix.append(matrix_i)
            matrix = np.array(matrix)
            matrix = matrix[::-1, :]  # reverse
            plt.imshow(matrix, cmap='RdBu', interpolation='nearest')
            ## add x labeland y label and keep round 2 and y_ticks need to be reversed because I want north is positive
            xticks = []
            for i in range(len(aridity_bin)):
                if i % 2 == 0:
                    xticks.append(f'{aridity_bin[i]:.1f}')
                else:
                    xticks.append('')
            plt.xticks(range(len(xticks)), xticks, rotation=45, ha='right')
            yticks = []
            ## y_ticks need to be reversed negative to positive
            for i in range(len(SM_bin)):
                if i % 2 == 0:
                    yticks.append(f'{SM_bin[::-1][i]:.3f}')
                else:
                    yticks.append('')
            plt.yticks(range(len(yticks)), yticks)
            # plt.xlabel('Aridity')
            # plt.ylabel('SM trend')
            plt.title(z_value)
            ## color bar scale from -0.1 to 0.1
            plt.clim(-0.008, 0.008)



        plt.colorbar()


        plt.show()
        # plt.savefig(self.outdir + f'{region}_{z_val_name}.pdf', dpi=300)
        # plt.close()


class plt_moving_dataframe():
    def run(self):
        self.plot_moving_window()
        pass
    def plot_moving_window(self):  ##plot moving window bar
        df=result_root+rf'Dataframe\extract_moving_window_trend\\extract_moving_window_trend.df'
        df=T.load_df(df)
        T.print_head_n(df)

        df=df.dropna(subset=['LAI4g'])
        pixel_list=T.get_df_unique_val_list(df,'pix')

        # df=df[df['landcover_classfication']!='Cropland']

        color_list = ['green', 'red', '#D3D3D3']

        fig = plt.figure()
        ii = 1
        for landcover in ['Evergreen', 'Deciduous', 'Mixed','Grass','Shrub','Cropland']:

        # for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
        #     df_region = df[df['AI_classfication'] == region]

            df_region = df[df['landcover_classfication'] == landcover]
            ax = fig.add_subplot(2, 3, ii)
            flag = 0

            window_list = []
            dic={}

            for i in range(1, 25):
                window_list.append(i)
            print(window_list)

            for window in tqdm(window_list):  # 构造字典的键值，并且字典的键：值初始化
                dic[window] = []

            ## plt moving window bar based on trend and p_value
            for window in window_list:
                df_pick = df_region[df_region['window'] == window]
                greening_area= 0
                browning_area= 0
                no_change_area= 0

                for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                    pix = row.pix
                    trend = row['LAI4g']
                    p_value = row['LAI4g_p_value']
                    if trend > 0 and p_value < 0.1:
                        greening_area += 1
                    elif trend < 0 and p_value < 0.1:
                        browning_area += 1
                    else:
                        no_change_area += 1

                total_area = greening_area + browning_area + no_change_area
                if total_area == 0:
                    continue
                greening_area = greening_area / len(pixel_list) * 100
                browning_area = browning_area / len(pixel_list) * 100
                no_change_area = no_change_area / len(pixel_list) * 100
                dic[window] = [greening_area, browning_area]
            df_new = pd.DataFrame(dic)
            df_new_T = df_new.T
        ## if no number in the window, continue

            df_new_T.plot.bar(ax=ax, stacked=True, color=color_list, legend=False)
            plt.title(landcover)
            plt.xlabel('window size (year)')
            xticks= []

            ## set xticks with 1982-1997, 1998-2013,.. 2014-2020
            # for i in range(1, 25):
            #     if i % 2 == 0:
            #         xticks.append(f'{1982 + 16 * (i - 1)}-{1982 + 16 * i - 1}')
            #     else:
            #         xticks.append(f'{1982 + 16 * (i - 1)}-{1982 + 16 * i - 1}')
            # plt.xticks(range(len(xticks)), xticks, rotation=45, ha='right')

            plt.ylabel('percentage')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0,15)
            plt.tight_layout()

            ii = ii + 1
        plt.legend(['Greening', 'Browning'])

        plt.tight_layout()
        plt.show()

    pass

    def plot_moving_window_vs_wetting_drying_bin(self):  ### each winwow,plot pdf of NDVI as bin of wetting and drying
        fdir_trend = result_root + rf'extract_window\\extract_original_window_trend\\15\\GPCC\\'

        dic_trend = T.load_npy(fdir_trend + 'GPCC.npy')

        area_dic = {}
        for ss in range(39 - 15):
            print(ss)

            trend_value_list = []

            for pix in tqdm(dic_trend):
                # print(len(dic_trend[pix]))
                if len(dic_trend[pix]) < 24:
                    continue
                trend_value = dic_trend[pix][ss]

                trend_value_list.append(trend_value)

            area_dic[ss] = [trend_value_list]

        #### load wetting and drying
        fdir_wetting_drying = result_root + rf'extract_window\extract_original_window_trend\15\wetting_drying\\'
        dic_wetting_drying = T.load_npy(fdir_wetting_drying + 'wetting_drying.npy')
        wetting_drying_list = []
        for pix in dic_wetting_drying:
            trend = dic_wetting_drying[pix]
            wetting_drying_list.append(trend)
        wetting_drying_list = np.array(wetting_drying_list)
        ##### creat df
        df_new = pd.DataFrame(area_dic)
        df_new = df_new.T
        df_new['wetting_drying'] = wetting_drying_list
        df_new = df_new.rename(columns={0: 'trend'})

        ####


class check_data():
    def run (self):
        self.plot_sptial()
        # self.testrobinson()
        # self.plot_time_series()
        # self.plot_bar()


        pass
    def plot_sptial(self):

        f =  'D:\Project3\Data\monthly_data\Precip\\'

        fdir=rf'D:\Project3\Data\monthly_data\Precip\\'
        # dic=T.load_npy(f)
        dic=T.load_npy_dir(fdir)

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
        plt.imshow(arr,cmap='RdBu',interpolation='nearest',vmin=20,vmax=40)
        plt.colorbar()
        plt.title('')
        plt.show()
    def testrobinson(self):

        f = result_root + rf'D:\Project3\Result\Dataframe\anomaly_trends\anomaly_trends_map.tif'
        plt.title('GPCC_trend(mm/mm/year)')
        # f = data_root + rf'split\NDVI4g\2001_2020.npy'
        Plot().plot_Robinson(f, vmin=9,vmax=0,is_discrete=True,colormap_n=5)


        plt.show()



    def plot_time_series(self):
        f=result_root + rf'anomaly\OBS\GIMMS3g.npy'

        # f=result_root + rf'extract_GS\OBS_LAI_extend\\Tmax.npy'
        # f=data_root + rf'monthly_data\\Precip\\DIC\\per_pix_dic_004.npy'
        # f= result_root+ rf'detrend_zscore_Yang\LAI4g\\1982_2000.npy'
        dic=T.load_npy(f)

        for pix in dic:
            vals=dic[pix]


            print(vals)



            vals=np.array(vals)
            print(vals)
            print(len(vals))

            # if not len(vals)==19*12:
            #     continue
            # if True in np.isnan(vals):
            #     continue
            # print(len(vals))
            if np.isnan(np.nanmean(vals)):
                continue
            if np.nanmean(vals)<-20:
                continue
            plt.plot(vals)

            plt.title(pix)
            plt.legend([f'{f}'])
            plt.show()
    def plot_bar(self):
        fdir=rf'D:\Project3\Result\trend_analysis\original\\\OBS\\'
        variable_list=['LAI4g_trend','LAI4g_xcludion_LaNina_trend','LAI4g_excludion_LaNina_EINino_trend']
        GPP_list=['GPP_CFE_trend','GPP_CFE_excludion_LaNina_trend','GPP_CFE_eexcludion_LaNina_EINino_trend']
        average_dic = {}

        for f in os.listdir(fdir):
            if not f.endswith(('.npy')):
                continue
            if not 'LAI4g' in f:
                continue
            if 'tiff' in f:
                continue
            if 'p_value' in f:
                continue


            array=np.load(fdir+f)
            array=np.array(array)
            array[array<-99]=np.nan

            array=array[array!=np.nan]
            ## calculate mean
            average_dic[f.split('.')[0]]=np.nanmean(array)


        #     average_dic[f.split('.')[0]]=np.nanmean(array,axis=0)
        df=pd.DataFrame(average_dic,index=['OBS'])
        df.plot.bar()
        #
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
    # data_processing().run()
    # statistic_analysis().run()
    # ResponseFunction().run()
    # bivariate_analysis().run()
    # CCI_LC_preprocess().run()
    # calculating_variables().run()
    # pick_event().run()
    # selection().run()
    # multi_regression().run()

    # quadratic_regression().run()
    # quadratic_regression1().run()
    Seasonal_sensitivity().run()
    # residual_method().run()
    # Contribution_Lixin().run()
    # fingerprint().run()
    # moving_window().run()
    # multi_regression_window().run()
    # build_dataframe().run()
    # plot_dataframe().run()
    # plt_moving_dataframe().run()
    # check_data().run()
    # Dataframe_func().run()


    pass

if __name__ == '__main__':
    main()