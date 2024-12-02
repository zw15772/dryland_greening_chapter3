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



this_root = 'E:\Project3\\'
data_root = 'E:/Project3/Data/'
result_root = 'E:/Project3/Result/'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

class data_processing():
    def __init__(self):
        pass
    def run(self):
        # self.download_CCI_landcover()
        # self.download_ERA_precip()
        # self.download_CCI_ozone()
        # self.nc_to_tif_fire()

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
        # self.resample_GIMMS4g()
        self.resample_MODIS_LUCC()
        # self.resample_glc_LUCC()
        # self.resample_inversion()
        # self.aggregate_GIMMS3g()
        # self.aggreate_AVHRR_LAI() ## this method is used to aggregate AVHRR LAI to monthly
        # self.unify_TIFF()
        # self.extract_dryland_tiff()
        # self.tif_to_dic()
        # self.aggreate_CO2()
        # self.average_temperature()
        # self.scales_Inversion()
        # self.scale_LAI()
        # self.extract_dryland_tiff()

        # self.trendy_ensemble_calculation()  ##这个函数不用 因为ensemble original data 会出现最后一年加入数据不全，使得最后一年得知降低



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
    def download_CCI_ozone(self):

        import cdsapi

        c = cdsapi.Client()

        c.retrieve(
            'satellite-ozone-v1',
            {
                'processing_level': 'level_4',
                'variable': 'atmosphere_mole_content_of_ozone',
                'vertical_aggregation': 'total_column',
                'sensor': 'msr',
                'year': [
                    '1982', '1983', '1984',
                    '1985', '1986', '1987',
                    '1988', '1989', '1990',
                    '1991', '1992', '1993',
                    '1994', '1995', '1996',
                    '1997', '1998', '1999',
                    '2000', '2001', '2002',
                    '2003', '2004', '2005',
                    '2006', '2007', '2008',
                    '2009', '2010', '2011',
                    '2012', '2013', '2014',
                    '2015', '2016', '2017',
                    '2018',
                ],
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'version': 'v0021',
                'format': 'zip',
            },
            rf'C:\Users\wenzhang1\Desktop\CCI_ozone/ozone_v0021.zip')
        pass

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
                'format': 'netcdf',
                'variable': [
                    '2m_temperature', 'skin_temperature', 'total_precipitation',
                ],
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'year': '1982',
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
            },
            'D:\Project3\Data\ERA5\\nc\\hourly_precipitation.nc')



        pass




        pass
    def nc_to_tif(self):

        # fdir=data_root+f'\GPP\\NIRvGPP\\nc\\'
        fdir=rf'D:\Project3\Data\deposition\\'
        outdir=rf'D:\Project3\Data\deposition\\TIFF\\'
        Tools().mk_dir(outdir,force=True)
        for f in os.listdir(fdir):

            outdir_name = f.split('.')[0]
            print(outdir_name)

            yearlist = list(range(1982, 2021))


            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir+f, var_name='GPP', outdir=outdir, yearlist=yearlist)
            except Exception as e:
                print(e)
                continue

    def nc_to_tif_fire(self):


        fdir=rf'C:\Users\wenzhang1\Desktop\CCI_ozone\ozone_v0021\\'
        outdir=rf'C:\Users\wenzhang1\Desktop\CCI_ozone\TIFF\\'
        Tools().mk_dir(outdir,force=True)
        # print(isfile(f))
        # exit()


        yearlist = list(range(1982, 2021))
        for f in os.listdir(fdir):

            self.nc_to_tif_template(fdir+f,var_name='total_ozone_column',outdir=outdir,yearlist=yearlist)
        # try:

        # except Exception as e:
        #     print(e)
        #     pass

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
        fdir =rf'E:\Carbontracker\nc\\'
        outdir = 'E:\Carbontracker\TIFF\\'
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
    def extract_dryland_tiff(self):
        self.datadir='E:/Project3/Data/'
        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask05.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan
        outdir = rf'D:\Project3\TRENDY\\dryland_tiff\\'
        T.mk_dir(outdir, force=True)

        fdir_all = rf'D:\Project3\TRENDY\unify\\'
        for fdir in T.listdir(fdir_all):
            fdir_i = join(fdir_all, fdir)
            outdir_i = join(outdir, fdir)
            T.mk_dir(outdir_i)
            for fi in tqdm(T.listdir(fdir_i), desc=fdir):
                if not fi.endswith('.tif'):
                    continue
                fpath = join(fdir_i, fi)
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr[np.isnan(array_mask)] = np.nan
                # plt.imshow(arr)
                # plt.show()
                outpath = join(outdir_i, fi)

                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)

        pass

        self.tif_to_dic()


        # self.extract_GS()
        # self.extract_GS_return_monthly_data()
        # self.extract_GS_return_monthly_data_yang()
        # self.extract_seasonality()
        #
        # self.extend_GS() ## for SDGVM， it has 37 year GS, to align with other models, we add one more year
        # self.extend_nan()  ##  南北半球的数据不一样，需要后面加nan
        # self.scales_GPP_Trendy()
        # self.split_data()
        # self.weighted_LAI()

    def tif_to_dic(self):

        fdir_all = rf'D:\Project3\TRENDY\dryland_tiff\\'



        year_list = list(range(1982, 2021))


        # 作为筛选条件
        for fdir in os.listdir(fdir_all):


            outdir=rf'D:\Project3\TRENDY\dic\\{fdir}\\'
            # if os.path.isdir(outdir):
            #     pass

            T.mk_dir(outdir, force=True)
            all_array = []  #### so important  it should be go with T.mk_dic

            for f in os.listdir(fdir_all+ fdir):
                if not f.endswith('.tif'):
                    continue
                if int(f.split('.')[0][0:4]) not in year_list:
                    continue


                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir_all + fdir + '\\' + f)
                array = np.array(array, dtype=float)


                # array_unify = array[:720][:720,
                #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
                array_unify = array[:360][:360,
                              :720]
                array_unify[array_unify < -999] = np.nan
                # array_unify[array_unify > 7] = np.nan
                # array[array ==0] = np.nan

                # array_unify[array_unify < 0] = np.nan

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
                    np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                    temp_dic = {}
            np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)



    def extract_GS(self):  ## here using new extraction method: 240<r<480 all year growing season

        fdir_all = data_root + rf'LAI4g\biweekly_dic\\'
        outdir = result_root + f'extract_GS\\OBS_LAI\\'
        Tools().mk_dir(outdir, force=True)
        date_list=[]

        # print(date_list)
        # exit()

        for year in range(1982, 2021):
            for mon in range(1, 13):
                date_list.append(datetime.datetime(year, mon, 1))

        for fdir in os.listdir(fdir_all):
            if not 'DIC' in fdir:
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


                vals[vals < -999] = np.nan

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
                    # mean = np.nanmean(vals_gs)
                    total_amount= np.nansum(vals_gs)  ### 降雨需要求和

                    # annual_gs_list.append(mean)
                    annual_gs_list.append(total_amount)

                annual_gs_list = np.array(annual_gs_list)

                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            np.save(outf, annual_spatial_dict)

        pass






    def extract_GS_return_monthly_data(self):  ## extract growing season but return monthly data

        fdir_all = data_root + rf'LAI4g\biweekly_dic\\'
        outdir = result_root + f'extract_GS_return_biweekly\\OBS_LAI\\'
        Tools().mk_dir(outdir, force=True)
        date_list=[]

        # print(date_list)
        # exit()

        for year in range(1982, 2021):
            for mon in range(1, 13):
                date_list.append(datetime.datetime(year, mon, 1))

        for f in os.listdir(fdir_all):


            outf = outdir + f.split('.')[0] + '.npy'

            # T.open_path_and_file(fdir_all + fdir)
            # print(outf)
            # exit()

            spatial_dict = T.load_npy_dir(fdir_all )
            # spatial_dict_new = {}
            # for pix in spatial_dict:
            #     vals = spatial_dict[pix]
            #     if np.nanstd(vals) == 0:
            #         continue
            #     vals_new = T.remove_np_nan(vals)
            #     spatial_dict_new[pix] = len(vals_new)
            # arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict_new)
            # plt.imshow(arr,interpolation='nearest',cmap='jet')
            # plt.colorbar()
            # plt.show()
            # exit()

            annual_spatial_dict = {}
            annual_spatial_dict_index = {}
            for pix in tqdm(spatial_dict):
                r,c=pix
                gs_mon = global_get_gs(pix)

                vals = spatial_dict[pix]
                vals = np.array(vals)
                if np.nanstd(vals) == 0:
                    continue
                vals = T.remove_np_nan(vals)
                if not len(vals) == 936:
                    continue
                # plt.plot(vals)
                # plt.show()

                # vals[vals == 65535] = np.nan
                if T.is_all_nan(vals):
                    continue


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
                annual_gs_list_idx = []

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
                    ## return month_index
                    annual_gs_list_idx.append(idx)

                annual_gs_list = np.array(annual_gs_list)

                # if T.is_all_nan(annual_gs_list):
                #     continue
                annual_spatial_dict[pix] = annual_gs_list
                # print(len(annual_gs_list))
                #
                # plt.imshow(annual_gs_list)
                # plt.show()
                annual_spatial_dict_index[pix] = annual_gs_list_idx

            np.save(outf, annual_spatial_dict)
            np.save(outf+'_index', annual_spatial_dict_index)

        pass
    def extract_GS_return_monthly_data_yang(self):  ## extract growing season but return monthly data

        fdir_all = data_root + rf'LAI4g\biweekly_dic\\'
        # print(fdir_all);exit()
        outdir = result_root + f'extract_GS_return_biweekly\\OBS_LAI\\'
        # print(outdir);exit()
        Tools().mk_dir(outdir, force=True)
        date_list=[]

        for year in range(1982, 2021):
            for mon in range(1, 13):
                for day in [1,15]:
                    date_list.append(datetime.datetime(year, mon, day))

        for f in tqdm(os.listdir(fdir_all)):

            outf = outdir + f.split('.')[0] + '.npy'

            fpath = join(fdir_all, f)

            spatial_dict = T.load_npy(fpath)
            # pprint(spatial_dict);exit()
            annual_spatial_dict = {}
            annual_spatial_dict_index = {}
            for pix in spatial_dict:
                gs_mon = global_get_gs(pix)

                vals = spatial_dict[pix]
                vals = np.array(vals)
                if T.is_all_nan(vals):
                    continue
                if np.nanstd(vals) == 0:
                    continue

                vals[vals < -999] = np.nan

                vals_dict = T.dict_zip(date_list, vals)
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)
                        date_list_index.append(i)

                consecutive_ranges = self.group_consecutive_vals(date_list_index)
                # date_dict = dict(zip(list(range(len(date_list))), date_list))
                date_dict = T.dict_zip(list(range(len(date_list))), date_list)
                # pprint(date_dict);exit()


                # annual_vals_dict = {}
                annual_gs_list = []
                annual_gs_list_idx = []
                # print(consecutive_ranges);exit()
                # print(len(consecutive_ranges[0]))
                # print(len(consecutive_ranges))

                if len(consecutive_ranges[0])>12: # tropical
                    consecutive_ranges=np.reshape(consecutive_ranges,(-1,24))
                # try:
                #     print(np.shape(consecutive_ranges))
                # except:
                #     print(consecutive_ranges)
                #     exit()
                # print(consecutive_ranges);exit()

                for idx in consecutive_ranges:
                    # print(len(idx))
                    date_gs = [date_dict[i] for i in idx]
                    # print(date_gs);exit()
                    # print(len(gs_mon)*2)
                    if not len(date_gs) == len(gs_mon)*2:
                        continue
                    year = date_gs[0].year

                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    # print(len(vals_gs))
                    ## return monthly data
                    annual_gs_list.append(vals_gs)
                    ## return month_index
                    annual_gs_list_idx.append(idx)

                annual_gs_list = np.array(annual_gs_list)
                # pprint(annual_gs_list)
                # print(annual_gs_list.shape)
                # exit()

                # if T.is_all_nan(annual_gs_list):
                #     continue
                annual_spatial_dict[pix] = annual_gs_list
                # print(len(annual_gs_list))
                #
                # plt.imshow(annual_gs_list)
                # plt.show()
                annual_spatial_dict_index[pix] = annual_gs_list_idx

            np.save(outf, annual_spatial_dict)
            np.save(outf+'_index', annual_spatial_dict_index)

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

        fdir=result_root + rf'relative_change\OBS_LAI_extend\\'

        outdir=result_root+rf'split_relative_change\\'
        T.mk_dir(outdir,force=True)

        for f in os.listdir(fdir):
            if not f.split('.')[0] in ['CO2','GPCC','tmax','VPD','CRU','LAI4g','GLEAM_SMroot']:
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

                time_series_i=time_series[:19]
                time_series_ii=time_series[19:]
                print(len(time_series_i))
                print(len(time_series_ii))

                # plt.plot(time_series_i)
                # plt.plot(time_series_ii)
                #
                # plt.show()

                dic_i[pix]=time_series_i
                dic_ii[pix]=time_series_ii

            np.save(outf+'1982_2000.npy',dic_i)
            np.save(outf+'2001_2020.npy',dic_ii)

    def extend_nan(self):
        fdir= rf'D:\Project3\Result\growth_rate\\'
        outdir=rf'D:\Project3\Result\growth_rate\\extend_nan\\'
        T.mk_dir(outdir,force=True)
        for f in os.listdir(fdir):


            if not f.endswith('.npy'):
                continue


            outf=outdir+f.split('.')[0]+'.npy'
            # if os.path.isfile(outf):
            #     continue
            dic = dict(np.load(fdir +f, allow_pickle=True, ).item())
            dic_new={}

            for pix in tqdm(dic):
                r,c=pix

                time_series=dic[pix]
                # print((time_series))

                time_series=np.array(time_series)
                if np.isnan(time_series).all():
                    continue
                time_series[time_series<-999]=np.nan
                if np.isnan(np.nanmean(time_series)):
                    continue
                if len(time_series)<38:

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

        f= result_root + rf'\\Detrend\detrend_relative_change\non_extend\\noy.npy'
        outf=result_root + rf'Detrend\detrend_relative_change\extend\\noy.npy'
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

    def weighted_LAI(self):
        area_dic = DIC_and_TIF(pixelsize=0.25).calculate_pixel_area()
        f_LAI = result_root + rf'\relative_change\OBS_LAI_extend\LAI4g.npy'

        dic_LAI = dict(np.load(f_LAI, allow_pickle=True, ).item())
        dic_leaf_area= {}
        for pix in tqdm(dic_LAI):
            area_val = area_dic[pix]
            r, c = pix
            LAI = dic_LAI[pix]
            LAI = np.array(LAI)
            leaf_area=LAI*area_val
            dic_leaf_area[pix] = leaf_area
        np.save(result_root + rf'relative_change\OBS_LAI_extend\leaf_area.npy', dic_leaf_area)



        pass

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
        fdir_all = 'D:\Project3\TRENDY\TIFF\\'
        for fdir in tqdm(os.listdir(fdir_all)):


            outdir = rf'D:\Project3\TRENDY\resample\\{fdir}\\'
            # if os.path.isdir(outdir):
            #     continue


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
                    gdal.Warp(outdir + '{}.tif'.format(date_2), dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
                # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
                # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
                except Exception as e:
                    pass
    def resample_AVHRR_LAI(self):
        fdir_all = rf'D:\Project3\Data\CO2\CO2_TIFF\original\\'

        outdir =rf'D:\Project3\Data\CO2\CO2_TIFF\original\\'


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

    def resample_GIMMS4g(self):
        fdir_all = rf'D:\Project3\Data\LAI4g\GIMMS_LAI4g_updated_20230216\\'

        outdir = rf'E:\Project3\Data\\LAI4g\\resample\\'


        T.mk_dir(outdir, force=True)
        year = list(range(1982, 2021))
        # print(year)
        # exit()
        for fdir in tqdm(os.listdir(fdir_all)):
            if not 'TIFF' in fdir:
                continue
            for f in tqdm(os.listdir(fdir_all + fdir)):
                if not f.endswith('.tif'):
                    continue



                # fname=f.split('.')[0]
                fname=f.split('.')[1].split('_')[1]
                print(fname)
                # exit()

                # fname=f.split('.')[1][:7]
                # print(fname)

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
                date = f.split('.')[1].split('_')[1]
                # date=f.split('.')[0]

                #
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
                    gdal.Warp(outdir + '{}.tif'.format(date), dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
                # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
                # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
                except Exception as e:
                    pass

    def resample_MODIS_LUCC(self):
        f=rf'D:\Project3\Data\Base_data\lc_trend\\max_trend.tif'

        outf = rf'E:\Project3\Data\\Base_data\MODIS_LUCC\\MODIS_LUCC_resample_05.tif'

        dataset = gdal.Open(f)



        try:
            gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
        # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
        # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
        except Exception as e:
            pass

    def resample_glc_LUCC(self):
        fpath=rf'E:\Project3\Data\Base_data\glc_025\\glc2000_025.tif'
        outf=rf'E:\Project3\Data\Base_data\glc_025\\glc2000_05.tif'


        dataset = gdal.Open(fpath)
        # print(dataset.GetGeoTransform())

        try:
            gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
        # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
        # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
        except Exception as e:
            pass



        pass
    def resample_inversion(self):
        fdir_all = rf'E:\Carbontracker\TIFF\\'

        outdir = rf'E:\Carbontracker\resample\\'
        T.mk_dir(outdir, force=True)
        year = list(range(2000, 2021))
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

        fdir_all =  rf'D:\Project4\Data\monthly_LAI4g\resample\\'
        outdir = rf'D:\Project4\Data\aggregated\\'
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
        fdir_all = rf'D:\Project4\Data\monthly_NDVI4g\scales_NDVI4g\\'
        outdir =  rf'D:\Project4\Data\\monthly_NDVI4g\\monthly_NDVI4g\\'
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
                    # arr=arr/1000 ###
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
                arr_average[arr_average <0] = np.nan
                arr_average[arr_average > 7] = np.nan
                if np.isnan(np.nanmean(arr_average)):
                    continue
                if np.nanmean(arr_average) < 0.:
                    continue
                # plt.imshow(arr_average)
                # plt.title(f'{year}{month}')
                # plt.show()

                # save

                DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_average, outdir + '{}{:02d}.tif'.format(year, month))
    def unify_TIFF(self):
        fdir_all=rf'D:\Project3\TRENDY\resample\\'


        for fdir in os.listdir(fdir_all):
            outdir = rf'D:\Project3\TRENDY\unify\\{fdir}\\'
            Tools().mk_dir(outdir, force=True)

            for f in tqdm(os.listdir(fdir_all+fdir)):
                fpath=join(fdir_all,fdir,f)

                outpath=join(outdir,f)

                if not f.endswith('.tif'):
                    continue
                if f.startswith('._'):
                    continue
                unify_tiff=DIC_and_TIF().unify_raster1(fpath,outpath,0.5)


    def aggreate_CO2(self):  # aggregate biweekly data to monthly
        fdir_all = rf'E:\Project3\Data\CO2_TIFF\unify\SSP245\\'
        outdir = rf'E:\Project3\Data\CO2_TIFF\annual\SSP245\\'
        Tools().mk_dir(outdir, force=True)

        year_list = list(range(2015, 2021))


        for year in tqdm(year_list):
            data_list = []


            for f in tqdm(os.listdir(fdir_all)):
                if not f.endswith('.tif'):
                    continue

                data_year = f.split('.')[0][0:4]


                if not int(data_year) == year:
                    continue


                arr = ToRaster().raster2array(fdir_all + f)[0]
                # arr=arr/1000 ###
                arr_unify = arr[:720][:720,
                            :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
                arr_unify = np.array(arr_unify)

                data_list.append(arr_unify)
            data_list = np.array(data_list)
            print(data_list.shape)
            # print(len(data_list))
            # exit()

            ##define arr_average and calculate arr_average

            arr_average = np.nanmean(data_list, axis=0)
            arr_average = np.array(arr_average)

            if np.isnan(np.nanmean(arr_average)):
                continue
            # if np.nanmean(arr_average) < 0.:
            #     continue
            # plt.imshow(arr_average)
            # plt.title(f'{year}{month}')
            # plt.show()

            # save

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_average, outdir+f'{year}.tif')



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
        fdir = rf'E:\Carbontracker\unify\\'

        outdir = rf'E:\Carbontracker\scales\\'
        Tools().mk_dir(outdir, force=True)

        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)

            outf=outdir+f
            print(outf)
            array[array < -999] = np.nan
            if np.isnan(np.nanmean(array)):
                continue

            array_new=array*12*60*60*24*365   #origianl unit is mol/m2/s, now is gC/m2/yr
            #1 mol=12g
            ToRaster().array2raster(outdir+f,originX,originY,pixelWidth,pixelHeight,array_new)



    pass
    def scale_LAI(self):
        fdir = rf'E:\Project3\Data\LAI4g\resample\\'
        outdir=rf'E:\Project3\Data\LAI4g\scale\\'
        Tools().mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)

            array=array*0.01
            array[array >7] = np.nan
            # array=array/10000


            array[array < -999] = np.nan

            outf=outdir+f
            ToRaster().array2raster(outf,originX,originY,pixelWidth,pixelHeight,array)







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
        # self.foo()
        self.plt_hist()

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
    def plt_hist(self):
        f=r'D:\Project3\Result\anomaly\OBS_extend\\Tempmean.npy'
        dic=np.load(f,allow_pickle=True).item()
        time_series_list=[]
        for pix in dic:
            vals=dic[pix]
            time_series_list.append(vals)
        time_series_list=np.array(time_series_list)
        plt.hist(time_series_list.flatten(),bins=100)
        plt.show()





    pass
class maximum_trend():
    def __init__(self):
        pass



    def run(self):
        # self.calculate_relative_change_monthly()
        # self.calculate_relative_change_monthly_trend()
        # self.convert_monthly()
        # self.gereate_individual_month()
        self.cal_trend_for_each_month()
        # self.pieplot()

    def calculate_relative_change_monthly(self):

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)

        fdir=result_root+rf'extract_GS_return_monthly_data\OBS_LAI\\'

        outdir = result_root + rf'extract_GS_return_monthly_data\OBS_LAI_relative_change\\'
        T.mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir),desc='relative_change'):

            outf=outdir+f
            if isfile(outf):
                continue


            dic_monthly_data = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            result_dic = {}
            for pix in dic_monthly_data:
                relative_change_list= []
                time_series = dic_monthly_data[pix]

                # plt.imshow(time_series)
                # plt.show()
                dryland = dic_dryland_mask[pix]
                if np.isnan(dryland):
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue

                if len(time_series) == 0:
                    pass
                if np.isnan(np.nanmean(time_series)):
                    continue
                time_series = np.array(time_series).T
                # print(time_series)

                for idx in range(len(time_series)):
                    time_series_month = time_series[idx]

                    if len(time_series_month) == 0:
                        continue
                    ###if all nan, fill data with nan and continue
                    if np.isnan(np.nanmean(time_series_month)):

                        time_series_month = np.nan * np.ones(len(time_series_month))
                        relative_change_list.append(time_series_month)
                        continue

                    average = np.nanmean(time_series_month)
                    relative_change = (time_series_month - average) / average * 100
                    relative_change_list.append(relative_change)

                # for i in range(len(relative_change_list)):
                #     print(relative_change_list[i])
                #     print(len(relative_change_list[i]))
                #     print('-----------------')

                relative_change_list=np.array(relative_change_list).T
                # plt.imshow(relative_change_list)
                # plt.show()
                result_dic[pix] = relative_change_list
            np.save(outdir + f, result_dic)



    def calculate_relative_change_monthly_trend(self):  ####### LAI max trend
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)

        f_monthly_data = result_root + rf'extract_GS_return_monthly_data\OBS_LAI\\LAI4g.npy'

        outdir = result_root + rf'extract_GS_return_monthly_data\OBS_LAI_trend\\'
        T.mk_dir(outdir, force=True)

        dic_monthly_data = np.load(f_monthly_data, allow_pickle=True, encoding='latin1').item()

        dic_max_trend= {}

        max_month_dict = {}
        for pix in tqdm(dic_monthly_data, desc='trend'):
            time_series = dic_monthly_data[pix]
            dryland=dic_dryland_mask[pix]
            if np.isnan(dryland):
                continue
            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue


            if len(time_series) == 0:
                pass
            if np.isnan(np.nanmean(time_series)):
                continue
            time_series = np.array(time_series).T


            # plt.imshow(time_series)
            # plt.show()
            trend_dict = {}

            for idx in range(len(time_series)):
                time_series_month = time_series[idx]

                if len(time_series_month) == 0:
                    continue
                if np.isnan(np.nanmean(time_series_month)):
                    continue
                average=np.nanmean(time_series_month)
                relative_change= (time_series_month-average)/average *100
                # print(relative_change)

                ##### trend
                a, b, r, p, std_err = stats.linregress(np.arange(len(time_series_month)), relative_change)
                if np.isnan(a):
                    continue
                trend_dict[idx] = abs(a)
            ###### calculate absolute maximum trend

            # max_month = np.nanmax(list(trend_dict.keys()))
            if len(trend_dict) == 0:
                continue
            max_month = T.get_max_key_from_dict(trend_dict)
            if max_month == None:
                print(trend_dict)


            actual_value_trend = trend_dict[max_month]
            # print(max_month)
            # print(max_month_index)
            # print(max_trend)
            # print(actual_value_trend)
            # exit()

            dic_max_trend[pix] = actual_value_trend
            max_month_dict[pix] = max_month
            # print(max_month)


        np.save(outdir + 'LAI4g_trend.npy', dic_max_trend)
        np.save(outdir + 'LAI4g_max_month.npy', max_month_dict)
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_max_trend, outdir + 'LAI4g_trend.tif')
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(max_month_dict, outdir + 'LAI4g_max_month.tif')
    def gereate_individual_month(self):
        fdir=result_root+'\extract_GS_return_monthly_data\OBS_LAI_relative_change\\'
        outdir=result_root+'\extract_GS_return_monthly_data\individual_month_relative_change\\'
        T.mk_dir(outdir,force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dic_monthly_data = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

                ##### extract each month and save each month seperately
            for month in range(12):
                result_dic = {}

                for pix in dic_monthly_data:
                    time_series = dic_monthly_data[pix]


                    if type(time_series) == np.float64:

                        continue
                    if len(time_series) == 0:
                        continue
                    if np.isnan(np.nanmean(time_series)):
                        continue
                    time_series_month_T = np.array(time_series).T
                    ##if length<=6, fill with nan and 12*39
                    if len(time_series_month_T) <= 6: ## fill with nan after 6
                        time_series_month_T = np.concatenate([time_series_month_T, np.nan * np.ones((12 - len(time_series_month_T), len(time_series_month_T[0])))], axis=0)

                    time_series_month = time_series_month_T[month]
                    result_dic[pix] = time_series_month
                file_month= f'{month:02d}'
                np.save(outdir + f'{f.split(".")[0]}_{file_month}.npy', result_dic)


    def cal_trend_for_each_month(self):
        fdir=result_root + rf'extract_GS_return_monthly_data\\individual_month_relative_change\\X\\'
        outdir = result_root + rf'extract_GS_return_monthly_data\\OBS_LAI_trend\\'
        T.mk_dir(outdir, force=True)
        x_list=['VPD','GLEAM_SMroot','Tempmean','GPCC','GLEAM_SMsurf','tmin','tmax']
        # x_list=['LAI4g']
        for x in x_list:
            for month in range(0,12):
                month=f'{month:02d}'
                f_monthly_data = fdir + f'\\{x}_{month}.npy'
                dic_monthly_data = np.load(f_monthly_data, allow_pickle=True, encoding='latin1').item()
                dic_trend = {}
                dic_pvalue = {}
                for pix in tqdm(dic_monthly_data, desc='trend'):
                    time_series = dic_monthly_data[pix]
                    if len(time_series) == 0:
                        continue
                    if np.isnan(np.nanmean(time_series)):
                        continue
                    a, b, r, p, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    if np.isnan(a):
                        continue
                    dic_trend[pix] = a
                    dic_pvalue[pix] = p
                np.save(outdir + f'{x}_{month}_trend.npy', dic_trend)
                np.save(outdir + f'{x}_{month}_pvalue.npy', dic_pvalue)
                DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_trend, outdir + f'{x}_{month}_trend.tif')
                DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_pvalue, outdir + f'{x}_{month}_pvalue.tif')




    def cal_trend_for_each_month_maximum(self):
        fdir=result_root + rf'extract_GS_return_monthly_data\OBS_LAI_trend\\'
        outdir = result_root + rf'extract_GS_return_monthly_data\\OBS_LAI_trend\\'
        T.mk_dir(outdir, force=True)
        x_list=['VPD','GLEAM_SMroot','Tempmean','GPCC','GLEAM_SMsurf','tmin','tmax']
        # x_list=['LAI4g']
        for x in x_list:

            xval_list=[]
            for month in range(0,12):
                month=f'{month:02d}'
                f_monthly_data = fdir + f'\\{x}_{month}_trend.npy'
                dic_monthly_data = np.load(f_monthly_data, allow_pickle=True, encoding='latin1').item()
                dic_trend = {}
                dic_pvalue = {}


        pass


        pass
    def convert_monthly(self):
        f_monthly_data = result_root + rf'extract_GS_return_monthly_data\\OBS_LAI_trend\\LAI4g_max_month.npy'
        dic_monthly_data = np.load(f_monthly_data, allow_pickle=True, encoding='latin1').item()
        dic_monthly_data_new = {}
        for pix in dic_monthly_data:
            r,c=pix
            idx=dic_monthly_data[pix]
            if r<240:
                month_real=idx+5

            elif r>480:
                month_real=idx+11
                if month_real>12:
                    month_real=month_real-12
            else: # 0-240
                month_real=idx+1
            dic_monthly_data_new[pix]=month_real
        np.save(result_root + rf'extract_GS_return_monthly_data\OBS_LAI_trend\max_month_real.npy', dic_monthly_data_new)
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_monthly_data_new, result_root + rf'extract_GS_return_monthly_data\OBS_LAI_trend\max_month_real.tif')

    def pieplot(self):
        f_monthly_data = result_root + rf'extract_GS_return_monthly_data\\OBS_LAI_trend\\max_month_real.npy'
        dic_monthly_data = np.load(f_monthly_data, allow_pickle=True, encoding='latin1').item()


        ##generate list of month
        month_list_northern = []
        month_list_southern = []
        month_list_tropical = []


        for pix in dic_monthly_data:
            r,c=pix
            if r <120:
                continue
            month_real=dic_monthly_data[pix]
            if r<240:
                month_list_northern.append(month_real)
            elif r>480:
                month_list_southern.append(month_real)
            else: # 0-240
                month_list_tropical.append(month_real)
                #### calculate the percentage of each month
        month_list_northern = np.array(month_list_northern)
        month_list_southern = np.array(month_list_southern)
        month_list_tropical = np.array(month_list_tropical)
        northern_hemisphere = {}
        southern_hemisphere = {}
        tropical = {}

        ##northern hemisphere only data from 5-10
        ## southern hemisphere only data from 11-4
        ## tropical hemisphere only data from 1-12
        for i in month_list_northern:
            northern_hemisphere[i] = month_list_northern.tolist().count(i)
        for i in month_list_southern:
            southern_hemisphere[i] = month_list_southern.tolist().count(i)
        for i in month_list_tropical:
            tropical[i] = month_list_tropical.tolist().count(i)
        ##plot pie for norther, southern, and tropical seperately

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].pie(northern_hemisphere.values(), labels=northern_hemisphere.keys(), autopct='%1.1f%%')
        ax[0].set_title('Northern Hemisphere')
        ax[1].pie(southern_hemisphere.values(), labels=southern_hemisphere.keys(), autopct='%1.1f%%')
        ax[1].set_title('Southern Hemisphere')
        ax[2].pie(tropical.values(), labels=tropical.keys(), autopct='%1.1f%%')
        ax[2].set_title('Tropical')
        plt.show()





        pass
class partial_correlation():
    def __init__(self):
        pass

        self.fdir_X = result_root + rf'\anomaly\OBS_extend\\'
        self.fdir_Y = result_root + rf'\anomaly\OBS_extend\\'
        self.fy_list = ['LAI4g']
        self.fx_list=['VPD','tmax','GPCC']
        self.outdir = result_root + rf'\partial_correlation\anomaly\\'
        T.mk_dir(self.outdir, force=True)

        self.outpartial = result_root + rf'\partial_correlation\anomaly\\partial_corr.npy'
        self.outpartial_pvalue = result_root + rf'\partial_correlation\anomaly\\partial_corr_pvalue.npy'

    def run(self):
        # df=self.build_df(self.fdir_X,self.fdir_Y,self.fx_list,self.fy_list)

        # self.cal_partial_corr(df,self.fx_list)
        # self.cal_single_correlation()
        # self.cal_single_correlation_ly()
        self.plot_partial_correlation()


    def build_df(self,fdir_X,fdir_Y,fx_list,fy):
        df = pd.DataFrame()

        filey = fdir_Y + fy[0] + '.npy'
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
            yvals=yvals[0:38]
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
            filex = fdir_X + xvar + '.npy'
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
                xvals = xvals[0:38]
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

        # exit()

        return df

    def cal_partial_corr(self,df,x_var_list, ):


        outf_corr = self.outpartial
        outf_pvalue = self.outpartial_pvalue

        partial_correlation_dic= {}
        partial_p_value_dic = {}
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

            if len(df_new) <= 3:
                continue
            partial_correlation = {}
            partial_correlation_p_value = {}
            for x in x_var_list_valid_new:
                x_var_list_valid_new_cov = copy.copy(x_var_list_valid_new)
                x_var_list_valid_new_cov.remove(x)
                r, p = self.partial_corr(df_new, x, 'y', x_var_list_valid_new_cov)
                partial_correlation[x] = r
                partial_correlation_p_value[x] = p

            partial_correlation_dic[pix] = partial_correlation
            partial_p_value_dic[pix] = partial_correlation_p_value
        T.save_npy(partial_correlation_dic, outf_corr)
        T.save_npy(partial_p_value_dic, outf_pvalue)





            # print(df_new)


    def cal_single_correlation(self):
        f_x= result_root + rf'\anomaly\OBS_extend\\wet_frequency_90th.npy'
        f_y = result_root + rf'\anomaly\OBS_extend\\LAI4g.npy'
        dic_x = T.load_npy(f_x)
        dic_y = T.load_npy(f_y)

        spatial_r_dic = {}

        for pix in tqdm(dic_x):
            if not pix in dic_y:
                continue
            x_val = dic_x[pix]

            y_val = dic_y[pix]

            x_val = T.interp_nan(x_val)
            y_val = T.interp_nan(y_val)
            if x_val[0] == None:
                continue
            y_val = y_val[0:38]

            if len(y_val) == 0:
                continue

            if np.isnan(np.nanmean(x_val)):
                continue
            if len(x_val) != len(y_val):
                continue
            ## remove nan

       ####
            r, p = stats.pearsonr(x_val, y_val)
            # print(r)
            # print(p)
            spatial_r_dic[pix] = r
        arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_r_dic)
        plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='jet',interpolation='nearest')
        plt.colorbar()
        plt.show()


    def cal_single_correlation_ly(self):
        f_x= result_root + rf'\anomaly\OBS_extend\\CV_rainfall.npy'
        f_y = result_root + rf'\anomaly\OBS_extend\\LAI4g.npy'
        outdir = join(result_root, 'anomaly', 'cal_single_correlation_ly')
        T.mk_dir(outdir, force=True)
        dic_x = T.load_npy(f_x)
        dic_y = T.load_npy(f_y)

        spatial_r_dic_cv = {}
        spatial_r_dic_lai = {}
        correlation_dict = {}

        for pix in tqdm(dic_x):
            if not pix in dic_y:
                continue
            x_val = dic_x[pix]

            y_val = dic_y[pix]

            x_val = T.interp_nan(x_val)
            y_val = T.interp_nan(y_val)
            if x_val[0] == None:
                continue
            y_val = y_val[0:38]

            if len(y_val) == 0:
                continue

            if np.isnan(np.nanmean(x_val)):
                continue
            if len(x_val) != len(y_val):
                continue
            ## remove nan

       ####
            # r, p = stats.pearsonr(x_val, y_val)
            # print(r)
            r_lai,_ = stats.pearsonr(list(range(len(y_val))), y_val)
            r_cv,_ = stats.pearsonr(list(range(len(x_val))), x_val)
            r,p = stats.pearsonr(x_val, y_val)
            # print(p)
            # spatial_r_dic[pix] = r
            spatial_r_dic_cv[pix] = r_cv
            spatial_r_dic_lai[pix] = r_lai
            correlation_dict[pix] = r
        outf_cv = join(outdir, 'CV_trend.tif')
        outf_lai = join(outdir, 'LAI_trend.tif')
        outf_corr = join(outdir, 'correlation.tif')
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(spatial_r_dic_cv, outf_cv)
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(spatial_r_dic_lai, outf_lai)
        DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(correlation_dict, outf_corr)



    def plot_partial_correlation(self):

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)


        fdir = result_root + rf'partial_correlation\anomaly\\'
        f_partial = fdir + 'partial_corr.npy'
        f_pvalue = fdir + 'partial_corr_pvalue.npy'


        partial_correlation_dic = np.load(f_partial, allow_pickle=True, encoding='latin1').item()
        partial_correlation_p_value_dic = np.load(f_pvalue, allow_pickle=True, encoding='latin1').item()


        var_list = []
        for pix in partial_correlation_dic:

            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue

            vals = partial_correlation_dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in partial_correlation_dic:

                dic_i = partial_correlation_dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            arr = arr * array_mask
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, fdir + f'partial_corr_{var_i}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr, vmin=-1, vmax=1)

            plt.title(var_i)
            plt.colorbar()

        plt.show()












    def partial_corr(self, df, x, y, cov):
        df = pd.DataFrame(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        # print(df)
        df = df.dropna()
        # try:
        # print(x)
        # print(y)
        stats_result = pg.partial_corr(data=df, x=x, y=y, covar=cov, method='pearson').round(3)
        r = float(stats_result['r'])
        p = float(stats_result['p-val'])
        return r, p


    pass




























    def maximum_trend(self):
        fdir = result_root + rf'extract_GS_return_monthly_data\OBS_LAI\\'
        outdir = result_root + rf'extract_GS_return_monthly_data\OBS_LAI_maximum_trend\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
            dic_maximum_trend = {}
            for pix in tqdm(dic, desc=f):
                time_series = dic[pix]
                if len(time_series) == 0:
                    pass

class single_correlation():
    def __init__(self):
        pass
    def run(self):
        # self.gereate_individual_month()
        self.cal_single_correlation()
        # self.plot_single_correlation()


        pass
    def cal_single_correlation(self):
        fdir_Y = result_root + rf'growth_rate\\'
        fdir_X = result_root + rf'growth_rate\\'
        outdir = result_root + rf'growth_rate\\\single_correlation\\'
        T.mk_dir(outdir, force=True)

        for fx in os.listdir(fdir_X):
            print(fx)
            if not  'GPCC' in fx:
                continue


            if not fx.endswith('.npy'):
                continue
            for fy in os.listdir(fdir_Y):
                if not 'LAI4g' in fy:
                    continue

                if not fy.endswith('.npy'):
                    continue
                outf=outdir + f'{fx.split(".")[0]}_{fy.split(".")[0]}.npy'

                dic_monthly_data_X = np.load(fdir_X + fx, allow_pickle=True, encoding='latin1').item()
                dic_monthly_data_Y = np.load(fdir_Y + fy, allow_pickle=True, encoding='latin1').item()
                result_dic = {}
                result_dic_pvalue = {}
                for pix in tqdm(dic_monthly_data_X, desc=fy):
                    if not pix in dic_monthly_data_Y:
                        continue


                    time_series_X = dic_monthly_data_X[pix]

                    time_series_Y = dic_monthly_data_Y[pix]
                    # print(time_series_X)
                    # print(time_series_Y)

                    if type(time_series_Y) == np.float64:
                        continue
                    if type(time_series_X) == np.float64:
                        continue
                    if len(time_series_X) == 0:
                        continue

                    if len(time_series_Y) == 0:
                        continue
                    if np.isnan(np.nanmean(time_series_X)):
                        continue
                    if np.isnan(np.nanmean(time_series_Y)):
                        continue

                    r, p = T.nan_correlation(time_series_X, time_series_Y)
                    result_dic[pix] = r
                    result_dic_pvalue[pix] = p


                np.save(outf, result_dic)

                array = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic)
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(array, outf.replace('.npy', '.tif'))
                # plt.imshow(array, interpolation='nearest', cmap='jet')
                # plt.colorbar()
                # plt.title(f'{fx.split("_")[0]}_{fy.split("_")[0]}_{month_y}')
                # plt.show()
    def plot_single_correlation(self):
        fdir = result_root + rf'extract_GS_return_monthly_data\single_correlation\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
            spatial_dic = {}
            for pix in dic:
                spatial_dic[pix] = dic[pix]
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr, interpolation='nearest', cmap='jet')
            plt.colorbar()
            plt.title(f)
            plt.show()





class data_preprocess_for_random_forest():

    def __init__(self):
        pass
    def run(self):
        # self.events_extraction()
        self.calculate_event_frequency()
    def events_extraction(self): ### extract wet and dry events
        fdir = result_root + rf'\relative_change\OBS_LAI_extend\\'
        outdir = result_root + rf'\relative_change\events_extraction\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'VPD' in f:
                continue
            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
            dic_event = {}

            for pix in tqdm(dic, desc=f):
                time_series = dic[pix]
                if len(time_series) == 0:
                    continue
                if np.isnan(np.nanmean(time_series)):
                    continue
                event = self.extract_event(time_series)
                dic_event[pix] = event
                fname=f.split('.')[0]+'_level.npy'
            np.save(outdir + fname, dic_event)

    def extract_event_CRU(self,time_series):
        slight_wet = []
        moderate_wet = []
        significant_wet = []
        extreme_wet = []
        slight_dry = []
        moderate_dry = []
        significant_dry = []
        extreme_dry = []
        no_events = []


        for idx in range(len(time_series)):
            ### wet threshold events slight 10%, moderation 20%, significant 30%, extreme 40%
            ### dry threshold events slight 10%, moderation 20%, significant 30%, extreme 40%
            ## return index of events
            if time_series[idx] >=10 and time_series[idx] < 20:
                slight_wet.append(idx)
            elif time_series[idx] >= 20 and time_series[idx] < 30:
                moderate_wet.append(idx)
            elif time_series[idx] >= 30 and time_series[idx] < 40:
                significant_wet.append(idx)
            elif time_series[idx] >= 40:
                extreme_wet.append(idx)
            elif time_series[idx] < -10 and time_series[idx] > -20:
                slight_dry.append(idx)
            elif time_series[idx] <= -20 and time_series[idx] > -30:
                moderate_dry.append(idx)
            elif time_series[idx] <= -30 and time_series[idx] > -40:
                significant_dry.append(idx)
            elif time_series[idx] <= -40:
                extreme_dry.append(idx)
            else:
                no_events.append(idx)
        # return {'slight_wet':slight_wet,'moderate_wet':moderate_wet,'significant_wet':significant_wet,'extreme_wet':extreme_wet,
        #         'slight_dry':slight_dry,'moderate_dry':moderate_dry,'significant_dry':significant_dry,'extreme_dry':extreme_dry,'no_events':no_events}

        return {1: slight_wet, 2: moderate_wet, 3: significant_wet,
                4: extreme_wet,
                -1: slight_dry, -2: moderate_dry, -3: significant_dry,
                -4: extreme_dry, 0: no_events}

    def extract_event_VPD(self,time_series):
        slight_wet = []
        moderate_wet = []
        significant_wet = []
        extreme_wet = []
        slight_dry = []
        moderate_dry = []
        significant_dry = []
        extreme_dry = []
        no_events = []


        for idx in range(len(time_series)):
            ### wet threshold events slight 10%, moderation 20%, significant 30%, extreme 40%
            ### dry threshold events slight 10%, moderation 20%, significant 30%, extreme 40%
            ## return index of events
            if time_series[idx] >=5 and time_series[idx] < 10:
                slight_wet.append(idx)
            elif time_series[idx] >= 10 and time_series[idx] < 15:
                moderate_wet.append(idx)
            elif time_series[idx] >= 15 and time_series[idx] < 20:
                significant_wet.append(idx)

            elif time_series[idx] < -5 and time_series[idx] > -10:
                slight_dry.append(idx)
            elif time_series[idx] <= -10 and time_series[idx] > -15:
                moderate_dry.append(idx)
            elif time_series[idx] <= -15 and time_series[idx] > -20:
                significant_dry.append(idx)

            else:
                no_events.append(idx)
        # return {'slight_wet':slight_wet,'moderate_wet':moderate_wet,'significant_wet':significant_wet,'extreme_wet':extreme_wet,
        #         'slight_dry':slight_dry,'moderate_dry':moderate_dry,'significant_dry':significant_dry,'extreme_dry':extreme_dry,'no_events':no_events}

        return {1: slight_wet, 2: moderate_wet, 3: significant_wet,
                4: extreme_wet,
                -1: slight_dry, -2: moderate_dry, -3: significant_dry,
                -4: extreme_dry, 0: no_events}

    def calculate_event_frequency(self):
        fdir = result_root + rf'\relative_change\events_extraction\\'
        outdir = result_root + rf'\relative_change\events_extraction_frequency\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if  'level' in f:
                continue
            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
            dic_event = {}
            for pix in tqdm(dic, desc=f):
                event = dic[pix]
                event_frequency = {}
                for key in event:
                    event_frequency[key] = len(event[key])/len(dic[pix])
                dic_event[pix] = event_frequency
            np.save(outdir + f, dic_event)




class statistic_analysis():
    def __init__(self):
        pass
    def run(self):

        # self.detrend()  ##original
        # self.detrend_zscore_monthly()
        # self.relative_change()
        # self.normalised_variables()
        # self.calculate_CV()
        # self.zscore()
        # self.detrend()
        # self.LAI_baseline()

        # self.anomaly_GS()
        # self.growth_rate_GS()
        # self.anomaly_GS_ensemble()
        # self.zscore_GS()

        self.trend_analysis()
        # self.trend_analysis_for_event()
        # self.trend_differences()
        # self.trend_average_TRENDY()
        # self.trend_analysis_landcover_composition()
        # self.LUCC_LAI_correlation()


        # self.scerios_analysis() ## this method tried to calculate different scenarios
        # self.calculate_annual_growth_rate()
        # self.growth_rate_relative_change()


    def detrend(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir=result_root + rf'growth_rate\extend_nan\\'
        outdir=result_root + rf'growth_rate\detrend\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            print(f)




            outf=outdir+f.split('.')[0]+'.npy'
            # if isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')
            dic = dict(np.load( fdir+f, allow_pickle=True, ).item())

            detrend_zscore_dic={}

            for pix in tqdm(dic):
                dryland_values=dic_dryland_mask[pix]
                if np.isnan(dryland_values):
                    continue
                r, c= pix


                # print(len(dic[pix]))
                time_series = dic[pix]
                # print(len(time_series))
                # print(time_series)

                time_series=np.array(time_series)
                # plt.plot(time_series)
                # plt.show()

                time_series[time_series < -999] = np.nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                if np.std(time_series) == 0:
                    continue
                ##### if count of nan is more than 50%, then skip
                if np.sum(np.isnan(time_series))/len(time_series) > 0.5:
                    continue


                # mean = np.nanmean(time_series)
                # std=np.nanstd(time_series)
                # if std == 0:
                #     continue
                # delta_time_series = (time_series - mean) / std
                # if np.isnan(time_series).any():
                #     continue
                if r<480:
                    # print(time_series)
                    ### interpolate
                    time_series=T.interp_nan(time_series)
                    # print(np.nanmean(time_series))
                    # plt.plot(time_series)



                    detrend_delta_time_series = signal.detrend(time_series)+np.nanmean(time_series)
                    # plt.plot(detrend_delta_time_series)
                    # plt.show()


                    detrend_zscore_dic[pix] = detrend_delta_time_series
                else:
                    time_series=time_series[0:38]
                    print(time_series)


                    if np.isnan(time_series).any():
                        continue
                    # print(time_series)
                    detrend_delta_time_series = signal.detrend(time_series)+np.nanmean(time_series)
                    ###add nan to the end if length is less than time_series
                    if len(detrend_delta_time_series) < 38:

                        detrend_delta_time_series=np.append(detrend_delta_time_series, [np.nan]*(38-len(detrend_delta_time_series)))

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

    def relative_change(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root+'extract_GS\OBS_LAI_extend\\\\'
        outdir=result_root + rf'relative_change\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'carbontracker' in f:
                continue

            outf=outdir+f.split('.')[0]+'.npy'
            if isfile(outf):
                continue
            print(outf)


            dic=T.load_npy(fdir+f)

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

                time_series=time_series
                mean=np.nanmean(time_series)
                relative_change=(time_series-mean)/abs(mean) *100

                zscore_dic[pix] = relative_change
                ## plot
                # plt.plot(time_series)
                # plt.show()
                # plt.plot(relative_change)
                # plt.legend(['original','relative_change'])
                plt.show()

                ## save
            np.save(outf, zscore_dic)

    def normalised_variables(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root + 'extract_GS\OBS_LAI_extend\\'
        outdir = result_root + rf'normalised\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            outf = outdir + f.split('.')[0]
            print(outf)

            dic = dict(np.load(fdir + f, allow_pickle=True, ).item())

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

                mean = np.nanmean(time_series)

                ### normalised

                ##(x - X_min) / (X_max - X_min)

                normalised=(time_series-np.nanmin(time_series))/(np.nanmax(time_series)-np.nanmin(time_series))
                zscore_dic[pix] = normalised
                ## plot
                # plt.plot(time_series)
                # plt.plot(relative_change)
                # plt.legend(['original','relative_change'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)


        pass
    def calculate_CV(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root + 'extract_GS\OBS_LAI_extend\\'
        outdir=result_root + rf'CV\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not 'CRU' in f:
                continue

            outf = outdir + f.split('.')[0]
            print(outf)

            dic = dict(np.load(fdir + f, allow_pickle=True, ).item())

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

                mean = np.nanmean(time_series)

                CV=np.nanstd(time_series)/mean
                zscore_dic[pix] = CV
                ## plot
                # plt.plot(time_series)
                # plt.plot(relative_change)
                # plt.legend(['original','relative_change'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)
            DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(zscore_dic, outf+'.tif')



        pass
    def zscore(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)


        fdir_all=result_root + 'extract_GS\OBS_LAI_extend\\'


        for f in os.listdir(fdir_all):
            if not 'GLEAM_SMroot' in f:
                continue


            outdir = result_root + rf'zscore\\'
            # if os.path.isdir(outdir):
            #     continue
            Tools().mk_dir(outdir, force=True)


            outf = outdir + f.split('.')[0]
            print(outf)

            dic = dict(np.load(fdir_all + f, allow_pickle=True, ).item())

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

        fdir = result_root + '\split_original\\'

        outdir = result_root + f'\split_anomaly\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)
            # if os.path.isfile(outf):
            #     continue
            # dic=T.load_npy_dir(fdir+f+'\\')

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

        fdir = result_root + 'extract_GS\\OBS_LAI_extend\\'
        outdir = result_root + f'zscore\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'GLEAM_SMroot' in f:
                continue


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


    def trend_analysis_for_event(self):  ## trend anaylsis for event
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        # crop_mask[crop_mask == 16] = 0
        # crop_mask[crop_mask == 17] = 0
        # crop_mask[crop_mask == 18] = 0

        fdir = rf'E:\Data\ERA5\ERA5_daily\dict\rainfall_frequency\\'
        outdir = result_root + rf'trend_analysis\\event\\'
        Tools().mk_dir(outdir, force=True)
        outf=outdir+'wet_frequency_90th'
        # for f in os.listdir(fdir):
        #     # if f.split('.')[0] not in ['CO2','CRU','VPD','tmax','noy','nhx','GLEAM_SMroot','GPCC']:
        #     #     continue
        #
        #     outf=outdir+f.split('.')[0]
        #     if os.path.isfile(outf+'_trend.tif'):
        #         continue
        #     print(outf)
        #
        #     if not f.endswith('.npy'):
        #         continue
        #     dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
        dic=T.load_npy_dir(fdir)
        trend_dic = {}
        p_value_dic = {}
        for pix in tqdm(dic):
            landcover_value=crop_mask[pix]
            if landcover_value==16 or landcover_value==17 or landcover_value==18:
                continue
            dict_i = dic[pix]
            if len(dict_i) == 0:
                continue
            # pprint(dict_i)
            # exit()
            value_list = []
            for year in dic[pix]:

                value = dic[pix][year]['frequency_wet']
                value_list.append(value)


                # time_series = dic[pix]['average_dry_spell']

            time_series = np.array(value_list)



                # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
            slope,b,r,p_value=T.nan_line_fit(np.arange(len(time_series)), time_series)
            trend_dic[pix] = slope
            p_value_dic[pix] = p_value

        arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
        arr_trend_dryland = arr_trend * array_mask
        p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
        p_value_arr_dryland = p_value_arr * array_mask


        plt.imshow(arr_trend_dryland, cmap='jet', vmin=-0.01, vmax=0.01)

        plt.colorbar()
        plt.title('frequency_wet')
        plt.show()

        DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend_dryland, outf + '_trend.tif')
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr_dryland, outf + '_p_value.tif')

        np.save(outf + '_trend', arr_trend_dryland)
        np.save(outf + '_p_value', p_value_arr_dryland)



        pass


    def trend_analysis(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        fdir = result_root+rf'growth_rate\growth_rate_raw\\'
        outdir = result_root + rf'growth_rate\\growth_rate_raw\\trend_analysis\\'
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



    def trend_differences(self):
        fdir=result_root + rf'trend_analysis\split_anomaly\\'
        outdir=result_root + rf'trend_analysis\split_anomaly\\'
        Tools().mk_dir(outdir, force=True)

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        # crop_mask[crop_mask == 16] = 0
        # crop_mask[crop_mask == 17] = 0
        # crop_mask[crop_mask == 18] = 0




        for f in os.listdir(fdir):

            if not f.endswith('.tif'):
                continue
            if not '1982_2001' in f:
                continue

            f_trend_p1=fdir+f

            f_trend_p2=fdir+f.replace('1982_2001','2002_2020')

            arr_trend_p1, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_trend_p1)
            arr_trend_p2, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_trend_p2)
            arr_trend_differences=(arr_trend_p2-arr_trend_p1)
            ###
            arr_trend_differences[arr_trend_differences==0]=np.nan
            arr_trend_differences*=array_mask

            dic_trend_differences=DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(arr_trend_differences)
            new_spatial_dic={}
            for pix in dic_trend_differences:
                if crop_mask[pix]==16 or crop_mask[pix]==17 or crop_mask[pix]==18:
                    continue
                new_spatial_dic[pix]=dic_trend_differences[pix]
            arr_trend_differences=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(new_spatial_dic)

            # outfile name joint
            fname=f.split('_')[0]+'_'+f.split('_')[1]+'_differences.tif'
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend_differences, outdir+fname)
            np.save(outdir+fname.replace('.tif',''),arr_trend_differences)



    def trend_average_TRENDY(self):
        fdir = result_root + rf'\\trend_analysis\relative_change\TRENDY_LAI\S2\\'
        outdir = result_root + rf'trend_analysis\\relative_change\\TRENDY_LAI\\S2\\'
        Tools().mk_dir(outdir, force=True)
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        crop_mask[crop_mask == 16] = 0
        crop_mask[crop_mask == 17] = 0
        crop_mask[crop_mask == 18] = 0

        array_list=[]
        for f in os.listdir(fdir):
            if not 'tif' in f:
                continue
            if 'p_value' in f:
                continue
            if 'xml' in f:
                continue

            print(f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
            array=array*array_mask
            array=array*crop_mask
            array_list.append(array)


        array_list=np.array(array_list)
        average=np.nanmean(array_list,axis=0)
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(average, outdir+'average.tif')
        np.save(outdir+'average',average)








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



    def bivariate_analysis(self):
        growthrate_trend=result_root+'CV\\CV.npy'

        trend_trend=result_root+rf'trend_analysis\relative_change\OBS_extend\\LAI4g_trend.npy'
        trend_pvalue=result_root+rf'trend_analysis\relative_change\OBS_extend\\LAI4g_p_value.npy'

        growthrate_trend_dic=np.load(growthrate_trend,allow_pickle=True).item()


        trend_dic=np.load(trend_trend,allow_pickle=True).item()
        trend_pvalue_dic=np.load(trend_pvalue,allow_pickle=True).item()

        growthrate_list=[]
        trend_list=[]
        result_dic={}
        for pix in growthrate_trend_dic:
            trend_value=trend_dic[pix]
            trend_pvalue=trend_pvalue_dic[pix]
            growthrate_value=growthrate_trend_dic[pix]







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


class classification():
    def __init__(self):
        pass
    def run(self):
        # self.classify()
        # self.cal_percentage()
        # self.build_df_LAI_trend_based_classification()
        # self.calculate_LAI_trend_based_classification()
        # self.heatmap()
          ##
        # self.classfy_greening_browning()
        # self.classfy_wetting_drying()
        # self.classfy_drought_wet_years() #plot kernal plot for drought and wet years
        # self.classfy_drought_wet_years_whole_period_LAI()
        # self.classfy_drought_wet_years_whole_period()
        self.classfy_drought_wet_years_seperate_period()
        # self.classfy_drought_wet_years_seperate_period_map()
        # self.plot_wet_year_dry_year_bar()
        # self.negative_postive_response()
        # self.negative_postive_response_curve()
        # self.negative_postive_response_curve_transition()
        # self.transition_bin()
        # self.plot_transition()
        # self.plot_transition_matrix()

        pass
    def classify(self):

        outdir=result_root + rf'classification\\'
        Tools().mk_dir(outdir, force=True)
        f_SM_trend_p2 = result_root + rf'trend_analysis\\split_relative_change\\OBS_LAI_extend\\GLEAM_SMroot_2002_2020_trend.npy'
        f_SM_trend_p1 = result_root + rf'trend_analysis\\split_relative_change\\OBS_LAI_extend\\GLEAM_SMroot_1982_2001_trend.npy'
        f_SM_pvalue_p2 = result_root + rf'trend_analysis\\split_relative_change\\OBS_LAI_extend\\GLEAM_SMroot_2002_2020_p_value.npy'
        f_SM_pvalue_p1 = result_root + rf'trend_analysis\\split_relative_change\\OBS_LAI_extend\\GLEAM_SMroot_1982_2001_p_value.npy'

        SM_trend_p2 = np.load(f_SM_trend_p2, allow_pickle=True)
        SM_trend_p1 = np.load(f_SM_trend_p1, allow_pickle=True)
        SM_pvalue_p2 = np.load(f_SM_pvalue_p2, allow_pickle=True)
        SM_pvalue_p1 = np.load(f_SM_pvalue_p1, allow_pickle=True)
        ### mask -9999
        SM_trend_p2[SM_trend_p2 < -999] = np.nan
        SM_trend_p1[SM_trend_p1 < -999] = np.nan
        SM_pvalue_p2[SM_pvalue_p2 < -999] = np.nan
        SM_pvalue_p1[SM_pvalue_p1 < -999] = np.nan

        SM_trend_p2=DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(SM_trend_p2)
        SM_trend_p1=DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(SM_trend_p1)
        SM_pvalue_p2=DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(SM_pvalue_p2)
        SM_pvalue_p1=DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(SM_pvalue_p1)

        result={}
        tiff_dic={}

        for pix in SM_trend_p2:
            if pix not in SM_trend_p1:
                continue
            if pix not in SM_pvalue_p2:
                continue
            if pix not in SM_pvalue_p1:
                continue
            if pix not in SM_trend_p2:
                continue
            trend_p1=SM_trend_p1[pix]
            trend_p2=SM_trend_p2[pix]
            pvalue_p1=SM_pvalue_p1[pix]
            pvalue_p2=SM_pvalue_p2[pix]
            ###### classification
            ## first period significant trend and second period significant trend
            ## first period significant trend and second period not significant trend
            ## first period not significant trend and second period significant trend
            ## first period not significant trend and second period not significant trend
            ## non-value = np.nan

            ## also increasing and decreasing trend

            if np.isnan(trend_p1):
                continue
            if np.isnan(trend_p2):
                continue

            if pvalue_p1<0.05 and pvalue_p2<0.05:
                if trend_p1>0 and trend_p2>0:
                    result[pix]='sig_wetting_sig_wetting'
                    tiff_dic[pix]=0
                elif trend_p1<0 and trend_p2<0:
                    result[pix]='sig_drying_sig_drying'
                    tiff_dic[pix]=1
                elif trend_p1>0 and trend_p2<0:
                    result[pix]='sig_wetting_sig_drying'
                    tiff_dic[pix]=2
                elif trend_p1<0 and trend_p2>0:
                    result[pix]='sig_drying_sig_wetting'
                    tiff_dic[pix]=3
            elif pvalue_p1<0.05 and pvalue_p2>0.05:
                if trend_p1>0:
                    result[pix]='sig_wetting_no_sig'
                    tiff_dic[pix]=4
                elif trend_p1<0:
                    result[pix]='sig_drying_no_sig'
                    tiff_dic[pix]=5
            elif pvalue_p1>0.05 and pvalue_p2<0.05:
                if trend_p2>0:
                    result[pix]='no_sig_sig_wetting'
                    tiff_dic[pix]=6
                elif trend_p2<0:
                    result[pix]='no_sig_sig_drying'
                    tiff_dic[pix]=7
            elif pvalue_p1>0.05 and pvalue_p2>0.05:
                result[pix]='no_sig_no_sig'
                tiff_dic[pix]=8
            else:
                result[pix]='other'
                tiff_dic[pix]=9
        array=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(tiff_dic)
        plt.imshow(array)
        plt.colorbar()
        plt.close()
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(array,outdir+'classification.tif')

        np.save(outdir+'classification',result)


    ### #### convert to spatial array


    def cal_percentage(self): ## calculate the percentage of each classification
        fdir=result_root + rf'classification\\'
        result_dic = np.load(fdir+'classification.npy', allow_pickle=True).item()
        sig_wetting_sig_wetting=0
        sig_drying_sig_drying=0
        sig_wetting_sig_drying=0
        sig_drying_sig_wetting=0
        sig_wetting_no_sig=0
        sig_drying_no_sig=0
        no_sig_sig_wetting=0
        no_sig_sig_drying=0
        no_sig_no_sig=0
        other=0
        for pix in result_dic:
            r, c = pix
            if r<120:
                continue
            if result_dic[pix]=='sig_wetting_sig_wetting':
                sig_wetting_sig_wetting+=1
            elif result_dic[pix]=='sig_drying_sig_drying':
                sig_drying_sig_drying+=1
            elif result_dic[pix]=='sig_wetting_sig_drying':
                sig_wetting_sig_drying+=1
            elif result_dic[pix]=='sig_drying_sig_wetting':
                sig_drying_sig_wetting+=1
            elif result_dic[pix]=='sig_wetting_no_sig':
                sig_wetting_no_sig+=1
            elif result_dic[pix]=='sig_drying_no_sig':
                sig_drying_no_sig+=1
            elif result_dic[pix]=='no_sig_sig_wetting':
                no_sig_sig_wetting+=1
            elif result_dic[pix]=='no_sig_sig_drying':
                no_sig_sig_drying+=1
            elif result_dic[pix]=='no_sig_no_sig':
                no_sig_no_sig+=1
            elif result_dic[pix]=='other':
                other+=1

        total=sig_wetting_sig_wetting+sig_drying_sig_drying+sig_wetting_sig_drying+sig_drying_sig_wetting+sig_wetting_no_sig+sig_drying_no_sig+no_sig_sig_wetting+no_sig_sig_drying+no_sig_no_sig+other
        percentage_dic={}
        percentage_dic['sig_wetting_sig_wetting']=sig_wetting_sig_wetting/total
        percentage_dic['sig_drying_sig_drying']=sig_drying_sig_drying/total
        percentage_dic['sig_wetting_sig_drying']=sig_wetting_sig_drying/total
        percentage_dic['sig_drying_sig_wetting']=sig_drying_sig_wetting/total
        percentage_dic['sig_wetting_no_sig']=sig_wetting_no_sig/total
        percentage_dic['sig_drying_no_sig']=sig_drying_no_sig/total
        percentage_dic['no_sig_sig_wetting']=no_sig_sig_wetting/total
        percentage_dic['no_sig_sig_drying']=no_sig_sig_drying/total
        percentage_dic['no_sig_no_sig']=no_sig_no_sig/total
        percentage_dic['other']=other/total
        print(percentage_dic)
        ## plot pie chart
        # figure, ax = plt.subplots()
        labels = 'sig_wetting_sig_wetting', 'sig_drying_sig_drying', 'sig_wetting_sig_drying', 'sig_drying_sig_wetting', 'sig_wetting_no_sig', 'sig_drying_no_sig', 'no_sig_sig_wetting', 'no_sig_sig_drying', 'no_sig_no_sig', 'other'
        sizes = [sig_wetting_sig_wetting, sig_drying_sig_drying, sig_wetting_sig_drying, sig_drying_sig_wetting, sig_wetting_no_sig, sig_drying_no_sig, no_sig_sig_wetting, no_sig_sig_drying, no_sig_no_sig, other]
        explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        ## label distance


        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90, labeldistance=1.2)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()
        pass

    def build_df_LAI_trend_based_classification(self):
        fdir=result_root + rf'trend_analysis\\split_relative_change\\OBS_LAI_extend\\'
        outdir=result_root + rf'classification\\'
        f_classify=result_root + rf'classification\\classification.npy'
        f_LAI_p1=fdir+'LAI4g_1982_2001_trend.npy'
        f_LAI_p2=fdir+'LAI4g_2002_2020_trend.npy'
        f_LAI_p1_pvalue=fdir+'LAI4g_1982_2001_p_value.npy'
        f_LAI_p2_pvalue=fdir+'LAI4g_2002_2020_p_value.npy'

        Tools().mk_dir(outdir, force=True)
        classify_dic=np.load(f_classify,allow_pickle=True).item()
        LAI_p1=np.load(f_LAI_p1,allow_pickle=True)
        LAI_p1[LAI_p1<-999]=np.nan

        LAI_p2=np.load(f_LAI_p2,allow_pickle=True)
        LAI_p2[LAI_p2<-999]=np.nan
        LAI_p1_pvalue=np.load(f_LAI_p1_pvalue,allow_pickle=True)
        LAI_p1_pvalue[LAI_p1_pvalue<-999]=np.nan

        LAI_p2_pvalue=np.load(f_LAI_p2_pvalue,allow_pickle=True)
        LAI_p2_pvalue[LAI_p2_pvalue<-999]=np.nan
        LAI_p1_dic=DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(LAI_p1)
        LAI_p2_dic=DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(LAI_p2)
        LAI_p1_pvalue_dic=DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(LAI_p1_pvalue)
        LAI_p2_pvalue_dic=DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(LAI_p2_pvalue)

        ######### df
        spatial_dic_all={
            'classfication_SM':classify_dic,
            'LAI_p1_trend':LAI_p1_dic,
            'LAI_p2_trend':LAI_p2_dic,
            'LAI_p1_p_value':LAI_p1_pvalue_dic,
            'LAI_p2_p_value':LAI_p2_pvalue_dic,

        }

        df=T.spatial_dics_to_df(spatial_dic_all)
        T.print_head_n(df)
        ## save df
        T.save_df(df,outdir+'classification.df')
        ## excel
        df.to_excel(outdir+'classification.xlsx')
        pass
        exit()

    def calculate_LAI_trend_based_classification(self):
        dff=result_root + rf'classification\\classification.df'
        df=T.load_df(dff)
        T.print_head_n(df)
        df=df[df['row']>120]
        ## get the unique classification
        classification_list=df['classfication_SM'].tolist()
        classification_list_unique=T.get_df_unique_val_list(df,'classfication_SM')
        print(classification_list_unique)
        result_dic={}
        for classification in classification_list_unique:
            df_temp=df[df['classfication_SM']==classification]
            print(classification)
            p1_trend=df_temp['LAI_p1_trend'].tolist()
            p2_trend=df_temp['LAI_p2_trend'].tolist()
            result_dic[classification]={}

            result_dic[classification]['p1_trend_mean']=np.nanmean(p1_trend)
            result_dic[classification]['p2_trend_mean']=np.nanmean(p2_trend)
            result_dic[classification]['p1_trend_std']=np.nanstd(p1_trend)
            result_dic[classification]['p2_trend_std']=np.nanstd(p2_trend)
        ## plot the trend
        df_result=T.dic_to_df(result_dic)
        T.print_head_n(df_result)
        ## plot bar based on the classification and two periods
        fig, ax = plt.subplots()
        barWidth = 0.3
        r1 = np.arange(len(classification_list_unique))
        r2 = [x + barWidth for x in r1]
        plt.bar(r1, df_result['p1_trend_mean'], color='b', width=barWidth, edgecolor='grey', label='p1_trend_mean')
        plt.bar(r2, df_result['p2_trend_mean'], color='r', width=barWidth, edgecolor='grey', label='p2_trend_mean')
        ## xticks is the classification
        x_ticks_list=[r + barWidth for r in range(len(classification_list_unique))]
        plt.xticks(x_ticks_list, classification_list_unique, fontweight='bold', rotation=40)
        ## Xlabel is the classification
        # plt.xlabel('classification', fontweight='bold')
        plt.ylabel('LAI_relative_change(%/year)', fontweight='bold')
        plt.tight_layout()

        plt.legend()
        plt.show()
        pass

    def heatmap(self):
        dff=result_root + rf'classification\\classification.df'
        df=T.load_df(dff)
        T.print_head_n(df)
        df=df[df['row']>120]
        df=df[df['classfication_SM']!='other']
        # df=df[df['classfication_SM']!='no_sig_no_sig']
        print(len(df))
        ## get the unique landcover_classfications
        # landcover_classfications_list=df['landcover_classfication'].tolist()
        # landcover_classfications_list_unique=T.get_df_unique_val_list(df,'landcover_classfication')
        # print(landcover_classfications_list_unique)
        # exit()
        # df=df[df['landcover_classfication']=='Shrub']
        # print(len(df))

        SM_p1_bin_list=np.linspace(-1.5,1.5,15)
        SM_p2_bin_list=np.linspace(-1.5,1.5,15)
        df_group1, bins_list_str1=T.df_bin(df,'GLEAM_SMroot_1982_2001_trend',SM_p1_bin_list)
        matrix=[]
        y_labels=[]
        for name1,df_group_i1 in df_group1:
            df_group2, bins_list_str2=T.df_bin(df_group_i1,'GLEAM_SMroot_2002_2020_trend',SM_p2_bin_list)
            name1_=name1[0].left

            matrix_i=[]
            x_labels = []

            for name2,df_group_i2 in df_group2:
                name2_=name2[0].left
                x_labels.append(name2_)
                # print(name1,name2)
                # print(len(df_group_i2))
                val1=np.nanmean(df_group_i2['LAI_p1_trend'])
                val2=np.nanmean(df_group_i2['LAI_p2_trend'])
                val=val2-val1
                matrix_i.append(val)

                # count=len(df_group_i2)
                # matrix_i.append(count)

                # matrix_i.append(val)
            matrix.append(matrix_i)
            y_labels.append(name1_)
        matrix=np.array(matrix)
        matrix=matrix[::-1,:]
        plt.imshow(matrix,cmap='RdBu',vmin=-1.5,vmax=1.5)
        # plt.imshow(matrix,cmap='RdBu')
        plt.xticks(np.arange(len(SM_p2_bin_list)-1), x_labels, rotation=45)
        plt.yticks(np.arange(len(SM_p1_bin_list)-1), y_labels[::-1])
        plt.xlabel('SM_p2_trend')
        plt.ylabel('SM_p1_trend')
        ## draw 1:1 line
        # plt.plot([20, 0], [0, 20], 'k-', lw=2)

        plt.colorbar()
        plt.show()

    def classfy_greening_browning(self):
        period=['1982_2001','2002_2020']
        for p in period:
            df=T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_{p}.df')

            ## sig greeing, browning
            for i, row in df.iterrows():
                if row[rf'LAI4g_{p}_trend'] > 0 and row[rf'LAI4g_{p}_p_value'] < 0.05:
                    df.at[i, rf'LAI4g_{p}_trend_class'] = 'sig_greening'
                elif row[rf'LAI4g_{p}_trend'] < 0 and row[rf'LAI4g_{p}_p_value'] < 0.05:
                    df.at[i, rf'LAI4g_{p}_trend_class'] = 'sig_browning'
                    ## non sig greening, browning
                elif row[rf'LAI4g_{p}_trend'] > 0 and row[rf'LAI4g_{p}_p_value'] > 0.05:
                    df.at[i, rf'LAI4g_{p}_trend_class'] = 'non_sig_greening'
                elif row[rf'LAI4g_{p}_trend'] < 0 and row[rf'LAI4g_{p}_p_value'] > 0.05:
                    df.at[i, rf'LAI4g_{p}_trend_class'] = 'non_sig_browning'
                else:
                    df.at[i, 'LAI4g_{p}_trend_class'] = 'other'

            T.save_df(df, result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_{p}.df')
            T.df_to_excel(df, result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_{p}')


    def classfy_wetting_drying(self):
        period=['1982_2001','2002_2020']
        for p in period:
            df=T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_{p}.df')

            ## sig greeing, browning
            for i, row in df.iterrows():
                if row[rf'GLEAM_SMroot_{p}_trend'] > 0 and row[rf'GLEAM_SMroot_{p}_p_value'] < 0.05:
                    df.at[i, rf'GLEAM_SMroot_{p}_trend_class'] = 'sig_wetting'
                elif row[rf'GLEAM_SMroot_{p}_trend'] < 0 and row[rf'GLEAM_SMroot_{p}_p_value'] < 0.05:
                    df.at[i, rf'GLEAM_SMroot_{p}_trend_class'] = 'sig_drying'
                    ## non sig greening, browning
                elif row[rf'GLEAM_SMroot_{p}_trend'] > 0 and row[rf'GLEAM_SMroot_{p}_p_value'] > 0.05:
                    df.at[i, rf'GLEAM_SMroot_{p}_trend_class'] = 'non_sig_wetting'
                elif row[rf'GLEAM_SMroot_{p}_trend'] < 0 and row[rf'GLEAM_SMroot_{p}_p_value'] > 0.05:
                    df.at[i, rf'GLEAM_SMroot_{p}_trend_class'] = 'non_sig_drying'
                else:
                    df.at[i, rf'GLEAM_SMroot_{p}_trend_class'] = 'other'


            T.save_df(df, result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_{p}.df')
            T.df_to_excel(df, result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_{p}')



    def classfy_drought_wet_years_seperate_period(self):
        period=['1982_2001','2002_2020']
        flag = 1
        CRU_thresholds = [20, 30, 40, 50, 60, 70, 80, ]
        SM_thresholds=[10,20,30,40]

        for p in period:

            df=T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_{p}.df')

            # df_Trans=Dataframe_per_value_transform(df,['LAI4g','GLEAM_SMroot','GLEAM_SMsurf'],1982,2020).df

            # SM_value=df['GLEAM_SMroot'].tolist()
            # plt.hist(SM_value,bins=100)
            # plt.show()
            outdir=result_root + rf'classification\\'
            Tools().mk_dir(outdir, force=True)
            ##df_clean
            df=df[df['row']>120]
            print(len(df))
            T.print_head_n(df)
            ## drop nan
            # df=df.dropna()
            # cols = df.columns.tolist()
            # for col in cols:
            #     print(col)

            # exit()
            ## get the unique landcover_classfications
            landcover_classfications_list=df['AI_classfication'].tolist()
            landcover_classfications_list_unique=T.get_df_unique_val_list(df,'AI_classfication')
            print(landcover_classfications_list_unique)

            df_clean=df[df['landcover_classfication']!='Cropland']
            # df_clean=df_clean[df_clean['row']>120]
            # df_clean=df_clean[df_clean['row']<240]
            # df_clean = df[df['landcover_classfication'] == 'Grass']
            # print(len(df_clean))



            Lai_col = rf'LAI4g'

            SM_col = rf'GPCC_{p}_detrend'


            for threshold in SM_thresholds:
                x_list = []
                y_list = []



                ax = plt.subplot(2, 4, flag)

                df_group = T.df_groupby(df_clean, 'pix')
                for pix in tqdm(df_group):
                    df_pix = df_group[pix]
                    # T.print_head_n(df_pix)
                    # exit()
                    df_pix_dry = df_pix[df_pix[SM_col] <-threshold]
                    df_pix_wet = df_pix[df_pix[SM_col] >=threshold]
                    ### screening normal year
                    # df_pix_normal=df_pix[df_pix[SM_col] >-20]
                    # df_pix_normal=df_pix_normal[df_pix_normal[SM_col] <20]

                    LAI_vals_dry = df_pix_dry[Lai_col].tolist()
                    LAI_vals_wet = df_pix_wet[Lai_col].tolist()
                    # LAI_vals_normal = df_pix_normal[Lai_col].tolist()
                    LAI_vals_dry_mean = np.nanmean(LAI_vals_dry)
                    LAI_vals_wet_mean = np.nanmean(LAI_vals_wet)
                    # LAI_vals_normal_mean = np.nanmean(LAI_vals_normal)

                    # LAI_vals_dry_mean = abs(LAI_vals_dry_mean)
                    # LAI_vals_wet_mean = abs(LAI_vals_wet_mean)
                    x_list.append(LAI_vals_dry_mean)
                    y_list.append(LAI_vals_wet_mean)
                    # c_list.append(LAI_vals_normal_mean)

                KDE_plot().plot_scatter(x_list, y_list, s=2, ax=ax, title=f'{p}_{threshold}')
                plt.grid(True)
                # plt.scatter(x_list, y_list, s=2)

                plt.axis('equal')
                # plt.xlim(x_lim)
                # plt.ylim(y_lim)

                ### plot -1:1 line

                plt.plot([-50, 20], [50,-20], 'k-', lw=2)


                flag+=1




        plt.show()

    def classfy_drought_wet_years_seperate_period_map(self):
        period=['1982_2001','2002_2020']
        flag = 1
        CRU_thresholds = [20, 30, 40, 50, 60, 70, 80, ]
        # SM_thresholds=[5,10,20,]

        period = ['1982_2001', '2002_2020']

        flag = 1
        for p in period:

            df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_{p}.df')
            df = df[df['lon'] > -125]
            df = df[df['lon'] < -105]
            df = df[df['lat'] > 0]
            df = df[df['lat'] < 45]
            df = df[df['row'] > 120]
            df = df[df['continent'] == 'North_America']

            ##df_clean
            df = df[df['row'] > 120]


            ##df_clean
            df=df[df['row']>120]
            print(len(df))
            T.print_head_n(df)
            ## drop nan
            # df=df.dropna()
            # cols = df.columns.tolist()
            # for col in cols:
            #     print(col)

            # exit()
            ## get the unique landcover_classfications
            landcover_classfications_list=df['AI_classfication'].tolist()
            landcover_classfications_list_unique=T.get_df_unique_val_list(df,'AI_classfication')
            print(landcover_classfications_list_unique)

            df_clean=df[df['landcover_classfication']!='Cropland']


            Lai_col = rf'LAI4g'

            SM_col = rf'GPCC_{p}_detrend'


            for threshold in CRU_thresholds:

                result_dic={}
                result_count_wet={}
                result_count_dry={}


                df_group = T.df_groupby(df_clean, 'pix')
                for pix in tqdm(df_group):
                    df_pix = df_group[pix]
                    # T.print_head_n(df_pix)
                    # exit()
                    df_pix_dry = df_pix[df_pix[SM_col] <-threshold]
                    df_pix_wet = df_pix[df_pix[SM_col] >=threshold]
                    ### screening normal year
                    # df_pix_normal=df_pix[df_pix[SM_col] >-20]
                    # df_pix_normal=df_pix_normal[df_pix_normal[SM_col] <20]

                    LAI_vals_dry = df_pix_dry[Lai_col].tolist()
                    LAI_vals_wet = df_pix_wet[Lai_col].tolist()
                    # LAI_vals_normal = df_pix_normal[Lai_col].tolist()
                    LAI_vals_dry_mean = np.nanmean(LAI_vals_dry)
                    LAI_vals_wet_mean = np.nanmean(LAI_vals_wet)
                    vals_differences=LAI_vals_wet_mean-LAI_vals_dry_mean
                    result_dic[pix]=vals_differences
                    count_wet=len(LAI_vals_dry)
                    count_dry=len(LAI_vals_wet)
                    result_count_wet[pix]=count_wet
                    result_count_dry[pix]=count_dry


                ### array
                # array=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic)
                array_count_wet=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_count_wet)
                array_count_dry=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_count_dry)
                array_difference=array_count_wet-array_count_dry

                # plt.imshow(array,cmap='RdBu',vmin=-10,vmax=10,interpolation='nearest')
                # plt.colorbar()
                # plt.title(f'{p}_{threshold}')
                # plt.show()
                flag+=1

                ##save array
                outf=outdir+f'{p}_{threshold}.npy'
                # np.save(outf, array)
                # DIC_and_TIF(pixelsize=0.25).arr_to_tif(array,outdir+f'{p}_{threshold}_CPCC.tif')

                DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_count_wet,outdir+f'{p}_{threshold}_count_wet.tif')
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_count_dry,outdir+f'{p}_{threshold}_count_dry.tif')
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_difference,outdir+f'{p}_{threshold}_difference.tif')








    def classfy_drought_wet_years_whole_period(self): ###

        # thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        thresholds=[50,60,70,80]

        df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly.df')

        # df_Trans=Dataframe_per_value_transform(df,['LAI4g','GLEAM_SMroot','GLEAM_SMsurf'],1982,2020).df

        # SM_value=df['CRU_detrend'].tolist()
        # plt.hist(SM_value,bins=100)
        # plt.show()
        outdir = result_root + rf'classification\\'
        Tools().mk_dir(outdir, force=True)
        ##df_clean
        df = df[df['row'] > 120]
        print(len(df))
        T.print_head_n(df)
        ## drop nan
        # df=df.dropna()
        # cols = df.columns.tolist()
        # for col in cols:
        #     print(col)

        # exit()
        ## get the unique landcover_classfications
        landcover_classfications_list = df['AI_classfication'].tolist()
        landcover_classfications_list_unique = T.get_df_unique_val_list(df, 'AI_classfication')
        print(landcover_classfications_list_unique)

        df_clean = df[df['landcover_classfication'] != 'Cropland']
        # df_clean=df_clean[df_clean['row']>120]
        # df_clean=df_clean[df_clean['row']<240]
        # df_clean = df[df['landcover_classfication'] == 'Grass']
        # print(len(df_clean))

        Lai_col = rf'LAI4g'

        SM_col = rf'CRU_detrend'
        flag = 1

        for threshold in thresholds:

            x_list = []
            y_list = []
            ax = plt.subplot(2, 5, flag)

            df_group = T.df_groupby(df_clean, 'pix')
            for pix in tqdm(df_group):
                df_pix = df_group[pix]
                # T.print_head_n(df_pix)
                # exit()
                df_pix_dry = df_pix[df_pix[SM_col] < -threshold]
                df_pix_wet = df_pix[df_pix[SM_col] >= threshold]
                ### screening normal year
                # df_pix_normal=df_pix[df_pix[SM_col] >-20]
                # df_pix_normal=df_pix_normal[df_pix_normal[SM_col] <20]

                LAI_vals_dry = df_pix_dry[Lai_col].tolist()
                LAI_vals_wet = df_pix_wet[Lai_col].tolist()
                # LAI_vals_normal = df_pix_normal[Lai_col].tolist()
                LAI_vals_dry_mean = np.nanmean(LAI_vals_dry)
                LAI_vals_wet_mean = np.nanmean(LAI_vals_wet)
                # LAI_vals_normal_mean = np.nanmean(LAI_vals_normal)

                LAI_vals_dry_mean = abs(LAI_vals_dry_mean)
                LAI_vals_wet_mean = abs(LAI_vals_wet_mean)
                x_list.append(LAI_vals_dry_mean)
                y_list.append(LAI_vals_wet_mean)
                # c_list.append(LAI_vals_normal_mean)

            KDE_plot().plot_scatter(x_list, y_list, s=2, ax=ax)
            plt.grid(True)
            plt.xlabel('LAI_dry_years')
            plt.ylabel('LAI_wet_years')
            plt.xlim(0, 50)
            plt.ylim(0, 50)

            plt.axis('equal')
            plt.title(f'{threshold}')

            # plt.plot([-50, 20], [50,-20], 'k-', lw=2)
            ### plot 1:1 line
            plt.plot([0, 50], [0, 50], 'k-', lw=2)
            flag = flag + 1

        plt.show()

    def classfy_drought_wet_years_whole_period_LAI(self): ###


        thresholds=[50,60,70,80]



        df=T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly.df')

        # df_Trans=Dataframe_per_value_transform(df,['LAI4g','GLEAM_SMroot','GLEAM_SMsurf'],1982,2020).df

        # SM_value=df['CRU_detrend'].tolist()
        # plt.hist(SM_value,bins=100)
        # plt.show()
        outdir=result_root + rf'classification\\'
        Tools().mk_dir(outdir, force=True)
        ##df_clean
        df=df[df['row']>120]
        print(len(df))
        T.print_head_n(df)
        ## drop nan
        # df=df.dropna()
        # cols = df.columns.tolist()
        # for col in cols:
        #     print(col)

        # exit()
        ## get the unique landcover_classfications
        landcover_classfications_list=df['AI_classfication'].tolist()
        landcover_classfications_list_unique=T.get_df_unique_val_list(df,'AI_classfication')
        print(landcover_classfications_list_unique)

        df_clean=df[df['landcover_classfication']!='Cropland']
        # df_clean=df_clean[df_clean['row']>120]
        # df_clean=df_clean[df_clean['row']<240]
        # df_clean = df[df['landcover_classfication'] == 'Grass']
        # print(len(df_clean))


        Lai_col = rf'LAI4g'

        SM_col = rf'CRU_detrend'
        flag=1


        for threshold in thresholds:

            x_list = []
            y_list = []
            ax = plt.subplot(2, 5, flag)

            df_group = T.df_groupby(df_clean, 'pix')
            for pix in tqdm(df_group):
                df_pix = df_group[pix]
                # T.print_head_n(df_pix)
                # exit()
                df_pix_dry = df_pix[df_pix[SM_col] <-threshold]
                df_pix_wet = df_pix[df_pix[SM_col] >=threshold]
                ### screening normal year
                # df_pix_normal=df_pix[df_pix[SM_col] >-20]
                # df_pix_normal=df_pix_normal[df_pix_normal[SM_col] <20]

                LAI_vals_dry = df_pix_dry[Lai_col].tolist()
                LAI_vals_wet = df_pix_wet[Lai_col].tolist()
                # LAI_vals_normal = df_pix_normal[Lai_col].tolist()
                LAI_vals_dry_mean = np.nanmean(LAI_vals_dry)
                LAI_vals_wet_mean = np.nanmean(LAI_vals_wet)
                # LAI_vals_normal_mean = np.nanmean(LAI_vals_normal)

                LAI_vals_dry_mean = abs(LAI_vals_dry_mean)
                LAI_vals_wet_mean = abs(LAI_vals_wet_mean)
                x_list.append(LAI_vals_dry_mean)
                y_list.append(LAI_vals_wet_mean)
                # c_list.append(LAI_vals_normal_mean)

            KDE_plot().plot_scatter(x_list, y_list, s=2, ax=ax)
            plt.grid(True)
            plt.xlabel('LAI_dry_years')
            plt.ylabel('LAI_wet_years')
            plt.xlim(0, 50)
            plt.ylim(0, 50)

            plt.axis('equal')
            plt.title(f'{threshold}')

            # plt.plot([-50, 20], [50,-20], 'k-', lw=2)
            ### plot 1:1 line
            plt.plot([0, 50], [0, 50], 'k-', lw=2)
            flag=flag+1



        plt.show()

    def plot_wet_year_dry_year_bar(self):
        thresholds_list=[20,30, 40, 50,60,70,80]
        period=['1982_2001','2002_2020']
        result_threshold_dic={}
        for threhold in thresholds_list:
            val_list=[]
            result_p_dic = {}
            for p in period:

                df=T.load_df(result_root + rf'\Dataframe\CRU_thresholds\\CRU_thresholds.df')
                df=df[df['row']>120]
                df = df[df['lon'] > -125]
                df = df[df['lon'] < -105]
                df = df[df['lat'] > 0]
                df = df[df['lat'] < 45]
                df = df[df['row'] > 120]
                df = df[df['continent'] == 'North_America']
                ## landcover  not crop
                df=df[df['landcover_classfication']!='Cropland']

                vals=df[f'{p}_{threhold}_difference'].tolist()

                val_list.append(vals)
                result_p_dic[p]=vals
            result_threshold_dic[threhold]=result_p_dic
       ## plot violin
        vals_list=[]
        positions_list=[]
        for threhold in thresholds_list:
            for p in period:
                vals=result_threshold_dic[threhold][p]
                vals=T.remove_np_nan(vals)
                plt.hist(vals,bins=100)
                plt.title(f'{p}_{threhold}')
                plt.show()
                vals_list.append(vals)
                positions_list.append(f'{p}_{threhold}')
        # plt.violinplot(vals, positions=[threhold], showmeans=False, showmedians=True)
        plt.boxplot(vals_list, labels=positions_list,)
        plt.show()




    def negative_postive_response(self):
        #### plot the negative and positive response map


        period = ['1982_2001', '2002_2020']



        for p in period:

            df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_{p}.df')

            ## western US
            df = df[df['lon'] > -125]
            df = df[df['lon'] < -105]
            df = df[df['lat'] > 0]
            df = df[df['lat'] < 45]
            df=df[df['row']>120]
            df=df[df['continent']=='North_America']


            outdir = result_root + rf'classification\\'
            Tools().mk_dir(outdir, force=True)
            ##df_clean

            print(len(df))
            T.print_head_n(df)
            ## drop nan
            # df = df.dropna()
            cols = df.columns.tolist()
            for col in cols:
                print(col)

            Lai_col = rf'LAI4g'

            SM_col = rf'GLEAM_SMroot_{p}_detrend'

            df_group = T.df_groupby(df, 'pix')
            response_dic={}
            wet_average_list_LAI=[]
            dry_average_list_LAI=[]
            for pix in tqdm(df_group):
                df_pix = df_group[pix]
                # T.print_head_n(df_pix)
                # exit()
                df_pix_dry = df_pix[df_pix[SM_col] < -20]
                df_pix_wet = df_pix[df_pix[SM_col] > 20]
                ### screening normal year
                # df_pix_normal=df_pix[df_pix[SM_col] >-20]
                # df_pix_normal=df_pix_normal[df_pix_normal[SM_col] <20]

                LAI_vals_dry = df_pix_dry[Lai_col].tolist()
                LAI_vals_wet = df_pix_wet[Lai_col].tolist()
                # LAI_vals_normal = df_pix_normal[Lai_col].tolist()
                LAI_vals_dry_mean = np.nanmean(LAI_vals_dry)
                LAI_vals_wet_mean = np.nanmean(LAI_vals_wet)
                wet_average_list_LAI.append(LAI_vals_wet_mean)
                dry_average_list_LAI.append(LAI_vals_dry_mean)


                # if LAI_vals_wet_mean> LAI_vals_dry_mean:
                #     response_dic[pix]=1  ##postive
                # elif LAI_vals_wet_mean<= LAI_vals_dry_mean:
                #     response_dic[pix]=2 # negative
                # else:
                #     print(LAI_vals_dry_mean,LAI_vals_wet_mean)
                #     response_dic[pix]=0  ## no response
            plt.plot(wet_average_list_LAI,dry_average_list_LAI,'.')
            plt.show()



            # array=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(response_dic)
            # plt.imshow(array)
            # plt.colorbar()
            # plt.title(p)
            # plt.show()
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(array,outdir+f'response_{p}.tif')
            # np.save(outdir+f'response_{p}.npy',response_dic)
            # exit()

    def negative_postive_response_curve(self): ####
        #### plot W.S. ,method

        period = ['1982_2001', '2002_2020']

        flag = 1
        for p in period:


            df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_{p}.df')
            df = df[df['lon'] > -125]
            df = df[df['lon'] < -105]
            df = df[df['lat'] > 0]
            df = df[df['lat'] < 45]
            df = df[df['row'] > 120]
            df = df[df['continent'] == 'North_America']

            ##df_clean
            df = df[df['row'] > 120]
            print(len(df))
            T.print_head_n(df)
            ## drop nan
            # df = df.dropna()
            # cols = df.columns.tolist()
            # for col in cols:
            #     print(col)

            for region in ['Arid','Semi-Arid', 'Sub-Humid']:
                plt.subplot(2, 3, flag)
                df_region = df[df['AI_classfication'] == region]

                SM_col = rf'GLEAM_SMroot_{p}_detrend'
                sm_value=df_region[SM_col].tolist()
                start=np.nanmin(sm_value)
                end=np.nanmax(sm_value)
                
                sm_bin= np.linspace(start,end,10)
                df_group, bins_list_str = T.df_bin(df_region, SM_col, sm_bin)

                average_LAI_list=[]
                x_list=[]
                for name, df_group_i in df_group:
                    SM_list = df_group_i[SM_col].tolist()
                    LAI_list = df_group_i['LAI4g'].tolist()

                    average= np.nanmean(LAI_list)
                    average_LAI_list.append(average)
                    x_list.append(name[0].left)
                plt.plot(x_list,average_LAI_list)

                # plt.show()
                flag+=1
                Lai_col = rf'LAI4g'

                SM_list=df_region[SM_col].tolist()
                LAI_list=   df_region[Lai_col].tolist()


                plt.scatter(SM_list,LAI_list,s=2,zorder=-1,c='k',alpha=0.2)
                # plt.show()
                plt.ylim(-60,60)
                plt.xlim(-60,60)
                plt.xlabel('SM')
                plt.ylabel('LAI')
                plt.title(region)


        plt.show()

    def negative_postive_response_curve_transition(self):  ####
        #### plot W.S. ,method


        transition_class=['sig_wetting_sig_wetting','sig_drying_sig_drying','sig_wetting_sig_drying','sig_drying_sig_wetting','sig_wetting_no_sig','sig_drying_no_sig','no_sig_sig_wetting','no_sig_sig_drying','no_sig_no_sig',]


        flag = 1


        df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly.df')

            ##df_clean
        df = df[df['row'] > 120]
        print(len(df))
        T.print_head_n(df)

        for region in transition_class:
            df_region=df[df['wetting_drying_transition']==region]

            plt.subplot(3, 4, flag)

            SM_col = rf'GLEAM_SMroot_detrend'
            sm_value = df_region[SM_col].tolist()
            start = np.nanmin(sm_value)
            end = np.nanmax(sm_value)

            sm_bin = np.linspace(start, end, 20)
            df_group, bins_list_str = T.df_bin(df_region, SM_col, sm_bin)

            average_LAI_list = []
            x_list = []
            for name, df_group_i in df_group:
                SM_list = df_group_i[SM_col].tolist()
                LAI_list = df_group_i['LAI4g'].tolist()

                average = np.nanmean(LAI_list)
                average_LAI_list.append(average)
                x_list.append(name[0].left)
            plt.plot(x_list, average_LAI_list)
            plt.title(region)

            # plt.show()
            flag += 1

            Lai_col = rf'LAI4g'

            SM_list = df_region[SM_col].tolist()
            LAI_list = df_region[Lai_col].tolist()

            plt.scatter(SM_list, LAI_list, s=2, zorder=-1, c='k', alpha=0.2)
            # plt.show()
            plt.ylim(-100, 100)


        plt.show()




    def transition_bin(self):
        outdir = result_root+'transition\\'
        T.mk_dir(outdir)
        dff = result_root+rf'Dataframe\relative_changes/relative_changes_yearly.df'
        df = T.load_df(dff)
        step = 14
        sm_col = 'CRU_detrend'
        ## plot hist
        # plt.hist(df[sm_col], bins=100)
        # plt.show()
        # exit()


        df = df.dropna(subset=[sm_col])

        sm_bin = np.arange(-50, 50.001, step)
        df_group, bins_list_str = T.df_bin(df, sm_col, sm_bin)
        df['mode'] = np.nan
        level_list = []
        for name, df_group_i in df_group:
            left = name[0].left
            right = name[0].right
            if left < 0:
                mode_i = 'dry'
                level = -left / step - 1
            else:
                mode_i = 'wet'
                level = left / step
            # print(mode_i, level, left, right)
            level = int(level)
            level_list.append(level)
            if level == 0:
                mode = 'normal'
            else:
                mode = f'{mode_i}-{level:0d}'
            index_list = df_group_i.index.tolist()
            df['mode'][index_list] = mode
        max_level = max(level_list)
        # print(max_level)
        # print(sm_bin)
        index_list_extreme_wet = df[df[sm_col] > sm_bin[-1]].index.tolist()
        index_list_extreme_dry = df[df[sm_col] < sm_bin[0]].index.tolist()
        df['mode'][index_list_extreme_wet] = f'wet-{max_level + 1:0d}'
        df['mode'][index_list_extreme_dry] = f'dry-{max_level + 1:0d}'
        # T.print_head_n(df)
        # exit()

        df_group_dict = T.df_groupby(df, 'pix')
        result_dict = {}
        flag = 0
        for pix in tqdm(df_group_dict):
            df_pix = df_group_dict[pix]
            year_list = df_pix['year'].tolist()
            mode = df_pix['mode'].tolist()
            mode_dict = T.dict_zip(year_list, mode)
            year_list.sort()
            for year in year_list:
                if year + 1 not in mode_dict:
                    continue
                mode1 = mode_dict[year]
                mode2 = mode_dict[year + 1]
                mode_transition = f'{mode1}_{mode2}'
                year_str = f'{year}_{year + 1}'
                flag += 1
                result_dict[flag] = {
                    'pix': pix,
                    'year': year_str,
                    'mode': mode_transition,
                }
        df_reslult = T.dic_to_df(result_dict, key_col_str='idx')
        # T.print_head_n(df_reslult)
        # exit()
        outf = join(outdir, 'transition.df')
        T.save_df(df_reslult, outf)
        T.df_to_excel(df_reslult, outf)

    def plot_transition(self):
        dff = result_root + 'transition\\transition.df'
        df = T.load_df(dff)
        T.print_head_n(df)

        df_group_dict = T.df_groupby(df, 'mode')
        for mode in tqdm(df_group_dict):
            df_mode = df_group_dict[mode]
            df_group_pix = T.df_groupby(df_mode, 'pix')
            spatial_dict = {}
            for pix in df_group_pix:
                df_pix = df_group_pix[pix]
                spatial_dict[pix] = len(df_pix)
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict)
            plt.figure(figsize=(20, 10))
            plt.imshow(arr, interpolation='nearest', cmap='jet', vmin=0, vmax=5)
            plt.title(mode)
            plt.colorbar()
            plt.show()

    def trans_condition_to_int(self, condition):
        if 'dry' in condition:
            level = int(condition.split('-')[1])
            level = -level
        elif 'wet' in condition:
            level = int(condition.split('-')[1])
        else:
            level = 0
        return level

    pass


    def plot_transition_matrix(self):
        dff = result_root + 'transition\\transition.df'
        df = T.load_df(dff)
        T.print_head_n(df)

        region_list =['Arid','Semi-Arid','Sub-Humid']
        flag=1
        for region in region_list:
            plt.subplot(1,3,flag)
            df_region = df[df['AI_classfication']==region]
            condition1_list = []
            condition2_list = []
            for i, row in tqdm(df_region.iterrows(), total=len(df_region)):
                mode = row['mode']
                mode1 = mode.split('_')[0]
                mode2 = mode.split('_')[1]
                condition1 = self.trans_condition_to_int(mode1)
                condition2 = self.trans_condition_to_int(mode2)
                condition1_list.append(condition1)
                condition2_list.append(condition2)
            df_region['year1'] = condition1_list
            df_region['year2'] = condition2_list

            year1_level_list = T.get_df_unique_val_list(df_region, 'year1')
            year2_level_list = T.get_df_unique_val_list(df_region, 'year2')
            year1_level_list = list(set(year1_level_list))
            year2_level_list = list(set(year2_level_list))
            year1_level_list.sort()
            year2_level_list.sort()

            matrix = []
            y_ticks = []
            for year1 in year1_level_list:
                matrix_i = []
                x_ticks = []
                for year2 in year2_level_list:
                    df_i = df_region[(df_region['year1'] == year1) & (df_region['year2'] == year2)]
                    n = len(df_i)
                    if n == 0:
                        n_log = np.nan
                    else:
                        n_log = math.log10(n)
                    matrix_i.append(n)
                    # matrix_i.append(n_log)
                    x_ticks.append(year2)
                y_ticks.append(year1)
                matrix.append(matrix_i)
            matrix = np.array(matrix)
            # plt.figure(figsize=(20, 10))
            plt.imshow(matrix, interpolation='nearest', cmap='Blues',vmin=0, vmax=5000)
            plt.colorbar()
            plt.xticks(range(len(x_ticks)), x_ticks)
            plt.yticks(range(len(y_ticks)), y_ticks)
            plt.xlabel('year2')
            plt.ylabel('year1')
            plt.title(region)
            flag=flag+1
        plt.show()
        exit()
























class plot_response_function:###### 03/18
    def __init__(self):
        self.this_class_arr = result_root + rf'Dataframe\relative_changes_trend\\'
        self.dff = self.this_class_arr + 'relative_changes_trend.df'
        self.outdir = result_root + 'response_function/'
        T.mkdir(self.outdir, force=True)
        pass
    def run(self):
        # self.build_df()

        # self.plot_response_func()
        self.heatmap()
        pass
    def plot_response_func(self):
        dff = result_root + rf'Dataframe\relative_changes_trend\\relative_changes_trend.df'
        df = T.load_df(dff)
        T.print_head_n(df)
        period_list=['1982_2001','2002_2020']

        for period in period_list:

            SM_trend_col = rf'GLEAM_SMroot_{period}_trend'
            LAI_trend_col = rf'LAI4g_{period}_trend'
            VPD_trend_col = rf'VPD_{period}_trend'

            df = df.dropna(subset=[SM_trend_col, LAI_trend_col, VPD_trend_col])
            df = df[df['row'] > 120]

            df = df[df['landcover_classfication'] != 'Cropland']
            ### bin the SM
            SM_bin = np.linspace(-1, 1, 10)
            df_group, bins_list_str = T.df_bin(df, SM_trend_col, SM_bin)
            LAI_trend_mean_list = []

            for name, df_group_i in df_group:
                SM_bin_name = name[0].left
                LAI_trend_mean = np.nanmean(df_group_i[LAI_trend_col].tolist())
                LAI_trend_mean_list.append(LAI_trend_mean)
            plt.plot(SM_bin[0:-1], LAI_trend_mean_list, label=period)
        plt.xlabel('SM_trend')
        plt.ylabel('LAI_trend')
        plt.legend()
        plt.show()

    def heatmap(self):
        dff = result_root + rf'Dataframe\relative_changes_trend\\relative_changes_trend.df'
        df = T.load_df(dff)
        T.print_head_n(df)
        p = '1982_2001'
        df = df[df['row'] > 120]

        # df=df[df['classfication_SM']!='no_sig_no_sig']
        print(len(df))
        ## get the unique landcover_classfications
        # landcover_classfications_list=df['landcover_classfication'].tolist()
        # landcover_classfications_list_unique=T.get_df_unique_val_list(df,'landcover_classfication')
        # print(landcover_classfications_list_unique)
        # exit()
        # df=df[df['landcover_classfication']=='Shrub']
        # print(len(df))
        # plt.hist(df[rf'GLEAM_SMroot_{p}_trend'], bins=100)
        # plt.show()
        # plt.hist(df[rf'LAI4g_{p}_trend'], bins=100)
        # plt.show()
        # plt.hist(df[rf'VPD_{p}_trend'], bins=100)
        # plt.show()
        # exit()

        SM_p1_bin_list = np.linspace(-1.5, 1.5, 15)
        VPD_p1_bin_list = np.linspace(-1, 1, 15)
        df_group1, bins_list_str1 = T.df_bin(df, rf'GLEAM_SMroot_{p}_trend', SM_p1_bin_list)
        matrix = []
        y_labels = []
        for name1, df_group_i1 in df_group1:
            df_group2, bins_list_str2 = T.df_bin(df_group_i1, rf'GLEAM_SMroot_{p}_trend', VPD_p1_bin_list)
            name1_ = name1[0].left

            matrix_i = []
            x_labels = []

            for name2, df_group_i2 in df_group2:
                name2_ = name2[0].left
                x_labels.append(name2_)
                # print(name1,name2)
                # print(len(df_group_i2))
                val = np.nanmean(df_group_i2[rf'LAI4g_{p}_trend'])

                matrix_i.append(val)

                # count=len(df_group_i2)
                # matrix_i.append(count)

                # matrix_i.append(val)
            matrix.append(matrix_i)
            y_labels.append(name1_)
        matrix = np.array(matrix)
        matrix = matrix[::-1, :]
        plt.imshow(matrix, cmap='RdBu', vmin=-1, vmax=1)
        # plt.imshow(matrix,cmap='RdBu')
        plt.xticks(np.arange(len(SM_p1_bin_list) - 1), x_labels, rotation=45)
        plt.yticks(np.arange(len(VPD_p1_bin_list) - 1), y_labels[::-1])
        plt.xlabel(rf'SM_{p}_trend')
        plt.ylabel(rf'VPD_{p}_trend')
        ## draw 1:1 line
        # plt.plot([20, 0], [0, 20], 'k-', lw=2)

        plt.colorbar()
        plt.show()

    def plot_asymmetry(self):
        f=result_root + rf'Detrend\detrend_relative_change\\GLEAM_SMroot.df'
        dic=T.load_npy(f)
        for pix in dic:
            vals=dic[pix]
            vals=np.array(vals)















        pass
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





class CCI_LC_preprocess():

    def __init__(self):
        self.datadir = rf'C:\Users\wenzhang1\Desktop\CCI_landcover\\'
        pass

    def run(self):
        # self.lccs_class_count()
        # self.lc_ratio_025_individal()

        # self.flags()  #not used
        # self.composition()
        # self.trend_analysis_LC()
        self.trend_maxmimum()


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
    def trend_analysis_LC(self):

        fdir_all = rf'E:\CCI_landcover\landcover_composition_DIC\\'
        outdir = rf'E:\\CCI_landcover\\trend_analysis_LC\\'
        T.mk_dir(outdir,force=True)
        for fdir in T.listdir(fdir_all):
            if not 'grassland' in fdir:
                continue
            outf_trend = join(outdir, f'{fdir}_trend.tif')
            outf_p_value = join(outdir, f'{fdir}_p_value.tif')
            outf_change = join(outdir, f'{fdir}_change.tif')
            if isfile(outf_trend):
                continue
            if isfile(outf_p_value):
                continue
            if isfile(outf_change):
                continue
            spatial_dic=T.load_npy_dir(join(fdir_all,fdir))
            trend_dic={}
            p_value_dic={}
            change_dic={}

            for pix in tqdm(spatial_dic):
                time_series=spatial_dic[pix]
                # print(time_series)


                time_series=np.array(time_series)
                if T.is_all_nan(time_series):
                    continue
                ## if nan in the time series
                if np.isnan(time_series).any():
                    continue

                # print(time_series)
                ## if all the values are the same
                if len(set(time_series))==1:
                    continue
                ## trend

                slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series)), time_series)
                trend_dic[pix] = slope
                p_value_dic[pix] = p_value
                ### calculate change based on the trend
                change_rate = slope * len(time_series)
                change_dic[pix] = change_rate
            arr_trend=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            arr_p_value=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            arr_change=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(change_dic)

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend,outf_trend)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_p_value,outf_p_value)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_change,outf_change)

            outf_trend=join(outdir,f'{fdir}_trend.npy')
            outf_p_value=join(outdir,f'{fdir}_p_value.npy')
            outf_change=join(outdir,f'{fdir}_change.npy')
            np.save(outf_trend,trend_dic)
            np.save(outf_p_value,p_value_dic)
            np.save(outf_change,change_dic)
        pass

    def trend_maxmimum(self):
        fdir_all = rf'E:\CCI_landcover\trend_analysis_LC\\'
        outdir = rf'E:\\CCI_landcover\\trend_analysis_LC\\'
        T.mk_dir(outdir,force=True)
        LC_list=[]
        for f in T.listdir(fdir_all):
            if not f.endswith('.tif'):
                continue
            if not 'change' in f:
                continue
            if 'p_value' in f:
                continue

            fpath=join(fdir_all,f)
            print(fpath)
            array, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(fpath)
            ## get absolute value
            array[array>99]=np.nan
            array[array<-99]=np.nan
            array_abs=np.absolute(array)

            LC_list.append(array_abs)
        LC_array=np.array(LC_list)
        ### get the maximum change rate absolute value

        LC_max=np.nanmax(LC_array,axis=0)
        outf=join(outdir,'LC_max.tif')
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(LC_max,outf)






        pass





class calculating_variables:  ###

    def run(self):
        self.calculate_average_GPP()
        # self.calculate_CV()
        # self.calculate_monthly_CV()
        # self.convert_tiff_to_npy()
        # self.create_CO2_dic()

        # self.create_GMST_dic()
        pass

    def calculate_CV(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)

        fdir = result_root + rf'extract_GS\OBS_LAI_extend\\'
        outdir = result_root + rf'state_variables\\CV_yearly\\'
        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.split('.')[0]  in ['GPCC' ,'LAI4g','CRU']:
                continue
            CV_dic = {}


            variable = f.split('.')[0]
            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)
            for pix in tqdm(val_dic, desc=variable):
                r,c=pix
                if r<120:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue

                vals = val_dic[pix]
                if T.is_all_nan(vals):
                    continue
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                CV = std / mean
                CV_dic[pix] = CV
            outf = outdir + rf'\{variable}_CV.npy'
            np.save(outf, CV_dic)
            array=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(CV_dic)
            arr_CV_dryland = array * array_mask


            ##save
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_CV_dryland,outdir+rf'\{variable}_CV.tif')

        pass
    def calculate_average_GPP(self):
        fdir = result_root + rf'\extract_GS\OBS_LAI\\'
        outdir = result_root + rf'state_variables\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'CRU' in f:
                continue
            dic=T.load_npy(fdir+f)
            for pix in dic:
                vals=dic[pix]
                if T.is_all_nan(vals):
                    continue
                dic[pix]=np.nanmean(vals)
            array=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(dic)
            plt.imshow(array)
            plt.colorbar()
            plt.show()
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(array,outdir+f.split('.')[0]+'.tif')
            outf=outdir+f.split('.')[0]+'.npy'
            np.save(outf,dic)
        pass


    def calculate_monthly_CV(self):## to calculate the monthly CV of precipitation
        fdir = data_root + rf'monthly_data\\\tmin\\'
        outdir = result_root + rf'CV\\tmin_monthly\\'
        T.mk_dir(outdir, force=True)
        dic_precip=T.load_npy_dir(fdir)

        result_dic={}
        for pix in tqdm(dic_precip):
            vals=dic_precip[pix]
            if T.is_all_nan(vals):
                continue
            vals=np.array(vals)
            vals_reshape=vals.reshape(-1,12)

            CV_list_i=[]
            for i in range(len(vals_reshape)):
                vals_i=vals_reshape[i]
                mean=np.nanmean(vals_i)
                std=np.nanstd(vals_i)
                CV=std/mean
                CV_list_i.append(CV)

            result_dic[pix]=CV_list_i
        outf=outdir+'tmin_monthly.npy'
        np.save(outf,result_dic)










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
        CO2_f=data_root+rf'GMST.xlsx'
        df=pd.read_excel(CO2_f,sheet_name='GMST')
        print(df)
        print(df.columns)

        for col in df.columns:
            new_col=col.replace(' ','')
            df.rename(columns={col:new_col},inplace=True)
            if new_col=='Anomaly (deg C)':
                df[new_col]=df[new_col].astype(float)
            else:
                df[new_col]=df[new_col].astype(int)

        df=df[['Time','Anomaly(degC)']]
        df=df[df['Time']>=1982]
        df=df[df['Time']<=2020]
        year_list=df['Time'].unique()
        average_CO2_list=[]
        for i in range(len(df)):
            for year in year_list:
                if df.iloc[i]['Time']==year:
                    average_CO2=(df[df['Time']==year]['Anomaly(degC)'])
                    average_CO2_list.append(average_CO2)
        df['average_GMST']=average_CO2_list
        ### CO2 dic
        CO2_list=[]
        for yr in year_list:

            for i in range(len(df)):
                if df.iloc[i]['Time']==yr:
                    CO2_value=df.iloc[i]['Anomaly(degC)']

            CO2_list.append(CO2_value)

        ########create spatial dic with CO2 dic

        dic=DIC_and_TIF(pixelsize=0.25).void_spatial_dic()

        for pix in dic:

            dic[pix]=CO2_list


        outf=result_root+rf'\extract_GS\OBS_LAI\GMST.npy'
        np.save(outf,dic)


    def create_GMST_dic(self):
        CO2_f=data_root+rf'GMST.xlsx'
        df=pd.read_excel(CO2_f,sheet_name='GMST')
        print(df)
        print(df.columns)


        df=df[['Time','Anomaly (deg C)']]
        df=df[df['Time']>=1982]
        df=df[df['Time']<=2020]
        year_list=df['Time'].unique()

        ### CO2 dic
        CO2_list=[]
        for yr in year_list:

            for i in range(len(df)):
                if df.iloc[i]['Time']==yr:
                    CO2_value=df.iloc[i]['Anomaly (deg C)']

            CO2_list.append(CO2_value)

        ########create spatial dic with CO2 dic

        dic=DIC_and_TIF(pixelsize=0.25).void_spatial_dic()

        for pix in dic:

            dic[pix]=CO2_list


        outf=result_root+rf'\extract_GS\OBS_LAI\GMST.npy'
        np.save(outf,dic)














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
        # self.cal_monthly_trend()
        # self.post_proprocessing()
        # self.plot_seasonal_trend()
        self.plot_monthly_trend()
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
        # product='Precip'
        product = 'LAI'
        # SM_dir = 'D:\Project3\Data\monthly_data\GLEAM_SMroot_anomaly\\'
        SM_dir = 'D:\Project3\Data\monthly_data\LAI4g\\'
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan
        array_mask[array_mask > 0] = 1



        SM_dic = T.load_npy_dir(SM_dir)
        # LAI_dic = T.load_npy_dir(LAI_dir)
        result_dict = {}
        for pix in tqdm(SM_dic):

            SM = SM_dic[pix]



            if np.nanstd(SM) == 0:
                continue
            # if np.nanstd(LAI) == 0:
            #     continue
            df = pd.DataFrame()
            df['SM'] = SM
            # df['LAI'] = LAI
            df['month'] = list(range(1, 13)) * int(len(SM) / 12)

            # df = df.dropna()

            result_dict_i = {}
            for month in range(1,13):
                df_month = df[df['month'] == month]
                SM = df_month['SM'].to_list()
                ## calculate the trend
                if np.nanstd(SM) == 0:
                    continue
                SM = np.array(SM)
                a,b,r,p = T.nan_line_fit(list(range(len(SM))),SM)
                result_dict_i[f'{month}_a'] = a
                result_dict_i[f'{month}_p'] = p


            result_dict[pix] = result_dict_i
        df_result = T.dic_to_df(result_dict,'pix')
        T.print_head_n(df_result)
        outf = join(outdir, f'{product}.df')
        T.save_df(df_result, outf)
        T.df_to_excel(df_result, outf)

    def post_proprocessing(self):
        dff=join(self.this_class_arr,'monthly_trend','LAI.df')
        df=T.load_df(dff)
        # df['pix'] = df['__key__'].to_list()
        df = df.dropna()
        NDVI_mask= data_root + rf'/Base_data/dryland_AI.tif\\dryland.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask)
        array_mask[array_mask < 0] = np.nan
        array_mask[array_mask > 0] = 1
        figure, axs = plt.subplots(3, 4, figsize=(20, 15))
        for month in range(1,13):
            col_name = f'{month}_a'
            spatial_dict = T.df_to_spatial_dic(df, col_name)
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict)
            arr = arr*array_mask
            ax = axs[int((month-1)/4), (month-1)%4]
            ax.set_title(col_name)


            ax.imshow(arr, vmin=-0.5, vmax=0.5, cmap='RdBu', interpolation='nearest')
            cax = figure.add_axes([0.2, 0.05, 0.6, 0.02])
            figure.colorbar(ax.imshow(arr, vmin=-0.1, vmax=0.1, cmap='RdBu', interpolation='nearest'), cax=cax,
                            orientation='horizontal')

        plt.show()

        pass
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
        # dff = join(self.this_class_arr, 'monthly_trend', 'LAI.df')
        # dff = join(self.this_class_arr, 'monthly_trend', 'SM.df')
        dff = join(self.this_class_arr, 'monthly_trend', 'Precip.df')
        df = T.load_df(dff)
        # df['pix'] = df['__key__'].to_list()
        month_list = list(range(1,13))
        figure, axs = plt.subplots(3, 4, figsize=(20, 15))
        for month in month_list:
            col_name = f'{month}_a'
            spatial_dict = T.df_to_spatial_dic(df, col_name)
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict)
            ax = axs[int((month-1)/4), (month-1)%4]
            ax.imshow(arr, vmin=-0.5, vmax=0.5, cmap='RdBu', interpolation='nearest')

            ax.set_title(col_name)
            ## add colorbar
            ## bottom
            cax = figure.add_axes([0.2, 0.05, 0.6, 0.02])
            figure.colorbar(ax.imshow(arr, vmin=-0.5, vmax=0.5, cmap='RdBu', interpolation='nearest'), cax=cax, orientation='horizontal')
        plt.show()








        pass


class multi_regression_anomaly():
    def __init__(self):

        self.fdirX = result_root+rf'extract_GS\OBS_LAI_extend\\'
        self.fdirY = result_root+rf'extract_GS\OBS_LAI_extend\\'

        self.period = ('1982_2020')
        # self.period=('1982_2001')
        # self.period=('2002_2020')
        # self.y_var = [f'LAI4g_{self.period}']
        # self.xvar = [f'Tmax_{self.period}', f'CRU_{self.period}', f'CO2_{self.period}', f'VPD_{self.period}']
        self.y_var = ['LAI4g']
        self.xvar = [ 'CO2', 'CRU']

        self.multi_regression_result_dir = result_root + rf'multi_regression\\original\\{self.period}\\'
        T.mk_dir(self.multi_regression_result_dir, force=True)

        self.multi_regression_result_f = result_root + rf'multi_regression\\original\\{self.period}\\{self.y_var[0]}.npy'

        pass

    def run(self):

        # step 1 build dataframe

        # df=self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var)

        # # # step 2 cal correlation
        self.cal_multi_regression_beta()

        # step 3 plot
        self.plt_multi_regression_result(self.multi_regression_result_dir,self.y_var[0],self.period)

        ## step 4 convert m2/m2/ppm to %/100ppm
        # self.convert_CO2_sensitivity_unit()

        # step 5
        # self.calculate_trend_contribution()

        pass

    def build_df(self, fdir_X, fdir_Y, fx_list, fy):

        df = pd.DataFrame()

        filey = fdir_Y + fy[0] + '.npy'
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
            filex = fdir_X + xvar + '.npy'
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
        ## save df
        T.save_df(df, self.multi_regression_result_dir + fy[0] + '.df')
        T.df_to_excel(df, self.multi_regression_result_dir + fy[0] + '.xlsx')

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

    def cal_multi_regression_beta(self):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd
        import joblib

        df = T.load_df(rf'D:\Project3\Result\\multi_regression\original\1982_2020\\LAI4g.df')

        x_var_list = self.xvar

        outf = self.multi_regression_result_f

        multi_derivative = {}

        for i, row in tqdm(df.iterrows(), total=len(df)):
            # print(row);exit()
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

                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue

                df_new[x] = x_vals

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue
            if len(x_var_list_valid) < 2:
                continue
            # T.print_head_n(df_new)

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
            # x_var_list_valid_new.append('CO2:CRU')
            # # x_var_list_valid_new.append('tmax:CRU')

            df_new = df_new.dropna()
            ## build multiregression model and consider interactioon

            # model = smf.ols('y ~ CO2 + CRU +  CO2:CRU ',
            #                 data=df_new).fit()
            model = smf.ols('y ~ CO2 + CRU ',
                            data=df_new).fit()

            coef_ = np.array(model.params)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))
            # print(df_new['y'])
            # exit()
            multi_derivative[pix] = coef_dic
        T.save_npy(multi_derivative, outf)

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir, y_var, period):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        f = self.multi_regression_result_f

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:


            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            # print(var_i)
            spatial_dic = {}
            for pix in dic:
                r, c = pix
                if r < 120:
                    continue

                landcover_value = crop_mask[pix]

                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            arr = arr * array_mask
            print(var_i)
            if var_i=='tmax:CRU':
                var_i = 'tmax_CRU'
            elif var_i=='CO2:CRU':
                var_i = 'CO2_CRU'
            else:
                var_i = var_i

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f'{multi_regression_result_dir}\\{var_i}_{y_var}_{period}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr, vmin=-5, vmax=5)

            plt.title(var_i)
            plt.colorbar()

        plt.show()

    def convert_CO2_sensitivity_unit(self):
        period_list = ['1982_2020']
        for period in period_list:
            CO2_sensitivity_f = result_root + rf'multi_regression\\anomaly\\{period}\\CO2_LAI4g_{period}.tif'
            average_LAI4g_f = result_root + rf'\state_variables\\\\LAI4g_{period}.npy'
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(CO2_sensitivity_f)
            arr[arr < -99] = np.nan
            dic_CO2_sensitivity = DIC_and_TIF().spatial_arr_to_dic(arr)

            dic_LAI4g_average = T.load_npy(average_LAI4g_f)

            for pix in dic_CO2_sensitivity:
                CO2_sensitivity = dic_CO2_sensitivity[pix]
                CO2_sensitivity = np.array(CO2_sensitivity, dtype=float)
                if np.isnan(CO2_sensitivity):
                    continue
                if not pix in dic_LAI4g_average:
                    continue
                LAI_average = dic_LAI4g_average[pix]
                LAI_average = np.array(LAI_average, dtype=float)

                if np.isnan(LAI_average):
                    continue
                CO2_sensitivity = CO2_sensitivity / LAI_average * 100
                if CO2_sensitivity < -99999:
                    continue
                dic_CO2_sensitivity[pix] = CO2_sensitivity
            arr_new = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(dic_CO2_sensitivity)
            arr_new[arr_new < -99] = np.nan
            arr_new[arr_new > 99] = np.nan

            # plt.imshow(arr_new)
            # plt.colorbar()
            # plt.show()

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_new, f'{CO2_sensitivity_f.replace(".tif", "_scale.tif")}')
            # DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_CO2_sensitivity, f'{CO2_sensitivity_f.replace(".tif","_new.tif")}')
            # T.save_npy(dic_CO2_sensitivity, CO2_sensitivity_f.replace('.tif', '.npy'))

    def calculate_trend_contribution(self):
        ## here I would like to calculate the trend contribution of each variable
        ## the trend contribution is defined as the slope of the linear regression between the variable and the target variable mutiplied by trends of the variable
        ## load the trend of each variable
        ## load the trend of the target variable
        ## load multi regression result
        ## calculate the trend contribution
        trend_dir = result_root + rf'\trend_analysis\anomaly\OBS_extend\\'

        selected_vairables_list = [
            'CRU_trend',
            'CO2_trend',
            'tmax_trend',
            'VPD_trend',
        ]

        trend_dict = {}
        for variable in selected_vairables_list:
            fpath = join(trend_dir, f'{variable}.npy')
            array = np.load(fpath, allow_pickle=True)
            array[array < -9999] = np.nan
            spatial_dict = D.spatial_arr_to_dic(array)
            for pix in tqdm(spatial_dict, desc=variable):
                r, c = pix
                if r < 120:
                    continue
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue
                if not pix in trend_dict:
                    trend_dict[pix] = {}
                key = variable.replace('_trend', '')
                trend_dict[pix][key] = spatial_dict[pix]

        f = self.multi_regression_result_f
        print(f)
        print(isfile(f))
        # exit()
        dic_multiregression = T.load_npy(f)
        var_list = []
        for pix in dic_multiregression:

            # landcover_value = crop_mask[pix]
            # if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
            #     continue

            vals = dic_multiregression[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        # print(var_list)
        # exit()
        for var_i in var_list:
            spatial_dic = {}
            for pix in dic_multiregression:
                if not pix in trend_dict:
                    continue

                dic_i = dic_multiregression[pix]
                if not var_i in dic_i:
                    continue
                val_multireg = dic_i[var_i]
                val_trend = trend_dict[pix][var_i]
                val_contrib = val_multireg * val_trend
                spatial_dic[pix] = val_contrib
            arr_contrib = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr_contrib, cmap='RdBu', interpolation='nearest')
            plt.colorbar()
            plt.title(var_i)
            plt.show()
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_contrib,
                                                   f'{self.multi_regression_result_dir}\\{var_i}_trend_contribution.tif')


class multi_regression_detrended_anomaly():
    def __init__(self):
        # self.fdirX=result_root+rf'anomaly\OBS_extend\\'
        # self.fdirY=result_root+rf'anomaly\OBS_extend\\'
        self.fdirX = result_root + rf'Detrend\detrend_anomaly\\1982_2020\\'
        self.fdirY = result_root + rf'Detrend\detrend_anomaly\\1982_2020\\'


        self.period=('1982_2020')
        # self.period=('1982_2001')
        # self.period=('1982_2001')
        # self.y_var = [f'LAI4g_{self.period}']
        # self.xvar = [f'Tmax_{self.period}', f'GLEAM_SMroot_{self.period}', f'VPD_{self.period}']
        self.y_var = ['LAI4g']
        self.xvar = ['tmax', 'VPD', 'CRU']




        self.multi_regression_result_dir=result_root+rf'multi_regression\\detrended_anomaly\\{self.period}\\'
        T.mk_dir(self.multi_regression_result_dir,force=True)

        self.multi_regression_result_f = result_root + rf'multi_regression\\\\detrended_anomaly\\{self.period}\\{self.y_var[0]}.npy'
        pass

    def run(self):

        #step 1 build dataframe
        df = self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var,self.period)
        # df=self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var)

        # # # step 2 cal correlation
        self.cal_multi_regression_beta(df, self.xvar)  # 修改参数

        # step 3 plot
        # self.plt_multi_regression_result(self.multi_regression_result_dir,self.y_var[0],self.period)


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
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)


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
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue


                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            arr=arr*array_mask
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



class pick_event():



    ##

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
class time_series_to_grid():
    def __init__(self):
        pass
    def run(self):
        self.time_series_to_grid()
    def time_series_to_grid(self):
        f=result_root+'pick_event\extract_variables_during_droughts_GS\\concat_df.df'
        df=T.load_df(f)


        pass






    pass





class plot_dataframe():
    def __init__(self):
        scenario='S1'
        self.product_list = ['LAI4g', 'GIMMS3g_LAI','GIMMS_AVHRR_LAI', f'CABLE-POP_{scenario}_lai', f'CLASSIC_{scenario}_lai', 'CLM5',  f'IBIS_{scenario}_lai', f'ISAM_{scenario}_lai',
             f'ISBA-CTRIP_{scenario}_lai', f'JSBACH_{scenario}_lai', f'JULES_{scenario}_lai',  f'LPJ-GUESS_{scenario}_lai', f'LPX-Bern_{scenario}_lai',
             f'ORCHIDEE_{scenario}_lai', f'SDGVM_{scenario}_lai', f'YIBs_{scenario}_Monthly_lai']

        # self.product_list = ['GPP_baseline', 'GPP_CFE','Ensemble', f'CABLE-POP_{scenario}_gpp', f'CLASSIC_{scenario}_gpp', 'CLM5',  f'IBIS_{scenario}_gpp', f'ISAM_{scenario}_gpp',
        #      f'ISBA-CTRIP_{scenario}_gpp', f'JSBACH_{scenario}_gpp', f'JULES_{scenario}_gpp',   f'LPX-Bern_{scenario}_gpp',
        #      f'ORCHIDEE_{scenario}_gpp', f'SDGVM_{scenario}_gpp', f'YIBs_{scenario}_Monthly_gpp']

        # self.product_list = [ 'GPP_NIRv','GPP_baseline', 'GPP_CFE',]
        self.product_list = ['LAI4g', 'GIMMS_AVHRR_LAI', 'GIMMS3g_LAI'  ]



        pass
    def run(self):


        # self.plot_annual_zscore_based_region()

        # self.plot_monthly_zscore_based_region()
        # self.plot_greening_trend_moisture_trend_heatmap()
        # self.plot_greening_trend_moisture_relative_change_heatmap()

        # self.plot_anomaly_trendy_LAI()
        self.plot_anomaly_LAI_based_on_cluster() #### widely used
        # self.plot_anomaly_LAI_global()

        # self.plot_asymetrical_response_based_on_cluster()
        # self.plot_anomaly_trendy_GPP()
        # self.plot_anomaly_vegetaton_indices()
        # self.plot_climatic_factors()
        # self.plot_plant_fuctional_types_trend()
        # self.plot_trend_spatial_all()
        # self.plot_trend_regional()
        # self.trend_bar()
        # self.trend_percentage_bar()
        # self.bin_trend()

        # self.plot_anomaly_bar()
        # self.plot_bin_sensitivity_for_each_bin()
        # self.plot_drying_greening()
        self.plot_drying_greening_map()
        # self.plot_cluster_variables_trend()
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
        df=df[df['row']>120]



        return df



    def plot_annual_zscore_based_region(self):   #based on semi-arid, arid and sub-humid
        df= T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_new.df')
        print(len(df))
        df=df[df['landcover_classfication']!='Cropland']
        print(len(df))

        product_list = ['LAI4g', 'carbontracker']

        # product_list = ['LAI4g','GIMMS3g','GIMMS_AVHRR_LAI']

        fig = plt.figure()
        i = 1

        for region in ['Africa', 'Asia', 'Australia', 'South_America', 'North_America']:

            ax = fig.add_subplot(2, 3, i)

            flag = 0
            color_list=['blue','green','red','orange']

            for variable in product_list:

                colunm_name = variable
                df_region = df[df['continent'] == region]
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


            major_yticks = np.arange(-30, 30, 5)
            ax.set_yticks(major_yticks)

            plt.grid(which='major', alpha=0.5)
            plt.tight_layout()
            i = i + 1

        plt.show()


    def plot_greening_trend_moisture_trend_heatmap(self):   #based on semi-arid, arid and sub-humid
        df= T.load_df(result_root + rf'Dataframe\relative_changes_trend\\relative_changes_trend.df')
        print(len(df))
        cols=df.columns
        for col in cols:
            print(col)

        # exit()
        df=df[df['landcover_classfication']!='Cropland']
        print(len(df))
        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 0]
        df = df[df['lat'] < 45]


        df=df[df['continent']=='North_America']
        # df=df[df['continent']=='Asia']
        # df=df[df['continent']=='Australia']
        LAI_trend_col='LAI4g_trend_1982_2020'
        VPD_col='VPD_trend'
        sm_col='GLEAM_SMroot_trend_1982_2020'
        # LAI_trend_col = 'LAI4g_2002_2020_trend'
        # VPD_col = 'VPD_2002_2020_trend'
        # sm_col = 'GLEAM_SMroot_2002_2020_trend'


        # sm_col='CRU_trend_1982_2020'


        fig = plt.figure()
        fig.set_size_inches(12, 3)
        i = 1
        # unique_landcover_classfication = df['landcover_classfication'].unique()
        # print(unique_landcover_classfication)
        # exit()


        for region in [ 'Arid', 'Semi-Arid','Sub-Humid']:
        # for region in ['Grass', 'Shrub', 'Deciduous', 'Evergreen']:
            df_region = df[df['AI_classfication'] == region]
            ax = fig.add_subplot(1, 3, i)

            # vals = df_region[sm_col].tolist()
            # plt.hist(vals, bins=20, color='blue', alpha=0.5, label='LAI_trend')
            # plt.show()
            # exit()

            sm_trend_bin =np.linspace(-0.25,0.25,7)
            df_group1, bins_list_str = T.df_bin(df_region, sm_col, sm_trend_bin)
            print(bins_list_str)
            matrix = []
            y_labels = []


            for name1, df_group_i1 in df_group1:
                df_group2, bins_list_str2 = T.df_bin(df_group_i1, VPD_col, sm_trend_bin)
                name1_ = name1[0].left

                matrix_i = []
                x_labels = []

                for name2, df_group_i2 in df_group2:
                    name2_ = name2[0].left
                    x_labels.append(name2_)
                    # print(name1,name2)
                    # print(len(df_group_i2))
                    val = np.nanmean(df_group_i2[LAI_trend_col])
                    matrix_i.append(val)


                    # count=len(df_group_i2)
                    # matrix_i.append(count)

                    # matrix_i.append(val)
                matrix.append(matrix_i)
                y_labels.append(name1_)
            matrix = np.array(matrix)
            matrix = matrix[::-1, :]
            plt.imshow(matrix, cmap='RdBu',vmin=-0.5,vmax=0.5)
            # plt.imshow(matrix,cmap='RdBu')
            plt.colorbar()
            plt.xlabel('VPD_trend')
            plt.ylabel('SM_trend')


            plt.xticks(range(len(x_labels)), x_labels, rotation=45)
            plt.yticks(range(len(y_labels)), y_labels, rotation=45)
            plt.title(region)
            i = i + 1
        plt.show()

    def plot_greening_trend_moisture_relative_change_heatmap(self):   #based on semi-arid, arid and sub-humid
        df= T.load_df(result_root + rf'Dataframe\\relative_changes\\\\relative_changes_yearly_new.df')
        print(len(df))
        cols=df.columns
        for col in cols:
            print(col)

        # exit()
        df=df[df['landcover_classfication']!='Cropland']
        print(len(df))
        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 0]
        df = df[df['lat'] < 45]


        df=df[df['continent']=='North_America']
        # df=df[df['continent']=='Asia']
        # df=df[df['continent']=='Australia']
        LAI_trend_col='LAI4g'
        VPD_col='VPD'
        sm_col='GLEAM_SMroot'


        fig = plt.figure()
        fig.set_size_inches(12, 3)
        i = 1
        # unique_landcover_classfication = df['landcover_classfication'].unique()
        # print(unique_landcover_classfication)
        # exit()


        for region in [ 'Arid', 'Semi-Arid','Sub-Humid']:
        # for region in ['Grass', 'Shrub', 'Deciduous', 'Evergreen']:
            df_region = df[df['AI_classfication'] == region]
            ax = fig.add_subplot(1, 3, i)

            # vals = df_region[sm_col].tolist()
            # plt.hist(vals, bins=20, color='blue', alpha=0.5, label='LAI_trend')
            # plt.show()
            # exit()

            sm_trend_bin =np.linspace(-10,10,7)
            df_group1, bins_list_str = T.df_bin(df_region, sm_col, sm_trend_bin)
            print(bins_list_str)
            matrix = []
            y_labels = []


            for name1, df_group_i1 in df_group1:
                df_group2, bins_list_str2 = T.df_bin(df_group_i1, VPD_col, sm_trend_bin)
                name1_ = name1[0].left

                matrix_i = []
                x_labels = []

                for name2, df_group_i2 in df_group2:
                    name2_ = name2[0].left
                    x_labels.append(name2_)
                    # print(name1,name2)
                    # print(len(df_group_i2))
                    val = np.nanmean(df_group_i2[LAI_trend_col])
                    matrix_i.append(val)


                    # count=len(df_group_i2)
                    # matrix_i.append(count)

                    # matrix_i.append(val)
                matrix.append(matrix_i)
                y_labels.append(name1_)
            matrix = np.array(matrix)
            # matrix = matrix[::-1, :]
            plt.imshow(matrix, cmap='RdBu',vmin=-10,vmax=10)
            # plt.imshow(matrix,cmap='RdBu')
            plt.colorbar()
            plt.xlabel('VPD')
            plt.ylabel('SM')


            plt.xticks(range(len(x_labels)), x_labels, rotation=45)
            plt.yticks(range(len(y_labels)), y_labels, rotation=45)
            plt.title(region)
            i = i + 1
        plt.show()

    def plot_monthly_zscore_based_region(self):   #based on semi-arid, arid and sub-humid
            df= T.load_df(result_root + rf'Dataframe\moving_window_sensitivity\\moving_window_sensitivity.df')
            print(len(df))
            df=df[df['landcover_classfication']!='Cropland']
            print(len(df))
            df=df[df['row']>120]
            df= df[df['row'] < 240]
            variable_list=['GLEAM_SMroot','GLEAM_SMsurf','GPCC','VPD','tmin','tmax','Tempmean']
            variable_list=['GLEAM_SMroot','VPD','Tempmean']


            monthly_list=['00','01','02','03','04','05','06','07','08','09','10','11']


            # create color list with one green and another 14 are grey


            color_list=['red','blue','green','orange','black','yellow','purple','pink','grey','brown','cyan','magenta','olive','lime','teal','aqua']



            for variable in variable_list:
                fig = plt.figure()
                print(variable)
                i=1
                for region in ['Africa', 'Asia', 'Australia', 'South_America', 'North_America']:
                    df_region = df[df['continent'] == region]
                    ax = fig.add_subplot(2, 3, i)

                    for month in monthly_list:
                        month=int(month)
                        month=f'{month:02d}'

                        column_name = f'{variable}_LAI'
                        vals=df_region[column_name].tolist()

                        # print(vals)

                        vals_nonnan = []
                        for val in vals:
                            if type(val) == float:  ## only screening
                                continue
                            vals_nonnan.append(list(val))
                            if not len(val) == 39:
                                print(val)
                                print(len(val))
                                exit()
                            # print(type(val))
                            # print(len(val))

                        ###### calculate mean
                        vals_mean = np.array(vals_nonnan)  ## axis=0, mean of each row  竖着加
                        vals_mean = np.nanmean(vals_mean, axis=0)
                        val_std = np.nanstd(vals_mean, axis=0)

                        plt.plot(vals_mean, label=column_name, color=color_list[monthly_list.index(month)])
                        # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])

                        # plt.scatter(range(len(vals_mean)),vals_mean)
                        # plt.text(0,vals_mean[0],product,fontsize=8)

                    i=i+1

                    ax.set_xticks(range(0, 40, 4))
                    ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
                    # plt.ylim(-0.2, 0.2)
                    plt.ylim(-20, 20)

                    plt.xlabel('year')

                    plt.ylabel(f'{variable}_relative_change')
                    # plt.legend()

                    plt.title(region)
                    # plt.legend()
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

        df = T.load_df(result_root + rf'\growth_rate\DataFrame\\growth_rate_all_years.df')
        print(len(df))
        df=self.clean_df(df)

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


        #create color list with one green and another 14 are grey

        color_list=['grey']*16
        color_list[0]='green'
        # color_list[1] = 'red'
        # color_list[2]='blue'
        # color_list=['green','blue','red','orange','aqua','brown','cyan', 'black', 'yellow', 'purple', 'pink', 'grey', 'brown','lime','teal','magenta']
        linewidth_list=[1]*16
        linewidth_list[0]=3
        # linewidth_list[1]=2
        # linewidth_list[2]=2

        fig = plt.figure()
        i = 1
        variable_list=['GLEAM_SMroot',]
        # variable_list=['VPD',]
        # variable_list=['leaf_area']
        variable_list=['LAI4g_relative_change']
        # scenario='S2'
        # variable_list= ['LAI4g',f'CABLE-POP_{scenario}_lai', f'CLASSIC_{scenario}_lai', 'CLM5',  f'IBIS_{scenario}_lai', f'ISAM_{scenario}_lai',
        #      f'ISBA-CTRIP_{scenario}_lai', f'JSBACH_{scenario}_lai', f'JULES_{scenario}_lai',  f'LPJ-GUESS_{scenario}_lai', f'LPX-Bern_{scenario}_lai',
        #      f'ORCHIDEE_{scenario}_lai', f'SDGVM_{scenario}_lai', f'YIBs_{scenario}_Monthly_lai']
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

        # exit()

        for continent in ['Arid', 'Semi-Arid','Sub-Humid','global']:
            ax = fig.add_subplot(2, 2, i)
            if continent=='global':
                df_continent=df
            else:

                df_continent = df[df['AI_classfication'] == continent]



            for product in variable_list:
                # print('=========')
                # print(product)
                # print(df_continent.columns.tolist())
                # exit()
                # T.print_head_n(df_continent)
                # exit()


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
                        # print(val)
                        # print(len(val))


                    vals_nonnan.append(list(val))

                        # exit()
                    # print(type(val))
                    # print(len(val))
                    # print(vals)

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


            plt.xlabel('year')

            plt.ylabel(f'relative change(%/year)')
            # plt.legend()

            plt.title(f'{continent}')
            plt.grid(which='major', alpha=0.5)
        # plt.legend()
        plt.show()

    def plot_anomaly_LAI_global(self):  ##### plot for 4 clusters

        df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes.df')
        print(len(df))
        df=self.clean_df(df)

        print(len(df))
        T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        # # western US  0–45° N, 105–125°W.


        ### Australia, 10–45° S, 110–155° E

        df=df[df['landcover_classfication']!='Cropland']
        # df=df[df['continent']=='Africa']
        df=df[df['Aridity']<0.65]

        # print(len(df))
        # exit()
        #


        color_list=['blue','green','red','orange','aqua','brown','cyan', 'black']
        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=2
        linewidth_list[2]=2

        fig = plt.figure()
        i = 1
        variable_list=['GLEAM_SMroot',]
        # variable_list=['VPD',]
        # variable_list=['CRU','GPCC']
        variable_list=['LAI4g','GIMMS_AVHRR_LAI','GIMMS3g_LAI']
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

        # exit()


        for product in variable_list:


            print(product)
            vals = df[product].tolist()



            # print(vals)

            vals_nonnan=[]
            for val in vals:
                if type(val)==float: ## only screening
                    continue
                if len(val) ==0:
                    continue
                val[val<-99]=np.nan

                vals_nonnan.append(list(val))
                # if not len(val) == 34:
                #     print(val)
                #     print(len(val))
                #     exit()
                # print(type(val))
                # print(len(val))
                # print(vals)

            ###### calculate mean
            vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
            vals_mean=np.nanmean(vals_mean,axis=0)
            val_std=np.nanstd(vals_mean,axis=0)

            # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
            plt.plot(vals_mean,label=product,color=color_list[variable_list.index(product)],linewidth=linewidth_list[variable_list.index(product)])
            # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])


            # plt.scatter(range(len(vals_mean)),vals_mean)
            # plt.text(0,vals_mean[0],product,fontsize=8)
        i=i+1

        plt.xticks(range(0, 40, 4), range(1982, 2021, 4), rotation=45)

        plt.ylim(-30,30)


        plt.xlabel('year')

        plt.ylabel(f'relative change(%/year)')
        # plt.legend()

        plt.title(f'global')
        plt.grid(which='major', alpha=0.5)
        plt.legend()
        plt.show()



    def plot_anomaly_LAI_one_figure(self):  ##### plot all_continent in one figure

        df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes.df')
        print(len(df))
        df=self.clean_df(df)

        print(len(df))
        T.print_head_n(df)
        # exit()
        df=df[df['row']>120]


        df=df[df['landcover_classfication']!='Cropland']
        # df=df[df['continent']=='Africa']
        df=df[df['Aridity']<0.65]


        color_list=['blue','green','red','orange','aqua','brown','cyan', 'black']
        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=2
        linewidth_list[2]=2


        i = 1
        variable_list=['GLEAM_SMroot',]
        # variable_list=['VPD',]
        # variable_list=['CRU','GPCC']
        variable_list=['LAI4g',]
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

        # exit()
        continent_dict = {}

        for continent in ['Africa', 'Asia', 'Australia', 'South_America', 'North_America']:

            if continent == 'North_America':
                df_continent = df[df['lon'] > -125]
                df_continent = df_continent[df_continent['lon'] < -105]
                df_continent = df_continent[df_continent['lat'] > 0]
                df_continent = df_continent[df_continent['lat'] < 45]
            else:
                df_continent = df[df['continent'] == continent]


            for product in variable_list:


                print(product)
                vals = df_continent[product].tolist()


                # print(vals)

                vals_nonnan=[]
                for val in vals:
                    if type(val)==float: ## only screening
                        continue
                    if len(val) ==0:
                        continue
                    val[val<-99]=np.nan

                    vals_nonnan.append(list(val))
                    # if not len(val) == 34:
                    #     print(val)
                    #     print(len(val))
                    #     exit()
                    # print(type(val))
                    # print(len(val))
                    # print(vals)

                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)
                val_std=np.nanstd(vals_mean,axis=0)
                continent_dict[continent] = vals_mean



        ## plot all continent in one figure
        fig = plt.figure()
        fig.set_size_inches(6, 6)

        for continent in ['Africa', 'Asia', 'Australia', 'South_America', 'North_America']:

            continent_vals=continent_dict[continent]
            plt.plot(continent_vals,label=continent)
            plt.xticks(range(0, 40, 4), range(1982, 2021, 4), rotation=45)
            plt.ylim(-20, 20)
            plt.xlabel('year')
            plt.ylabel(f'relative change(%/year)')


        plt.legend()
        plt.grid(which='major', alpha=0.5)
        plt.show()

    #
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
    def trend_bar(self): ## calculate average
        df = T.load_df(result_root + rf'Dataframe\relative_changes_trend\relative_changes_trend.df')
        # for col in df.columns:
        #     print(col)
        # exit()
        variable='GLEAM_SMroot'
        color_list=['black','red','green','blue','orange','aqua','brown','cyan', ]

        fig = plt.figure()
        flag = 1
        trend_list=[]

        global_trend_mean = df[f'{variable}_trend'].tolist()
        global_trend_mean = np.nanmean(global_trend_mean)
        trend_list.append(global_trend_mean)

        for continent in ['Africa', 'Asia', 'Australia', 'South_America', 'North_America']:
            if continent == 'North_America':
                df_region = df[df['lon'] > -125]
                df_region = df_region[df_region['lon'] < -105]
                df_region = df_region[df_region['lat'] > 0]
                df_region = df_region[df_region['lat'] < 45]
            else:


                df_region = df[df['continent'] == continent]


            ##calculate trend
            vals=df_region[f'{variable}_trend'].tolist()
            average_val=np.nanmean(vals)
            trend_list.append(average_val)



        df_new=pd.DataFrame(trend_list,index=['global','Africa', 'Asia', 'Australia', 'South_America', 'North_America'])

        # df_new.plot.bar()

        ## bar color
        plt.bar(df_new.index,df_new[0],color=color_list)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.ylabel('trend (%/year)')
        plt.show()


    def trend_percentage_bar(self): ## calculate average
        df = T.load_df(result_root + rf'Dataframe\relative_changes_trend\relative_changes_trend.df')
        df=df[df['landcover_classfication']!='Cropland']
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]

        ## wester US  0–45° N, 105–125°W.
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        df=df[df['continent']=='Africa']
        #
        # for col in df.columns:
        #     print(col)
        # exit()
        variable_list=['LAI4g','GPCC','GLEAM_SMroot',]
        color_list=['red','green','blue','orange']
        period_list=['1982_2020','1982_2001','2002_2020']

        fig = plt.figure()
        fig.set_size_inches(12, 6)
        flag = 1


        # landcover_list=['Evergreen','Deciduous','Shrub','Grass',]


        ##calculate trend
        ## significant trend, non-significant trend
        average_dic = {}
        for variable in variable_list:
            period_dic = {}

            for period in period_list:


                dic_landcover = {}
                average_list_significant_positive = []
                average_list_significant_negative = []

                for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
                    df_region = df[df['AI_classfication'] == region]


                    vals_trend = df_region[f'{variable}_{period}_trend'].tolist()
                    vals_p_value = df_region[f'{variable}_{period}_p_value'].tolist()
                    vals_trend = np.array(vals_trend)
                    vals_p_value = np.array(vals_p_value)

                    vals_trend[vals_p_value>0.1]=np.nan


                    ## #percentage of areas of positive and negative trend
                    positive_percentage = np.nansum(vals_trend>0)/len(vals_trend)
                    negative_percentage = np.nansum(vals_trend<0)/len(vals_trend)

                    average_list_significant_positive.append(positive_percentage)
                    average_list_significant_negative.append(negative_percentage)
                    dic_landcover[region] = [positive_percentage, negative_percentage]
                period_dic[period] = dic_landcover
            average_dic[variable] = period_dic

        for period in period_list:



            for variable in variable_list:
                ax = fig.add_subplot(3, 3, flag)

                for landcover in ['Arid', 'Semi-Arid', 'Sub-Humid']:
                    positive_percentage = average_dic[variable][period][region][0]
                    negative_percentage = average_dic[variable][period][region][1]
                    ## plot stacked bar
                    plt.bar(landcover,positive_percentage,color='green')
                    plt.bar(landcover,negative_percentage,bottom=positive_percentage,color='red')


                plt.title(f'{variable}_{period}')
                plt.ylabel('percentage')
                plt.ylim(0,1)
                plt.xticks(rotation=45)
                flag=flag+1
            plt.legend(['sig_positive','sig_negative'])
        plt.tight_layout()
        plt.show()



    def bin_trend(self): ## calculate average

        ### to bin trend and plot average bin to
        df = T.load_df(result_root + rf'Dataframe\relative_changes_trend\relative_changes_trend.df')
        df=df[df['landcover_classfication']!='Cropland']
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]

        ## plot percentile of trend
        ## plot hist
        LAI_trend=df['LAI4g_1982_2020_trend'].tolist()
        LAI_trend=np.array(LAI_trend)
        LAI_trend[LAI_trend>2]=np.nan
        LAI_trend[LAI_trend<-2]=np.nan
        LAI_trend=T.remove_np_nan(LAI_trend)
        threhold_list=[-2,-1.5, -1,-0.5,0,0.5,1,1.5, 2]
        ratio_list=[]
        x_ticks=[]

        for i in range(len(threhold_list)-1):
            threhold1=threhold_list[i]
            threhold2=threhold_list[i+1]
            df_threshold=df[(df['LAI4g_1982_2020_trend']>=threhold1) & (df['LAI4g_1982_2020_trend']<threhold2)]
            ratio=len(df_threshold)/len(df)*100
            ratio_list.append(ratio)
            x_ticks.append(f'{threhold1}_{threhold2}')
        ## plot bar
        plt.bar(x_ticks,ratio_list)
        plt.xticks(rotation=45)
        plt.ylabel('percentage')
        plt.figure()

        ##### plot map of trend
        spatial_dic={}

        for i in range(len(threhold_list) - 1):
            threhold1 = threhold_list[i]
            threhold2 = threhold_list[i + 1]
            df_threshold = df[(df['LAI4g_1982_2020_trend'] >= threhold1) & (df['LAI4g_1982_2020_trend'] < threhold2)]
            pix_list=df_threshold['pix'].tolist()
            for pix in pix_list:
                spatial_dic[pix]=(threhold_list[i+1]+threhold_list[i])/2
        spatial_arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(spatial_arr,cmap='RdBu',interpolation='nearest')
        plt.colorbar()
        plt.show()


        ## plot map











        pass
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
        df= T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes.df')

        z_values_list=self.product_list
        ### plot heatmap of aridity, SM trend, greening trend
        df=df[df['landcover_classfication']!='Cropland']
        aridity_bin=np.arange(0.05,0.7,0.1)
        CV_bin=np.arange(0.05,0.7,0.1)
        SM_bin=np.arange(-2,2,0.4)

        flag=  1
        ##plt figure
        fig = plt.figure()
        # ##figure size
        fig.set_size_inches(8, 10)
        for z_value in tqdm(z_values_list):
            df=df[df[f'{z_value}_trend']>-999]
            matrix = []
            z_value=z_value+'_trend'
            ax = fig.add_subplot(4, 4, flag)

            print(flag)

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

                    df_temp_i = df[df['CRU_trend'] >= y_left]
                    df_temp_i = df_temp_i[df_temp_i['CRU_trend'] < y_right]
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
                    xticks.append(f'{aridity_bin[i]:.1f}')
                    # xticks.append('')
            # plt.xticks(range(len(xticks)), xticks, rotation=45, ha='right')
            yticks = []
            ## y_ticks need to be reversed negative to positive
            for i in range(len(SM_bin)):
                if i % 2 == 0:
                    yticks.append(f'{SM_bin[::-1][i]:.3f}')
                else:
                    yticks.append('')
            # plt.yticks(range(len(yticks)), yticks)


            fig.text(0.5, 0.05, 'Aridity', ha='center')
            fig.text(0.05, 0.5, 'Precip trend (mm/year)', va='center', rotation='vertical')

            ###add x_tick and y_tick only left and bottom
            labels = [0.05,0.1,0.2,0.3,0.4,0.5,0.65]
            labels_str = [f'{i:.2f}' for i in labels]
            if flag in [13,14,15,16]:
                plt.xticks(np.array(range(len(xticks)))-0.5, labels_str, rotation=90)
            else: ##remove x_tick
                plt.xticks([])
            #####y tick
            if flag in [1,5,9,13]:
                plt.yticks(range(len(yticks)), yticks)
            else:
                plt.yticks([])
            flag = flag + 1

            ##z value name.split by _ and drop the last one join rest of them
            z_val_name = '_'.join(z_value.split('_')[:-1])
            plt.title(z_val_name)
            ## color bar scale from -0.1 to 0.1
            plt.clim(-0.5, 0.5)
            cax = fig.add_axes([0.2, 0.03, 0.6, 0.02])
            fig.colorbar(ax.imshow(matrix, vmin=-0.5, vmax=0.5, cmap='RdBu', interpolation='nearest'), cax=cax,
                             orientation='horizontal')

        plt.show()



        # plt.savefig(self.outdir + f'{region}_{z_val_name}.pdf', dpi=300)
        # plt.close()


class plt_moving_dataframe():
    def run(self):


        # self.plot_moving_window_area_bar()
        self.plot_CV_LAI()
        # self.plot_CV_LAI_all_together()
        # self.plot_moving_window_area_bar_trend_level()
        # self.plot_moving_window_area_bar_trend_level_for_regions()
        # self.plot_moving_window_area_bar_trend_level_all()
        # self.plot_moving_window_average()
        # self.plot_multiregression_moving_window()
        # self.plot_multiregression_moving_window_pdf()
        # self.plt_CO2_function_of_VPD()
        pass



    def plot_moving_window_average(self): ## based on yearly data
        df=result_root+rf'Dataframe\moving_window_CV\\moving_window_CV.df'
        df=T.load_df(df)
        T.print_head_n(df)
        print(len(df))
        ## define western US 0–45° N, 105–125°W
        df=df[df['lon']>-125]
        df=df[df['lon']<-105]
        df=df[df['lat']>0]
        df=df[df['lat']<45]
        # print(len(df))

        df=df[df['landcover_classfication']!='Cropland']
        df=df[df['continent']=='North_America']
        # df = df.dropna(subset=['LAI4g_trend_01'])
        fig = plt.figure()
        fig.set_size_inches(9, 3)
        flag=1



        for region in ['North_America','South_America','Australia','Africa','Asia']:
            ax = fig.add_subplot(2, 3, flag)
            df_region = df[df['continent'] == region]


            average_list=[]
            for i in range(1, 25):
                print(f'LAI4g_trend_{i:02d}')
                ##### tiwen
                # df_region = df_region[df_region[f'LAI4g_p_value_{i:02d}'] < 0.1]

                vals=df_region[f'LAI4g_trend_{i:02d}'].tolist()
                vals=np.array(vals)
                vals[vals<-999]=np.nan
                mean=np.nanmean(vals)
                print(mean)
                average_list.append(mean)
            ax.plot(average_list,color='green',label='LAI4g')
            ax.set_title(f'{region}_count_{len(df_region)}')
            ax.set_xlabel('window size (year)')
            ax.set_ylabel('trend (%/year)')
            ax.set_xticks(range(0, 24, 2), )

            # xticks = []
            # for i in range(1, 25):
            #     if i % 2 == 0:
            #         xticks.append(f'{1982 + 15 * (i - 1)}-{1982 + 15  * i - 1}')
            #     else:
            #         xticks.append(f'{1982 + 15 * (i - 1)}-{1982 + 15  * i - 1}')


            ax.set_ylim(-5,5)
            plt.xticks(rotation=45, ha='right')
            flag=flag+1
        plt.legend()
        plt.show()


    def plot_CV_LAI(self):  ##### plot based on all years together

        df = T.load_df(result_root + rf'Dataframe\moving_window_CV\\moving_window_CV.df')
        print(len(df))
        df=df[df['landcover_classfication']!='Cropland']

        print(len(df))
        T.print_head_n(df)
        # exit()
        df=df[df['row']>120]

        df=df[df['Aridity']<0.65]



        #create color list with one green and another 14 are grey

        # color_list=['grey']*16
        # color_list[0]='green'
        # color_list[1] = 'red'
        # color_list[2]='blue'
        color_list=['blue','green','red','orange','aqua','brown','cyan', 'black']
        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=2
        linewidth_list[2]=2

        fig = plt.figure()
        i = 1
        # variable_list=['GLEAM_SMroot',]
        # variable_list=['VPD',]
        variable_list=['GPCC_trend']
        # variable_list=['LAI4g',]

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

        # exit()

        for continent in ['Africa', 'Asia', 'Australia', 'South_America', 'North_America']:
            ax = fig.add_subplot(2, 3, i)
            if continent == 'North_America':
                df_continent = df[df['lon'] > -125]
                df_continent = df_continent[df_continent['lon'] < -105]
                df_continent = df_continent[df_continent['lat'] > 0]
                df_continent = df_continent[df_continent['lat'] < 45]
            else:
                df_continent = df[df['continent'] == continent]


            for product in variable_list:


                print(product)
                vals = df_continent[product].tolist()


                vals_nonnan=[]
                for val in vals:

                    val=np.array(val)

                    if type(val)==float: ## only screening
                        continue
                    val[val<-99]=np.nan
                    # print(len(val))
                    if not len(val) == 24:
                        continue

                    vals_nonnan.append(list(val))
                    # if not len(val) == 34:
                    #     print(val)
                    #     print(len(val))
                    #     exit()
                    # print(type(val))
                    # print(len(val))
                    # print(vals)

                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)
                val_std=np.nanstd(vals_mean,axis=0)

                # plt.plot(vals_mean,label=product,color=color_list[self.product_list.index(product)],linewidth=linewidth_list[self.product_list.index(product)])
                plt.plot(vals_mean,label=product,color=color_list[variable_list.index(product)],linewidth=linewidth_list[variable_list.index(product)])
                # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])


                # plt.scatter(range(len(vals_mean)),vals_mean)
                # plt.text(0,vals_mean[0],product,fontsize=8)
            i=i+1

            ax.set_xticks(range(0, 24, 4))
            window_size = 15

            # set xticks with 1982-1997, 1998-2013,.. 2014-2020
            year_range = range(1982, 2021)
            year_range_str = []
            for year in year_range:

                start_year = year
                end_year = year + window_size - 1
                if end_year > 2020:
                    break
                year_range_str.append(f'{start_year}-{end_year}')
            plt.xticks(range(len(year_range_str))[::4], year_range_str[::4], rotation=45, ha='right')

            # plt.ylim(-0.2, 0.2)



            plt.xlabel('year')

            plt.ylabel(f'relative change(%/year)')
            # plt.legend()

            plt.title(f'{continent}')
            plt.grid(which='major', alpha=0.5)
        plt.legend()
        plt.show()

    def plot_CV_LAI_all_together(self):  ##### plot based on all years together

        df = T.load_df(result_root + rf'Dataframe\moving_window_CV\\moving_window_CV.df')
        print(len(df))
        df=df[df['landcover_classfication']!='Cropland']

        print(len(df))
        T.print_head_n(df)
        # exit()
        df=df[df['row']>120]

        df=df[df['Aridity']<0.65]



        #create color list with one green and another 14 are grey


        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=2
        linewidth_list[2]=2

        fig = plt.figure()
        i = 1

        variable_list=['LAI4g',]

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

        # exit()
        dic_CV = {}

        for continent in ['Africa', 'Asia', 'Australia', 'South_America', 'North_America']:

            if continent == 'North_America':
                df_continent = df[df['lon'] > -125]
                df_continent = df_continent[df_continent['lon'] < -105]
                df_continent = df_continent[df_continent['lat'] > 0]
                df_continent = df_continent[df_continent['lat'] < 45]
            else:
                df_continent = df[df['continent'] == continent]


            for product in variable_list:


                print(product)
                vals = df_continent[product].tolist()


                vals_nonnan=[]
                for val in vals:

                    val=np.array(val)

                    if type(val)==float: ## only screening
                        continue
                    val[val<-99]=np.nan
                    # print(len(val))
                    if not len(val) == 24:
                        continue

                    vals_nonnan.append(list(val))
                    # if not len(val) == 34:
                    #     print(val)
                    #     print(len(val))
                    #     exit()
                    # print(type(val))
                    # print(len(val))
                    # print(vals)

                ###### calculate mean
                vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
                vals_mean=np.nanmean(vals_mean,axis=0)
                val_std=np.nanstd(vals_mean,axis=0)

                dic_CV[continent] = vals_mean
        ##### global
        vals=df[variable_list[0]].tolist()
        vals_nonnan=[]
        for val in vals:
            val=np.array(val)
            if type(val)==float: ## only screening
                continue
            val[val<-99]=np.nan
            if not len(val) == 24:
                continue
            vals_nonnan.append(list(val))
        vals_mean=np.array(vals_nonnan)## axis=0, mean of each row  竖着加
        vals_mean=np.nanmean(vals_mean,axis=0)
        dic_CV['Global'] = vals_mean

        ##plot
        color_list=['green', 'lime', 'orange', 'red', 'blue', 'black', 'grey', 'yellow', 'pink', 'purple']
        ##set line width



        df_new = pd.DataFrame(dic_CV)
        df_new.plot(color=color_list,linewidth=1.5)

        plt.xticks(range(0, 24, 4), )
        ## set x_tick
        year_range= range(1982, 2021)
        year_range_str = []
        for year in year_range:
            start_year = year
            end_year = year + 15 - 1
            if end_year > 2020:
                break
            year_range_str.append(f'{start_year}-{end_year}')

        plt.xticks(range(len(year_range_str))[::4], year_range_str[::4], rotation=45, ha='right')
        plt.xlabel('year')
        plt.ylabel('CV')
        plt.grid(which='major', alpha=0.5)
        plt.tight_layout()

        ### plt continent on the line



        plt.show()




    def plot_moving_window_area_bar(self):  ##plot moving window bar
        df=result_root+rf'Dataframe\extract_moving_window_trend_relative_change\\extract_moving_window_trend_relative_change.df'
        df=T.load_df(df)
        T.print_head_n(df)
        print(len(df))
        ## define western US 0–45° N, 105–125°W
        df=df[df['lon']>-125]
        df=df[df['lon']<-105]
        df=df[df['lat']>0]
        df=df[df['lat']<45]
        df=df[df['continent']=='North_America']
        df=df.dropna(subset=['LAI4g_trend_01'])
       # print(len(df))
       # exit()
       #  global_land_tif='D:\Project3\Data\Base_data\\MAP.tif'
       #  ## df to spatial dic
       #  DIC_and_TIF(pixelsize=0.25).plot_df_spatial_pix(df,global_land_tif)
       #  plt.show()



        # df=df.dropna(subset=['LAI4g_trend_01'])


        df=df[df['landcover_classfication']!='Cropland']

        color_list = ['green', 'lime', 'orange', 'red', 'blue', 'black', 'grey', 'yellow', 'pink', 'purple']

        fig = plt.figure()
        ii = 1
        # for landcover in ['Evergreen', 'Deciduous', 'Mixed','Grass','Shrub','Cropland']:
        # for continent in [ 'North_America', ]:
        #     df_continent = df[df['continent'] == continent]

        for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
            df_region = df[df['AI_classfication'] == region]


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

                sig_greening_area = 0
                sig_browning_area = 0
                no_change_area = 0
                non_sig_greening_area = 0
                non_sig_browning_area = 0


                for i, row in tqdm(df_region.iterrows(), total=len(df_region)):
                    pix = row.pix
                    trend = row[f'LAI4g_trend_{window:02d}']
                    p_value = row[f'LAI4g_p_value_{window:02d}']
                    ## significant greening, significant browning, non-significant greening, non-significant browning
                    if p_value < 0.1:
                        if trend > 0:
                            sig_greening_area += 1
                        else:
                            sig_browning_area += 1
                    elif p_value > 0.1:
                        if trend > 0:
                            non_sig_greening_area += 1
                        else:
                            non_sig_browning_area += 1
                    else:
                        no_change_area += 1

                total_area = sig_greening_area + sig_browning_area + no_change_area + non_sig_greening_area + non_sig_browning_area
                if total_area == 0:
                    continue

                sig_greening_area = sig_greening_area / total_area * 100

                sig_browning_area = sig_browning_area / total_area * 100
                no_change_area = no_change_area / total_area * 100
                non_sig_greening_area = non_sig_greening_area / total_area * 100
                non_sig_browning_area = non_sig_browning_area / total_area * 100
                dic[window] = [sig_greening_area, non_sig_greening_area, non_sig_browning_area,  sig_browning_area,]
            df_new = pd.DataFrame(dic)
            df_new_T = df_new.T
        ## if no number in the window, continu


            df_new_T.plot.bar(ax=ax, stacked=True, color=color_list, legend=False)
            plt.title(f'{region}_count_{len(df_region)}')
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
            plt.ylim(0,100)
            plt.tight_layout()

            ii = ii + 1
        plt.legend(['sig_greening', 'non_sig_greening', 'non_sig_browning', 'sig_browning'])

        plt.tight_layout()
        plt.show()

    pass


    def plot_moving_window_area_bar_trend_level(self):  ##plot moving window bar
        df=result_root+rf'Dataframe\extract_moving_window_trend_relative_change\\extract_moving_window_trend_relative_change.df'
        df=T.load_df(df)
        T.print_head_n(df)
        print(len(df))
        ## define western US 0–45° N, 105–125°W

        df=df.dropna(subset=['LAI4g_trend_01'])
       # print(len(df))
       # exit()
       #  global_land_tif='D:\Project3\Data\Base_data\\MAP.tif'
       #  ## df to spatial dic
       #  DIC_and_TIF(pixelsize=0.25).plot_df_spatial_pix(df,global_land_tif)
       #  plt.show()



        # df=df.dropna(subset=['LAI4g_trend_01'])


        df=df[df['landcover_classfication']!='Cropland']

        color_list = ['yellowgreen', 'lime', 'green','lightseagreen', 'yellow', 'orange', 'red', 'brown' ]

        fig = plt.figure()
        ii = 1
        # for landcover in ['Evergreen', 'Deciduous', 'Mixed','Grass','Shrub','Cropland']:
        for continent in [ 'North_America',  'Asia', 'Australia', 'Africa', 'South_America']:
            if continent=='North_America':
                df_continent=df[df['continent']=='North_America']
                df_continent=df_continent[df_continent['lon']<-105]
                df_continent=df_continent[df_continent['lat']>0]
                df_continent=df_continent[df_continent['lat']<45]
            else:
                df_continent = df[df['continent'] == continent]

            ax = fig.add_subplot(2, 3, ii)
            flag = 0

            window_list = []
            dic={}

            for i in range(1, 25):
                window_list.append(i)
            print(window_list)

            for window in tqdm(window_list):  # 构造字典的键值，并且字典的键：值初始化
                dic[window] = []

            ## plt moving window bar based on trend level and p_value

            for window in window_list:

                slight_greening_area = 0 ### 0-0.5
                moderate_greening_area = 0 ## 0.5-1
                severe_greening_area = 0 # >1-1.5
                extreme_greening_area = 0 ## >1.5
                slight_browning_area = 0
                moderate_browning_area = 0
                severe_browning_area = 0
                extreme_browning_area = 0
                no_change_area = 0


                for i, row in tqdm(df_continent.iterrows(), total=len(df_continent)):
                    pix = row.pix
                    trend = row[f'LAI4g_trend_{window:02d}']
                    p_value = row[f'LAI4g_p_value_{window:02d}']
                    ## significant greening, significant browning, non-significant greening, non-significant browning
                    if p_value < 0.1:
                        if trend > 0 and trend < 0.5:
                            slight_greening_area += 1
                        elif trend >= 0.5 and trend < 1:
                            moderate_greening_area += 1
                        elif trend >= 1 and trend < 1.5:
                            severe_greening_area += 1
                        elif trend >= 1.5:
                            extreme_greening_area += 1
                        elif trend < 0 and trend > -0.5:
                            slight_browning_area += 1
                        elif trend <= -0.5 and trend > -1:
                            moderate_browning_area += 1
                        elif trend <= -1 and trend > -1.5:
                            severe_browning_area += 1
                        elif trend <= -1.5:
                            extreme_browning_area += 1
                    else:
                        no_change_area += 1


                total_area = slight_greening_area + moderate_greening_area + severe_greening_area + extreme_greening_area + slight_browning_area + moderate_browning_area + severe_browning_area + extreme_browning_area + no_change_area
                if total_area == 0:
                    continue

                slight_greening_area = slight_greening_area / total_area * 100
                moderate_greening_area = moderate_greening_area / total_area * 100
                significant_greening_area = severe_greening_area / total_area * 100
                extreme_greening_area = extreme_greening_area / total_area * 100
                slight_browning_area = slight_browning_area / total_area * 100
                moderate_browning_area = moderate_browning_area / total_area * 100
                significant_browning_area = severe_browning_area / total_area * 100
                extreme_browning_area = extreme_browning_area / total_area * 100
                no_change_area = no_change_area / total_area * 100
                dic[window] = [slight_greening_area, moderate_greening_area, significant_greening_area, extreme_greening_area, slight_browning_area, moderate_browning_area, significant_browning_area, extreme_browning_area]

            df_new = pd.DataFrame(dic)
            df_new_T = df_new.T
        ## if no number in the window, continu


            df_new_T.plot.bar(ax=ax, stacked=True, color=color_list, legend=False)
            plt.title(f'{continent}_count_{len(df_continent)}')
            plt.xlabel('window size (year)')
            xticks= []
            window_size=15

            # set xticks with 1982-1997, 1998-2013,.. 2014-2020
            year_range=range(1982,2021)
            year_range_str=[]
            for year in year_range:

                start_year=year
                end_year=year+window_size-1
                if end_year>2020:
                    break
                year_range_str.append(f'{start_year}-{end_year}')
            plt.xticks(range(len(year_range_str))[::4], year_range_str[::4], rotation=45, ha='right')



            plt.ylabel('percentage')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0,90)
            plt.tight_layout()

            ii = ii + 1
        # plt.legend(['slight_greening', 'moderate_greening', 'significant_greening', 'extreme_greening', 'slight_browning', 'moderate_browning', 'significant_browning', 'extreme_browning', ])

        plt.tight_layout()
        plt.show()

    def plot_moving_window_area_bar_trend_level_for_regions(self):  ##plot moving window bar
        df=result_root+rf'Dataframe\extract_moving_window_trend_relative_change\\extract_moving_window_trend_relative_change.df'
        df=T.load_df(df)
        T.print_head_n(df)
        print(len(df))
        ## define western US 0–45° N, 105–125°W

        df=df.dropna(subset=['LAI4g_trend_01'])
       # print(len(df))
       # exit()
       #  global_land_tif='D:\Project3\Data\Base_data\\MAP.tif'
       #  ## df to spatial dic
       #  DIC_and_TIF(pixelsize=0.25).plot_df_spatial_pix(df,global_land_tif)
       #  plt.show()



        # df=df.dropna(subset=['LAI4g_trend_01'])


        df=df[df['landcover_classfication']!='Cropland']

        color_list = ['yellowgreen', 'lime', 'green','lightseagreen', 'yellow', 'orange', 'red', 'brown' ]



        # for landcover in ['Evergreen', 'Deciduous', 'Mixed','Grass','Shrub','Cropland']:
        for continent in [ 'North_America',  'Asia', 'Australia', 'Africa', 'South_America']:
            if continent=='North_America':
                df_continent=df[df['continent']=='North_America']
                df_continent=df_continent[df_continent['lon']<-105]
                df_continent=df_continent[df_continent['lat']>0]
                df_continent=df_continent[df_continent['lat']<45]
            else:
                df_continent = df[df['continent'] == continent]
            fig = plt.figure()
            ii = 1
            for region in ['Arid', 'Semi-Arid', 'Sub-Humid']:
                df_region = df_continent[df_continent['AI_classfication'] == region]


                ax = fig.add_subplot(2, 3, ii)


                window_list = []
                dic={}

                for i in range(1, 25):
                    window_list.append(i)
                print(window_list)

                for window in tqdm(window_list):  # 构造字典的键值，并且字典的键：值初始化
                    dic[window] = []

                ## plt moving window bar based on trend level and p_value

                for window in window_list:

                    slight_greening_area = 0 ### 0-0.5
                    moderate_greening_area = 0 ## 0.5-1
                    severe_greening_area = 0 # >1-1.5
                    extreme_greening_area = 0 ## >1.5
                    slight_browning_area = 0
                    moderate_browning_area = 0
                    severe_browning_area = 0
                    extreme_browning_area = 0
                    no_change_area = 0


                    for i, row in tqdm(df_region.iterrows(), total=len(df_continent)):
                        pix = row.pix
                        trend = row[f'LAI4g_trend_{window:02d}']
                        p_value = row[f'LAI4g_p_value_{window:02d}']
                        ## significant greening, significant browning, non-significant greening, non-significant browning
                        if p_value < 0.1:
                            if trend > 0 and trend < 0.5:
                                slight_greening_area += 1
                            elif trend >= 0.5 and trend < 1:
                                moderate_greening_area += 1
                            elif trend >= 1 and trend < 1.5:
                                severe_greening_area += 1
                            elif trend >= 1.5:
                                extreme_greening_area += 1
                            elif trend < 0 and trend > -0.5:
                                slight_browning_area += 1
                            elif trend <= -0.5 and trend > -1:
                                moderate_browning_area += 1
                            elif trend <= -1 and trend > -1.5:
                                severe_browning_area += 1
                            elif trend <= -1.5:
                                extreme_browning_area += 1
                        else:
                            no_change_area += 1


                    total_area = slight_greening_area + moderate_greening_area + severe_greening_area + extreme_greening_area + slight_browning_area + moderate_browning_area + severe_browning_area + extreme_browning_area + no_change_area
                    if total_area == 0:
                        continue

                    slight_greening_area = slight_greening_area / total_area * 100
                    moderate_greening_area = moderate_greening_area / total_area * 100
                    significant_greening_area = severe_greening_area / total_area * 100
                    extreme_greening_area = extreme_greening_area / total_area * 100
                    slight_browning_area = slight_browning_area / total_area * 100
                    moderate_browning_area = moderate_browning_area / total_area * 100
                    significant_browning_area = severe_browning_area / total_area * 100
                    extreme_browning_area = extreme_browning_area / total_area * 100
                    no_change_area = no_change_area / total_area * 100
                    dic[window] = [slight_greening_area, moderate_greening_area, significant_greening_area, extreme_greening_area, slight_browning_area, moderate_browning_area, significant_browning_area, extreme_browning_area]

                df_new = pd.DataFrame(dic)
                df_new_T = df_new.T
            ## if no number in the window, continu


                df_new_T.plot.bar(ax=ax, stacked=True, color=color_list, legend=False)
                plt.title(f'{continent}_{region}_count_{len(df_region)}')
                plt.xlabel('window size (year)')
                xticks= []
                window_size=15

                # set xticks with 1982-1997, 1998-2013,.. 2014-2020
                year_range=range(1982,2021)
                year_range_str=[]
                for year in year_range:

                    start_year=year
                    end_year=year+window_size-1
                    if end_year>2020:
                        break
                    year_range_str.append(f'{start_year}-{end_year}')
                plt.xticks(range(len(year_range_str))[::4], year_range_str[::4], rotation=45, ha='right')



                plt.ylabel('percentage')
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0,90)
                plt.tight_layout()

                ii = ii + 1
            # plt.legend(['slight_greening', 'moderate_greening', 'significant_greening', 'extreme_greening', 'slight_browning', 'moderate_browning', 'significant_browning', 'extreme_browning', ])

            plt.tight_layout()
            plt.show()

        pass




    def plot_moving_window_area_bar_trend_level_all(self):  ##plot moving window bar
        df=result_root+rf'Dataframe\extract_moving_window_trend_relative_change\\extract_moving_window_trend_relative_change.df'
        df=T.load_df(df)
        T.print_head_n(df)
        print(len(df))
        ## define western US 0–45° N, 105–125°W

        df=df.dropna(subset=['LAI4g_trend_01'])
       # print(len(df))
       # exit()
       #  global_land_tif='D:\Project3\Data\Base_data\\MAP.tif'
       #  ## df to spatial dic
       #  DIC_and_TIF(pixelsize=0.25).plot_df_spatial_pix(df,global_land_tif)
       #  plt.show()


        df=df[df['landcover_classfication']!='Cropland']

        color_list = ['yellowgreen', 'lime', 'green','lightseagreen', 'yellow', 'orange', 'red', 'brown' ]

        fig = plt.figure()
        ii = 1

        ax = fig.add_subplot(2, 3, ii)
        flag = 0

        window_list = []
        dic={}

        for i in range(1, 25):
            window_list.append(i)
        print(window_list)

        for window in tqdm(window_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[window] = []

        ## plt moving window bar based on trend level and p_value

        for window in window_list:

            slight_greening_area = 0 ### 0-0.5
            moderate_greening_area = 0 ## 0.5-1
            severe_greening_area = 0 # >1-1.5
            extreme_greening_area = 0 ## >1.5
            slight_browning_area = 0
            moderate_browning_area = 0
            severe_browning_area = 0
            extreme_browning_area = 0
            no_change_area = 0


            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row.pix
                trend = row[f'LAI4g_trend_{window:02d}']
                p_value = row[f'LAI4g_p_value_{window:02d}']
                ## significant greening, significant browning, non-significant greening, non-significant browning
                if p_value < 0.1:
                    if trend > 0 and trend < 0.5:
                        slight_greening_area += 1
                    elif trend >= 0.5 and trend < 1:
                        moderate_greening_area += 1
                    elif trend >= 1 and trend < 1.5:
                        severe_greening_area += 1
                    elif trend >= 1.5:
                        extreme_greening_area += 1
                    elif trend < 0 and trend > -0.5:
                        slight_browning_area += 1
                    elif trend <= -0.5 and trend > -1:
                        moderate_browning_area += 1
                    elif trend <= -1 and trend > -1.5:
                        severe_browning_area += 1
                    elif trend <= -1.5:
                        extreme_browning_area += 1
                else:
                    no_change_area += 1


            total_area = slight_greening_area + moderate_greening_area + severe_greening_area + extreme_greening_area + slight_browning_area + moderate_browning_area + severe_browning_area + extreme_browning_area + no_change_area
            if total_area == 0:
                continue

            slight_greening_area = slight_greening_area / total_area * 100
            moderate_greening_area = moderate_greening_area / total_area * 100
            significant_greening_area = severe_greening_area / total_area * 100
            extreme_greening_area = extreme_greening_area / total_area * 100
            slight_browning_area = slight_browning_area / total_area * 100
            moderate_browning_area = moderate_browning_area / total_area * 100
            significant_browning_area = severe_browning_area / total_area * 100
            extreme_browning_area = extreme_browning_area / total_area * 100
            no_change_area = no_change_area / total_area * 100
            dic[window] = [slight_greening_area, moderate_greening_area, significant_greening_area, extreme_greening_area, slight_browning_area, moderate_browning_area, significant_browning_area, extreme_browning_area]

        df_new = pd.DataFrame(dic)
        df_new_T = df_new.T
    ## if no number in the window, continu


        df_new_T.plot.bar(ax=ax, stacked=True, color=color_list, legend=False)
        plt.title(f'Global_count_{len(df)}')
        plt.xlabel('window size (year)')
        xticks= []
        window_size=15

        # set xticks with 1982-1997, 1998-2013,.. 2014-2020
        year_range=range(1982,2021)
        year_range_str=[]
        for year in year_range:

            start_year=year
            end_year=year+window_size-1
            if end_year>2020:
                break
            year_range_str.append(f'{start_year}-{end_year}')
        plt.xticks(range(len(year_range_str))[::4], year_range_str[::4], rotation=45, ha='right')


        plt.ylabel('percentage')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0,90)
        plt.tight_layout()

        ii = ii + 1
        plt.legend(['slight_greening', 'moderate_greening', 'significant_greening', 'extreme_greening', 'slight_browning', 'moderate_browning', 'significant_browning', 'extreme_browning', ])

        plt.tight_layout()
        plt.show()


    def plot_multiregression_moving_window_pdf(self):
        ## here I would like to plot distribution of the multi-regression coefficient for each pixel

        df = T.load_df(result_root + rf'Dataframe\moving_window_multiregression_anomaly\\moving_window_multiregression_anomaly.df')
        T.print_head_n(df)
        print(len(df))
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['Aridity'] < 0.65]
        df=df[df['LC_max']<20]
        df = df.dropna(subset=['window00_CO2_LAI4g'])
        continent_list = ['Global','Africa', 'Asia', 'Australia', 'South_America', 'North_America']
        all_continent_dict = {}
        for continent in continent_list:
            if continent == 'North_America':
                df_continent = df[df['lon'] > -125]
                df_continent = df_continent[df_continent['lon'] < -105]
                df_continent = df_continent[df_continent['lat'] > 0]
                df_continent = df_continent[df_continent['lat'] < 45]
            elif continent == 'Global':
                df_continent = df
            else:
                df_continent = df[df['continent'] == continent]


            window_vals_list=[]
            for window in range(0, 24):
                window=format(window,'02d')
                print(window)

                vals=df_continent[f'window{window}_CO2_LAI4g'].tolist()
                window_vals_list.append(vals)
            all_continent_dict[continent]=window_vals_list


        fig = plt.figure()
        ii = 1
            ## plot
        for continent in all_continent_dict:
            window_vals_list=all_continent_dict[continent]

            ax = fig.add_subplot(2, 3, ii)
            color_list = T.gen_colors(len(window_vals_list))
            flag = 0
            for window_vals in window_vals_list:

                vals = window_vals
                ## plt distribution of the multi-regression coefficient for each pixel
                vals = np.array(vals)
                vals = vals[~np.isnan(vals)]
                # plt.hist(vals,bins=100, alpha=0.5, label=f'window{window}')
                x_i,y_i = Plot().plot_hist_smooth(vals, bins=100,range=(-0.01,0.01),alpha=0)
                plt.plot(x_i, y_i, label=f'{flag}',c=color_list[flag])
                flag += 1
            plt.legend()
            plt.show()
                # sns.kdeplot(vals, ax=ax, label=f'window{window}')
            plt.title(f'{continent}')
            plt.xlabel('multi-regression coefficient')
            plt.ylabel('density')
            # plt.xlim(-0.5, 0.5)

            plt.tight_layout()
            ii = ii + 1
            plt.show()


    def plt_CO2_function_of_VPD(self):
        ## bin plot
        df=T.load_df(result_root+rf'Dataframe\moving_window_multiregression_anomaly\\moving_window_multiregression_anomaly.df')
        T.print_head_n(df)
        print(len(df))
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 20]
        df = df.dropna(subset=['CO2_LAI4g'])
        # continent_list = ['Global', 'Africa', 'Asia', 'Australia', 'South_America', 'North_America']
        continent_list = ['Global']
        vals_sensitivity=df['CO2_LAI4g_1982_2020_long_term_average'].tolist()
        vals_VPD=df['VPD_trend'].tolist()
        plt.hist(vals_VPD,bins=100)

        plt.show()
        plt.hist(vals_sensitivity,bins=100)
        plt.show()
        VPD_bins=np.linspace(0,0.015,21)
        ### bin plot

        vals_list = []
        err_list = []
        pixel_number_list= []
        df_group1, bins_list_str1 = T.df_bin(df, 'VPD_trend', VPD_bins)

        ## plot bin also add pixel number as bar height double y axis

        for name1, df_group_i1 in df_group1:
            vals = df_group_i1['CO2_LAI4g_1982_2020_long_term_average'].tolist()
            vals = np.array(vals, dtype=float)
            pixel_number=len(vals)
            pixel_number_list.append(pixel_number)

            #vals = vals*100
            val_mean = np.nanmean(vals)
            err, _, _ = T.uncertainty_err(vals)
            err_list.append(err)

            print(name1, val_mean)
            vals_list.append(val_mean)

        ## plot double y axis y 1 is CO2 sensitivity to LAI, y2 is pixel number
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(bins_list_str1, vals_list, 'g-')
        plt.fill_between(bins_list_str1, np.array(vals_list) - err_list, np.array(vals_list) + err_list, alpha=0.3)
        ax2.bar(bins_list_str1, pixel_number_list, width=0.001, alpha=0.3)
        ax1.set_xlabel('VPD trend')
        ax1.set_ylabel('CO2 sensitivity to LAI (%/100ppm)', color='g')
        ax2.set_ylabel('pixel number', color='b')
        plt.show()



        # plt.plot(bins_list_str1, vals_list)
        # err_list = np.array(err_list)
        # plt.fill_between(bins_list_str1, np.array(vals_list) - err_list, np.array(vals_list) + err_list, alpha=0.3)
        # plt.bar(bins_list_str1, pixel_number_list, width=0.001, alpha=0.3)
        # plt.xlabel('VPD trend')
        # plt.xticks(rotation=45, ha='right')
        # plt.ylabel('CO2 sensitivity to LAI (m2/m2/100ppm)')
        #
        # plt.show()


class check_data():
    def run (self):
        self.plot_sptial()

        # self.testrobinson()
        # self.plot_time_series()
        # self.plot_bar()


        pass
    def plot_sptial(self):

        fdir = rf'D:\Project3\Data\monthly_data\GPCC\dic\\'


        # dic=T.load_npy(fdir)
        dic=T.load_npy_dir(fdir)

            # for f in os.listdir(fdir):
            #     if not f.endswith(('.npy')):
            #         continue
            #
            #     dic_i=T.load_npy(fdir+f)
            #     dic.update(dic_i)

        len_dic={}

        for pix in dic:
            r,c=pix
            # (r,c)=(817,444)
            # if not r==444:
            #     continue
            # if not c==817:
            #     continue
            # if r<480:
            #     continue
            vals=dic[pix]
            # date_list = []
            # date_base = datetime.datetime(1982, 1, 1)
            # for i in range(len(vals)):
            #     # date_list.append(date_base + datetime.timedelta(months=i))
            #     date_obj = T.month_index_to_date_obj(i, date_base,)
            #     date_list.append(date_obj)
            # # exit()
            #
            # plt.plot(date_list,vals)
            # plt.title(pix)
            # plt.figure()
            # tif = r"D:\Project3\Data\monthly_data\CRU\unify\19900816.tif"
            # arr = DIC_and_TIF().spatial_tif_to_arr(tif)
            # arr[arr>999999] = np.nan
            # plt.imshow(arr,cmap='RdBu',interpolation='nearest',vmax=2000)
            # plt.colorbar()
            # plt.scatter(pix[1],pix[0],c='k',s=200)
            # plt.show()



            if len(vals)<1:
                continue
            if np.isnan(np.nanmean(vals)):
                continue


            # plt.plot(vals)
            # plt.show()


            # len_dic[pix]=np.nanmean(vals)
            # len_dic[pix]=np.nanstd(vals)
            vals=np.array(vals)

            vals=vals[~np.isnan(vals)]

            len_dic[pix] = len(vals)
        arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(len_dic)


        plt.imshow(arr,cmap='RdBu',interpolation='nearest',vmin=460,vmax=468)
        plt.colorbar()
        plt.title(fdir)
        plt.show()
    def testrobinson(self):
        fdir=rf'D:\Project3\Result\multi_regression_moving_window\window15_anomaly_GPCC\trend_analysis\100mm_unit\\'
        period='1982_2020'
        fpath_p_value=result_root+rf'D:\Project3\Result\multi_regression_moving_window\window15_anomaly_GPCC\trend_analysis\100mm_unit\\\\GPCC_LAI4g_p_value.tif'
        temp_root=r'trend_analysis\anomaly\\temp_root\\'
        T.mk_dir(temp_root,force=True)

        # f =  rf'D:\Project3\Data\Base_data\dryland_AI.tif\dryland.tif'
        for f in os.listdir(fdir):
            if not f.endswith('GPCC_LAI4g_trend.tif'):
                continue




            fname=f.split('.')[0]
            print(fname)


            m,ret=Plot().plot_Robinson(fdir+f, vmin=-5,vmax=5,is_discrete=True,colormap_n=7,)
            # Plot().plot_Robinson_significance_scatter(m,fpath_p_value,temp_root,0.05)


            fname='ERA5 Precipitation CV (%)'
            plt.title(f'{fname}')

            plt.show()



    def plot_time_series(self):
        f=rf'D:\Project3\Result\zscore\GPCC.npy'
        f2=rf'D:\Project3\Result\anomaly\OBS_extend\GPCC.npy'

        # f=result_root + rf'extract_GS\OBS_LAI_extend\\Tmax.npy'
        # f=data_root + rf'monthly_data\\Precip\\DIC\\per_pix_dic_004.npy'
        # f= result_root+ rf'detrend_zscore_Yang\LAI4g\\1982_2000.npy'
        # dic=T.load_npy(f)
        dic=T.load_npy(f)
        dic2=T.load_npy(f2)

        for pix in dic:
            if not pix in dic2:
                continue
            vals=dic[pix]
            vals2=dic2[pix]

            print(vals)


            vals=np.array(vals)
            vals2=np.array(vals2)
            print(vals)
            print(len(vals))

            # if not len(vals)==19*12:
            #     continue
            # if True in np.isnan(vals):
            #     continue
            # print(len(vals))
            if np.isnan(np.nanmean(vals)):
                continue
            # if np.nanmean(vals)<-20:
            #     continue
            plt.plot(vals)
            plt.twinx()
            plt.plot(vals2)


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





class growth_rate:
    from scipy import stats, linalg
    def run(self):
        # self.calculate_annual_growth_rate()

        self.plot_growth_rate()
        # self.bar_plot()
    pass

    def calculate_annual_growth_rate(self):
        fdir=result_root + rf'\extract_GS\OBS_LAI_extend\\'
        outdir=result_root + rf'growth_rate\\growth_rate_raw\\'
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if f.split('.')[0] not in ['LAI4g','CO2','CRU','GPCC','VPD','tmax']:
                continue

            dict=np.load(fdir+f,allow_pickle=True).item()
            growth_rate_dic={}
            for pix in tqdm(dict):
                time_series=dict[pix]
                print(len(time_series))
                growth_rate_time_series=np.zeros(len(time_series)-1)
                for i in range(len(time_series)-1):
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

            vals_growth_rate = df_continent['LAI4g_growth_rate_raw'].tolist()
            # print(vals_growth_rate_relative_change);exit()
            vals_relative_change = df_continent['LAI4g_relative_change'].tolist()

            vals_growth_rate_list = []
            vals_relative_change_list = []
            for val_growth_rate in vals_growth_rate:

                if type(val_growth_rate) == float:  ## only screening
                    continue
                if len(val_growth_rate) == 0:
                    continue
                val_growth_rate[val_growth_rate < -99] = np.nan
                val_growth_rate = np.array(val_growth_rate)
                # print(val_growth_rate)

                vals_growth_rate_list.append(list(val_growth_rate))

            for val_relative_change in vals_relative_change:
                if type(val_relative_change) == float:  ## only screening
                    continue
                if len(val_relative_change) == 0:
                    continue
                val_relative_change[val_relative_change < -99] = np.nan


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
            x = np.arange(0, 39)
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

    def bar_plot(self):
        ## layer 1 LAI4g layer 2 LAI4g growth rate
        # 1 long term LAI4g >0 and growth rate >0 [significant]
        # 2 long term LAI4g >0 and growth rate <0 [significant]
        # 3 long term LAI4g <0 and growth rate >0 [significant]
        # 4 long term LAI4g <0 and growth rate <0 [significant]
        # 5 long term LAI4g >0 and growth rate is not significant
        # 6 long term LAI4g <0 and growth rate is not significant
        df = T.load_df(rf'D:\Project3\Result\growth_rate\DataFrame\growth_rate_all_years.df')
        print(len(df))
        T.print_head_n(df)
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['Aridity'] < 0.65]
        df = df[df['MODIS_LUCC'] != 12]
        print(len(df))

        df=df.dropna(subset=['LAI4g_trend'])
        # print(len(df))
        # exit()
        T.print_head_n(df)
        # exit()
        color_list = ['yellowgreen', 'lime', 'green', 'lightseagreen', 'yellow', 'orange', 'red', 'brown']
        fig = plt.figure()
        ii = 1
        ax = fig.add_subplot(2, 3, ii)
        flag = 0
        number1=0
        number2=0
        number3=0
        number4=0
        number5=0
        number6=0
        number7=0
        number8=0
        number9=0


        for i, row in tqdm(df.iterrows(), total=len(df)):
            LAI4g_trend = row['LAI4g_trend']
            LAI4g_p_value = row['LAI4g_p_value']
            growth_rate = row['LAI4g_trend_growth_rate']
            growth_rate_p_value = row['LAI4g_p_value_growth_rate']
            if LAI4g_p_value < 0.1 and growth_rate_p_value < 0.1:  ## significant long term and significant growth rate
                if LAI4g_trend > 0 and growth_rate > 0:  ## significant greening and significant increase growth rate
                    number1+=1
                elif LAI4g_trend > 0 and growth_rate < 0:  ## significant greening and significant decrease growth rate
                    number2+=1
                elif LAI4g_trend < 0 and growth_rate > 0:  ## significant browning and significant increase growth rate
                    number3+=1
                elif LAI4g_trend < 0 and growth_rate < 0:  ## significant browning and significant decrease growth rate
                    number4+=1
            elif LAI4g_p_value < 0.1 and growth_rate_p_value > 0.1:  ## significant long term and non-significant growth rate
                if LAI4g_trend > 0:  ## significant greening and non-significant growth rate
                    number5+=1
                elif LAI4g_trend < 0:  ## significant browning and non-significant growth rate
                    number6+=1
            elif LAI4g_p_value > 0.1 and growth_rate_p_value < 0.1:  ## non-significant long term and significant growth rate
                if growth_rate > 0:  ## non-significant long term and significant increase growth rate
                    number7+=1
                elif growth_rate < 0:  ## non-significant long term and significant decrease growth rate
                    number8+=1
            else:
                number9+=1  ## non-significant long term and non-significant growth rate

        percentage1=number1/len(df)*100
        percentage2=number2/len(df)*100
        percentage3=number3/len(df)*100
        percentage4=number4/len(df)*100
        percentage5=number5/len(df)*100
        percentage6=number6/len(df)*100
        percentage7=number7/len(df)*100
        percentage8=number8/len(df)*100
        percentage9=number9/len(df)*100
        number_list=[number1,number2,number3,number4,number5,number6,number7,number8,number9]
        percentage_list=[percentage1,percentage2,percentage3,percentage4,percentage5,percentage6,percentage7,percentage8,percentage9]


        ax.bar(range(1,10),percentage_list)
        plt.xticks(range(1,10),['1','2','3','4','5','6','7','8','9'])
        plt.ylabel('percentage')
        plt.show()
            # exit()












class Dataframe_func:

    def run (self):
        fdir = result_root + rf'ERA_precip_CV_trend\cv_trend\\'
        #
        for f in os.listdir(fdir):
            if not f.endswith('.df'):

                continue
            outf=fdir+f
            print(outf)

            df=T.load_df(fdir+f)
            print('add aridity')
            df=self.add_aridity_to_df(df)

            print('add landcover_data')
            df=self.add_landcover_data_to_df(df)
            df=self.add_landcover_classfication_to_df(df)
            print('add MODIS landcover')
            df=self.add_MODIS_LUCC_to_df(df)
            print('add continent')
            df=self.add_continent_to_df(df)
            print('add row')
            df=self.add_row(df)
            print('CCI_max')
            df=self.add_maxmium_LC_change(df)
            print('add average_lai')
            df=self.add_average_lai(df)

            print('add aridity_classfication')
            df=self.add_AI_classfication(df)
            # print('add percipitation_zscore')
            # df=self.add_precipitation_zscore(df)
            # print('add LAI4g_zscore')
            # df=self.add_LAI4g_zscore(df)



            T.save_df(df,outf)
            # T.print_head_n(df)
            T.df_to_excel(df,outf)


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

    def add_aridity_to_df(self, df):  ## here is original aridity index not classification

        f = data_root + rf'Base_data\dryland_AI.tif\\dryland.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)

        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        # val_array = np.load(fdir + f)

        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        f_name = 'Aridity'
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


    def add_MODIS_LUCC_to_df(self, df):
        f = data_root + rf'\Base_data\MODIS_LUCC\\MODIS_LUCC_resample.tif'
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

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, DIC_and_TIF())
        return df
    def add_row(self, df):
        r_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix
            r_list.append(r)
        df['row'] = r_list
        return df




    def add_SM_trend_label(self, df):  ##

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

    def add_continent_to_df(self, df):
        tiff = rf'D:\Project3\Data\Base_data\\continent.tif'
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

    def add_maxmium_LC_change(self, df): ## #### CCI landcover maximum change

        f = rf'E:\CCI_landcover\trend_analysis_LC\\LC_max.tif'

        array, origin, pixelWidth, pixelHeight, extent = ToRaster().raster2array(f)
        array[array <-99] = np.nan

        LC_dic =DIC_and_TIF().spatial_arr_to_dic(array)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix

            val= LC_dic[pix]
            df.loc[i,'LC_max'] = val
        return df

    def add_average_lai(self,df):

        f = rf'D:\Project3\Result\state_variables\\LAI4g_1982_2020.npy'
        lai_mean_dic = T.load_npy(f)


        f_name = 'growing_season_LAI_mean'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in lai_mean_dic:
                val_list.append(np.nan)
                continue
            val = lai_mean_dic[pix]
            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f_name] = val_list
        return df

        pass

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
            elif val==2:
                label='Sub-Humid'
            elif val<-99:
                label=np.nan
            else:
                raise
            val_list.append(label)

        df['AI_classfication'] = val_list
        return df

    def add_precipitation_zscore(self,df):
        f = result_root + rf'zscore\GPCC.npy'

        val_dic = T.load_npy(f)
        f_name = 'GPCC_zscore'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            year=row['year']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix][year-1982]
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def add_LAI4g_zscore(self,df):
        f = result_root + rf'zscore\LAI4g.npy'

        val_dic = T.load_npy(f)
        f_name = 'LAI4g_zscore'
        print(f_name)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            year=row['year']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix][year-1982]
            val_list.append(vals)
        df[f_name] = val_list
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


class moving_window():
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass
    def run(self):
        # self.moving_window_extraction()
        # self.moving_window_extraction_for_LAI()
        # self.moving_window_trend_anaysis()
        # self.moving_window_CV_extraction_anaysis()
        # self.moving_window_CV_trends()
        # self.moving_window_average_anaysis()
        self.produce_trend_for_each_slides()
        # self.calculate_trend_spatial()
        # self.calculate_trend_trend()
        # self.convert_trend_trend_to_tif()

        # self.plot_moving_window_time_series_area()
        # self.calculate_browning_greening_average_trend()
        # self.plot_moving_window_time_series()
        pass
    def moving_window_extraction(self):

        fdir = result_root + rf'D:\Project3\Result\extract_GS\OBS_LAI_extend\\'
        outdir = result_root + rf'extract_window\extract_anomaly_window\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if f.split('.')[0] not in ['CRU']:
                continue

            outf = outdir + f.split('.')[0] + '.npy'
            print(outf)
            # if os.path.isfile(outf):
            #     continue

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

                # new_x_extraction_by_window[pix] = self.forward_window_extraction_detrend_anomaly(time_series, window)
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
                relative_change_list=[]
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
                    # x_anomaly=(x_vals[i]-x_mean)
                    # relative_change = (x_vals[i] - x_mean) / x_mean

                    # relative_change_list.append(x_vals)
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

                # for i in range(len(x_vals)):
                #     if x_vals[0]==None:
                #         continue
                #     x_anomaly=x_vals[i]-x_mean
                #
                #     anomaly.append(x_anomaly)
                # if np.isnan(anomaly).any():
                #     continue
                # detrend_anomaly=signal.detrend(anomaly)+x_mean
                detrend_original=signal.detrend(x_vals)+x_mean


                new_x_extraction_by_window.append(detrend_original)
        return new_x_extraction_by_window

    def moving_window_trend_anaysis(self):
        window_size=15
        fdir = result_root + rf'extract_window\extract_relative_change_window_CV\\'
        outdir = result_root + rf'\\extract_window\\extract_relative_change_window_CV_trend\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):

            dic = T.load_npy(fdir + f)
            slides = 39-window_size
            outf = outdir + f.split('.')[0] + f'.npy'
            print(outf)
            if os.path.isfile(outf):
                continue

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

    def moving_window_CV_extraction_anaysis(self):
        window_size=15
        fdir = result_root + rf'extract_window\extract_detrend_original_window\\15\\'
        outdir = result_root + rf'\\extract_window\\extract_detrend_original_window_CV\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if f.split('.')[0] not in ['GPCC']:
                continue

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


                time_series_all = dic[pix]
                if len(time_series_all)<23:
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

                    if np.nanmean(time_series)==0:
                        continue
                    cv=np.nanstd(time_series)/np.nanmean(time_series)*100
                    trend_list.append(cv)

                trend_dic[pix]=trend_list

            np.save(outf, trend_dic)

            ##tiff
            # arr_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(trend_dic)
            #
            # p_value_arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(p_value_dic)
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_trend, outf + '_trend.tif')
            # DIC_and_TIF(pixelsize=0.25).arr_to_tif(p_value_arr, outf + '_p_value.tif')

    def moving_window_CV_trends(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)
        variable='GPCC'

        f = result_root + rf'extract_window\extract_detrend_original_window_CV\\{variable}.npy'
        outdir = result_root + rf'\\extract_window\\extract_detrend_original_window_CV\\'
        T.mk_dir(outdir, force=True)
        dic=T.load_npy(f)
        result_dic_trend={}
        result_dic_p_value={}
        for pix in dic:
            r,c=pix
            if r<120:
                continue
            vals=dic[pix]
            land_cover_val=crop_mask[pix]
            if land_cover_val==16 or land_cover_val==17 or land_cover_val==18:
                continue
            modis_val=dic_modis_mask[pix]
            if modis_val==12:
                continue
            if np.isnan(np.nanmean(vals)):
                continue
            slope, b, r, p_value = T.nan_line_fit(np.arange(len(vals)), vals)
            result_dic_trend[pix]=slope
            result_dic_p_value[pix]=p_value
        array_slope=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic_trend)
        array_slope_mask=array_slope*array_mask
        array_p_value=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic_p_value)
        array_p_value_mask=array_p_value*array_mask

        DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_slope_mask,outdir+f'{variable}_CV_trend.tif')
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_p_value_mask,outdir+f'{variable}_CV_p_value.tif')

        outf=outdir+f'{variable}_CV_trend.npy'
        np.save(outf,result_dic_trend)







        pass

    def moving_window_average_anaysis(self):
        window_size = 15
        fdir = self.result_root + rf'extract_window\\extract_original_window\\{window_size}\\'
        outdir = self.result_root + rf'\\extract_window\\extract_original_window_average\\{window_size}\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not 'LAI4g' in f:
                continue

            dic = T.load_npy(fdir + f)
            slides = 39 - window_size
            outf = outdir + f.split('.')[0] + f'.npy'
            print(outf)
            # if os.path.isfile(outf):
            #     continue

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




    def produce_trend_for_each_slides(self):  ## 从上一个函数生成的一个像素24个trend, 变成 一个trend 一张图
        fdir=rf'D:\Project3\Result\extract_window\extract_relative_change_window_trend\\'
        dryland_mask=join(data_root,'Base_data','dryland_mask.tif')
        dic_dryland=DIC_and_TIF().spatial_tif_to_dic(dryland_mask)


        for f in os.listdir(fdir):
            if not 'LAI4g_p_value' in f:
                continue


            dic=T.load_npy(fdir+f)
            result_dic={}


            for slide in range(1,25):
                slide_f=f'{slide:02d}'

                for pix in dic:
                    dryland_val=dic_dryland[pix]

                    vals=dic[pix]
                    vals=np.array(vals)
                    vals=vals*dryland_val
                    vals=np.array(vals)
                    if len(vals)!=24:
                        continue
                    result_dic[pix]=vals[slide-1]
                DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(result_dic,fdir+f.split('.')[0]+f'_{slide_f}.tif')


        pass
    def calculate_trend_spatial(self):
        fdir = result_root + rf'multi_regression_moving_window\window15_relative_change\TIFF\\'
        outdir = result_root + rf'multi_regression_moving_window\window15_relative_change_trend\\'
        T.mk_dir(outdir, force=True)
        val_list=['GLEAM_SMroot_LAI4g','VPD_LAI4g']

        for val in val_list:
            array_list=[]
            for f in os.listdir(fdir):
                if not f.endswith('.tif'):
                    continue

                fname=f.split('.')[0]
                if not val in fname:
                    continue


                print(f)
                array=ToRaster().raster2array(fdir+f)[0]
                array=np.array(array)
                array[array<-999]=np.nan
                array_list.append(array)
            array_list=np.array(array_list)

            trend_list=[]


            ### calculate trend for each pixel across all slides
            for i in range(array_list.shape[1]):
                for j in range(array_list.shape[2]):
                    vals=array_list[:,i,j]
                    if np.isnan(np.nanmean(vals)):
                        trend_list.append(np.nan)
                        continue
                    slope, b, r, p_value = T.nan_line_fit(np.arange(len(vals)), vals)
                    trend_list.append(slope)
            trend_list=np.array(trend_list)
            trend_list=trend_list.reshape(array_list.shape[1],array_list.shape[2])
            outf=outdir+val+'.tif'
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(trend_list,outf)












            ##save








        T.mk_dir(outdir, force=True)
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



def main():
    data_processing().run()
    # statistic_analysis().run()
    # classification().run()
    # calculating_variables().run()
    # plot_response_function().run()
    # maximum_trend().run()
    # partial_correlation().run()
    # single_correlation().run()
    # ResponseFunction().run()
    # bivariate_analysis().run()
    # CCI_LC_preprocess().run()
    # calculating_variables().run()
    # pick_event().run()
    # selection().run()
    # multi_regression_anomaly().run()
    # multi_regression_detrended_anomaly().run()
    # data_preprocess_for_random_forest().run()

    # monte_carlo().run()
    # moving_window().run()

    # fingerprint().run()


    # plot_dataframe().run()
    # growth_rate().run()
    # plt_moving_dataframe().run()
    # check_data().run()
    # Dataframe_func().run()
    # Check_plot().run()


    pass

if __name__ == '__main__':
    main()