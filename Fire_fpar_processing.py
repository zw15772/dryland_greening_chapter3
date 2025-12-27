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

class download_FVC():
    def __init__(self):
        pass

    def run(self):
        self.nc_to_tif_time_series_fast()
        pass

    def download_fpar(self):
        import os
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin
        year_list=list(range(1982,2021))

        # Target URL
        for year in year_list:
            base_url = f"http://www.glass.umd.edu/05D/FVC/glass_fvc_avhrr_netcdf/{year}/"

            output_dir = data_root + f"glass_fvc_avhrr\\netcdf/{year}/"
            T.mk_dir(output_dir, force=True)


            # Loop through all links and download .nc files
            response = requests.get(base_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all .nc4 files
            file_links = [urljoin(base_url, a['href']) for a in soup.find_all('a') if a['href'].endswith('.nc4')]

            # Download each file
            for file_url in file_links:
                filename = file_url.split('/')[-1]
                out_path = os.path.join(output_dir, filename)

                if os.path.exists(out_path):
                    print(f"Already downloaded: {filename}")
                    continue

                print(f"Downloading: {filename}")
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(out_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

        print("Download complete.")

class download_NOAA_AVHRR():
    def __init__(self):
        pass

    def run(self):
        self.download_AVHRR_NOAA()
        # self.nc_to_tif_time_series_fast()
        pass

    def download_AVHRR_NOAA(self):
        import os
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin
        outdir=data_root + 'NOAA_AVHRR\\'
        T.mk_dir(outdir, force=True)


        # Target URL
        ## https://www.ncei.noaa.gov/thredds/fileServer/cdr/lai/1981/AVHRR-Land_v005_AVH15C1_NOAA-07_19810628_c20181025194441.nc
        base_url = 'https://www.ncei.noaa.gov/thredds/fileServer/cdr/lai/'
        index_url = 'https://www.ncei.noaa.gov/thredds/catalog/ncFC/cdr/lai-fapar-fc/files/catalog.html'
        response = requests.get(index_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all .nc files
        file_links = [urljoin(index_url, a['href']) for a in soup.find_all('a') if a['href'].endswith('.nc')]

        # Download each file
        for file_url in tqdm(file_links):
            # print(file_url)
            fname = file_url.split('/')[-2:]
            year = fname[0]
            outdir_i = join(outdir, year)
            T.mk_dir(outdir_i, force=True)
            fname = '/'.join(fname)
            if not fname.endswith('.nc'):
                continue
            # print(fname)
            new_url = base_url + fname
            outpath=join(outdir,fname)
            if os.path.isfile(outpath):
                continue

            try:
                with requests.get(new_url, stream=True) as r:
                    r.raise_for_status()
                    with open(outpath, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except:
                print(f"Failed to download: {fname}")




    def nc_to_tif_time_series_fast(self):

        fdir_all=rf'D:\Project3\Data\glass_fvc_avhrr\netcdf\\'
        outdir=rf'D:\Project3\Data\glass_fvc_avhrr\\\\TIFF\\'
        Tools().mk_dir(outdir,force=True)
        for fdir in tqdm(os.listdir(fdir_all)):

            for f in os.listdir(fdir_all+fdir):
                if not f.endswith('.nc4'):
                    continue

                outdir_name = f.split('.')[2][1:]
                # print(outdir_name);exit()

                yearlist = list(range(1982, 2021))
                fpath = join(fdir_all+fdir,f)
                nc_in = xarray.open_dataset(fpath)
                print(nc_in)

                outf = join(outdir, f'{outdir_name}.tif')
                array = nc_in['FVC']
                array = np.array(array)
                array[array < 0] = np.nan
                longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.5, -0.5
                ToRaster().array2raster(outf, longitude_start, latitude_start,
                                        pixelWidth, pixelHeight, array, ndv=-999999)
                # exit()


            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir+f, var_name='FVC', outdir=outdir, yearlist=yearlist)
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

class processing_CCI_landcover_change:
    def __init__(self):
        pass

    def run(self):
        # self.tif_to_dic()
        self.relative_change()
        pass


    def CCI_landcover_preprocess(self):
        fdir_all=data_root+'\ESA_CCI_LC_tif05\\'
        dic_composite={'SHRUBS-BD':'shrubs',
                       'SHRUBS-BE':'shrubs',
                       'SHRUBS-ND':'shrubs',
                       'SHRUBS-NE':'shrubs',
                       'GRASS-MAN':'grass',
                       'GRASS-NAT':'grass',
                       'TREES-BD':'trees',
                       'TREES-BE':'trees',
                       'TREES-ND':'trees',
                       'TREES-NE':'trees',
                       'BARE':'bare',
                       'WATER':'water',
                       'BUILT':'built',


                       }


        ## each landcover group I want to append and sum them

        for fdir in os.listdir(fdir_all):
            outdir=join(fdir_all,fdir,'composite')
            T.mk_dir(outdir,force=True)
            shrubslist = []
            grasslist = []
            trees_list = []
            water_list=[]
            built_list=[]
            bare_list=[]

            results_class_dict={}
            for f in os.listdir(fdir_all+fdir):
                if not f.endswith('.tif'):
                    continue
                if 'LAND' in f:
                    continue
                if 'SNOWICE' in f:
                    continue

                fpath=join(fdir_all+fdir,f)
                landcover=f.split('_')[-1].split('.')[0]
                landcover=dic_composite[landcover]

                if landcover=='shrubs':
                    shrubslist.append(fpath)
                elif landcover=='grass':
                    grasslist.append(fpath)
                elif landcover=='trees':
                    trees_list.append(fpath)
                elif landcover=='water':
                    water_list.append(fpath)
                elif landcover=='built':
                    built_list.append(fpath)
                elif landcover=='bare':
                    bare_list.append(fpath)
                else:
                    raise
            results_class_dict['shrubs'] = shrubslist
            results_class_dict['grass'] = grasslist
            results_class_dict['trees'] = trees_list
            results_class_dict['water'] = water_list
            results_class_dict['built'] = built_list
            results_class_dict['bare'] = bare_list

            for key in results_class_dict.keys():
                array_list=[]

                outf = join(outdir, key + '.tif')
                for f in results_class_dict[key]:
                    array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
                    array_list.append(array)
                array_list=np.array(array_list)
                array_sum=np.sum(array_list,axis=0)
                DIC_and_TIF().arr_to_tif(array_sum, outf)


    def tif_to_dic(self):
        fdir=rf'D:\Project3\Data\ESA_CCI_LC_tif05\\'
        outdir = rf'D:\Project3\Data\ESA_CCI_LC_dic_dict\\'
        T.mk_dir(outdir)
        veg_types = ["grass", "trees", "shrubs"]
        years = range(1993, 2021)

        # 初始化结果字典
        veg_dict = {veg: [] for veg in veg_types}

        # 拼接每个植被类型的时间序列
        for veg in veg_types:
            print(veg)
            fpath_list=[]
            for year in years:
                file_path = os.path.join(fdir, str(year),'composite', f"{veg}.tif")
                # array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(file_path)
                fpath_list.append(file_path)
            # print(fpath_list);exit()
            outdir_i = join(outdir,veg)
            T.mk_dir(outdir_i)
            spatial_dict = self.compose_tif_list(fpath_list, method='array')
            T.save_distributed_perpix_dic(spatial_dict, outdir_i)

        ## 保存结果字典

    pass
    def relative_change(self):
        fdir_all = rf'D:\Project3\Data\ESA_CCI_LC_tif05\DIC\\'
        outdir =result_root + rf'3mm\ESA_CCI_LC\\'
        T.mk_dir(outdir)
        result_dic={}
        for fdir in os.listdir(fdir_all):
            fdir_i = join(fdir_all,fdir)
            outdir_i = join(outdir,fdir)
            dic=T.load_npy_dir(fdir_i)
            for pix in dic:
                vals=dic[pix]
                vals=np.array(vals)
                if vals[0]==0:
                    continue
                relative_changes=(vals[0]-vals[-1])/vals[0]*100
                # print(relative_changes)
                result_dic[pix]=relative_changes

            T.save_npy(result_dic, outdir_i)

            pass




        pass
    def compose_tif_list(self, flist, less_than=-9999, method='mean'):
        # less_than -9999, mask as np.nan
        if len(flist) == 0:
            return
        tif_template = flist[0]
        void_dic = DIC_and_TIF(tif_template=tif_template).void_spatial_dic()
        for f in tqdm(flist, desc='transforming...'):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
            for r in range(len(array)):
                for c in range(len(array[0])):
                    pix = (r, c)
                    val = array[r][c]
                    void_dic[pix].append(val)
        spatial_dic = {}
        for pix in tqdm(void_dic, desc='calculating mean...'):
            vals = void_dic[pix]
            vals = np.array(vals, dtype=float)
            vals[vals < less_than] = np.nan
            if method == 'mean':
                compose_val = np.nanmean(vals)
            elif method == 'max':
                compose_val = np.nanmax(vals)
            elif method == 'sum':
                compose_val = np.nansum(vals)
            elif method == 'array':
                compose_val = vals
            else:
                raise UserWarning(f'{method} is invalid, should be "mean" "max" or "sum" or "array"')
            spatial_dic[pix] = compose_val
        return spatial_dic


class Data_processing:

    def __init__(self):


        pass

    def run(self):
        # self.nc_to_tif_time_series_fast()

        # self.resample()
        # self.scale()
        # self.extract_dryland_tiff()
        # self.generate_nan_map()
        # self.composite_fpar()


        # self.tif_to_dic()
        # self.extract_phenology_monthly_variables()
        self.extract_annual_growing_season_fire_mean()
        # self.extract_phenology_fire_mean()
        # self.weighted_fire()
        # self.interpolation()



        pass



    def nc_to_tif_time_series_fast(self):

        fdir_all=rf'D:\Project3\Data\glass_fvc_avhrr\netcdf\\'
        outdir=rf'D:\Project3\Data\glass_fvc_avhrr\\\\TIFF\\'
        Tools().mk_dir(outdir,force=True)
        for fdir in tqdm(os.listdir(fdir_all)):

            for f in os.listdir(fdir_all+fdir):
                if not f.endswith('.nc4'):
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
                    array = nc_in['FVC'][t]
                    array = np.array(array)
                    array[array < 0] = np.nan
                    longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.5, -0.5
                    ToRaster().array2raster(outf, longitude_start, latitude_start,
                                            pixelWidth, pixelHeight, array, ndv=-999999)
                    # exit()


                # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
                try:
                    self.nc_to_tif_template(fdir+f, var_name='FVC', outdir=outdir, yearlist=yearlist)
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
        outdir = rf'D:\Project3\Data\Fire\sum\\'
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


        fdir_all = rf'D:\Project3\Data\glass_fvc_avhrr\\'

        for fdir in T.listdir(fdir_all):



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
            outf=outdir+'1994'+i+'01.tif'
            # print(outf);exit()
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_void,outf)

        pass

    def composite_fpar(self):
        fdir=rf'D:\Project3\Data\glass_fvc_avhrr\dryland_tiff\\'
        outdir=rf'D:\Project3\Data\glass_fvc_avhrr\composite_fpar\\'
        T.mk_dir(outdir)


        Pre_Process().monthly_compose(fdir,outdir,date_fmt='doy',method='max')

        pass
    def tif_to_dic(self):

        fdir_all = rf'D:\Project3\Data\glass_fvc_avhrr\\'
        year_list = list(range(1982, 2021))

        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            if not 'composite_fpar' in fdir:
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
        fdir = rf'D:\Project3\Data\Fire\dic\\'

        outdir = rf'D:\Project3\Data\Fire\\extract_phenology_monthly\\'

        Tools().mk_dir(outdir, force=True)
        f_phenology = rf'D:\Project3\Data\LAI4g\4GST\\4GST.npy'
        phenology_dic = T.load_npy(f_phenology)
        new_spatial_dic={}
        # for pix in phenology_dic:
        #     # print(phenology_dic[pix]);exit()
        #     val=phenology_dic[pix]['SeasType']
        #     try:
        #         val=float(val)
        #     except:
        #         continue
        #
        #     new_spatial_dic[pix]=val
        # spatial_array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(new_spatial_dic)
        # plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
        # plt.show()
        # exit()
        spatial_dict_gs_count = {}

        for f in T.listdir(fdir):

            outf = outdir + f
            #
            # if os.path.isfile(outf):
            #     continue
            # print(outf)
            spatial_dict = dict(np.load(fdir + f, allow_pickle=True, encoding='latin1').item())
            dic_DOY={15: 1,
                     30: 1,
                     45: 2,
                     60: 2,
                     75: 3,
                     90: 3,
                     105: 4,
                     120: 4,
                     135: 5,
                     150: 5,
                     165: 6,
                     180: 6,
                     195: 7,
                     210: 7,
                     225: 8,
                     240: 8,
                     255: 9,
                     270: 9,
                     285: 10,
                     300: 10,
                     315: 11,
                     330: 11,
                     345: 12,
                     360: 12,
                   }


            result_dic = {}

            for pix in tqdm(spatial_dict):
                if not pix in phenology_dic:
                    continue
                # print(pix)

                r, c = pix

                SeasType=phenology_dic[pix]['SeasType']
                if SeasType==2:

                    SOS=phenology_dic[pix]['Onsets']
                    try:
                        SOS=float(SOS)

                    except:
                        continue

                    SOS=int(SOS)
                    SOS_monthly=dic_DOY[SOS]

                    EOS=phenology_dic[pix]['Offsets']
                    EOS=int(EOS)
                    EOS_monthly=dic_DOY[EOS]
                    # print(SOS_monthly,EOS_monthly)
                    # print(SOS,EOS)

                    time_series = spatial_dict[pix]

                    time_series = np.array(time_series)
                    if SOS_monthly>EOS_monthly:  ## south hemisphere
                        time_series_flatten=time_series.flatten()
                        time_series_reshape=time_series_flatten.reshape(-1,12)
                        time_series_dict={}
                        for y in range(len(time_series_reshape)):
                            if y+1==len(time_series_reshape):
                                break

                            time_series_dict[y]=np.concatenate((time_series_reshape[y][SOS_monthly-1:],time_series_reshape[y+1][:EOS_monthly]))

                    else:
                        time_series_flatten = time_series.flatten()
                        time_series_reshape = time_series_flatten.reshape(-1, 12)
                        time_series_dict = {}
                        for y in range(len(time_series_reshape)):
                            time_series_dict[y] = time_series_reshape[y][SOS_monthly - 1:EOS_monthly]
                    time_series_gs = []
                    for y in range(len(time_series_dict)):
                        time_series_gs.append(time_series_dict[y])
                    time_series_gs = np.array(time_series_gs)

                elif SeasType==3:
                    time_series = spatial_dict[pix]
                    time_series = np.array(time_series)
                    time_series_gs = np.reshape(time_series, (-1, 12))

                else:
                    SeasClss=phenology_dic[pix]['SeasClss']
                    print(SeasType,SeasClss)
                    continue
                spatial_dict_gs_count[pix] = time_series_gs.shape[1]
                result_dic[pix] = time_series_gs
        # print(spatial_dict_gs_count)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_gs_count)
        # # arr[arr<6] = np.nan
        # plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=12)
        # plt.colorbar()
        # plt.show()
            np.save(outf, result_dic)





    def extract_annual_growing_season_fire_mean(self):  ## extract LAI average
        fdir =rf'D:\Project3\Data\Fire\extract_phenology_monthly\\'

        outdir_CV = result_root +rf'\Fire\\extract_annual_growing_season_LAI_mean\\'

        # print(outdir_CV);exit()

        T.mk_dir(outdir_CV, force=True)

        spatial_dic = T.load_npy_dir(fdir)
        result_dic = {}

        for pix in tqdm(spatial_dic):
            ### ui==if northern hemisphere
            r, c = pix

            ### annual year

            vals_growing_season = spatial_dic[pix]
            print(vals_growing_season.shape[1])
            # plt.imshow(vals_growing_season)
            # plt.colorbar()
            # plt.show()
            growing_season_mean_list=[]



            for val in vals_growing_season:
                if T.is_all_nan(val):
                    continue
                val = np.array(val)

                # sum_growing_season = np.nanmean(val)
                sum_growing_season = np.nansum(val)

                growing_season_mean_list.append(sum_growing_season)



            result_dic[pix] = {
                             'growing_season':growing_season_mean_list,
                             }

        outf = outdir_CV + 'extract_annual_growing_season_fire_sum.npy'

        np.save(outf, result_dic)





    def weighted_fire(self):
        ## fire should be weighted because each pixel has different area
        f=rf'D:\Project3\Result\3mm\extract_fire_ecosystem_year\\fire_ecosystem_year.npy'
        outf=rf'D:\Project3\Result\3mm\extract_fire_ecosystem_year\\fire_weighted_ecosystem_year.npy'
        result_dic=T.load_npy(f)
        spatial_dic = {}
        area_dict = DIC_and_TIF().calculate_pixel_area()
        for pix in tqdm(result_dic):

            value=result_dic[pix]['ecosystem_year']
            value=np.array(value)

            ecosystem_year_weighted=value/area_dict[pix]*100
            if np.isnan(ecosystem_year_weighted).all():
                continue
            plt.plot(ecosystem_year_weighted)
            plt.title(pix)



            spatial_dic[pix]=ecosystem_year_weighted
        np.save(outf,spatial_dic)


    pass
class moving_window():
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass
    def run(self):
        # self.moving_window_extraction()

        self.moving_window_average_anaysis()
        # self.moving_window_max_min_anaysis()
        # self.moving_window_std_anaysis()
        # self.moving_window_trend_anaysis()
        # self.trend_analysis()
        # self.robinson()

        pass

    def moving_window_extraction(self):

        fdir_all = self.result_root + rf'\3mm\extract_FVC_ecosystem_year\\'

        growing_season_mode_list=['growing_season', 'non_growing_season','ecosystem_year',]
        # growing_season_mode_list = ['growing_season', ]

        for mode in growing_season_mode_list:

            outdir = self.result_root + rf'\3mm\extract_VOD_phenology_year\\\moving_window_extraction\\{mode}\\'
            # outdir = self.result_root + rf'\3mm\extract_LAI4g_phenology_year\moving_window_extraction\\'
            T.mk_dir(outdir, force=True)
            for f in os.listdir(fdir_all):
                if not 'detrended' in f:
                    continue

                if not f.endswith('.npy'):
                    continue

                outf = outdir + f.split('.')[0] + '.npy'
                print(outf)

                # if os.path.isfile(outf):
                #     continue
                # if os.path.isfile(outf):
                #     continue

                dic = T.load_npy(fdir_all + f)
                window = 15

                new_x_extraction_by_window = {}
                for pix in tqdm(dic):

                    time_series = dic[pix][mode]
                    # time_series = dic[pix]

                    time_series = np.array(time_series)
                    # if T.is_all_nan(time_series):
                    #     continue
                    if len(time_series) == 0:
                        continue

                    # time_series[time_series < -999] = np.nan
                    if np.isnan(np.nanmean(time_series)):
                        print('error')
                        continue
                    # print((len(time_series)))
                    ## if all values are identical, then continue
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
        for i in range(len(x)+1):
            if i + window >= len(x)+1:
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

    def moving_window_average_anaysis(self): ## each window calculating the average
        window_size = 15

        f = rf'D:\Project3\Result\3mm\extract_FVC_phenology_year\moving_window_extraction\\FVC.npy'
        dic = T.load_npy(f)


        outf = rf'D:\Project3\Result\3mm\extract_FVC_phenology_year\moving_window_extraction_average\\FVC.npy'


        slides = 36 - window_size+1   ## revise!!

        print(outf)

        trend_dic = {}


        for pix in tqdm(dic):
            trend_list = []

            time_series_all = dic[pix]
            time_series_all = np.array(time_series_all)
            # print(time_series_all)
            if np.isnan(np.nanmean(time_series_all)):
                print('error')
                continue
            for ss in range(slides):


                ### if all values are identical, then continue
                if len(time_series_all)<22:
                    continue


                time_series = time_series_all[ss]
                # print(time_series)
                # if np.nanmax(time_series) == np.nanmin(time_series):
                #     continue
                # print(len(time_series))
                ##average
                average=np.nanmean(time_series)
                # print(average)

                trend_list.append(average)

            trend_dic[pix] = trend_list

            ## save
        np.save(outf, trend_dic)




    def moving_window_trend_anaysis(self): ## each window calculating the trend
        window_size = 15

        fdir=rf'D:\Project3\Result\3mm\extract_VOD_phenology_year\moving_window_extraction\moving_window_average_anaysis\\'
        outdir = rf'D:\Project3\Result\3mm\extract_VOD_phenology_year\moving_window_extraction\moving_window_average_anaysis\\trend\\'
        T.mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            # if not f.split('.')[0] in ['average_heat_spell', 'heat_event_frequency',
            #    'maxmum_heat_spell']:
            #     continue


            dic = T.load_npy(fdir + f)

            slides = 32 - window_size
            outf = outdir + f.split('.')[0] + f'_trend.npy'
            print(outf)

            trend_dic = {}


            for pix in tqdm(dic):
                trend_list = []

                time_series_all = dic[pix]
                time_series_all = np.array(time_series_all)
                # print(time_series_all)
                if np.isnan(np.nanmean(time_series_all)):
                    print('error')
                    continue
                for ss in range(slides):


                    ### if all values are identical, then continue
                    if len(time_series_all)<23:
                        continue


                    time_series = time_series_all[ss]
                    # print(time_series)
                    if np.nanmax(time_series) == np.nanmin(time_series):
                        continue
                    print(len(time_series))
                    ### calculate slope and intercept

                    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    print(slope)

                    trend_list.append(slope)

                trend_dic[pix] = trend_list

                ## save
            np.save(outf, trend_dic)
    def trend_analysis(self):  ##each window average trend

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\moving_window_min_max_anaysis\\'
        outdir = rf'D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\moving_window_min_max_anaysis\\trend\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):


            # if not f.split('.')[0] in ['detrended_sum_rainfall_CV', 'heat_event_frenquency',
            #                            'rainfall_intensity','rainfall_frenquency',
            #     'rainfall_seasonality_all_year']:
            #     continue
            #     continue
            outf = outdir + f.split('.')[0]
            # if os.path.isfile(outf + '_trend.tif'):
            #     continue
            print(outf)

            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()

            trend_dic = {}
            p_value_dic = {}
            for pix in tqdm(dic):
                r, c = pix
                # if r < 120:
                #     continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                    ## ignore the last one year

                # time_series = dic[pix][:-1]
                time_series = dic[pix]
                # print((time_series))
                # exit()
                time_series = np.array(time_series)
                average = np.nanmean(time_series)
                # print(time_series)

                if len(time_series) == 0:
                    continue
                # print(time_series)
                ### if all valus are the same, then skip
                # if len(set(time_series)) == 1:
                #     continue
                # print(time_series)

                if np.nanstd(time_series) == 0:
                    continue
                try:

                    # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(time_series)), time_series)
                    slope, b, r, p_value = T.nan_line_fit(np.arange(len(time_series)), time_series)
                    trend_dic[pix] = slope
                    p_value_dic[pix] = p_value
                except:
                    continue



            arr_trend = D.pix_dic_to_spatial_arr(trend_dic)


            p_value_arr = D.pix_dic_to_spatial_arr(p_value_dic)

            # plt.imshow(arr_trend, cmap='jet', vmin=1, vmax=1)
            #
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

            D.arr_to_tif(arr_trend, outf + '_trend.tif')
            D.arr_to_tif(p_value_arr, outf + '_p_value.tif')

            np.save(outf + '_trend', arr_trend)
            np.save(outf + '_p_value', p_value_arr)

        T.open_path_and_file(outdir)

    pass




    def robinson(self):
        fdir=rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\trend\\'
        temp_root=rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\trend\\'
        out_pdf_fdir=rf'D:\Project3\ERA5_025\extract_LAI4g_phenology_year\moving_window_extraction_average\growing_season\trend\\pdf\\'

        T.mk_dir(out_pdf_fdir,force=True)


        variable='detrended_growing_season_LAI_mean_CV'
        f_trend=fdir+variable+'_trend.tif'

        f_p_value=fdir+variable+'_p_value.tif'
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        m,ret=Plot().plot_Robinson(f_trend, ax=ax, vmin=-1, vmax=1, is_plot_colorbar=True, is_discrete=True,colormap_n=5)

        self.plot_Robinson_significance_scatter(m, f_p_value,temp_root,0.05,s=0.1)
        # m.colorbar(location='bottom',label='(LAI CV(%/window (15 years per window)))')
        # cbar.set_label(fontsize=6, label='(LAI CV(%/window (15 years per window)))')

        # plt.title(f'{variable}_(%/yr2)')
        # m.title(f'LAI CV(%/window (15 years per window)')



        # plt.title(f'{variable}_(day/yr)')
        # plt.title('r')
        # plt.show()
        ## save fig pdf
        #save pdf
        plt.savefig(out_pdf_fdir+variable+'.png', dpi=900, bbox_inches='tight')
        plt.close()
        T.open_path_and_file(out_pdf_fdir)

    def plot_Robinson(self, fpath, ax=None, cmap=None, vmin=None, vmax=None, is_plot_colorbar=True, is_reproj=True,res=25000,is_discrete=False,colormap_n=11):
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
        if not is_reproj:
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        else:
            fpath_robinson = self.Robinson_reproj(fpath, fpath + '_robinson-reproj.tif',res=res)
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
        m = Basemap(projection='robin', lon_0=0, lat_0=90., ax=ax, resolution='i')
        ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax,)
        m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        for obj in meridict:
            line = meridict[obj][0][0]
        coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        polys = m.fillcontinents(color='#D1D1D1', lake_color='#EFEFEF',zorder=90)
        if is_plot_colorbar:
            if is_discrete:
                bounds = np.linspace(vmin, vmax, colormap_n)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.5)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation='horizontal')

            else:
                cbar = plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom', pad=0.05)
        return m, ret


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


class Fire_percentage():
    def __init__(self):
        pass
    def run(self):
        # self.calculating_percentage()
        self.statistics()
        pass
    def calculating_percentage(self):

        dff=result_root+rf'Nov\Dataframe\Fire\\Fire.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        for col in df.columns:
            print(col)
        # exit()
        ratio_dict = {}
        area_dict = DIC_and_TIF().calculate_pixel_area()

        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix=row['pix']
            lat = row['lat']

            # 计算该像素的实际面积（单位：km²）
            area_km2 =area_dict[pix]/1000000
            # print(area_km2);exit()
            fire_area = row['extract_annual_growing_season_fire_mean']

            ratio = fire_area / area_km2 if area_km2 > 0 else np.nan
            ratio_dict[pix] = ratio*100
        arr=DIC_and_TIF().pix_dic_to_spatial_arr(ratio_dict)
        plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=30)
        # plt.colorbar()
        # plt.show()
        DIC_and_TIF().arr_to_tif(arr, result_root+rf'Nov\Fire\\Fire_percentage_annual_mean.tif')





        pass

    def statistics(self):
        dff = result_root + rf'Nov\Dataframe\Fire\\Fire.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df.dropna(subset=['Fire_percentage_annual_sum_percent'],inplace=True)
        ## calculate extract_annual_growing_season_fire_mean
        vals = df['Fire_percentage_annual_sum_percent'].dropna().values

        # 定义 bins（有物理意义）
        bins = np.linspace(0, 100, 51)  # 每 2%

        counts, bin_edges = np.histogram(vals, bins=bins)

        # 转成百分比
        percentages = counts / counts.sum() * 100

        plt.figure(figsize=(6, 4))

        plt.bar(
            bin_edges[:-1],
            percentages,
            width=np.diff(bin_edges),
            align='edge',
            color='#4C72B0',
            edgecolor='black',
            linewidth=0.6
        )

        plt.xlabel('Annual burned area fraction per pixel (%)', fontsize=12)
        plt.ylabel('Percentage of pixels', fontsize=12)
        plt.xlim(0, 20)



        plt.tight_layout()
        # plt.show()
        plt.savefig(result_root + rf'Nov\Fire\\Fire_percentage_annual_sum_percent.pdf')
        plt.close()



        df_extract_annual_growing_season_fire_mean = (
            df[df['Fire_percentage_annual_sum_percent'] <=1]
        )

        percentage=len(df_extract_annual_growing_season_fire_mean)/len(df)*100
        print(percentage)



    def df_clean(self, df):
            T.print_head_n(df)
            # df = df.dropna(subset=[self.y_variable])
            # T.print_head_n(df)
            # exit()
            df = df[df['row'] > 60]
            df = df[df['Aridity'] < 0.65]
            df = df[df['LC_max'] < 10]
            df = df[df['MODIS_LUCC'] != 12]
            # df = df[df['composite_LAI_detrend_CV_zscore_trend'] > 0]

            df = df[df['landcover_classfication'] != 'Cropland']

            return df



    pass


def main():

    # download_fpar().run()
    # download_NOAA_AVHRR().run()
    # Data_processing().run()
    Fire_percentage().run()

    # moving_window().run()


if __name__ == '__main__':
    main()

