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




this_root = 'E:\\'
data_root = 'E:\Data\\'
result_root = 'E:\Result\\'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

class data_processing():
    def __init__(self):
        pass
    def run (self):
        # self.nc_to_tif()
        # self.aggreate_tiff()
        # self.clipped()
        # self.plot()
        self.plot_spatial()

        pass
    def nc_to_tif(self):
        fdir= 'E:\Data\\GPP\\nc\\S2\\'

        for f in os.listdir(fdir):


            outdir_name = f.split('.')[0]

            outdir = data_root + rf'GPP\\tif\\{outdir_name}\\'
            if not 'ISBA-CTRIP_S2_gpp' in f:
                continue
            # if isdir(outdir):
            #     continue
            # print(outdir)
            Tools().mk_dir(outdir, force=True)

            yearlist = list(range(1982, 2021))


            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir + f, var_name='gpp', outdir=outdir, yearlist=yearlist)
            except Exception as e:
                print(e)
                continue


        pass

    def nc_to_tif_template(self, fname, var_name, outdir, yearlist):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())
            # time=ncin.variables['time'][:]
            time=ncin.variables['time_counter'][:]

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
    def aggreate_tiff(self):
        fdir_all = 'E:\Data\\GPP\\tif\\'
        outdir = 'E:\Data\\GPP\\agg\\'
        Tools().mk_dir(outdir, force=True)
        yearlist=list(range(1982,2021))
        ###### annual * days of each month aggreate

        for fdir in os.listdir(fdir_all):

            outdir = rf'E:\Data\\GPP\\agg\\{fdir}\\'
            Tools().mk_dir(outdir, force=True)

            ## annual aggreate
            for year in yearlist:
                array_list = []
                outpath = outdir + f'{fdir}_{year}.tif'
                if isfile(outpath):
                    continue
                for f in os.listdir(fdir_all+fdir):
                    if not f.startswith(str(year)):
                        continue
                    if not f.endswith('.tif'):
                        continue
                    fpath=fdir_all+fdir+'\\'+f
                    print(fpath)

                    arr = ToRaster().raster2array(fpath)[0]
                    arr=arr*3600*24*30
                    arr[arr<0]=np.nan
                    arr[arr>10]=np.nan
                    ###### annual * days of each month aggreate

                    array_list.append(arr)

                array_list=np.array(array_list)
                array_mean=np.nansum(array_list,axis=0)
                DIC_and_TIF(tif_template=fpath).arr_to_tif(array_mean,outpath)
    def clipped(self):
        fdir_all = 'E:\Data\\GPP\\agg\\'


        input_shape = 'E:\Data\Basedata\AZ\\az.shp'
        for fdir in os.listdir(fdir_all):

            outdir = rf'E:\Data\\GPP\\clipped\\{fdir}\\'
            Tools().mk_dir(outdir, force=True)
            for f in os.listdir(fdir_all+fdir):
                if not f.endswith('.tif'):
                    continue
                outpath = outdir + f
                if isfile(outpath):
                    continue
                fpath=fdir_all+fdir+'\\'+f
                print(fpath)
                ToRaster().clip_array(fpath,  outpath,input_shape)


    def plot(self): ### plot annual time series
        fdir_all = 'E:\Data\\GPP\\clipped\\'
        for fdir in os.listdir(fdir_all):
            data_list=[]

            for f in os.listdir(fdir_all+fdir):
                fpath=fdir_all+fdir+'\\'+f
                print(fpath)
                arr,originX,originY,pixelWidth,pixelHeight=ToRaster().raster2array(fpath)
                arr[arr<0]=np.nan
                arr[arr>10]=np.nan
                arr_mean=np.nanmean(arr)
                data_list.append(arr_mean)
            data_mean=np.array(data_list)

            plt.title('annual GPP (kg/m2/year)')

            plt.plot(data_mean)
            year_list=list(range(1982,2021))
            plt.xticks(list(range(len(data_mean))),year_list,rotation=45)

        plt.legend(os.listdir(fdir_all))
        plt.show()


        pass


class Plot_AZ:
    def __init__(self):
        pass

    def plot_spatial(self): ###
        fdir_all = 'E:\Data\\GPP\\az_clipped\\'
        outdir = 'E:\Data\\GPP\\az\\'
        T.mk_dir(outdir, force=True)
        # plt.figure(figsize=(20,10))


        for fdir in os.listdir(fdir_all):
            print(fdir)

            data_list=[]

            # fig=plt.subplot(3,5,i)

            for f in os.listdir(fdir_all+fdir):
                if 'xml' in f:
                    continue
                fpath=fdir_all+fdir+'\\'+f
                # print(fpath)
                # self.plot_az(fpath, shp)
                # plt.show()
                arr,originX,originY,pixelWidth,pixelHeight=ToRaster().raster2array(fpath)
                arr[arr<0]=np.nan
                arr[arr>10]=np.nan

                data_list.append(arr)
            data_mean=np.array(data_list)
            data_mean=np.nanmean(data_mean,axis=0)
            # print(data_mean)
            # exit()
            # i=i+1

            # plt.title(f'{fdir}')
            #
            # plt.imshow(data_mean,vmin=0,vmax=1,cmap='jet')
            # plt.show()
            DIC_and_TIF(tif_template=fpath).arr_to_tif(data_mean,outdir+fdir+'.tif')

            ### unify the colorbar
            # vmin,vmax=np.nanmin(data_mean),np.nanmax(data_mean)
            # plt.clim(vmin,vmax)


    def plot_az(self, fpath, in_shpfile,ax=None, cmap=None, vmin=None, vmax=None, is_plot_colorbar=True, is_reproj=True,res=10000,is_discrete=False,colormap_n=11):
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
        arr_deg, originX_deg, originY_deg, pixelWidth_deg, pixelHeight_deg = ToRaster().raster2array(fpath)
        llcrnrlon = originX_deg
        urcrnrlat = originY_deg
        urcrnrlon = originX_deg + pixelWidth_deg * arr_deg.shape[1]
        llcrnrlat = originY_deg + pixelHeight_deg * arr_deg.shape[0]
        arr_deg = Tools().mask_999999_arr(arr_deg, warning=False)
        arr_m = ma.masked_where(np.isnan(arr_deg), arr_deg)
        # exit()
        lon_list = np.arange(originX_deg, originX_deg +  pixelWidth_deg * arr_deg.shape[1], pixelWidth_deg)
        lat_list = np.arange(originY_deg, originY_deg + pixelHeight_deg * arr_deg.shape[0], pixelHeight_deg)
        lat_list = lat_list + pixelHeight_deg / 2
        lon_list = lon_list + pixelWidth_deg / 2

        m = Basemap(projection='cyl', ax=ax, resolution='i',
                    llcrnrlon=-115, llcrnrlat=31.2, urcrnrlon=-109, urcrnrlat=37.2,
                    lon_0=-111,lat_0=0,lat_1=25,lat_2=47)
        m.drawparallels(np.arange(32., 37.1, 1.), zorder=99, linewidth=1)
        m.drawmeridians(np.arange(-115., -108.5, 1.), zorder=99, linewidth=1)

        # m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        lon_matrix = []
        lat_matrix = []
        for lon in tqdm(lon_list):
            lon_matrix_i = []
            lat_matrix_i = []
            for lat in lat_list:
                # print(lon,lat)
                lon_projtran, lat_projtran = m.projtran(lon,lat)
                lon_matrix_i.append(lon_projtran)
                lat_matrix_i.append(lat_projtran)

            lon_matrix.append(lon_matrix_i)
            lat_matrix.append(lat_matrix_i)
        lon_matrix = np.array(lon_matrix)
        lat_matrix = np.array(lat_matrix)

        ret = m.pcolormesh(lon_matrix, lat_matrix, arr_deg.T, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax)
        shp_f = in_shpfile
        m.readshapefile(shp_f,'a', drawbounds=True, linewidth=0.5, color='k', zorder=100)
        # m.readshapefile(shp_provinces_f, 'ooo', drawbounds=True, linewidth=0.3, color='k', zorder=100)
        # m.drawparallels(np.arange(-60., 90., 20.), zorder=99, dashes=[8, 8], linewidth=.5)
        # m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        # meridict = m.drawmeridians(np.arange(0., 420., 20.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        # meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        plt.axis('off')

        # for obj in meridict:
        #     line = meridict[obj][0][0]
        # coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        # polys = m.fillcontinents(color='#D1D1D1', lake_color='#EFEFEF',zorder=90)
        if is_plot_colorbar:
            if is_discrete:
                bounds = np.linspace(vmin, vmax, colormap_n)
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05,shrink=0.5)
                # cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation='horizontal')
            else:
                cbar = plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom', pad=0.05)
        return m, ret

def main():
    # data_processing().run()
    Plot_AZ().plot_spatial()


    pass

if __name__ == '__main__':
    main()