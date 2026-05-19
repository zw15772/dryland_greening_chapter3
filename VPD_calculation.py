# coding='utf-8'
import sys


import lytools
import pingouin
import pingouin as pg
import xymap
from fontTools.subset import subset
from matplotlib.mlab import detrend
from matplotlib.pyplot import xticks
from numba.core.compiler_machinery import pass_info
from numba.cuda.libdevice import fdiv_rd
from openpyxl.styles.builtins import percent, total
from scipy.ndimage import label
# from green_driver_trend_contribution import *
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import t
from statsmodels.sandbox.regression.gmm import results_class_dict


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
result_root = 'D:/Project3/Result/Nov//'

class preprocessing_daily_VPD():
    def __init__(self):
        pass

    def run(self):
        # self.unzip()
        # self.nc_to_tif_time_series_fast()
        # self.calculating_VPD()
        # self.aggregate_VPD_daily_method2()
        # self.extract_dryland_tiff()
        #  self.tiff_to_dict()
        # self.transform()
        # self.deseasonal() ## not use
        self.detrend()


    def unzip(self):
        import os
        import gzip
        import shutil
        fdir = rf'C:\Users\wenzhang1.BLUECAT\Desktop\CRU_JRA_tmp\\zip\\'
        outdir = rf'C:\Users\wenzhang1.BLUECAT\Desktop\CRU_JRA_tmp\\unzip\\'

        T.mk_dir(outdir, force=True)



        for f in os.listdir(fdir):


            if f.endswith(".nc.gz"):
                gz_path = os.path.join(fdir, f)

                nc_path = os.path.join(outdir, f[:-3])  # 去掉.gz后缀



                print(f"正在解压: {f} -> {f[:-3]}")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(nc_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        print("所有文件解压完成！")




    def nc_to_tif_time_series_fast(self):

        fdir = rf'C:\Users\wenzhang1.BLUECAT\Desktop\CRU_JRA_spfh\unzip\\'
        outdir=rf'C:\Users\wenzhang1.BLUECAT\Desktop\CRU_JRA_spfh\\tif\\'
        Tools().mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):



            fpath = join(fdir, f)
            nc_in = xarray.open_dataset(fpath)
            print(nc_in)
            time_bnds = nc_in['time']
            for t in range(len(time_bnds)):

                date = nc_in.indexes['time'][t]
                date_str = date.strftime('%Y%m%d_%H%M')

                date_str = date_str.split()[0]
                outf = join(outdir, f'{date_str}.tif')
                array = nc_in['spfh'][t]
                # array = nc_in['tmp'][t].values.astype(np.float32)

                array = np.array(array)

                # array = array - 273.15
                ## flip
                array=np.flipud(array)
                # plt.imshow(array)
                # plt.colorbar()
                # plt.show()

                array[array < 0] = np.nan
                array[array > 1e10] = np.nan  # 过滤掉 _FillValue (9.96921E36)
                longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.5, -0.5
                ToRaster().array2raster(outf, longitude_start, latitude_start,
                                        pixelWidth, pixelHeight, array, ndv=-999999)
                # exit()


    def calculating_VPD(self):
        fdir_temp=rf'C:\Users\wenzhang1.BLUECAT\Desktop\CRU_JRA_tmp\\tif\\\\'
        fdir_spfh=rf'C:\Users\wenzhang1.BLUECAT\Desktop\CRU_JRA_spfh\\tif\\\\'
        outdir=rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\tif\\'
        Tools().mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir_temp)):
            ftemp=fdir_temp+f
            fspath=fdir_spfh+f.replace('spfh','tmp')
            outf=outdir+f
            array_temp = ToRaster().raster2array(ftemp)[0]
            # plt.imshow(array_temp)
            # plt.colorbar()
            # plt.show()
            array_spfh = ToRaster().raster2array(fspath)[0]

            # temperature already in Celsius
            T = np.array(array_temp, dtype=float)

            # specific humidity (kg/kg)
            q = np.array(array_spfh, dtype=float)

            # standard pressure (kPa)
            P = 101.325

            # saturation vapor pressure (kPa)
            es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))

            # actual vapor pressure (kPa)
            ea = (q * P) / (0.622 + 0.378 * q)

            # VPD
            vpd = es - ea

            # avoid negative values
            vpd[vpd < 0] = 0
            # plt.imshow(vpd, vmin=0, vmax=3)
            # plt.colorbar()
            # plt.show()

            DIC_and_TIF().arr_to_tif(
                vpd,
                outf,

            )


        pass

    def aggregate_daily_VPD(self):
        from collections import defaultdict
        fdir=rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\tif\\'
        outdir=rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\aggregate_daily_VPD\\'


        Tools().mk_dir(outdir,force=True)
        flist=os.listdir(fdir)

        spatial_dic = defaultdict(list)

        for f in tqdm(flist):
            date=f.split('_')[0]

            fpath=fdir+f
            array = ToRaster().raster2array(fpath)[0]
            spatial_dic[date].append(array)
        for date in tqdm(spatial_dic):
            array=np.array(spatial_dic[date])
            array_mean=np.nanmean(array,axis=0)
            plt.imshow(array_mean)
            plt.colorbar()
            plt.show()
            outf=outdir+date+'.tif'
            DIC_and_TIF().arr_to_tif(array_mean,outf)



        pass

    def aggregate_VPD_daily_method2(self):

        fdir = rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\tif\\'
        outdir = rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\aggregate_daily_VPD\\'

        Tools().mk_dir(outdir, force=True)

        year_list = range(1982, 2021)
        month_list = range(1, 13)
        day_list = range(1, 32)

        for year in tqdm(year_list):

            for month in month_list:

                for day in day_list:

                    arr_day_list = []

                    date_str = f'{year:04d}{month:02d}{day:02d}'

                    for hour in [0, 6, 12, 18]:

                        date_str_hour = date_str + f'_{hour:02d}00'

                        fpath = fdir + date_str_hour + '.tif'

                        if os.path.exists(fpath):
                            array = ToRaster().raster2array(fpath)[0]

                            array = array.astype(np.float32)

                            array[array < -999] = np.nan
                            array[array > 1e10] = np.nan

                            arr_day_list.append(array)

                    if len(arr_day_list) == 0:
                        continue

                    arr_day = np.array(arr_day_list)

                    # daily maximum VPD
                    arr_day_max = np.nanmax(arr_day, axis=0)

                    outf = outdir + date_str + '.tif'

                    DIC_and_TIF().arr_to_tif(arr_day_max, outf)

    def extract_dryland_tiff(self):
        NDVI_mask_f = join(rf'D:\Project3\Data\\', 'Base_data', 'dryland_mask05.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan
        outdir = rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\dryland_tiff\\'
        T.mk_dir(outdir,force=True)

        fdir = rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\\aggregate_daily_VPD\\'



        for fi in tqdm(T.listdir(fdir)):

            if not fi.endswith('.tif'):
                continue

            fpath = join(fdir, fi)
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr[np.isnan(array_mask)] = np.nan


            # plt.imshow(arr)
            # plt.show()
            outpath = join(outdir,fi)

            ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)

    def tiff_to_dict(self):
        fdir_all=rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\\'
        outdir = rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\dryland_tiff_dict\\'
        T.mk_dir(outdir,force=True)


        year_list = list(range(1982, 2021))

        for fdir in T.listdir(fdir_all):

            for year in year_list:

                outdir_i = join(outdir, str(year))

                if isdir(outdir_i):
                    continue

                T.mk_dir(outdir_i, force=True)

                all_array = []

                # read all daily tif
                for f in sorted(T.listdir(join(fdir_all, fdir))):

                    if not f.endswith('.tif'):
                        continue

                    year_f = f[:4]

                    if year_f != str(year):
                        continue

                    fpath = join(fdir_all, fdir, f)

                    arr, originX, originY, pixelWidth, pixelHeight = \
                        ToRaster().raster2array(fpath)

                    arr = np.array(arr, dtype=np.float32)

                    arr = arr[:360, :720]

                    arr[arr < -999] = np.nan

                    all_array.append(arr)

                # stack
                all_array = np.array(all_array)

                # shape:
                # (time,row,col)

                print(all_array.shape)

                row = all_array.shape[1]
                col = all_array.shape[2]

                temp_dic = {}

                flag = 0

                for r in tqdm(range(row)):

                    for c in range(col):

                        ts = all_array[:, r, c]

                        temp_dic[(r, c)] = ts

                        flag += 1

                        if flag % 10000 == 0:
                            np.save(
                                join(outdir_i,
                                     f'per_pix_dic_{flag // 10000:03d}.npy'),
                                temp_dic
                            )

                            temp_dic = {}

                # save remain
                np.save(
                    join(outdir_i, 'per_pix_dic_000.npy'),
                    temp_dic
                )

    def transform(self):

        fdir_all = rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\dryland_tiff_dict\\'
        outdir =rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\\transform\\'
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(106):
            data_list.append(i)



        for data in data_list:

            dic_all_list = []
            for fdir_i in os.listdir(fdir_all):

                for f in os.listdir(fdir_all + fdir_i + '\\'):

                    if not f.endswith('.npy'):
                        continue
                    if f.split('.')[0].split('_')[-1] != '%03d' % data:
                        continue

                    dict = np.load(fdir_all + fdir_i + '\\' + f, allow_pickle=True).item()
                    dic_all_list.append(dict)

            result_dic = {}


            for pix in tqdm(dic_all_list[0].keys()):
                result_list = []
                for i in range(len(dic_all_list)):
                    if pix not in dic_all_list[i].keys():
                        continue
                    else:
                        # print(dic_all_list[i][pix])
                        result_list.append(dic_all_list[i][pix][0:365])

                result_dic[pix] = result_list
            ## save
            np.save(outdir + f'per_pix_dic_%03d' % data, result_dic)
            # print(result_dic)
    def detrend(self):
        fdir_all = rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\\transform\\'
        outdir = rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\detrend\\'
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(26):
            data_list.append(i)

        for data in data_list:

            dic_all = np.load(fdir_all + f'per_pix_dic_%03d.npy' % data, allow_pickle=True).item()
            result_dic = {}
            outf = outdir + f'per_pix_dic_%03d' % data + '.npy'
            print(outf)
            # if isfile(outf):
            #     continue
            for pix in tqdm(dic_all.keys()):
                vals = dic_all[pix]
                vals = np.array(vals)
                if np.isnan(np.mean(vals)):
                    continue
                # plt.imshow(vals)
                # plt.show()
                vals_flatten = vals.flatten()

                #
                if T.is_all_nan(vals_flatten):
                    continue
                if np.nanmean(vals_flatten) == 0:
                    continue

                detrend=T.detrend_vals(vals_flatten)
                # plt.bar(range(len(vals_flatten)),vals_flatten)


                # plt.bar(range(len(detrend)),detrend)
                # plt.show()

                detrend_reshape = detrend.reshape(-1, 365)
                # plt.imshow(anomaly_reshape,vmin=-.1,vmax=.1)
                # plt.show()

                result_dic[pix] = detrend_reshape
            np.save(outdir + f'per_pix_dic_%03d' % data, result_dic)
        pass

    def deseasonal(self):  ## temperature does not need to detrend
        fdir_all = rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\\transform\\'
        outdir = rf'C:\Users\wenzhang1.BLUECAT\Desktop\VPD\deseasonal\\'
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(26):

            data_list.append(i)

        for data in data_list:
            # if data !=5:
            #     continue
            dic_all = np.load(fdir_all + f'per_pix_dic_%03d.npy' % data, allow_pickle=True).item()
            result_dic = {}
            outf = outdir + f'per_pix_dic_%03d' % data + '.npy'
            print(outf)
            # if isfile(outf):
            #     continue
            for pix in tqdm(dic_all.keys()):
                vals = dic_all[pix]
                vals = np.array(vals)
                if np.isnan(np.mean(vals)):
                    continue
                # plt.imshow(vals)
                # plt.show()
                vals_flatten = vals.flatten()
                #
                if T.is_all_nan(vals_flatten):
                    continue

                anomaly = self.daily_climatology_anomaly(vals_flatten)

                anomaly_reshape = anomaly.reshape(-1, 365)
                # plt.imshow(anomaly_reshape,vmin=-.1,vmax=.1)
                # plt.show()

                result_dic[pix] = anomaly_reshape
            np.save(outdir + f'per_pix_dic_%03d' % data, result_dic)

        pass

    def daily_climatology_anomaly(self, vals):
        '''
        juping
        :param vals: 40 * 365
        :return:
        '''
        pix_anomaly = []
        climatology_means = []
        for day in range(1, 366):
            one_day = []
            for i in range(len(vals)):
                d = i % 365 + 1
                if day == d:
                    one_day.append(vals[i])
            mean = np.nanmean(one_day)
            std = np.nanstd(one_day)
            climatology_means.append(mean)
        for i in range(len(vals)):
            d_ind = i % 365
            mean_ = climatology_means[d_ind]
            anomaly = vals[i] - mean_
            pix_anomaly.append(anomaly)
        pix_anomaly = np.array(pix_anomaly)
        return pix_anomaly

    def detrend_deseasonal(self):
        fdir_all = data_root + rf'CRU-JRA\Tmax\transform\\'
        outdir = data_root+rf'CRU-JRA\Tmax\\\deseasonal_detrend\\'
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(106):

            data_list.append(i)

        for data in data_list:
            dic_all = np.load(fdir_all + f'per_pix_dic_%03d.npy' % data, allow_pickle=True).item()
            result_dic = {}
            outf=outdir + f'per_pix_dic_%03d' % data+'.npy'
            print(outf)
            # if isfile(outf):
            #     continue
            for pix in tqdm(dic_all.keys()):
                vals = dic_all[pix]
                vals=np.array(vals)
                vals_flatten = vals.flatten()
                #
                if T.is_all_nan(vals_flatten):
                    continue

                anomaly=self.daily_climatology_anomaly(vals_flatten)
                anomaly_detrend=T.detrend_vals(anomaly)
                # plt.bar(range(len(anomaly)),anomaly,color='red')
                #
                #
                # #
                # plt.bar(range(len(anomaly_detrend)),anomaly_detrend,color= 'b')
                # plt.show()
                anomaly_detrend_reshape = anomaly_detrend.reshape(-1, 365)

                result_dic[pix] = anomaly_detrend_reshape
            np.save(outdir + f'per_pix_dic_%03d' % data, result_dic)

        pass



def main():

    preprocessing_daily_VPD().run()


    pass

if __name__ == '__main__':
    main()



    pass