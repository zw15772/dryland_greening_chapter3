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

class US_soil_project(object):

    def __init__(self):

        pass
    def run (self):
        # self.clip_raster_using_shp()
        # self.scaling_GPP()
        # self.tif_to_dic()
        # self.annual_sum()  ## yearly sum
        # self.scaling_GPP_MODIS()
        self.clip_raster_using_shp_MODIS()
        # self.multi_year_average()


    def clip_raster_using_shp(self):

        f_shp = rf'E:\US_soil_project\\conus\\conus.shp'  # the shapefile you want to clip raster obj
        fdir = rf'E:\US_soil_project\GPP_CFE_NT\TIFF\\'
        outdir = rf'E:\US_soil_project\GPP_CFE_NT\\\conus\\'

        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            outf=outdir+f
            inputraster = fdir + f

            ToRaster().clip_array(inputraster, outf, f_shp)
            # gdal.Warp(
            #     outf,
            #     inputraster,
            #     cutlineDSName=f_shp,
            #     cropToCutline=True,
            #     dstAlpha=True
            # )


        pass
    def scaling_GPP(self): ##
        ## Yanghui's GPP scaling factor is 0.01 unit is gC/m2/day
        ## Songhan's GPP scaling factor is 0.001 unit is gC/m2/day
        fdir=rf'E:\US_soil_project\NIRvGPP\conus\\'
        outdir=rf'E:\US_soil_project\NIRvGPP\conus_scaling_monthly\\'
        Tools().mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array=array
            array[array<0]=np.nan
            array[array<-999]=np.nan
            ## convert it to annual mean GPP
            array_scaling=array*30

            DIC_and_TIF(tif_template=fdir + f).arr_to_tif(array_scaling, outdir+f)



        pass

    def scaling_GPP_MODIS(self): ##


        f=rf'E:\US_soil_project\MODIS_GPP_average\\2001-01-01_2023-1-1.tif'
        outdir=rf'E:\US_soil_project\MODIS_GPP_average\\multi_year_average\\'

        Tools().mk_dir(outdir,force=True)

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        array[array < 0] = np.nan
        array[array < -999] = np.nan
        ## convert it to annual mean GPP
        array_scaling=array/8*360/10000*1000
        outf=outdir+'GPP_multi_year_average.tif'
        DIC_and_TIF(tif_template=f).arr_to_tif(array_scaling, outf)



        pass

    def clip_raster_using_shp_MODIS(self):

        f_shp = rf'E:\US_soil_project\\conus\\conus.shp'  # the shapefile you want to clip raster obj
        fdir = rf'E:\US_soil_project\MODIS_GPP_average\\multi_year_average\\'
        outdir = rf'E:\US_soil_project\GPP_CFE_NT\\\conus\\'

        T.mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            outf=outdir+f
            inputraster = fdir + f

            ToRaster().clip_array(inputraster, outf, f_shp)
            # gdal.Warp(
            #     outf,
            #     inputraster,
            #     cutlineDSName=f_shp,
            #     cropToCutline=True,
            #     dstAlpha=True
            # )


        pass


    def tif_to_dic(self):

        fdir_all = rf'E:\US_soil_project\NIRvGPP\\'


        year_list = list(range(1982, 2021))


        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            if not 'conus_scaling_monthly' in fdir:
                continue

            outdir = rf'E:\US_soil_project\NIRvGPP\conus_dic\\'
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



                array[array < -999] = np.nan
                # array_unify[array_unify > 7] = np.nan
                # array[array ==0] = np.nan

                array[array < 0] = np.nan


                # plt.imshow(array)



                all_array.append(array)

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

    def annual_sum(self):
        fdir=rf'E:\US_soil_project\GPP_CFE_NT\conus_scaling_monthly\\'
        outdir=rf'E:\US_soil_project\GPP_CFE_NT\annual_sum\\'
        Tools().mk_dir(outdir,force=True)

        tif_template=r"E:\US_soil_project\GPP_CFE_NT\conus_scaling_monthly\19820101.tif"
        # DIC_and_TIF(tif_template=tif_template);exit()
        year_list = list(range(1982, 2021))

        for year in year_list:
            print(year)
            array_list=[]

            for f in os.listdir(fdir):
                if not f.endswith('.tif'):
                    continue
                if int(f.split('.')[0][0:4])!=year:
                    continue
                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
                array = np.array(array,dtype=np.float64)
                array[array < 0] = np.nan

                array_list.append(array)

            array_sum = np.nansum(array_list,axis=0)
            array_sum = np.array(array_sum,dtype=float)
            array_sum[array_sum <= 0] = np.nan
            # print(array_sum.shape);exit()
            # plt.imshow(array_sum);plt.show()
            outf=outdir+'GPP_%04d.tif'%year
            # print(outf)

            DIC_and_TIF(tif_template=tif_template).arr_to_tif(array_sum,outf)
    def multi_year_average(self):
        fdir=rf'E:\US_soil_project\GPP_CFE_NT\annual_sum\\'
        outdir=rf'E:\US_soil_project\GPP_CFE_NT\multi_year_average\\'
        template_tif=rf'E:\US_soil_project\GPP_CFE_NT\conus_scaling_monthly\19820101.tif'
        Tools().mk_dir(outdir,force=True)
        array_list=[]

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)

            array[array < 0] = np.nan

            array_list.append(array)

        array_sum = np.nanmean(array_list, axis=0)
        outf=outdir+'GPP_multi_year_average.tif'
        DIC_and_TIF(tif_template=template_tif).arr_to_tif(array_sum, outf)








def main():
    US_soil_project().run()




    pass

if __name__ == '__main__':
    main()

