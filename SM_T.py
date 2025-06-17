# coding='utf-8'
import sys

import lytools
import pingouin
import pingouin as pg
from openpyxl.styles.builtins import percent
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

class data_processing():
    def __init__(self):
        pass

    def run(self):
        self.nc_to_tif_time_series_ERA5land()
        # self.resample()
        # self.scale()
        # self.unify_TIFF()
        # self.extract_dryland_tiff()
        # self.tif_to_dic()
        # self.plot_sptial()
        self.calculate_Hp()
        # self.zscore()
        # self.SM_T()

    def nc_to_tif_time_series_ERA5land(self):

        fdir=rf'D:\Project3\Data\SM_T\unzip\\'
        outdir_ssr=rf'D:\Project3\Data\SM_T\\TIFF\\solar_radiation\\'
        outdir_e=rf'D:\Project3\Data\SM_T\\TIFF\\evapotranspiration\\'
        outdir_Temp=rf'D:\Project3\Data\SM_T\\TIFF\\temperature\\'
        outdir_PET=rf'D:\Project3\Data\SM_T\\TIFF\\PET\\'
        Tools().mk_dir(outdir_ssr,force=True)
        Tools().mk_dir(outdir_e,force=True)
        Tools().mk_dir(outdir_Temp,force=True)
        Tools().mk_dir(outdir_PET,force=True)

        for fdir_i in os.listdir(fdir):
            fdir_i = join(fdir,fdir_i)
            for f in tqdm(os.listdir(fdir_i)):

                outdir_name = f.split('.')[0]
                # print(outdir_name)

                yearlist = list(range(1982, 2021))
                fpath = join(fdir_i,f)
                nc_in = xarray.open_dataset(fpath)
                # print(nc_in)
                time_bnds = nc_in['valid_time']
                # print(time_bnds)
                date=time_bnds[0]
                date = pd.to_datetime(date.values)
                date_str = date.strftime('%Y%m%d')
                # print(date_str);exit()
                array_t2m=nc_in['t2m'][0]
                array_t2m=np.array(array_t2m)
                # print(array_t2m.shape)
                array_t2m1=array_t2m[:,:array_t2m.shape[1]//2]
                array_t2m2=array_t2m[:,array_t2m.shape[1]//2:]
                array_t2m = np.concatenate((array_t2m2, array_t2m1), axis=1)
                # print(array_t2m.shape)
                # plt.imshow(array_t2m,interpolation='nearest',cmap='jet');plt.show()
                # print(array_t2m);exit()
                array_ssr=nc_in['ssr'][0]
                array_ssr=np.array(array_ssr)
                array_ssr1=array_ssr[:,:array_ssr.shape[1]//2]
                array_ssr2=array_ssr[:,array_ssr.shape[1]//2:]
                array_ssr = np.concatenate((array_ssr2, array_ssr1), axis=1)


                array_pev = nc_in['pev'][0]
                array_pev = np.array(array_pev)
                array_pev1 = array_pev[:, :array_pev.shape[1] // 2]
                array_pev2 = array_pev[:, array_pev.shape[1] // 2:]
                array_pev = np.concatenate((array_pev2, array_pev1), axis=1)
                plt.imshow(array_pev, interpolation='nearest', cmap='jet')
                plt.show()


                array_e = nc_in['e'][0]
                array_e = np.array(array_e)
                array_e1 = array_e[:, :array_e.shape[1] // 2]
                array_e2 = array_e[:, array_e.shape[1] // 2:]
                array_e = np.concatenate((array_e2, array_e1), axis=1)


                outf_t2m = join(outdir_Temp, f'{date_str}.tif')
                outf_ssr = join(outdir_ssr, f'{date_str}.tif')
                outf_pev = join(outdir_PET, f'{date_str}.tif')
                outf_e = join(outdir_e, f'{date_str}.tif')

                ToRaster().array2raster(outf_t2m, -180, 90, 0.1, -0.1, array_t2m, )
                ToRaster().array2raster(outf_ssr, -180, 90, 0.1, -0.1, array_ssr, )
                ToRaster().array2raster(outf_pev, -180, 90, 0.1, -0.1, array_pev, )
                ToRaster().array2raster(outf_e, -180, 90, 0.1, -0.1, array_e, )

    def resample(self):
        fdir=rf'D:\Project3\Data\GLEAM\\TIFF_Ep\\'
        outdir=rf'D:\Project3\Data\GLEAM_resample_Ep\\'
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath=fdir+f
            outf=outdir+f
            dataset = gdal.Open(fpath)

            try:
                gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass


    def scale(self):
        fdir=rf'D:\Project3\Data\GLEAM\GLEAM_resample_Ep\\'
        outdir=rf'D:\Project3\Data\GLEAM\GLEAM_scale_Ep\\'
        T.mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath=join(fdir,f)

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array = np.array(array, dtype=float)

            array = array*30
            # array[array>0]=np.nan
            # array=abs(array) *100

            outf = outdir + f
            ToRaster().array2raster(outf, -180, 90, 0.5, -0.5, array, )

    def unify_TIFF(self):
        fdir_all=rf'D:\Project3\Data\SM_T\TIFF\\PET_scale\\'
        outdir=rf'D:\Project3\Data\SM_T\TIFF\\unify\\'
        Tools().mk_dir(outdir, force=True)


        for f in os.listdir(join(fdir_all)):
            fpath=join(fdir_all,f)
            outpath=join(outdir,f)

            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            unify_tiff=DIC_and_TIF().unify_raster(fpath,outpath,0.5)




    def extract_dryland_tiff(self):
        self.datadir=rf'D:\Project3\Data\\'
        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask05.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan


        fdir_all = rf'D:\Project3\Data\GLEAM\\'

        for fdir in T.listdir(fdir_all):
            if  'GLEAM_scale_Ep' in fdir:
                continue
            if fdir.endswith('.nc'):
                continue


            fdir_i = join(fdir_all, fdir)

            outdir_i = join(fdir_all, 'dryland_tiff_E')

            T.mk_dir(outdir_i)
            # print(outdir_i);exit()
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


    def tif_to_dic(self):

        fdir_all = rf'D:\Project3\Data\GLEAM\\'

        year_list = list(range(1982, 2021))

        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            if not 'dryland_tiff_AE' in fdir:
                continue
            outdir=join(fdir_all, 'dic_AE')


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

                # array_unify[array_unify < -999] = np.nan
                # # array_unify[array_unify > 10] = np.nan
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
                    np.save(outdir + '\\per_pix_dic_%03d' % (flag / 10000), temp_dic)
                    temp_dic = {}
            np.save(outdir + '\\per_pix_dic_%03d' % 0, temp_dic)

    def calculate_Hp(self):
        ## H = Rn - lambda_ * E
        # Hp = Rn - lambda_ * Ep
        # lambda_ = 2.45(MJ / kg)
        ## Rn unit =j/m2  E unit = cm

        f_Rn=result_root + rf'\3mm\SM_T\ERA5\\extract_solar_radiation_phenology_year\solar_radiation.npy'
        f_E=result_root + rf'3mm\SM_T\GLEAM\\AE.npy'
        f_PET=result_root + rf'\3mm\SM_T\GLEAM\\\Ep.npy'

        Rn_dic=T.load_npy(f_Rn)  ### 38 year data
        E_dic=T.load_npy(f_E)
        Ep_dic=T.load_npy(f_PET)
        result_H={}
        result_Hp={}

        for pix in Rn_dic:
            if not pix in E_dic:
                continue
            if not pix in Ep_dic:
                continue
            Rn=Rn_dic[pix]['growing_season']
            E=E_dic[pix]['growing_season']
            Ep=Ep_dic[pix]['growing_season']
            Rn=np.array(Rn)
            E=np.array(E)
            Ep=np.array(Ep)

            ## Convert Rn from J to MJ
            Rn_MJ = Rn / 1e6
            # Convert E, Ep from cm to mm
            # E_mm = E * 10
            # Ep_mm = Ep * 10

            E_mm = E/30
            Ep_mm = Ep/30

            # Compute H and Hp
            H = Rn_MJ - 2.45 * E_mm
            Hp = Rn_MJ - 2.45 * Ep_mm
            result_H[pix]=H
            result_Hp[pix]=Hp

            plt.plot(H)
            plt.plot(Hp)
            plt.plot(Rn_MJ)
            plt.legend(['H','Hp'])
            plt.show()
        outdir=result_root + rf'\3mm\SM_T\GLEAM\H_Hp\\'
        Tools().mk_dir(outdir,force=True)
        T.save_npy(result_Hp,outdir+f'Hp.npy',)
        T.save_npy(result_H,outdir+f'H.npy',)






        pass

    def zscore(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)
        fdir = result_root + rf'\3mm\SM_T\extract_temperature_phenology_year\\'
        outdir = result_root + rf'3mm\SM_T\extract_temperature_phenology_year\zscore\\'
        Tools().mk_dir(outdir, force=True)
        # growing_season_list=['growing_season','ecosystem_year','non_growing_season']
        growing_season_list = ['growing_season', ]
        for season in growing_season_list:

            for f in os.listdir(fdir):
                if not f.endswith('.npy'):
                    continue





                outf = outdir + f.split('.')[0]+f'_{season}_zscore.npy'
                if isfile(outf):
                    continue
                print(outf)

                dic = T.load_npy(fdir + f)

                zscore_dic = {}

                for pix in tqdm(dic):

                    if pix not in dic_dryland_mask:
                        continue

                    print(len(dic[pix]))
                    time_series = dic[pix][season]


                    time_series = np.array(time_series)
                    # time_series = time_series[3:37]

                    print(len(time_series))

                    if np.isnan(np.nanmean(time_series)):
                        continue

                    time_series = time_series
                    mean = np.nanmean(time_series)
                    std=np.nanstd(time_series)
                    relative_change = (time_series - mean) / std
                    anomaly = time_series - mean
                    zscore_dic[pix] = relative_change
                    # plot
                    # plt.plot(time_series)
                    # plt.legend(['raw'])
                    # # plt.show()
                    #
                    # plt.plot(relative_change)
                    # plt.legend(['raw', 'zscore'])
                    # plt.show()

                    ## save
                np.save(outf, zscore_dic)

    def plot_sptial(self):
        ##['CABLE-POP_S2_lai',

        fdir = result_root + rf'\3mm\SM_T\extract_temperature_phenology_year\temperature.npy'


        dic=T.load_npy(fdir)


        average_={}

        for pix in dic:
            r,c=pix
            # (r,c)=(817,444)
            # if not r==444:
            #     continue
            # if not c==817:
            #     continue
            # if r<480:
            #     continue
            vals=dic[pix]['growing_season']
            average_[pix]=np.nanmean(vals)
        arr=DIC_and_TIF().pix_dic_to_spatial_arr(average_)
        plt.imshow(arr)
        plt.show()

    def SM_T(self):
        f_Hp_zscore=result_root+rf'3mm\SM_T\H_Hp_phenology_year\zscore\\Hp_growing_season_zscore.npy'
        f_H_zscore=result_root+rf'\3mm\SM_T\H_Hp_phenology_year\zscore\\H_growing_season_zscore.npy'
        f_Temp_zscore=result_root+rf'\3mm\SM_T\extract_temperature_phenology_year\zscore\\temperature_growing_season_zscore.npy'

        Hp_zscore=T.load_npy(f_Hp_zscore)
        H_zscore=T.load_npy(f_H_zscore)
        Temp_zscore=T.load_npy(f_Temp_zscore)

        pi_dic={}
        pi_average={}

        for pix in Hp_zscore:
            if pix not in H_zscore:
                continue
            if pix not in Temp_zscore:
                continue

            ## pi =(H-Hp)*T
            pi_dic[pix]=(H_zscore[pix]-Hp_zscore[pix])*Temp_zscore[pix]
            # print(pi_dic[pix])
            pi_average[pix]=np.nanmean(pi_dic[pix])
        array=DIC_and_TIF().pix_dic_to_spatial_arr(pi_average)
        plt.imshow(array,vmin=0,vmax=1,cmap='jet')
        plt.show()

        T.save_npy(pi_dic,result_root+rf'\3mm\SM_T\pi.npy')



        pass





def main():
    data_processing().run()








if __name__ == '__main__':
    main()