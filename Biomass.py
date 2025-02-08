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
import sys
import xarray


this_root = 'E:\\'
data_root = 'E:/Biomass/Data/'
result_root = 'E:/Biomass/Result/'


class Data_processing_2:

    def __init__(self):

        pass

    def run(self):

        # self.nc_to_tif_time_series()

        # self.dryland_mask()
        # self.test_histogram()
        # self.resampleSOC()
        # self.reclassification_koppen()
        # self.aggregation_soil()


        # self.extract_tiff_by_shp()
        # self.average_tiff()
        # self.plot_bar()
        # self.rename_tiff()
        self.plot_biomass_vs_NPP()


        pass

    def nc_to_tif_time_series(self):
        import netCDF4

        fdir=data_root+rf'NPP\\nc\\'

        for f in os.listdir(fdir):
            if not 'ISAM_S2' in f:
                continue



            fname=f.split('.')[0]
            outdir = data_root + rf'NPP\\TIFF\\{fname}\\'

            Tools().mk_dir(outdir, force=True)

            outdir_name = f.split('.')[0]
            print(outdir_name)

            yearlist = list(range(1980, 2021))


            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            # try:
            #     self.nc_to_tif_template(fdir+f, var_name='npp', outdir=outdir, yearlist=yearlist)
            ncin = netCDF4.Dataset(fdir + f, 'r')
            ncin = Dataset(fdir + f, 'r')
            print(ncin.variables.keys())
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units
            print(basetime_str);exit()
            # except Exception as e:
            #     print(e)
            #     continue

    def nc_to_tif_template(self, fname, var_name, outdir, yearlist):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())
            # time=ncin.variables['time'][:]
            # time_counter=ncin.variables['time_counter'][:]

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
                    try:
                        lat = ncin.variables['lat_FULL_bnds'][:]
                        lon = ncin.variables['lon_FULL_bnds'][:]
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
                                basetime = datetime.datetime.strptime(basetime, '%Y')
                            except:
                                try:
                                    basetime = datetime.datetime.strptime(basetime, '%Y.%f')
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


    def dryland_mask(self):
        ## here we extract dryland tif
        fpath=rf'D:\Project3\Data\Base_data\aridity_index05.tif\\aridity_index.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        NDVI_mask_f = rf'D:\Greening\Data\Base_data\\NDVI_mask.tif'
        array_NDVI_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)

        array[array_NDVI_mask < 0.1] = np.nan


        array[array >= 0.65] = np.nan


        outf=rf'D:\Project3\Data\Base_data\aridity_index05.tif\\dryland_mask.tif'
        DIC_and_TIF().arr_to_tif(array, outf)

    def test_histogram(self):

        fpath = rf'E:\Project3\Data\ERA5_daily\dict\moving_window_average_anaysis\\detrended_annual_LAI4g_CV.npy'
        spatial_dic=np.load(fpath, allow_pickle=True, encoding='latin1').item()
        data_list=[]

        for pix in spatial_dic:
            val=spatial_dic[pix]
            data_list.append(val)

        data_list=np.array(data_list)
        ## histogram

        plt.hist(data_list,bins=50)

        plt.show()




    def resampleSOC(self):
        f=rf'E:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth.tif'

        outf = rf'E:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth_05.tif'

        dataset = gdal.Open(f)



        try:
            gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
        # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
        # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
        except Exception as e:
            pass

    def nc_to_tif(self):

        f=rf'E:\Project3\Data\GIMMS3g_plus_NDVI\\'
        outdir=rf'D:\Project3\Data\Base_data\HWSD\nc\\tif\\'
        Tools().mk_dir(outdir,force=True)

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

        year = int(f.split('.')[-3][0:4])
        month = int(f.split('.')[-3][4:6])

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
        array = array * 0.001  ### GPP need scale factor
        array[array < 0] = np.nan

        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()
        ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, )

    def reclassification_koppen(self):
        f=rf'E:\Project3\Data\Base_data\Koeppen_tif\\Koeppen.tif_unify.tif'
        dic={0: 'Tropical', 1: 'Tropical', 2: 'Tropical', 3: 'Tropical', 4: 'Dry', 5: 'Dry', 6: 'Dry', 7: 'Dry', 8: 'Temperate', 9: 'Temperate', 10: 'Temperate',
         11: 'Temperate', 12: 'Temperate', 13: 'Temperate', 14: 'Temperate', 15: 'Temperate', 16: 'Temperate', 17: 'Continental', 18: 'Continental', 19: 'Continental', 20: 'Continental',
         21: 'Continental', 22: 'Continental', 23: 'Continental', 24: 'Continental', 25: 'Continental', 26: 'Continental', 27: 'Continental', 28: 'Polar', 29: 'Polar'}

        reclassification_dic = {
            'Tropical': 1,
            'Dry': 2,
            'Temperate': 3,
            'Continental': 4,
            'Polar': 5

        }
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        spatial_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        for pix in spatial_dic:
            val = spatial_dic[pix]
            if val not in dic:
                spatial_dic[pix] = np.nan
                continue
            if val<-99:
                spatial_dic[pix]=np.nan
                continue

            class_i = dic[val]
            spatial_dic[pix] = reclassification_dic[class_i]


        arr_new = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr_new, interpolation='nearest', cmap='jet')
        plt.colorbar()
        plt.title('Koeppen')
        plt.show()
        outf=rf'E:\Project3\Data\Base_data\Koeppen_tif\\Koeppen_reclassification.tif'
        DIC_and_TIF().arr_to_tif(arr_new, outf)

        pass





    def aggregation_soil(self):
        ## aggregation soil sand, CEC etc

        fdir = data_root+rf'Base_data\SoilGrid\SOIL_Grid_05_unify\\'

        product_list=['soc', 'sand', 'nitrogen', 'cec']

        for product_i in product_list:
            ##cec_15-30cm_mean_5000_05
            f_layer1 = fdir + rf'{product_i}_0-5cm_mean_5000_05.tif'
            f_layer2 = fdir + rf'{product_i}_5-15cm_mean_5000_05.tif'
            f_layer3 = fdir + rf'{product_i}_15-30cm_mean_5000_05.tif'


            array1, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_layer1)
            array2, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_layer2)
            array3, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_layer3)

            array1 = np.array(array1, dtype=float)*1/6
            array2 = np.array(array2, dtype=float)*1/3
            array3 = np.array(array3, dtype=float)*1/2

            aggregation_array =array1+array2+array3
            aggregation_array[aggregation_array < -99] = np.nan
            ToRaster().array2raster(fdir + rf'{product_i}.tif', originX, originY, pixelWidth, pixelHeight, aggregation_array, )



        pass
    def extract_tiff_by_shp(self):

        shp_dryland = rf'E:\Biomass\Data\Basedata\\dryland.shp'
        fdir = rf'E:\Biomass\Data\NPP\TIFF\\'
        outdir = rf'E:\Biomass\Data\NPP\dryland_tiff\\'
        T.mk_dir(outdir)
        for fdir_i in tqdm(os.listdir(fdir)):
            outdir_i = join(outdir, fdir_i)
            T.mk_dir(outdir_i)

            fdir_i = join(fdir, fdir_i)
            print(fdir_i)

            for f in os.listdir(fdir_i):
                if not f.endswith('.tif'):
                    continue
                inf = join(fdir_i, f)

                outf=join(outdir_i,f)
                ToRaster().clip_array(inf, outf ,shp_dryland,)




        pass

    def average_tiff(self):
        ## calculate spatial average of long term
        fdir = rf'E:\Biomass\Data\NPP\dryland_tiff\\'
        outdir = rf'E:\Biomass\Data\NPP\dryland_tiff_average\\'
        T.mk_dir(outdir, force=True)
        for fdir_i in tqdm(os.listdir(fdir)):

            fdir_i = join(fdir, fdir_i)
            # print(fdir_i)

            all_array = []
            flag=0

            for f in os.listdir(fdir_i):
                flag+=1

                if not f.endswith('.tif'):
                    continue
                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_i, f))
                array = np.array(array, dtype=float)
                array[array <= 0] = np.nan
                array=array*24*60*60*30 ## GPP unit is kg/m2/month

                all_array.append(array)

            all_array = np.array(all_array)
            year_length = flag//12
            # print(year_length);exit()
            all_array_mean = np.nansum(all_array, axis=0)/year_length
            # all_array_mean = np.nanmean(all_array, axis=0)
            variable_name = fdir_i.split('/')[-1]
            print(outdir)

            outf = join(outdir, f'{variable_name}_mean.tif')
            # print(outf);exit()
            ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, all_array_mean)

    def plot_bar(self):
        fdir = rf'E:\Biomass\Data\Biomass\dryland_tiff_average\\'
        variable_list=['CABLE-POP','CLASSIC','CLM5','IBIS',
                       'ISBA-CTRIP','JSBACH','JULES','LPJ-GUESS','ORCHIDEE',
                       'SDGVM']

        data_all={}
        uncertainty_all={}
        for f in os.listdir(fdir):

            if not f.endswith('.tif'):
                continue
            fname=(f.split('.')[0].split('_')[0])
            if fname not in variable_list:
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir, f))
            array = np.array(array, dtype=float)
            array[array < 0] = np.nan
            array[array>9999] = np.nan
            #### average spatially
            average = np.nansum(array)
            std=np.nanstd(array)

            data_all[fname]=average
            uncertainty_all[fname]=std
        result_all={'average':data_all,'uncertainty':uncertainty_all}


        df=pd.DataFrame.from_dict(result_all,orient='index')

        df=df.T
        df=df.sort_index()

        df.plot(kind='bar',yerr='uncertainty',figsize=(6, 4),rot=45,legend=False,color='grey',alpha=0.7)

        # df.plot(kind='bar',figsize=(8, 8),rot=45,legend=False,color='grey')
        plt.xticks(rotation=45)
        # plt.ylim(0, 10000)
        plt.ylabel('Biomass (Kg/m2)')
        plt.tight_layout()

        plt.show()
        # plt.savefig(join(f,'plot_bar.png'))
        # plt.close()


    def rename_tiff(self):
        dic={
            'CABLE-POP_S2_npp_mean':'CABLE-POP',
        'CLASSIC_S2_npp_mean':'CLASSIC',
        'CLM5_S2_mean':'CLM5',
        'DLEM_S2_npp_mean':'DLEM',
        'IBIS_S2_npp_mean':'IBIS',
        'ISBA-CTRIP_S2_npp_mean':'ISBA-CTRIP',
        'JSBACH_S2_npp_mean':'JSBACH',
        'JULES_S2_npp_mean':'JULES',
        'LPJmL_S2_npp_mean':'LPJmL',
            'LPJwsl_S2_npp_mean':'LPJwsl',
            'OCN_S2_npp_mean':'OCN',
            'ORCHIDEE_S2_npp_mean':'ORCHIDEE',
            'SDGVM_S2_npp_mean':'SDGVM',
            'VISIT_S2_npp_mean':'VISIT',
            'YIBs_S2_Monthly_npp_mean':'YIBs',}

        # dic = {
        #     'CABLE-POP_S2_gpp_mean': 'CABLE-POP',
        #     'CLASSIC_S2_gpp_mean': 'CLASSIC',
        #     'CLM5_S2_gpp_mean': 'CLM5',
        #     'DLEM_S2_gpp_mean': 'DLEM',
        #     'IBIS_S2_gpp_mean': 'IBIS',
        #     'ISBA-CTRIP_S2_gpp_mean': 'ISBA-CTRIP',
        #     'JSBACH_S2_gpp_mean': 'JSBACH',
        #     'JULES_S2_gpp_mean': 'JULES',
        #     'LPJmL_S2_gpp_mean': 'LPJmL',
        #     'LPJwsl_S2_gpp_mean': 'LPJwsl',
        #     'OCN_S2_gpp_mean': 'OCN',
        #     'ORCHIDEE_S2_gpp_mean': 'ORCHIDEE',
        #     'SDGVM_S2_gpp_mean': 'SDGVM',
        #     'VISIT_S2_gpp_mean': 'VISIT',
        #     'YIBs_S2_Monthly_gpp_mean': 'YIBs', }
        #

        fdir = rf'E:\Biomass\Data\NPP\dryland_tiff_average\\'

        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fname=(f.split('.')[0].split('_')[0])

            if fname in dic:
                fname=dic[fname]
            os.rename(join(fdir,f),join(fdir,fname+'.tif'))
    def plot_biomass_vs_NPP(self):
        fdir_NPP = rf'E:\Biomass\Data\NPP\dryland_tiff_average\\'
        average_NPP={}
        average_biomass={}

        variable_list=['DLEM','VISIT','YIBs','LPJmL','OCN','LPJwsl']
        for f in os.listdir(fdir_NPP):
            if not f.endswith('.tif'):
                continue
            if f.split('.')[0].split('_')[0]  in variable_list:
                continue
            f_NPP_name=(f.split('.')[0].split('_')[0])
            f_NPP=join(fdir_NPP,f)
            f_bimass_name=(f.split('.')[0].split('_')[0])
            f_biomass=join(rf'E:\Biomass\Data\Biomass\dryland_tiff_average\\',f_bimass_name+'.tif')
            if not os.path.exists(f_NPP) or not os.path.exists(f_biomass):
                continue
            array_NPP, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_NPP)
            array_NPP = np.array(array_NPP, dtype=float)
            array_NPP[array_NPP <=0] = np.nan
            array_NPP[array_NPP>9999] = np.nan
            average_NPP[f_NPP_name]=np.nanmean(array_NPP)


            array_biomass, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_biomass)
            array_biomass = np.array(array_biomass, dtype=float)
            array_biomass[array_biomass <= 0] = np.nan
            array_biomass[array_biomass>9999] = np.nan
            average_biomass[f_bimass_name]=np.nansum(array_biomass)
        df=pd.DataFrame({'GPP':average_NPP,'biomass':average_biomass})

        color_list = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'gray',
                       'pink', 'brown', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'maroon', 'navy',
                      'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive',
                      'silver', 'aqua', 'fuchsia']

        marker_list = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D',]

        df=df.sort_index()
        ## using different markers


        for i in range(len(df)):
            plt.scatter(df.iloc[i]['biomass'],df.iloc[i]['GPP'],color=color_list[i],marker=marker_list[i],s=100)
        plt.legend(df.index)

        plt.ylabel('NPP (kg/m2/yr)',fontsize=10)
        plt.xlabel('Biomass (kg/m2)',fontsize=10)

        plt.show()
        plt.close()




        pass




def main():
    Data_processing_2().run()



    pass

if __name__ == '__main__':
    main()
