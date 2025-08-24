# coding='utf-8'
import sys
from math import isnan

import lytools
import pingouin
import pingouin as pg
from openpyxl.styles.builtins import percent
# from green_driver_trend_contribution import *
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import t
from sklearn.pipeline import make_pipeline
from sympy import expand_func

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

class products_check():
    def __init__(self):
        pass

    def run(self):
        # self.check_consistency()  ### selected regions showing differences

        # self.calculate_significance_count()
        # self.statistic_trend_CV()
        # self.plot_products_pdf()
        self.plot_time_series()




    def check_consistency(self):
        dff=result_root+rf'\3mm\Dataframe\relative_change_growing_season\\relative_change_growing_season_yearly.df'
        df=T.load_df(dff)
        T.print_head_n(df)
        variable_list=['LAI4g','NDVI4g','NDVI','GIMMS_plus_NDVI']
        result_dic={}

        regions = {
            "Western_US": {"lat_min": 24, "lat_max": 42, "lon_min": -112, "lon_max": -97},
            "Southern_South_America": {"lat_min": -18, "lat_max": -2, "lon_min": -45, "lon_max": -36},
            "Centra_Asia": {"lat_min": 44, "lat_max": 48, "lon_min": 43, "lon_max": 68},
            "Australia": {"lat_min": -33, "lat_max": -26, "lon_min": 143, "lon_max": 150},

        }
        year_range = range(1983, 2021)

        for region in regions:
            df_filtered = df[(df['lat'] > regions[region]['lat_min']) & (df['lat'] < regions[region]['lat_max']) & (
                    df['lon'] > regions[region]['lon_min']) & (df['lon'] < regions[region]['lon_max'])]


            for var in variable_list:

                result_dic[var] = {}
                data_dic = {}

                for year in year_range:
                    df_i = df_filtered[df_filtered['year'] == year]
                    # vals = df_i[f'weighted_avg_{var}'].tolist()
                    vals = df_i[f'{var}'].tolist()
                    data_dic[year] = np.nanmean(vals)
                result_dic[var] = data_dic
            ##dic to df



            ## plot
            color_list=['black','red','blue','green']

            for var in variable_list:


                plt.plot(year_range, result_dic[var].values(), label=var, color=color_list[variable_list.index(var)])



                plt.ylabel(f'Relative change (%)', fontsize=10, font='Arial')
                plt.xticks(font='Arial', fontsize=10)
                plt.yticks(font='Arial', fontsize=10)
                # plt.ylim(-20, 20)
                ##add line
                plt.axhline(y=0, color='grey', linestyle='--', alpha=0.2)
                # plt.grid(which='major', alpha=0.2)
                plt.legend()
            plt.title(region)
            plt.show()



            pass


        pass

    def calculate_significance_count(self):
        # f=rf'D:\Project3\Result\3mm\extract_GIMMS3g_plus_NDVI_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\detrended_GIMMS_plus_NDVI_CV_trend.npy'

        # pvalue_f=rf'D:\Project3\Result\3mm\extract_GIMMS3g_plus_NDVI_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\detrended_GIMMS_plus_NDVI_CV_p_value.npy'

        # f=rf'D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.npy'
        #
        # pvalue_f=rf'D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_p_value.npy'

        # f=rf'D:\Project3\Result\3mm\extract_NDVI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\detrended_NDVI4g_CV_trend.npy'
        #
        # pvalue_f=rf'D:\Project3\Result\3mm\extract_NDVI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\detrended_NDVI4g_CV_p_value.npy'

        # f=rf'D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\NDVI_detrend_CV_trend.npy'
        #
        # pvalue_f=rf'D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\NDVI_detrend_CV_p_value.npy'

        # f=rf'D:\Project3\Result\3mm\extract_MODIS_LAI_2002-2024_phenology_year\moving_window_extraction\trend\\detrended_MODIS_LAI_CV_trend_2002_2020.npy'
        #
        # pvalue_f=rf'D:\Project3\Result\3mm\extract_MODIS_LAI_2002-2024_phenology_year\moving_window_extraction\trend\\detrended_MODIS_LAI_CV_p_value_2002_2020.npy'
        f=rf'D:\Project3\Result\3mm\extract_GEODES_AVHRR_LAI_phenology_year\moving_window_extraction\trend\\detrended_GEODES_AVHRR_LAI_CV_trend.npy'

        pvalue_f=rf'D:\Project3\Result\3mm\extract_GEODES_AVHRR_LAI_phenology_year\moving_window_extraction\trend\\detrended_GEODES_AVHRR_LAI_CV_p_value.npy'
        array_trend=np.load(f)
        array_pvalue=np.load(pvalue_f)
        num=0
        total_count=0
        non_significant=0
        dic_trend=DIC_and_TIF().spatial_arr_to_dic(array_trend)
        dic_pvalue=DIC_and_TIF().spatial_arr_to_dic(array_pvalue)

        for pix in dic_trend:
            r,c=pix
            if r<60:
                continue

            trend=dic_trend[pix]
            if trend<-9999:
                continue
            total_count+=1



            p_value=dic_pvalue[pix]
            if trend>0:
                non_significant+=1

            if trend>0 and p_value<0.05:
                num+=1
        percentage=num/total_count
        non_significant_percentage=non_significant/total_count
        print(percentage)
        print(non_significant_percentage)





        pass

    def plot_products_pdf(self):
        fdir=rf'D:\Project3\Result\3mm\plot_products_CV_pdf\\'
        color_list=['r','g','b','k','m']
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if 'p_value' in f:
                continue
            product_name=f.split('.')[0]
            ## plot PDF
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
            array[array<-9999]=np.nan
            array=array.flatten()
            x,y=Plot().plot_hist_smooth(array, range=(-2,2),bins=100,alpha=0)

            plt.plot(x,y,label=product_name,color=color_list.pop(0))
        plt.legend()

        plt.show()


        pass

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]
        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['landcover_classfication'] != 'Cropland']


        return df

class NDVI_LAI():

    def __init__(self):
        pass

    def run(self):
        # NDVI_arr,LAI_arr=self.read_data()
        # self.NDVI_LAI_exponential_fitting(NDVI_arr,LAI_arr)
        # self.NDVI_LAI_polynomial_fitting(NDVI_arr,LAI_arr)
        # self.NDVI_LAI_linear_fitting(NDVI_arr,LAI_arr)
        self.NDVI_LAI_exponential_fitting_pixel()
        # self.NDVI_LAI_polynomial_fitting_pixel()
        # self.apply_model()
        # self.plot_data()
        # self.difference_predict_original()

        pass

    def read_data(self):

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        f = rf'D:\Project3\Result\3mm\NDVI_LAI\\SNU_NDVI.npy'
        outf = rf'D:\Project3\Result\3mm\NDVI_LAI\\SNU_LAI.npy'
        dic_NDVI = T.load_npy(f)
        dic_LAI = T.load_npy(outf)
        NDVI_list = []
        LAI_list = []
        for pix in dic_NDVI:
            r, c = pix
            if r < 60:
                continue
            if pix not in dic_LAI:
                continue

            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue
            if dic_modis_mask[pix] == 12:
                continue
            NDVI = dic_NDVI[pix]['growing_season']
            LAI = dic_LAI[pix]['growing_season']
            if len(NDVI) != len(LAI):
                continue
            if len(NDVI) != 38:
                continue
            if len(LAI) != 38:
                continue
            NDVI_list.append(NDVI)
            LAI_list.append(LAI)

        NDVI_arr = np.array(NDVI_list)
        LAI_arr = np.array(LAI_list)
        NDVI_arr = NDVI_arr.flatten()
        LAI_arr = LAI_arr.flatten()
        return NDVI_arr,LAI_arr

    pass

    def NDVI_LAI_polynomial_fitting(self,NDVI_arr,LAI_arr):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score



        ### fitting into polynomial model ##


        mask = NDVI_arr <= 0.9

        x = NDVI_arr[mask].reshape(-1, 1)
        y = LAI_arr[mask]
        # Try 2nd-degree polynomial

        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x)
        # print(x_poly)
        model=LinearRegression()


        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        r2_poly = r2_score(y, y_pred)
        ## plot equation

        a=model.coef_[2]
        b=model.coef_[1]

        c=model.intercept_

        LAI=f'{a:.2f} * NDVI² + {b:.2f} * NDVI + {c:.2f}'
        print(LAI)


        ## sort for clean plot


        # Plot
        plt.scatter(x, y, s=10, alpha=0.5)
        plt.scatter(x[:,np.newaxis],y_pred,color="red")
        print(r2_poly)
        R2=f'R2={r2_poly:.2f}'
        plt.text(0.5, 0.9, R2, transform=plt.gca().transAxes, fontsize=12)
        plt.xlabel('Annual growing season NDVI')
        plt.ylabel('Annual growing season LAI')
        plt.title(LAI)

        plt.show()


        pass

    def NDVI_LAI_exponential_fitting(self,NDVI_arr,LAI_arr):

        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score


        # mask = NDVI_arr <= 0.9
        #
        # NDVI_arr = NDVI_arr[mask]
        # LAI_arr = LAI_arr[mask]

        popt,_=curve_fit(self.exp_func,NDVI_arr,LAI_arr)
        a,b,c=popt
        T.save_dict_to_binary({'a':a,'b':b,'c':c},rf'D:\Project3\Result\3mm\NDVI_LAI\\SNU_LAI_exponential_fitting.pkl')

        LAI_predict=self.exp_func(NDVI_arr,a,b,c)

        R2=r2_score(LAI_arr,LAI_predict)

        plt.scatter(NDVI_arr,LAI_arr,s=10,alpha=0.5)
        plt.scatter(NDVI_arr,LAI_predict,color='red')
        R2=f'R2={R2:.2f}'
        plt.text(0.5, 0.9, R2, transform=plt.gca().transAxes, fontsize=12)
        plt.xlabel('Annual growing season NDVI')
        plt.ylabel('Annual growing season LAI')
        plt.title(f'{a:.2f} * exp({b:.2f} * NDVI) + {c:.2f}')
        plt.show()

        return a,b,c

    def NDVI_LAI_exponential_fitting_pixel(self):

        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        NDVI_fdir = rf'D:\Project3\Data\NDVI4g\dic_dryland\\'
        LAI_fdir = rf'D:\Project3\Data\LAI4g\dic_dryland\\'
        dic_NDVI = T.load_npy_dir(NDVI_fdir)
        dic_LAI = T.load_npy_dir(LAI_fdir)
        result_dic={}
        spatial_dict = {}
        R2_dic={}
        for pix in tqdm(dic_NDVI):
            r, c = pix
            if r < 60:
                continue
            if pix not in dic_LAI:
                continue
            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue
            if dic_modis_mask[pix] == 12:
                continue
            NDVI = dic_NDVI[pix]
            LAI = dic_LAI[pix]
            if len(NDVI) != len(LAI):
                continue
            if len(NDVI) != 936:
                continue
            if len(LAI) != 936:
                continue
            # print(NDVI)
            # print(LAI)
            NDVI = np.array(NDVI)
            LAI = np.array(LAI)
            # spatial_dict[pix] = 1

            try:

                popt, pcov = curve_fit(self.exp_func, NDVI, LAI,maxfev=5000)
                a, b, c = popt
                LAI_predict = self.exp_func(NDVI, a, b, c)
            except:
                continue

            R2 = r2_score(LAI, LAI_predict)
            result_dic[pix] = [a, b, c, R2]
            R2_dic[pix] = R2
            spatial_dict[pix] = 1






            # print(type(NDVI));exit()
            # LAI_predict=self.exp_func(NDVI,a,b,c)
            # R2=r2_score(LAI,LAI_predict)
            # result_dic[pix]=[a,b,c,R2]
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        arr_R2 = DIC_and_TIF().pix_dic_to_spatial_arr(R2_dic)

        # plt.imshow(arr_R2,interpolation='nearest',cmap='RdYlGn')
        # plt.colorbar()
        # plt.show()
        DIC_and_TIF().arr_to_tif(arr_R2, rf'D:\Project3\Result\3mm\NDVI_LAI\\LAI4g_exponential_fitting_pixel_R2.tif')

        T.save_npy(result_dic,rf'D:\Project3\Result\3mm\NDVI_LAI\\LAI4g_exponential_fitting_pixel.npy')


    def NDVI_LAI_polynomial_fitting_pixel(self):

        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score

        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_05.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        f = rf'D:\Project3\Result\3mm\NDVI_LAI\\SNU_NDVI.npy'
        outf = rf'D:\Project3\Result\3mm\NDVI_LAI\\SNU_LAI.npy'
        dic_NDVI = T.load_npy(f)
        dic_LAI = T.load_npy(outf)
        result_dic={}
        spatial_dict = {}
        R2_dic={}
        for pix in tqdm(dic_NDVI):
            r, c = pix
            if r < 60:
                continue
            if pix not in dic_LAI:
                continue
            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue
            if dic_modis_mask[pix] == 12:
                continue
            NDVI = dic_NDVI[pix]['growing_season']
            LAI = dic_LAI[pix]['growing_season']
            if len(NDVI) != len(LAI):
                continue
            if len(NDVI) != 38:
                continue
            if len(LAI) != 38:
                continue
            # print(NDVI)
            # print(LAI)
            NDVI = np.array(NDVI).reshape(-1, 1)
            LAI = np.array(LAI)
            # spatial_dict[pix] = 1

            poly = PolynomialFeatures(degree=2)
            x_poly = poly.fit_transform(NDVI)
            # print(x_poly)
            model = LinearRegression()

            model.fit(x_poly, LAI)
            y_pred = model.predict(x_poly)
            R2 = r2_score(LAI, y_pred)
            ## plot equation

            a = model.coef_[2]
            b = model.coef_[1]

            c = model.intercept_

            # LAI_predict = a * NDVI ** 2 + b * NDVI + c
            result_dic[pix] = [a, b, c, R2]
            R2_dic[pix] = R2
            spatial_dict[pix] = 1



        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        arr_R2 = DIC_and_TIF().pix_dic_to_spatial_arr(R2_dic)

        plt.imshow(arr_R2,interpolation='nearest',cmap='jet',vmin=0.9,vmax=1)
        plt.colorbar()
        plt.show()
        DIC_and_TIF().arr_to_tif(arr_R2, rf'D:\Project3\Result\3mm\NDVI_LAI\\SNU_LAI_Polynomial_fitting_pixel_R2.tif')

        T.save_npy(result_dic,rf'D:\Project3\Result\3mm\NDVI_LAI\\SNU_LAI_Polynomial_fitting_pixel.npy')



    def apply_model(self):
        f_NDVI_4g=rf'D:\Project3\Data\NDVI4g\dic_dryland\\'

        coff_f=rf'D:\Project3\Result\3mm\NDVI_LAI\\LAI4g_exponential_fitting_pixel.npy'

        LAI_predict_dic={}

        dict_coff=T.load_npy(coff_f)


        dic_NDVI=T.load_npy_dir(f_NDVI_4g)
        for pix in dic_NDVI:
            if pix not in dict_coff:
                continue
            r,col=pix
            if r<60:
                continue
            NDVI_arr=dic_NDVI[pix]
            a, b, c,r = dict_coff[pix]

            # print(NDVI_arr)
            LAI_predict_list=[]
            for i in range(len(NDVI_arr)):


                LAI_arr=self.exp_func(NDVI_arr[i],a,b,c)
                LAI_predict_list.append(LAI_arr)

            LAI_predict_dic[pix]=LAI_predict_list

        T.save_npy(LAI_predict_dic,rf'D:\Project3\Result\3mm\NDVI_LAI\\NDVI4g_predict.npy')




        pass
    def exp_func(self,x,a,b,c):
        LAI_arr = a * np.exp(b * x) + c
        # print(LAI_arr)
        return LAI_arr

    def plot_data(self):
        f_predict=rf'D:\Project3\Result\3mm\NDVI_LAI\\LAI4g_predict.npy'
        f_original=rf'D:\Project3\Data\LAI4g\dic_dryland\\'
        dic_predict=T.load_npy(f_predict)
        dic_original=T.load_npy_dir(f_original)
        for pix in dic_predict:
            r,col=pix
            if r<60:
                continue

            LAI_arr=dic_predict[pix]
            LAI_original=dic_original[pix]
            plt.plot(LAI_arr)
            plt.plot(LAI_original)

            plt.show()
        pass

    def difference_predict_original(self):
        fdir=rf'D:\Project3\Result\3mm\NDVI_LAI\LAI4g_predict\average_LAI4g_phenology_year\moving_window_extraction\trend\\'
        f_predic=fdir+rf'\\LAI4g_predict_detrend_CV_trend.tif'
        f_original=fdir+rf'\\LAI4g_detrend_CV_trend.tif'
        array_predict,originX,originY,pixelWidth,pixelHeight=ToRaster().raster2array(f_predic)
        array_original,originX,originY,pixelWidth,pixelHeight=ToRaster().raster2array(f_original)
        array_predict[array_predict<-9999]=np.nan
        array_original[array_original<-9999]=np.nan


        array_diff=array_predict-array_original
        array_diff[array_diff>100]=np.nan
        array_diff[array_diff<-10]=np.nan
        array_diff_relative_change=array_diff*100/array_original
        # plt.imshow(array_diff_relative_change,vmin=-50,vmax=50,cmap='jet')
        # plt.colorbar()
        # plt.show()
        DIC_and_TIF().arr_to_tif(array_diff_relative_change,rf'D:\Project3\Result\3mm\NDVI_LAI\\LAI4g_predict_original_difference.tif')
        pass





class build_dataframe():


    def __init__(self):

        self.this_class_arr = (result_root+rf'3mm\product_consistency\\dataframe\\')
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + rf'relative_change.df'

        pass

    def run(self):


        df = self.__gen_df_init(self.dff)
        # df=self.foo1(df)
        # df=self.foo2(df)
        # df=self.add_multiregression_to_df(df)
        # df=self.build_df(df)
        # df=self.build_df_monthly(df)
        # df=self.append_attributes(df)  ## 加属性
        # df=self.append_cluster(df)  ## 加属性
        # df=self.append_value(df)   ## insert or append value


        # df = self.add_detrend_zscore_to_df(df)
        # df=self.add_GPCP_lagged(df)
        # df=self.add_rainfall_characteristic_to_df(df)
        # df=self.add_lc_composition_to_df(df)


        # df=self.add_trend_to_df_scenarios(df)  ### add different scenarios of mild, moderate, extreme
        # df=self.add_trend_to_df(df)
        # df=self.add_mean_to_df(df)
        #

        df=self.add_aridity_to_df(df)
        df=self.add_dryland_nondryland_to_df(df)
        df=self.add_MODIS_LUCC_to_df(df)
        df = self.add_landcover_data_to_df(df)  # 这两行代码一起运行
        df=self.add_landcover_classfication_to_df(df)
        df=self.add_maxmium_LC_change(df)
        df=self.add_row(df)
        # df=self.add_continent_to_df(df)
        df=self.add_lat_lon_to_df(df)



        # df=self.rename_columns(df)
        # df = self.drop_field_df(df)
        df=self.show_field(df)


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

        fdir=rf'D:\Project3\Result\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\\'
        all_dic= {}
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            fname= f.split('.')[0]

            fpath=fdir+f

            dic = T.load_npy(fpath)
            key_name=fname
            print(key_name)
            all_dic[key_name]=dic
        # print(all_dic.keys())
        df=T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df

    def build_df_monthly(self, df):


        fdir = result_root+rf'extract_GS_return_monthly_data\individual_month_relative_change\X\\'
        all_dic= {}

        for fdir_ii in os.listdir(fdir):

            dic=T.load_npy(fdir+fdir_ii)

            key_name=fdir_ii.split('.')[0]
            all_dic[key_name] = dic
                # print(all_dic.keys())
        df = T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)
        return df



    def append_attributes(self, df):  ## add attributes
        fdir = result_root+ rf'\3mm\relative_change_growing_season\TRENDY\\'
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
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
        fdir = result_root + rf'growth_rate\\\growth_rate_raw\\'
        col_list=[]
        for f in os.listdir(fdir):
            if not 'LAI4g' in f:
                continue

            if not f.endswith('.npy'):
                continue


            col_name=f.split('.')[0]+'_growth_rate_raw'

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
                # if len(vals)==23:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     # print(len(vals))
                # elif len(vals)==38:
                #     for i in range(1):
                #         vals=np.append(vals,np.nan)
                #     print(len(vals))
                if len(vals)==38:

                    # vals=np.append(vals,np.nan)
                    ## append at the beginning
                    vals = np.insert(vals, 0, np.nan)


                vals_new.append(vals)

                # exit()
            df[col]=vals_new

        return df

        pass


    def foo1(self, df):

        f = rf'D:\Project3\Result\3mm\\extract_composite_phenology_year\\composite_LAI_relative_change_mean.npy'
        # array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        # array = np.array(array, dtype=float)
        # dic = DIC_and_TIF().spatial_arr_to_dic(array)


        dic = T.load_npy(f)

        pix_list = []
        change_rate_list = []
        year = []

        for pix in tqdm(dic):
            time_series = dic[pix]

            y = 1983
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y)
                y += 1


        df['pix'] = pix_list

        df['year'] = year
        # df['window'] = 'VPD_LAI4g_00'
        df['composite_LAI_relative_change_mean'] = change_rate_list
        return df

    def foo2(self, df):  # 新建trend

        f = result_root + rf'\3mm\relative_change_growing_season\TRENDY\trend_analysis_simple_linear_0206\LAI4g_trend.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        # val_array = np.load(f)
        # val_array[val_array<-99]=np.nan
        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        # plt.imshow(val_array)
        # plt.colorbar()
        # plt.show()

        # exit()

        pix_list = []
        for pix in tqdm(val_dic):
            val = val_dic[pix]
            if np.isnan(val):
                continue
            pix_list.append(pix)
        df['pix'] = pix_list
        T.print_head_n(df)


        return df

    def add_multiregression_to_df(self, df):
        fdir = result_root + rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\trend_ecosystem_year\\'
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if not 'sum_rainfall' in f:
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

    def add_GPCP_lagged(self,df): ##
        fdir=result_root+rf'\extract_GS\OBS_LAI_extend\\'
        for f in os.listdir(fdir):
            if f.split('.')[0] not in ['GPCC','CRU','tmax']:
                continue

            spatial_dic = T.load_npy(fdir+f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year

                pix = row['pix']
                r, c = pix

                if not pix in spatial_dic:
                    NDVI_list.append(np.nan)
                    continue

                vals = spatial_dic[pix][1:]


                v1=vals[year-1983]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)
            col_name=f.split('.')[0]
            print(col_name)
            df[col_name]=NDVI_list

            return df
            pass

    def add_detrend_zscore_to_df(self, df):

        fdir=rf'D:\Project3\Result\3mm\relative_change_growing_season\whole_period\\'


        for f in os.listdir(fdir):
            variable=f.split('.')[0]
            print(variable)
            # if not 'relative_change' in variable:
            #     continue
            # if 'detrend' in variable:
            #     continue




            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year
                # pix = row.pix
                pix = row['pix']
                r, c = pix

                if not pix in val_dic:
                    NDVI_list.append(np.nan)
                    continue

                vals = val_dic[pix]
                print(len(vals))

                if len(vals)==0:
                    NDVI_list.append(np.nan)
                    continue


                if len(vals)==33 :
                    nan_list=np.array([np.nan]*5)
                    vals=np.append(vals,nan_list)





                v1= vals[year - 1983]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)


            df[f'{variable}'] = NDVI_list
        # exit()
        return df

    def add_rainfall_characteristic_to_df(self, df):
        fdir = rf'E:\Data\ERA5_precip\ERA5_daily\dict\dry_spell\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            variable= f.split('.')[0]

            print(variable)


            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year
                # pix = row.pix
                pix = row['pix']
                r, c = pix


                if not pix in val_dic:
                    NDVI_list.append(np.nan)
                    continue


                vals = val_dic[pix]

                print(len(vals))
                ##### if len vals is 38, the end of list add np.nan

                # if len(vals) == 38:
                #     vals=np.append(vals,np.nan)
                #     v1 = vals[year - 1982]
                #     NDVI_list.append(v1)
                # if len(vals)==39:
                # v1 = vals[year - 1982]
                # v1 = vals[year - 1982]
                # if year < 2000:  ## fillwith nan
                #     NDVI_list.append(np.nan)
                #     continue


                v1= vals[year - 1982]
                # print(v1,year,len(vals))

                NDVI_list.append(v1)


            df[f'{variable}'] = NDVI_list
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
    def add_continent_to_df(self, df):
        tiff = rf'E:\Project3\Data\Base_data\\continent_05.tif'
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
    def add_lat_lon_to_df(self, df):
        D=DIC_and_TIF(pixelsize=0.5)
        df=T.add_lon_lat_to_df(df,D)
        return df


    def add_soil_texture_to_df(self, df):
        fdir = data_root + rf'\Base_data\\SoilGrid\SOIL_Grid_05_unify\\weighted_average\\'
        for f in (os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue

            tiff = fdir + f

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
            array = np.array(array, dtype=float)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
            fname=f.split('.')[0]
            print(fname)
            # exit()
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
            df[fname] = val_list
        return df
        pass

    def add_rooting_depth_to_df(self, df):
        tiff=rf'D:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth_05.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'rooting_depth'
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
    def add_Ndepostion_to_df(self, df):
        fdir='D:\Project3\Result\extract_GS\OBS_LAI_extend\\noy\\'
        val_dic = T.load_npy_dir(fdir)
        f_name = 'Noy'
        print(f_name)
        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            year = row.year
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
            # v1 = vals[year - 1982]
            v1 = vals[year - 0]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)

        df[f_name] = NDVI_list
        # exit()

        return df

    def add_area_to_df(self, df):
        area_dic=DIC_and_TIF(pixelsize=0.25).calculate_pixel_area()
        f_name = 'pixel_area'
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in area_dic:
                val_list.append(np.nan)
                continue
            val = area_dic[pix]
            if val < -99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f_name] = val_list
        return df


    def add_ozone_to_df(self, df):
            f=rf'D:\Project3\Result\extract_GS\OBS_LAI_extend\\ozone.npy'
            val_dic = T.load_npy(f)
            f_name = 'ozone'
            print(f_name)

            NDVI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                year = row.year_range
                # pix = row.pix
                pix = row['pix']
                r, c = pix

                if not pix in val_dic:
                    NDVI_list.append(np.nan)
                    continue

                vals = val_dic[pix]

                # print(len(vals))
                ##### if len vals is 38, the end of list add np.nan

                if len(vals) == 37:
                    ## append 2 nan
                    vals=np.append(vals,np.nan)
                    vals=np.append(vals,np.nan)

                    v1 = vals[year - 0]
                    NDVI_list.append(v1)


            df[f_name] = NDVI_list
            # exit()

            return df

            pass
    def add_root_depth_to_df(self, df):
        tiff=rf'D:\Project3\Data\Base_data\Rooting_Depth\tif_025_unify_merge\\rooting_depth.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tiff)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'rooting_depth'
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
        fdir='D:\Project3\Result\state_variables\\CV_monthly\\'
        for f in os.listdir(fdir):

            val_dic = T.load_npy(fdir + f)
            f_name = f.split('.')[0]+'_CV'
            print(f_name)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix'][0]
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

    def add_events(self, df):
        fdir = result_root + rf'relative_change\events_extraction\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dict_ = T.load_npy(fdir + f)
            key_name = f.split('.')[0]+'_event_level'
            print(key_name)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                year= row['year']
                if not pix in dict_:
                    val_list.append(np.nan)
                    continue
                dic_i = dict_[pix]
                mode=np.nan

                for key in dic_i:
                    idx_list=dic_i[key]
                    for idx in idx_list:
                        yeari=idx+1982
                        if yeari==year:
                            mode=key
                            break
                val_list.append(mode)
            df[key_name] = val_list

        T.print_head_n(df)
        return df


    def add_trend_to_df_scenarios(self,df):
        mode_list=['wet','dry']
        for mode in mode_list:
            period_list=['1982_2000','2001_2020','1982_2020']
            for period in period_list:

                fdir=result_root+rf'\monte_carlo\{mode}\\{period}\\'

                for f in os.listdir(fdir):
                    # print(f)
                    # exit()
                    if not 'trend' in f:
                        continue


                    if not f.endswith('.tif'):
                        continue



                    variable=(f.split('.')[0])



                    array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
                    array = np.array(array, dtype=float)

                    val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

                    # val_array = np.load(fdir + f)
                    # val_dic=T.load_npy(fdir+f)

                    # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                    f_name=f.split('.')[0]+'_'+period
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
                        if val>99:
                            val_list.append(np.nan)
                            continue
                        val_list.append(val)
                    df[f'{f_name}']=val_list

        return df

    def add_trend_to_df(self, df):
        fdir=data_root+ rf'\Base_data\SoilGrid\SOIL_Grid_05_unify\weighted_average\\'
        for f in os.listdir(fdir):


            if not f.endswith('.tif'):
                continue

            variable = (f.split('.')[0])

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

            # val_array = np.load(fdir + f)
            # val_dic=T.load_npy(fdir+f)

            # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
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
                # if val > 99:
                #     val_list.append(np.nan)
                #     continue
                val_list.append(val)


            df[f'{f_name}'] = val_list


        return df


    def add_mean_to_df(self, df):
        fdir=rf'D:\Project3\Result\state_variables\mean\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            variable = (f.split('.')[0])
            if not 'GPCC' in variable:
                continue


            val_dic=T.load_npy(fdir+f)

            # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)

            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < 0:
                    val_list.append(np.nan)
                    continue
                if val > 9999:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            # df[f'{f_name}'] = val_list
            df['MAP'] = val_list


        return df





    def rename_columns(self, df):
        df = df.rename(columns={'composite_LAI_median': 'composite_LAI_median_CV',





                            }

                               )



        return df
    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=[


                              'weighted_avg_GOSIF',

                              'weighted_avg_contribution_GOSIF',



                              ])
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
    def add_MODIS_LUCC_to_df(self, df):
        f = data_root + rf'\Base_data\MODIS_LUCC\\MODIS_LUCC_resample_05.tif'
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

        f = data_root + rf'\Base_data\\glc_025\\glc2000_05.tif'

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
    def add_maxmium_LC_change(self, df): ##

        f = data_root+rf'\Base_data\lc_trend\\max_trend.tif'

        array, origin, pixelWidth, pixelHeight, extent = ToRaster().raster2array(f)
        array[array <-99] = np.nan

        LC_dic =DIC_and_TIF().spatial_arr_to_dic(array)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix

            val= LC_dic[pix]
            df.loc[i,'LC_max'] = val
        return df

    def add_aridity_to_df(self,df):  ## here is original aridity index not classification

        f=data_root+rf'Base_data\\aridity_index_05\\aridity_index.tif'

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

        f = data_root + rf'\Base_data\aridity_index_05\\aridity_index.tif'

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
    def add_dryland_nondryland_to_df(self, df):
        fpath=data_root+rf'\\Base_data\\aridity_index_05\\aridity_index.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue

            val = val_dic[pix]
            if val <= 0.65:
                label = 'Dryland'
            elif 0.65 < val <= 0.8:
                label = 'Sub-humid'
            elif 0.8 < val <= 1.5:
                label = 'Humid'
            elif val > 1.5:
                label = 'Very Humid'
            elif val < -99:
                label = np.nan
            else:
                raise
            val_list.append(label)
        df['Dryland_Humid'] = val_list

        return df




    def show_field(self, df):
        for col in df.columns:
            print(col)
        return df
        pass

class PLOT_dataframe():
    def __init__(self):
        self.map_width = 13 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass
    def run (self):
        # self.plot_CV_LAI()
        self.plot_relative_change_LAI()
        # self.statistic_trend_CV_bar()
        # self.statistic_trend_bar()


        pass

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 60]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 10]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def plot_CV_LAI(self):  ##### plot for 4 clusters

        df = T.load_df(
            result_root + rf'\3mm\product_consistency\dataframe\\moving_window.df')
        print(len(df))
        df = self.df_clean(df)

        print(len(df))
        T.print_head_n(df)
        # exit()

        # create color list with one green and another 14 are grey



        color_list = ['black','green', 'blue',  'magenta', 'black','purple',  'purple', 'black', 'yellow', 'purple', 'pink', 'grey',
                      'brown', 'lime', 'teal', 'magenta']
        linewidth_list = [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]



        # variable_list = ['LAI4g', 'AVHRR_solely_relative_change','GEODES_AVHRR_LAI_relative_change',]
        # variable_list = ['NDVI', 'NDVI4g', 'GIMMS_plus_NDVI', ]
        #'detrended_SNU_LAI_CV','SNU_LAI_predict_detrend_CV','

        variable_list=['composite_LAI_CV',
                       'LAI4g_detrend_CV','detrended_SNU_LAI_CV',
            'GLOBMAP_LAI_detrend_CV',]
        dic_label={'composite_LAI_CV':'Composite LAI',
                   'LAI4g_detrend_CV':'LAI4g',
                   'detrended_SNU_LAI_CV':'SNUV',
                   'GLOBMAP_LAI_detrend_CV':'GLOBMAP',}
        year_list=range(0,24)


        result_dic = {}
        CI_dic={}
        std_dic={}

        for var in variable_list:

            result_dic[var] = {}
            CI_dic[var] = {}
            data_dic = {}
            CI_dic_data={}
            std_dic_data={}

            for year in year_list:
                df_i = df[df['year'] == year]

                vals = df_i[f'{var}'].tolist()
                SEM=stats.sem(vals)
                CI=stats.t.interval(0.95, len(vals)-1, loc=np.nanmean(vals), scale=SEM)
                std=np.nanstd(vals)
                CI_dic_data[year]=CI
                std_dic_data[year]=std

                data_dic[year] = np.nanmean(vals)




            result_dic[var] = data_dic
            CI_dic[var]=CI_dic_data
            std_dic[var]=std_dic_data
        ##dic to df

        df_new = pd.DataFrame(result_dic)

        flag = 0
        plt.figure(figsize=(self.map_width, self.map_height))

        for var in variable_list:
            if var == 'composite_LAI_CV':
                ## plot CI bar
                plt.plot(year_list, df_new[var], label=dic_label[var], linewidth=linewidth_list[flag], color=color_list[flag],
                         )
                ## fill std
                # std = [std_dic[var][y] for y in year_list]
                # plt.fill_between(year_list, df_new[var]-std, df_new[var]+std, alpha=0.3, color=color_list[flag])

                ## fill CI
                # ci_low = [CI_dic[var][y][0] for y in year_list]
                # ci_high = [CI_dic[var][y][1] for y in year_list]
                # plt.fill_between(year_list, ci_low, ci_high, color=color_list[flag], alpha=0.3, label='95% CI')
                slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, df_new[var])

                print(var, f'{slope:.2f}',  f'{p_value:.2f}')


                ## std



            else:
                plt.plot(year_list, df_new[var], label=dic_label[var], linewidth=linewidth_list[flag], color=color_list[flag])

                # std = [std_dic[var][y] for y in year_list]
                # plt.fill_between(year_list, df_new[var] - std, df_new[var] + std, alpha=0.3, color=color_list[flag])
                slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, df_new[var])
                print(var, f'{slope:.2f}',  f'{p_value:.2f}')

            flag = flag + 1
        ## if var == 'composite_LAI_CV': plot CI bar


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
        # plt.xticks(range(0, 23, 4))
        plt.xticks(range(len(year_range_str))[::4], year_range_str[::4], rotation=45, ha='right')
        plt.yticks(np.arange(5, 25, 5))
        # plt.yticks(np.arange(5, 50, 5))
        # plt.xticks(range(0, 23, 3))


        plt.ylabel(f'LAI CV (%)')

        plt.grid(which='major', alpha=0.5)
        plt.legend(loc='upper left')

        plt.show()
        # plt.tight_layout()
        # out_pdf_fdir = result_root + rf'\3mm\product_consistency\pdf\\'
        # plt.savefig(out_pdf_fdir + 'time_series_CV.pdf', dpi=300, bbox_inches='tight')
        # plt.close()


        #
        # plt.legend()
        # plt.show()

    def plot_relative_change_LAI(self):  ##### plot for 4 clusters

        df = T.load_df(
            result_root + rf'\3mm\product_consistency\dataframe\\relative_change.df')
        print(len(df))
        df = self.df_clean(df)

        print(len(df))
        T.print_head_n(df)
        # exit()

        # create color list with one green and another 14 are grey

        color_list = ['black','green', 'blue',  'magenta', 'black','purple',  'purple', 'black', 'yellow', 'purple', 'pink', 'grey',
                      'brown', 'lime', 'teal', 'magenta']
        linewidth_list = [2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


        fig = plt.figure()
        i = 1

        # variable_list = ['LAI4g', 'AVHRR_solely_relative_change','GEODES_AVHRR_LAI_relative_change',]
        # variable_list = ['NDVI', 'NDVI4g', 'GIMMS_plus_NDVI', ]
        #'detrended_SNU_LAI_CV','SNU_LAI_predict_detrend_CV','

        variable_list = [
                         'composite_LAI_relative_change_mean','LAI4g', 'SNU_LAI_relative_change',
            'GLOBMAP_LAI_relative_change',
                         ]
        dic_label={'LAI4g':'LAI4g','SNU_LAI_relative_change':'SNU_LAI',
                   'GLOBMAP_LAI_relative_change':'GLOBMAP_LAI',
                   'composite_LAI_relative_change_mean':'Composite LAI'}
        year_list=range(1983,2021)


        result_dic = {}
        for var in variable_list:

            result_dic[var] = {}
            data_dic = {}

            for year in year_list:
                df_i = df[df['year'] == year]

                vals = df_i[f'{var}'].tolist()
                data_dic[year] = np.nanmean(vals)
            result_dic[var] = data_dic
        ##dic to df

        df_new = pd.DataFrame(result_dic)
        flag=0
        plt.figure(figsize=(self.map_width, self.map_height))

        for var in variable_list:
            plt.plot(year_list, df_new[var], label=dic_label[var],linewidth=linewidth_list[flag], color=color_list[flag])
            flag=flag+1
            slope, intercept, r_value, p_value, std_err = stats.linregress(year_list, df_new[var])
            print(var, f'{slope:.2f}', f'{p_value:.2f}')
        plt.ylabel('Relative change LAI (%)')
        plt.grid(True)


        plt.legend()
        plt.show()
        # out_pdf_fdir = result_root + rf'\3mm\product_consistency\pdf\\'
        # plt.savefig(out_pdf_fdir + 'time_series_relative_change.pdf', dpi=300, bbox_inches='tight')
        # plt.close()


    def statistic_trend_CV_bar(self):
        fdir = result_root + rf'3mm\product_consistency\\CV\\'
        variable_list=['composite_LAI_CV','GlOBMAP_detrend_CV','LAI4g_detrend_CV','SNU_LAI_detrend_CV']
        for variable in variable_list:
            f_trend_path=fdir+f'{variable}_trend.tif'
            f_pvalue_path=fdir+f'{variable}_p_value.tif'
            result_dic={}

            arr_corr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_trend_path)
            arr_pvalue, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_pvalue_path)
            arr_corr[arr_corr<-99]=np.nan
            arr_corr[arr_corr>99]=np.nan
            arr_corr=arr_corr[~np.isnan(arr_corr)]

            arr_pvalue[arr_pvalue<-99]=np.nan
            arr_pvalue[arr_pvalue>99]=np.nan
            arr_pvalue=arr_pvalue[~np.isnan(arr_pvalue)]
            ## corr negative and positive
            arr_corr = arr_corr.flatten()
            arr_pvalue = arr_pvalue.flatten()
            arr_pos=len(arr_corr[arr_corr>0])/len(arr_corr)*100
            arr_neg=len(arr_corr[arr_corr<0])/len(arr_corr)*100


            ## significant positive and negative
            ## 1 is significant and 2 positive or negative

            mask_pos = (arr_corr > 0) & (arr_pvalue < 0.05)
            mask_neg = (arr_corr < 0) & (arr_pvalue < 0.05)


            # 满足条件的像元数
            count_positive_sig = np.sum(mask_pos)
            count_negative_sig = np.sum(mask_neg)

            # 百分比
            significant_positive = (count_positive_sig / len(arr_corr)) * 100
            significant_negative = (count_negative_sig / len(arr_corr)) * 100
            result_dic = {

                'sig neg': significant_negative,
                'non sig neg': arr_neg,
                'non sig pos': arr_pos,
                'sig pos': significant_positive



            }
            # df_new=pd.DataFrame(result_dic,index=[variable])
            # ## plot
            # df_new=df_new.T
            # df_new=df_new.reset_index()
            # df_new.columns=['Variable','Percentage']
            # df_new.plot.bar(x='Variable',y='Percentage',rot=45,color='green')
            # plt.show()
            color_list = [
                '#008837',
                'silver',

                'silver',
                '#7b3294',
            ]
            width = 0.4
            alpha_list = [1, 0.5, 0.5, 1]
            plt.figure(figsize=(3, 3))

            # 逐个画 bar
            for i, (key, val) in enumerate(result_dic.items()):
                plt.bar(i , val, color=color_list[i], alpha=alpha_list[i], width=width)
                plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
                plt.ylabel('Percentage (%)')
                # plt.title(variable)

            plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
            plt.tight_layout()
            # plt.show())
        ## save pdf
            plt.savefig(result_root + rf'3mm\product_consistency\pdf\{variable}_CV_bar.pdf')


    def statistic_trend_bar(self):
        fdir = result_root + rf'3mm\product_consistency\relative_change\Trend\\'
        variable_list=['GLOBMAP_LAI_relative_change','LAI4g','composite_LAI_relative_change_mean','SNU_LAI_relative_change']
        for variable in variable_list:
            f_trend_path=fdir+f'{variable}_trend.tif'
            f_pvalue_path=fdir+f'{variable}_p_value.tif'
            result_dic={}

            arr_corr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_trend_path)
            arr_pvalue, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_pvalue_path)
            arr_corr[arr_corr<-99]=np.nan
            arr_corr[arr_corr>99]=np.nan
            arr_corr=arr_corr[~np.isnan(arr_corr)]

            arr_pvalue[arr_pvalue<-99]=np.nan
            arr_pvalue[arr_pvalue>99]=np.nan
            arr_pvalue=arr_pvalue[~np.isnan(arr_pvalue)]
            ## corr negative and positive
            arr_corr = arr_corr.flatten()
            arr_pvalue = arr_pvalue.flatten()
            arr_pos=len(arr_corr[arr_corr>0])/len(arr_corr)*100
            arr_neg=len(arr_corr[arr_corr<0])/len(arr_corr)*100


            ## significant positive and negative
            ## 1 is significant and 2 positive or negative

            mask_pos = (arr_corr > 0) & (arr_pvalue < 0.05)
            mask_neg = (arr_corr < 0) & (arr_pvalue < 0.05)


            # 满足条件的像元数
            count_positive_sig = np.sum(mask_pos)
            count_negative_sig = np.sum(mask_neg)

            # 百分比
            significant_positive = (count_positive_sig / len(arr_corr)) * 100
            significant_negative = (count_negative_sig / len(arr_corr)) * 100
            result_dic = {

                'sig neg': significant_negative,
                'non sig neg': arr_neg,
                'non sig pos': arr_pos,
                'sig pos': significant_positive



            }
            # df_new=pd.DataFrame(result_dic,index=[variable])
            # ## plot
            # df_new=df_new.T
            # df_new=df_new.reset_index()
            # df_new.columns=['Variable','Percentage']
            # df_new.plot.bar(x='Variable',y='Percentage',rot=45,color='green')
            # plt.show()
            color_list = [
                 '#844000',
            'lightgray',

            'lightgray',
            '#064c6c',
            ]

            # color_list = [
            #     '#844000',
            #     '#fc9831',
            #     '#fffbd4',
            #     '#86b9d2',
            #     '#064c6c',
            # ]

            width = 0.4
            alpha_list = [1, 0.5, 0.5, 1]

            # 逐个画 bar
            for i, (key, val) in enumerate(result_dic.items()):
                plt.bar(i , val, color=color_list[i], alpha=alpha_list[i], width=width)
                plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')
                plt.ylabel('Percentage')
                plt.title(variable)

            plt.xticks(range(len(result_dic)), list(result_dic.keys()), rotation=0)
            plt.show()
            # plt.savefig(result_root + rf'3mm\product_consistency\pdf\{variable}_trend_bar.pdf')
            # plt.close()


def main():


    # build_dataframe().run()
    # products_check().run()

    PLOT_dataframe().run()
    # NDVI_LAI().run()

if __name__ == '__main__':
    main()