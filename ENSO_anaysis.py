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

class ENSO_anaysis():
    def __init__(self):


        self.weak_El_Nino_list = [2004,2005,2006,2007,2014,2015,2018,2019]

        self.weak_La_Nina_list = [1983,1984,1985, 2000,2001,2005, 2006,2008,2009,2016,2017,2018,]

        self.moderate_El_Nino_list = [1986,1987,1994,1995,2002,2003,2009,2010]
        self.moderate_La_Nina_list = [1995,1996,2011,2012,2020,]

        self.strong_El_Nino_list = [1982,1983, 1987,1988, 1991,1992,1997,1998,2015,2016]
        self.next_strong_El_Nino_list = [1983,1984, 1988,1989,1992,1993,1998,1999,2016,2017]
        self.strong_La_Nina_list = [1988, 1989,1998,1999, 2000,2007,2008,2010,2011]
        self.next_strong_El_Nino_list = [1989,1990,1999,2000,2001,2008,2009,2011,2012]

        ### netural year = all years - ENSO years
        self.netural_list = [i for i in range(1982,2021) if i not in self.strong_El_Nino_list+self.strong_La_Nina_list+self.moderate_El_Nino_list+self.moderate_La_Nina_list]
        # print('netural_list',self.netural_list)
        pass
    def run(self):
        # self.extract_data_based_on_ENSO()
        #self.calculate_relatve_change_based_on_ENSO()

        # self.extract_first_year_and_second_year_ENSO()
        # self.calculate_relatve_change_based_on_ENSO_first_second_year()
        self.plot_spatial_average_ENSO_LAI()


        pass

    def extract_data_based_on_ENSO(self):
        variable='tmax'
        f=result_root+rf'\Detrend\detrend_original\\{variable}.npy'
        outdir=result_root+'ENSO\\ENSO_original\\'
        T.mk_dir(outdir,force=True)
        dict=np.load(f,allow_pickle=True).item()



        result_dic_strong_El_Nino = {}
        result_dic_strong_La_Nina = {}

        result_dic_netural = {}

        for pix in tqdm(dict.keys()):
            result_strong_El_Nino = []
            result_strong_La_Nina = []

            result_netural = []

            vals=dict[pix]
            print(len(vals))
            ## get index of ENSO years
            if np.nanmean(vals) == 0:
                continue
            ##1) get number of ENSO years 2) substract 1982 to extract vals
            for year in self.strong_El_Nino_list:
                idx = year-1982
                result_strong_El_Nino.append(vals[idx])
            result_dic_strong_El_Nino[pix]=result_strong_El_Nino
            for year in self.strong_La_Nina_list:
                idx = year-1982
                result_strong_La_Nina.append(vals[idx])
            result_dic_strong_La_Nina[pix]=result_strong_La_Nina


            for year in self.netural_list:
                idx = year-1982
                result_netural.append(vals[idx])
            result_dic_netural[pix]=result_netural


        np.save(outdir+f'strong_El_Nino_{variable}.npy',result_dic_strong_El_Nino)
        np.save(outdir+f'strong_La_Nina_{variable}.npy',result_dic_strong_La_Nina)

        np.save(outdir+f'netural_{variable}.npy',result_dic_netural)



    def calculate_relatve_change_based_on_ENSO(self):
        fdir=result_root+'ENSO\\ENSO_original\\'
        outdir=result_root+'ENSO\\\\ENSO_relative_change\\'
        T.mk_dir(outdir,force=True)
        variable='tmax'
        f1=fdir+f'strong_El_Nino_{variable}.npy'
        f2=fdir+f'strong_La_Nina_{variable}.npy'

        f4=fdir+f'netural_{variable}.npy'
        strong_El_Nino=T.load_npy(f1)
        strong_La_Nina=T.load_npy(f2)

        netural=T.load_npy(f4)

        dic_strong_El_Nino = {}
        dic_strong_La_Nina = {}

        for pix in tqdm(strong_El_Nino.keys()):
            val1=strong_El_Nino[pix]
            val2=strong_La_Nina[pix]

            val4=netural[pix]
            val1=np.array(val1)
            val2=np.array(val2)

            val4=np.array(val4)

            ## relative change
            relative_val1=(val1-np.nanmean(val4))/np.nanmean(val4)*100
            relative_val2=(val2-np.nanmean(val4))/np.nanmean(val4)*100

            dic_strong_La_Nina[pix]=relative_val2
            dic_strong_El_Nino[pix]=relative_val1





            ## Save
        np.save(outdir+f'relative_strong_El_Nino_{variable}.npy',dic_strong_El_Nino)
        np.save(outdir+f'relative_strong_La_Nina_{variable}.npy',dic_strong_La_Nina)
    def calculate_relatve_change_based_on_ENSO_first_second_year(self):

        fdir = result_root + 'ENSO\\ENSO_first_year_second_year\\extraction\\'
        outdir = result_root + 'ENSO\\\\ENSO_first_year_second_year\\relative_change\\'
        T.mk_dir(outdir, force=True)
        variable = 'LAI4g'
        f1 = fdir + f'strong_El_Nino_first_{variable}.npy'
        f2 = fdir + f'strong_La_Nina_first_{variable}.npy'

        f3=fdir+f'strong_El_Nino_second_{variable}.npy'
        f4=fdir+f'strong_La_Nina_second_{variable}.npy'
        f5=fdir+f'netural_{variable}.npy'
        strong_El_Nino_first = T.load_npy(f1)
        strong_La_Nina_first = T.load_npy(f2)
        strong_El_Nino_second = T.load_npy(f3)
        strong_La_Nina_second = T.load_npy(f4)
        netural = T.load_npy(f5)



        dic_strong_El_Nino_first = {}
        dic_strong_La_Nina_first = {}
        dic_strong_El_Nino_second = {}
        dic_strong_La_Nina_second = {}


        for pix in tqdm(strong_El_Nino_first.keys()):
            val1 = strong_El_Nino_first[pix]
            val2 = strong_La_Nina_first[pix]
            val3 = strong_El_Nino_second[pix]
            val4 = strong_La_Nina_second[pix]
            val5 = netural[pix]
            val1 = np.array(val1)
            val2 = np.array(val2)
            val3 = np.array(val3)
            val4 = np.array(val4)
            val5 = np.array(val5)
            ## relative change
            relative_val1 = (val1 - np.nanmean(val5)) / np.nanmean(val5) * 100
            relative_val2 = (val2 - np.nanmean(val5)) / np.nanmean(val5) * 100
            relative_val3 = (val3 - np.nanmean(val5)) / np.nanmean(val5) * 100
            relative_val4 = (val4 - np.nanmean(val5)) / np.nanmean(val5) * 100
            dic_strong_La_Nina_first[pix] = relative_val2
            dic_strong_El_Nino_first[pix] = relative_val1
            dic_strong_La_Nina_second[pix] = relative_val4
            dic_strong_El_Nino_second[pix] = relative_val3


            ## Save
        np.save(outdir + f'relative_strong_El_Nino_first_{variable}.npy', dic_strong_El_Nino_first)
        np.save(outdir + f'relative_strong_La_Nina_first_{variable}.npy', dic_strong_La_Nina_first)
        np.save(outdir + f'relative_strong_El_Nino_second_{variable}.npy', dic_strong_El_Nino_second)
        np.save(outdir + f'relative_strong_La_Nina_second_{variable}.npy', dic_strong_La_Nina_second)


    def plot_spatial_average_ENSO_LAI(self):

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)


        fdir=result_root+rf'\ENSO\ENSO_first_year_second_year\relative_change\\'
        result_dic={}
        for f in os.listdir(fdir):
            if not 'LAI4g' in f:
                continue
            if not '.npy' in f:
                continue

            dic=T.load_npy(fdir+f)
            for pix in dic.keys():
                r,c=pix
                if r<120:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                vals=dic[pix]

                result_dic[pix]=np.nanmean(vals)
            array=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic)
            array=array*array_mask
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(array,fdir+f.replace('.npy','.tif'))

            plt.imshow(array,vmin=0,vmax=0.5,cmap='jet',interpolation='nearest')
            plt.title(f)
            plt.colorbar()
            plt.show()
            plt.close()
        pass

    def extract_first_year_and_second_year_ENSO(self):


        strong_first_El_Nino_list = [1987, 1991,1982,1997,2015]
        strong_second_El_Nino_list = [1988,1992,1983,1998,2016]


        strong_first_La_Nina_list = [1988, 1998, 2000,2007,2010]
        strong_second_La_Nina_list = [1989,1999,2001,2008,2011]

        ### netural year = all years - ENSO years
        netural_list = [i for i in range(1982,2021) if i not in self.strong_El_Nino_list+self.strong_La_Nina_list+self.moderate_El_Nino_list+self.moderate_La_Nina_list]

        variable='LAI4g'
        f=rf'\Detrend\detrend_original\\{variable}.npy'
        dict=np.load(result_root+f,allow_pickle=True).item()
        outdir=result_root+'ENSO\\ENSO_first_year_second_year\\'
        T.mk_dir(outdir,force=True)
        result_dic_strong_El_Nino_first = {}
        result_dic_strong_La_Nina_first = {}
        result_dic_strong_El_Nino_second = {}
        result_dic_strong_La_Nina_second = {}

        result_dic_netural = {}

        for pix in tqdm(dict.keys()):
            result_strong_El_Nino_first = []
            result_strong_La_Nina_first = []
            result_strong_El_Nino_second = []
            result_strong_La_Nina_second = []



            result_netural = []
            vals = dict[pix]
            print(len(vals))
            ## get index of ENSO years
            if np.nanmean(vals) == 0:
                continue
            ##1) get number of ENSO years 2) substract 1982 to extract vals
            for year in strong_first_El_Nino_list:
                idx = year - 1982
                result_strong_El_Nino_first.append(vals[idx])
            result_dic_strong_El_Nino_first[pix] = result_strong_El_Nino_first
            for year in strong_first_La_Nina_list:
                idx = year - 1982
                result_strong_La_Nina_first.append(vals[idx])
            result_dic_strong_La_Nina_first[pix] = result_strong_La_Nina_first
            for year in strong_second_El_Nino_list:
                idx = year - 1982
                result_strong_El_Nino_second.append(vals[idx])
            result_dic_strong_El_Nino_second[pix] = result_strong_El_Nino_second
            for year in strong_second_La_Nina_list:
                idx = year - 1982
                result_strong_La_Nina_second.append(vals[idx])
            result_dic_strong_La_Nina_second[pix] = result_strong_La_Nina_second


            for year in self.netural_list:
                idx = year - 1982
                result_netural.append(vals[idx])
            result_dic_netural[pix] = result_netural

        np.save(outdir + f'strong_El_Nino_first_{variable}.npy', result_dic_strong_El_Nino_first)
        np.save(outdir + f'strong_La_Nina_first_{variable}.npy', result_dic_strong_La_Nina_first)
        np.save(outdir + f'strong_El_Nino_second_{variable}.npy', result_dic_strong_El_Nino_second)
        np.save(outdir + f'strong_La_Nina_second_{variable}.npy', result_dic_strong_La_Nina_second)


        np.save(outdir + f'netural_{variable}.npy', result_dic_netural)

    pass





class plot_ENSO():
    def __init__(self):
        self.weak_El_Nino_list = [2004, 2005, 2006, 2007, 2014, 2015, 2018, 2019]

        self.weak_La_Nina_list = [1983, 1984, 1985, 2000, 2001, 2005, 2006, 2008, 2009, 2016, 2017, 2018, ]

        self.moderate_El_Nino_list = [1986, 1987, 1994, 1995, 2002, 2003, 2009, 2010]
        self.moderate_La_Nina_list = [1995, 1996, 2011, 2012, 2020, ]

        self.strong_El_Nino_list = [1987, 1988, 1991, 1992]
        self.strong_La_Nina_list = [1988, 1989, 1998, 1999, 2000, 2007, 2008, 2010, 2011]
        self.very_strong_El_Nino_list = [1982, 1983, 1997, 1998, 2015, 2016]

        ### netural year = all years - ENSO years



        pass
    def run(self):
        self.plot_anomaly_LAI_based_on_cluster()
        # self.plot_anomaly_LAI_based_on_cluster_test()
        # self.barplot_relative_change()
        pass
    def clean_df(self,df):
        df=df[df['landcover_classfication']!='Cropland']
        df=df[df['row']>120]


        return df

    def plot_anomaly_LAI_based_on_cluster(self):  ##### plot for 4 clusters

        df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes_yearly_new.df')
        print(len(df))
        df=self.clean_df(df)
        ## get uqniue AI_classfication
        # AI_classfication_unique = T.get_df_unique_val_list(df, 'continent')
        # print(AI_classfication_unique)
        # exit()


        print(len(df))
        T.print_head_n(df)
        # exit()
        df=df[df['row']>120]


        df=df[df['landcover_classfication']!='Cropland']

        df=df[df['Aridity']<0.65]



        color_list=['blue','green','red','orange','aqua','brown','cyan', 'black']
        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=2
        linewidth_list[2]=2

        fig = plt.figure()
        ## size
        fig.set_size_inches(12, 8)
        flag=1
        variable_list=['GLEAM_SMroot',]
        # variable_list=['VPD',]
        # variable_list=['CRU','GPCC']
        variable_list=['LAI4g','GIMMS_AVHRR_LAI','GIMMS3g_LAI']
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
        for continent in ['Africa','Australia','Asia','North_America','South_America']:
            ax = fig.add_subplot(2, 3, flag)
            if continent == 'North_America':
                df_NA = df[df['lon'] > -125]
                df_NA = df_NA[df_NA['lon'] < -95]
                df_NA = df_NA[df_NA['lat'] > 0]
                df_region = df_NA[df_NA['lat'] < 45]
            else:
                df_region = df[df['continent'] == continent]

            GPP_all = df_region['FLUXCOM'].tolist()
            GPP_min = np.nanmin(GPP_all)
            GPP_max = np.nanmax(GPP_all)


            for product in variable_list:

                print(product)
                ## creat year list dic
                dic_year={}
                yearlist=list(range(1982,2021))
                for year in yearlist:
                    dic_year[str(year)]=[]



                for i, row in df_region.iterrows():
                    year=row['year']
                    LAI_vals = row[product]
                    LAI_vals = np.array(LAI_vals)
                    gpp_vals = row['FLUXCOM']
                    gpp_vals = np.array(gpp_vals)
                    ## calculated weighted GPP
                    GPP_weights= (gpp_vals-GPP_min)/(GPP_max-GPP_min)
                    vals = np.array(LAI_vals)
                    vals = vals * GPP_weights
                    dic_year[str(year)].append(vals)
                ## calculate mean of each year

                for year in yearlist:
                    vals=dic_year[str(year)]
                    vals_mean = np.nanmean(vals, axis=0)
                    dic_year[str(year)]=vals_mean


            #plot time series
                val_list=[]
                for year in yearlist:
                    vals=dic_year[str(year)]
                    val_list.append(vals)
                val_list=np.array(val_list)
                plt.plot(val_list,label=product,color=color_list[variable_list.index(product)],linewidth=linewidth_list[variable_list.index(product)])
                plt.xticks(range(0, 40, 4), range(1982, 2021, 4), rotation=45)
                plt.xlabel('year')
                plt.ylabel(f'{product}')
                plt.title(f'{continent}')
                plt.ylim(-5, 5)
                plt.grid(which='major', alpha=0.5)


            for j in range(len(self.strong_La_Nina_list)):
                plt.axvspan(self.strong_La_Nina_list[j]-1982, self.strong_La_Nina_list[j]+1-1982, color='pink', alpha=0.5)

            for k in range(len(self.strong_El_Nino_list)):
                plt.axvspan(self.strong_El_Nino_list[k]-1982, self.strong_El_Nino_list[k]+1-1982, color='skyblue', alpha=0.5)
            for m in range(len(self.very_strong_El_Nino_list)):
                plt.axvspan(self.very_strong_El_Nino_list[m]-1982, self.very_strong_El_Nino_list[m]+1-1982, color='blue', alpha=0.5)

            ## show legend of ENSO events
            plt.plot([], [], color='pink', label='Strong La Nina')
            plt.plot([], [], color='skyblue', label='Strong El Nino')
            plt.plot([], [], color='blue', label='Very Strong El Nino')
            flag=flag+1


        plt.legend()
        plt.show()
        plt.close()

    def plot_anomaly_LAI_based_on_cluster_test(self):  ##### plot for 4 clusters

        df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes.df')
        print(len(df))
        df=self.clean_df(df)
        ## get uqniue AI_classfication
        # AI_classfication_unique = T.get_df_unique_val_list(df, 'continent')
        # print(AI_classfication_unique)
        # exit()


        print(len(df))
        T.print_head_n(df)
        # exit()
        df=df[df['row']>120]


        df=df[df['landcover_classfication']!='Cropland']

        df=df[df['Aridity']<0.65]



        color_list=['blue','green','red','orange','aqua','brown','cyan', 'black']
        linewidth_list=[1]*16
        linewidth_list[0]=3
        linewidth_list[1]=2
        linewidth_list[2]=2

        fig = plt.figure()
        ## size
        fig.set_size_inches(12, 8)
        i = 1
        variable_list=['GLEAM_SMroot',]
        # variable_list=['VPD',]
        # variable_list=['CRU','GPCC']
        variable_list=['LAI4g','GIMMS_AVHRR_LAI','GIMMS3g_LAI']
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
        for continent in ['Africa','Australia','Asia','North_America','South_America']:
            ax = fig.add_subplot(2, 3, i)
            if continent == 'North_America':
                df_NA = df[df['lon'] > -125]
                df_NA = df_NA[df_NA['lon'] < -95]
                df_NA = df_NA[df_NA['lat'] > 0]
                df_region = df_NA[df_NA['lat'] < 45]
            else:
                df_region = df[df['continent'] == continent]

            for product in variable_list:


                print(product)
                vals=df_region[product].tolist()

                ### weighted LAI by GPP
                vals = np.array(vals)






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

            ax.set_xticks(range(0, 40, 4))
            ax.set_xticklabels(range(1982, 2021, 4), rotation=45)
            # plt.ylim(-0.2, 0.2)
            plt.ylim(-25,25)


            plt.xlabel('year')

            plt.ylabel(f'relative change{product}(%/year)')
            # plt.legend()

            plt.title(f'{continent}')
            plt.grid(which='major', alpha=0.5)


            for j in range(len(self.strong_La_Nina_list)):
                plt.axvspan(self.strong_La_Nina_list[j]-1982, self.strong_La_Nina_list[j]+1-1982, color='pink', alpha=0.5)

            for k in range(len(self.strong_El_Nino_list)):
                plt.axvspan(self.strong_El_Nino_list[k]-1982, self.strong_El_Nino_list[k]+1-1982, color='skyblue', alpha=0.5)
            for m in range(len(self.very_strong_El_Nino_list)):
                plt.axvspan(self.very_strong_El_Nino_list[m]-1982, self.very_strong_El_Nino_list[m]+1-1982, color='blue', alpha=0.5)

            ## show legend of ENSO events
            plt.plot([], [], color='pink', label='Strong La Nina')
            plt.plot([], [], color='skyblue', label='Strong El Nino')
            plt.plot([], [], color='blue', label='Very Strong El Nino')

        plt.legend()
        plt.show()
        plt.close()
    def barplot_relative_change(self):
        df = T.load_df(result_root + rf'Dataframe\relative_changes\\relative_changes.df')
        print(len(df))
        df=self.clean_df(df)
        variable='LAI4g'

        print(len(df))
        T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        df=df[df['landcover_classfication']!='Cropland']
        df=df[df['Aridity']<0.65]


        fig = plt.figure()
        ## size
        fig.set_size_inches(9, 5)
        vals_list_El_Nino = {}
        vals_list_La_Nina = {}
        all_dic={}

        for continent in ['Africa','Australia','Asia','North_America','South_America']:

            if continent == 'North_America':
                df_NA = df[df['lon'] > -125]
                # df_NA = df_NA[df_NA['lon'] < -105]
                df_NA = df_NA[df_NA['lon'] < -95]
                df_NA = df_NA[df_NA['lat'] > 0]
                df_region = df_NA[df_NA['lat'] < 45]
            else:
                df_region = df[df['continent'] == continent]
            # pix_list= df_region['pix'].tolist()
            # spatial_dic= {}
            # for pix in pix_list:
            #     spatial_dic[pix]=1
            # array=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(array, cmap='jet', vmin=0, vmax=1, interpolation='nearest')
            # plt.colorbar()
            # plt.show()


            vals = df_region[f'relative_strong_La_Nina_{variable}'].tolist()
            #### group by lanina and elnino












def main():
    ENSO_anaysis().run()
    # plot_ENSO().run()


    pass

if __name__ == '__main__':
    main()

