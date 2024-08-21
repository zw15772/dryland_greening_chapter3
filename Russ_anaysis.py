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


class PLOT:
    def __init__(self):
        pass

    def __del__(self):
        pass
    def run(self):
        self.plot_anomaly_LAI_based_on_cluster()


    def clean_df(self, df):
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['Aridity'] < 0.65]

        df = df[df['MODIS_LUCC'] != 12]

        df = df[df['lon'] > -125]
        df = df[df['lon'] < -105]
        df = df[df['lat'] > 0]
        df = df[df['lat'] < 45]

        return df
    def plot_anomaly_LAI_based_on_cluster(self):  ##### plot for 4 clusters

        df = T.load_df(result_root + rf'\Dataframe\growing_season_original\\growing_season_original.df')
        print(len(df))


        print(len(df))
        T.print_head_n(df)
        df=self.clean_df(df)
        print(len(df))
        # exit()


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

        variable_list=['LAI4g']
        # scenario='S2'
        # variable_list= ['LAI4g',f'CABLE-POP_{scenario}_lai', f'CLASSIC_{scenario}_lai', 'CLM5',  f'IBIS_{scenario}_lai', f'ISAM_{scenario}_lai',
        #      f'ISBA-CTRIP_{scenario}_lai', f'JSBACH_{scenario}_lai', f'JULES_{scenario}_lai',  f'LPJ-GUESS_{scenario}_lai', f'LPX-Bern_{scenario}_lai',
        #      f'ORCHIDEE_{scenario}_lai', f'SDGVM_{scenario}_lai', f'YIBs_{scenario}_Monthly_lai']
        region_unique = T.get_df_unique_val_list(df, 'landcover_classfication')
        print(region_unique)


        for continent in ['Grass', 'Evergreen', 'Shrub', 'Deciduous','Mixed','North_America']:
            ax = fig.add_subplot(2, 3, i)
            if continent=='North_America':
                df_continent=df
            else:

                df_continent = df[df['landcover_classfication'] == continent]
            pixel_number = len(df_continent)

            for product in variable_list:


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


            # plt.xlabel('year')

            plt.ylabel(f'LAI4g(m2/m2)')
            plt.title(f'{continent}_{pixel_number}_pixels')
            plt.grid(which='major', alpha=0.5)
        # plt.legend()
        plt.show()

def main():
    PLOT().run()

    pass

if __name__ == '__main__':
    main()