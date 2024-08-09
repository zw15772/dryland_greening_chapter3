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

class aysmetry_response:
    def __init__(self):
        pass
    def run(self):
        # self.pick_wet_year_events()  ##pick mild, moderate, extreme, and random wet year
        # self.pick_dry_year_events()  ##pick mild, moderate, extreme, and random dry year
        # self.pick_dry_year_anomaly() ##pick wet year when the precipitation is less than 0.5 std
        # self.pick_wet_year_anomaly()
        self.pick_wet_year_anomaly_moving_window()  ##pick the wet year based on the moving window

        # self.calculate_frequency_wet_dry_event() ### calculate the frequency of wet and dry year spatially based on this and create df

        # self.calculate_frequency_wet_dry_anomaly()
        # self.plot_frequency_wet_dry_spatial()

        # self.plot_bar_frequency_wet_dry()
        # self.plot_bar_frequency_dry_regional_one_mode()  ##no differential between mild, moderate, and extreme
        # self.plot_bar_frequency_dry_regional_three_mode()




    def pick_wet_year_events(self):
        ## 1) build df for the extreme wet year and dry year for four period (1982-1990, 1991-2000, 2001-2010, 2011-2020)
        outdir = join(result_root,'asymmetry_response','anomaly','wet_event_CRU')

        T.mk_dir(outdir,force=True)
        ### define  mild, moderate, extreme, and random fluation
        ## mild: 0.5-1, moderate: 1-2, extreme: >2, random: 0-0.5

        f_precipitation_zscore = (result_root + rf'zscore\CRU.npy')
        dic_precip_relative_change = T.load_npy(f_precipitation_zscore)



        # ##plot the distribution of the relative change
        value_list = []
        for year in range(1982,2021):
            for pix in dic_precip_relative_change:
                value=dic_precip_relative_change[pix][year-1982]
                value_list.append(value)
        value_list=np.array(value_list)
        ### data_clean 1. remove 3std 2. remove nan
        value_list=value_list[np.isnan(value_list)==0]
        ## 3std clean
        value_list=value_list[value_list<np.nanmean(value_list)+3*np.nanstd(value_list)]
        value_list=value_list[value_list>np.nanmean(value_list)-3*np.nanstd(value_list)]
        ##calculate 99% percentile
        percentile_99=np.nanpercentile(value_list,99)
        print(percentile_99)

        # plt.hist(value_list,bins=100)
        # plt.show()
        # exit()
        #

        f_precip = rf'D:\Project3\Result\\anomaly\OBS_extend\\CRU.npy'
        f_lai = rf'D:\Project3\Result\\anomaly\OBS_extend\\LAI4g.npy'
        f_vpd = rf'D:\Project3\Result\\anomaly\OBS_extend\\VPD.npy'
        f_temp=rf'D:\Project3\Result\\anomaly\OBS_extend\\Tempmean.npy'
        f_tmax=rf'D:\Project3\Result\\anomaly\OBS_extend\\tmax.npy'
        dic_precip = T.load_npy(f_precip)
        dic_lai = T.load_npy(f_lai)
        dic_vpd=T.load_npy(f_vpd)
        dic_temp=T.load_npy(f_temp)
        dic_tmax=T.load_npy(f_tmax)
        year_list = range(1982,2021)
        year_list = np.array(year_list)

            ### find the extreme wet year and dry year

        result_dict={}

        flag = 0

        for pix in tqdm(dic_precip):
            if not pix in dic_precip_relative_change:
                continue
            if not pix in dic_vpd:
                continue
            if not pix in dic_lai:
                continue
            precip_relative_change=dic_precip_relative_change[pix]

            precip = dic_precip[pix]
            vpd = dic_vpd[pix]
            lai = dic_lai[pix]
            temp=dic_temp[pix]
            tmax=dic_tmax[pix]


            ## based on threshold to find the wet year


            picked_mild_wet_index = (precip_relative_change >= 0.5) & (precip_relative_change < 1)
            mild_wet_year = year_list[picked_mild_wet_index]  ## extract the year index based on the wet_index
            precip_mild_wet = precip[picked_mild_wet_index]
            vpd_mild_wet = vpd[picked_mild_wet_index]
            lai_mild_wet = lai[picked_mild_wet_index]
            temp_mild_wet=temp[picked_mild_wet_index]
            tmax_mild_wet=tmax[picked_mild_wet_index]
            mode_list_mild_wet = ['mild'] * len(mild_wet_year)


            picked_moderate_wet_index = (precip_relative_change >= 1) & (precip_relative_change < 2)
            moderate_wet_year = year_list[picked_moderate_wet_index]  ## extract the year index based on the wet_index
            precip_moderate_wet = precip[picked_moderate_wet_index]
            vpd_moderate_wet = vpd[picked_moderate_wet_index]
            lai_moderate_wet = lai[picked_moderate_wet_index]
            temp_moderate_wet = temp[picked_moderate_wet_index]
            tmax_moderate_wet = tmax[picked_moderate_wet_index]
            mode_list_moderate_wet = ['moderate'] * len(moderate_wet_year)

            picked_extreme_wet_index = (precip_relative_change >= 2) & (precip_relative_change <3)
            extreme_wet_year = year_list[picked_extreme_wet_index]  ## extract the year index based on the wet_index
            precip_extreme_wet = precip[picked_extreme_wet_index]
            vpd_extreme_wet = vpd[picked_extreme_wet_index]
            lai_extreme_wet = lai[picked_extreme_wet_index]
            temp_extreme_wet = temp[picked_extreme_wet_index]
            tmax_extreme_wet = tmax[picked_extreme_wet_index]
            mode_list_extreme_wet = ['extreme'] * len(extreme_wet_year)


            picked_random_wet_index = (precip_relative_change > 0) & (precip_relative_change < 0.5)
            random_wet_year = year_list[picked_random_wet_index]  ## extract the year index based on the wet_index
            precip_random_wet = precip[picked_random_wet_index]
            vpd_random_wet = vpd[picked_random_wet_index]
            lai_random_wet = lai[picked_random_wet_index]
            temp_random_wet = temp[picked_random_wet_index]
            tmax_random_wet = tmax[picked_random_wet_index]
            mode_list_random_wet = ['random'] * len(random_wet_year)


            for wet in range(len(mild_wet_year)):
                year = mild_wet_year[wet]
                mode = mode_list_mild_wet[wet]

                result_dict[flag] = {'year':year,'mode':mode,'precip':precip_mild_wet[wet],
            'pix':pix,'vpd':vpd_mild_wet[wet],'lai':lai_mild_wet[wet],'temp':temp_mild_wet[wet],'tmax':tmax_mild_wet[wet]}
                flag += 1

            for wet in range(len(moderate_wet_year)):
                year = moderate_wet_year[wet]
                mode = mode_list_moderate_wet[wet]

                result_dict[flag] = {'year':year,'mode':mode,'precip':precip_moderate_wet[wet],
            'pix':pix,'vpd':vpd_moderate_wet[wet],'lai':lai_moderate_wet[wet],'temp':temp_moderate_wet[wet],'tmax':tmax_moderate_wet[wet]}
                flag += 1



            for wet in range(len(extreme_wet_year)):
                year = extreme_wet_year[wet]
                mode = mode_list_extreme_wet[wet]
                result_dict[flag] = {'year':year,'mode':mode,'precip':precip_extreme_wet[wet],
            'pix':pix,'vpd':vpd_extreme_wet[wet],'lai':lai_extreme_wet[wet],'temp':temp_extreme_wet[wet],'tmax':tmax_extreme_wet[wet]}
                flag += 1


            for wet in range(len(random_wet_year)):
                year = random_wet_year[wet]
                mode = mode_list_random_wet[wet]
                result_dict[flag] = {'year':year,'mode':mode,'precip':precip_random_wet[wet],
            'pix':pix,'vpd':vpd_random_wet[wet],'lai':lai_random_wet[wet],'temp':temp_random_wet[wet],'tmax':tmax_random_wet[wet]}


                flag += 1
            # pprint(result_dict)
        df = T.dic_to_df(result_dict,'index')
        outf=join(outdir,'asymmetry_response_precip.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)


    def pick_dry_year_events(self):
        ## 1) build df for the extreme wet year and dry year for four period (1982-1990, 1991-2000, 2001-2010, 2011-2020)
        outdir = join(result_root,'asymmetry_response','dry_event_CRU')

        T.mk_dir(outdir)
        ### define  mild, moderate, extreme, and random fluation

        ##drought  mild -0.5 - -1, moderate drought -1 - -2, severe drought <-2, random -0.5 - 0

        f_precipitation_zscore = (result_root + rf'Detrend\detrend_zscore\1982_2020\CRU.npy')
        dic_precip_zscore = T.load_npy(f_precipitation_zscore)



        # ////plot the distribution of the zscore
        # value_list = []
        # for year in range(1982,2021):
        #     for pix in dic_precip_zscore:
        #         value=dic_precip_zscore[pix][year-1982]
        #         value_list.append(value)
        # value_list=np.array(value_list)
        # ### data_clean 1. remove 3std 2. remove nan
        # value_list=value_list[np.isnan(value_list)==0]
        # ## 3std clean
        # value_list=value_list[value_list<np.nanmean(value_list)+3*np.nanstd(value_list)]
        # value_list=value_list[value_list>np.nanmean(value_list)-3*np.nanstd(value_list)]
        # #
        #
        # plt.hist(value_list,bins=100)
        # plt.show()
        # exit()
        #


        f_precip=rf'D:\Project3\Result\Detrend\detrend_anomaly\1982_2020\\CRU.npy'
        f_lai = rf'D:\Project3\Result\Detrend\detrend_anomaly\1982_2020\\LAI4g.npy'
        f_vpd=rf'D:\Project3\Result\Detrend\detrend_anomaly\1982_2020\\VPD.npy'

        dic_precip = T.load_npy(f_precip)
        dic_lai = T.load_npy(f_lai)
        dic_vpd=T.load_npy(f_vpd)
        year_list = range(1982,2021)
        year_list = np.array(year_list)

            ### find the extreme wet year and dry year

        result_dict={}

        flag = 0

        for pix in tqdm(dic_precip):
            if not pix in dic_precip_zscore:
                continue
            if not pix in dic_vpd:
                continue
            if not pix in dic_lai:
                continue
            precip_zscore=dic_precip_zscore[pix]

            precip = dic_precip[pix]
            vpd = dic_vpd[pix]
            lai = dic_lai[pix]


            ## based on threshold to find the wet year


            picked_mild_dry_index = (precip_zscore >= -1) & (precip_zscore < -0.5)

            mild_dry_year = year_list[picked_mild_dry_index]  ## extract the year index based on the wet_index
            precip_mild_dry = precip[picked_mild_dry_index]
            vpd_mild_dry = vpd[picked_mild_dry_index]
            lai_mild_dry = lai[picked_mild_dry_index]
            mode_list_mild_dry = ['mild'] * len(mild_dry_year)
            picked_moderate_dry_index = (precip_zscore >= -2) & (precip_zscore < -1)
            moderate_dry_year = year_list[picked_moderate_dry_index]  ## extract the year index based on the wet_index
            precip_moderate_dry = precip[picked_moderate_dry_index]
            vpd_moderate_dry = vpd[picked_moderate_dry_index]
            lai_moderate_dry = lai[picked_moderate_dry_index]
            mode_list_moderate_dry = ['moderate'] * len(moderate_dry_year)
            picked_extreme_dry_index = (precip_zscore >= -3) & (precip_zscore < -2)
            extreme_dry_year = year_list[picked_extreme_dry_index]  ## extract the year index based on the wet_index
            precip_extreme_dry = precip[picked_extreme_dry_index]
            vpd_extreme_dry = vpd[picked_extreme_dry_index]
            lai_extreme_dry = lai[picked_extreme_dry_index]
            mode_list_extreme_dry = ['extreme'] * len(extreme_dry_year)
            picked_random_dry_index = (precip_zscore >= -0.5) & (precip_zscore < 0)
            random_dry_year = year_list[picked_random_dry_index]  ## extract the year index based on the wet_index
            precip_random_dry = precip[picked_random_dry_index]
            vpd_random_dry = vpd[picked_random_dry_index]
            lai_random_dry = lai[picked_random_dry_index]
            mode_list_random_dry = ['random'] * len(random_dry_year)

            for dry in range(len(mild_dry_year)):
                year = mild_dry_year[dry]
                mode = mode_list_mild_dry[dry]

                result_dict[flag] = {'year':year,'mode':mode,'precip':precip_mild_dry[dry],'pix':pix,'vpd':vpd_mild_dry[dry],'lai':lai_mild_dry[dry]}
                flag += 1
            for dry in range(len(moderate_dry_year)):
                year = moderate_dry_year[dry]
                mode = mode_list_moderate_dry[dry]
                result_dict[flag] = {'year':year,'mode':mode,'precip':precip_moderate_dry[dry],'pix':pix,'vpd':vpd_moderate_dry[dry],'lai':lai_moderate_dry[dry]}
                flag += 1
            for dry in range(len(extreme_dry_year)):
                year = extreme_dry_year[dry]
                mode = mode_list_extreme_dry[dry]
                result_dict[flag] = {'year':year,'mode':mode,'precip':precip_extreme_dry[dry],'pix':pix,'vpd':vpd_extreme_dry[dry],'lai':lai_extreme_dry[dry]}
                flag += 1
            for dry in range(len(random_dry_year)):
                year = random_dry_year[dry]
                mode = mode_list_random_dry[dry]
                result_dict[flag] = {'year':year,'mode':mode,'precip':precip_random_dry[dry],'pix':pix,'vpd':vpd_random_dry[dry],'lai':lai_random_dry[dry]}
                flag += 1
            # pprint(result_dict)
        df = T.dic_to_df(result_dict,'index')
        outf=join(outdir,'asymmetry_response_precip.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)







    def pick_wet_year_anomaly(self):


        outdir = join(result_root, 'asymmetry_response', 'wet_relative_change_CRU')

        T.mk_dir(outdir)
        ### define  mild, moderate, extreme, and random fluation

        ##drought  mild -0.5 - -1, moderate drought -1 - -2, severe drought <-2, random -0.5 - 0

        f_precipitation_zscore = (result_root + rf'zscore\CRU.npy')
        dic_precip_zscore = T.load_npy(f_precipitation_zscore)


        f_precip = rf'D:\Project3\Result\\relative_change\OBS_LAI_extend\\\CRU.npy'
        f_lai = rf'D:\Project3\Result\\relative_change\OBS_LAI_extend\\LAI4g.npy'
        f_vpd = rf'D:\Project3\Result\\relative_change\OBS_LAI_extend\\VPD.npy'

        dic_precip = T.load_npy(f_precip)
        dic_lai = T.load_npy(f_lai)
        dic_vpd = T.load_npy(f_vpd)
        # year_list = range(1982, 2021)
        year_list= range(1982, 2001)
        year_list=range(2001,2021)
        year_list=np.array(year_list)
        year_list = np.array(year_list)

        ### find the extreme wet year and dry year

        result_dict = {}

        flag = 0

        for pix in tqdm(dic_precip):
            if not pix in dic_precip_zscore:
                continue
            if not pix in dic_vpd:
                continue
            if not pix in dic_lai:
                continue
            precip_zscore = dic_precip_zscore[pix][19:]

            precip = dic_precip[pix][19:]
            vpd = dic_vpd[pix][19:]
            lai = dic_lai[pix][19:]

            ## based on threshold to find the wet year

            picked_mild_dry_index = (precip_zscore > 0.5) & (precip_zscore < 3)


            mild_dry_year = year_list[picked_mild_dry_index]  ## extract the year index based on the wet_index
            precip_mild_dry = precip[picked_mild_dry_index]
            vpd_mild_dry = vpd[picked_mild_dry_index]
            lai_mild_dry = lai[picked_mild_dry_index]
            mode_list_mild_dry = ['wet'] * len(mild_dry_year)

            for dry in range(len(mild_dry_year)):
                year = mild_dry_year[dry]
                mode = mode_list_mild_dry[dry]

                result_dict[flag] = {'year': year, 'mode': mode, 'precip': precip_mild_dry[dry], 'pix': pix,
                                     'vpd': vpd_mild_dry[dry], 'lai': lai_mild_dry[dry]}
                flag += 1

            # pprint(result_dict)
        df = T.dic_to_df(result_dict, 'index')
        outf = join(outdir, 'asymmetry_response_precip_2001_2020.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)



    def pick_wet_year_anomaly(self):


        outdir = join(result_root, 'asymmetry_response', 'wet_relative_change_CRU')

        T.mk_dir(outdir)
        ### define  mild, moderate, extreme, and random fluation

        ##drought  mild -0.5 - -1, moderate drought -1 - -2, severe drought <-2, random -0.5 - 0

        f_precipitation_zscore = (result_root + rf'zscore\CRU.npy')
        dic_precip_zscore = T.load_npy(f_precipitation_zscore)


        f_precip = rf'D:\Project3\Result\\relative_change\OBS_LAI_extend\\\CRU.npy'
        f_lai = rf'D:\Project3\Result\\relative_change\OBS_LAI_extend\\LAI4g.npy'
        f_vpd = rf'D:\Project3\Result\\relative_change\OBS_LAI_extend\\VPD.npy'

        dic_precip = T.load_npy(f_precip)
        dic_lai = T.load_npy(f_lai)
        dic_vpd = T.load_npy(f_vpd)
        # year_list = range(1982, 2021)
        year_list= range(1982, 2001)
        year_list=range(2001,2021)
        year_list=np.array(year_list)
        year_list = np.array(year_list)

        ### find the extreme wet year and dry year

        result_dict = {}

        flag = 0

        for pix in tqdm(dic_precip):
            if not pix in dic_precip_zscore:
                continue
            if not pix in dic_vpd:
                continue
            if not pix in dic_lai:
                continue
            precip_zscore = dic_precip_zscore[pix][19:]

            precip = dic_precip[pix][19:]
            vpd = dic_vpd[pix][19:]
            lai = dic_lai[pix][19:]

            ## based on threshold to find the wet year

            picked_mild_dry_index = (precip_zscore > 0.5) & (precip_zscore < 3)


            mild_dry_year = year_list[picked_mild_dry_index]  ## extract the year index based on the wet_index
            precip_mild_dry = precip[picked_mild_dry_index]
            vpd_mild_dry = vpd[picked_mild_dry_index]
            lai_mild_dry = lai[picked_mild_dry_index]
            mode_list_mild_dry = ['wet'] * len(mild_dry_year)

            for dry in range(len(mild_dry_year)):
                year = mild_dry_year[dry]
                mode = mode_list_mild_dry[dry]

                result_dict[flag] = {'year': year, 'mode': mode, 'precip': precip_mild_dry[dry], 'pix': pix,
                                     'vpd': vpd_mild_dry[dry], 'lai': lai_mild_dry[dry]}
                flag += 1

            # pprint(result_dict)
        df = T.dic_to_df(result_dict, 'index')
        outf = join(outdir, 'asymmetry_response_precip_2001_2020.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)




    def pick_dry_year_anomaly(self):

        outdir = join(result_root, 'asymmetry_response', 'dry_anomaly_CRU')

        T.mk_dir(outdir)
        ### define  mild, moderate, extreme, and random fluation

        ##drought  mild -0.5 - -1, moderate drought -1 - -2, severe drought <-2, random -0.5 - 0

        f_precipitation_zscore = (result_root + rf'zscore\CRU.npy')
        dic_precip_zscore = T.load_npy(f_precipitation_zscore)

        f_precip = rf'D:\Project3\Result\\anomaly\OBS_extend\\CRU.npy'
        f_lai = rf'D:\Project3\Result\\anomaly\OBS_extend\\LAI4g.npy'
        f_vpd = rf'D:\Project3\Result\\anomaly\OBS_extend\\VPD.npy'

        dic_precip = T.load_npy(f_precip)
        dic_lai = T.load_npy(f_lai)
        dic_vpd = T.load_npy(f_vpd)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        ### find the extreme wet year and dry year

        result_dict = {}

        flag = 0

        for pix in tqdm(dic_precip):
            if not pix in dic_precip_zscore:
                continue
            if not pix in dic_vpd:
                continue
            if not pix in dic_lai:
                continue
            precip_zscore = dic_precip_zscore[pix]

            precip = dic_precip[pix]
            vpd = dic_vpd[pix]
            lai = dic_lai[pix]

            ## based on threshold to find the wet year

            picked_mild_dry_index = (precip_zscore < -0.5) & (precip_zscore > -3)

            mild_dry_year = year_list[picked_mild_dry_index]  ## extract the year index based on the wet_index
            precip_mild_dry = precip[picked_mild_dry_index]
            vpd_mild_dry = vpd[picked_mild_dry_index]
            lai_mild_dry = lai[picked_mild_dry_index]
            mode_list_mild_dry = ['dry'] * len(mild_dry_year)

            for dry in range(len(mild_dry_year)):
                year = mild_dry_year[dry]
                mode = mode_list_mild_dry[dry]

                result_dict[flag] = {'year': year, 'mode': mode, 'precip': precip_mild_dry[dry], 'pix': pix,
                                     'vpd': vpd_mild_dry[dry], 'lai': lai_mild_dry[dry]}
                flag += 1

            # pprint(result_dict)
        df = T.dic_to_df(result_dict, 'index')
        outf = join(outdir, 'asymmetry_response_precip.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)





    def calculate_frequency_wet_dry_event(self):  ##based on pixel and then calculate the frequency of the wet and dry year
        f_precipitation = result_root+rf'zscore\CRU.npy'
        dic_precip = T.load_npy(f_precipitation)



        period_list=['1982-2000','1992-2010','2002-2020']
        # period_list = ['1982-1989', '1990-1997', '1998-2005', '2006-2013', '2014-2020']


        for period in period_list:
            period_upper = int(period.split('-')[1])+1
            period_lower = int(period.split('-')[0])
            year_list = range(period_lower, period_upper)
            year_list = np.array(year_list)

            result_dic = {}

            for pix in tqdm(dic_precip):


                precip_val = dic_precip[pix][period_lower-1982:period_upper-1982]
                    ### wet
                mild=precip_val[(precip_val>=0.5) & (precip_val<1)]
                moderate=precip_val[(precip_val>=1) & (precip_val<2)]
                extreme=precip_val[(precip_val>=2)& (precip_val<3)]
                random=precip_val[(precip_val>0) & (precip_val<0.5)]

                ##dry
                #
                # mild = precip_val[(precip_val <= -0.5) & (precip_val > -1)]
                # moderate = precip_val[(precip_val <= -1) & (precip_val > -2)]
                # extreme = precip_val[(precip_val <= -2)]
                # random = precip_val[(precip_val < 0) & (precip_val > -0.5)]
                # print(len(mild),len(moderate),len(extreme),len(random))
                # exit()
                mild_frenquency = len(mild)/len(year_list)*100
                moderate_frenquency = len(moderate)/len(year_list)*100
                extreme_frenquency = len(extreme)/len(year_list)*100
                random_frenquency = len(random)/len(year_list)*100

                result_dic[pix] = {'mild':mild_frenquency,'moderate':moderate_frenquency,'extreme':extreme_frenquency,'random':random_frenquency}


            outdir=join(result_root,'asymmetry_response','frequency_wet_anomaly_CRU')
            T.mk_dir(outdir)
            outf=join(outdir,f'frequency_{period}.npy')
            np.save(outf,result_dic)

    def plot_frequency_wet_dry_spatial(self):

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        period_list = ['1982-2000', '1992-2010', '2002-2020']
        mode_list = ['mild', 'moderate', 'extreme', 'random']
        mode_list=['wet']
        for period in period_list:
            f = join(result_root, 'asymmetry_response', 'frequency_wet_CRU', f'frequency_{period}.npy')
            dic = T.load_npy(f)
            spatial_dic = {}
            for mode in mode_list:

                for pix in dic:
                    r,c=pix
                    if r<120:
                        continue
                    landcover_value = crop_mask[pix]
                    if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                        continue
                    if dic_modis_mask[pix] == 12:
                        continue
                    spatial_dic[pix] = dic[pix][mode]
                array= DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
                array=array*array_mask
                # plt.imshow(array)
                # plt.colorbar()
                # plt.show()
                outf=join(result_root, 'asymmetry_response', 'frequency_wet_CRU', f'frequency_{period}_{mode}.tif')
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(array, outf)










                # outf = outf.replace('.npy', '.tif')
                # ##array to tif
                # DIC_and_TIF(pixelsize=0.25).arr_to_tif(array, outf)

    def plot_bar_frequency_wet_dry_event(self):  ###whole dryland
        period_list = ['1982-1990', '1991-2000', '2001-2010', '2011-2020']
        mode=['mild','moderate','extreme']

        dry_color_list=['peachpuff','orange','darkorange','chocolate','saddlebrown'] ## reverse

        wet_color_list = ['lightblue', 'cyan', 'deepskyblue', 'dodgerblue', 'navy']

        dff=rf'D:\Project3\Result\\asymmetry_response\Dataframe\wet_dry_frequency.df'

        df = T.load_df(dff)
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['MODIS_LUCC'] != 12]
        # df = df[df['partial_corr_GPCC'] < 0.5]
        df = df.dropna()
        # df = df[df['continent'] == 'Australia']
        # df = df[df['continent'] == 'North_America']
        # df = df[df['continent'] == 'South_America']
        # df = df[df['continent'] == 'Africa']

        result_dic={}

        for period in period_list:
            result_dic[period]={}
            lower_period=int(period.split('-')[0])
            upper_period=int(period.split('-')[1])
            for mode_ in mode:

                vals=df[f'frequency_{lower_period}-{upper_period}_{mode_}_wet'].to_list()
                vals_array=np.array(vals)
                average=np.nanmean(vals_array)
                result_dic[period][mode_]=average
        df_new = pd.DataFrame(result_dic)
        df_new = df_new.T
        df_new.plot(kind='bar', stacked=True, color=dry_color_list)
        plt.ylabel('Frequency of wet year (%)')
        plt.xticks(rotation=0)
        plt.ylim(0,40)
        plt.show()

    def plot_bar_frequency_dry_regional_three_mode(self):


        period_list = ['1982-1990', '1991-2000', '2001-2010', '2011-2020']
        # period_list=['1982-1989', '1990-1997', '1998-2005', '2006-2013', '2014-2020']
        mode=['mild','moderate','extreme']
        dry_color_list=['peachpuff','orange','darkorange','chocolate','saddlebrown'] ## reverse

        wet_color_list = [ 'cyan', 'deepskyblue', 'dodgerblue', 'navy']

        dff=rf'D:\Project3\Result\\asymmetry_response\\\relative_change_detrend\frequency_wet_event_CRU\\frequency_wet_event_CRU.df'

        df = T.load_df(dff)
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['MODIS_LUCC'] != 12]
        # df = df[df['partial_corr_GPCC'] < 0.5]
        df = df.dropna()


        AI_classfication = ['Africa',  'Asia','Australia','North_America', 'South_America','global']
        result_dic = {}
        for AI in AI_classfication:
            result_dic[AI] = {}
            if AI=='global':
                df_AI=df
            else:
                df_AI=df[df['continent']==AI]
            for period in period_list:


                result_dic[AI][period] = {}

                lower_period = int(period.split('-')[0])
                upper_period = int(period.split('-')[1])

                for mode_ in mode:

                    result_dic[AI][period][mode_] = {}

                    vals = df_AI[f'frequency_{lower_period}-{upper_period}_{mode_}'].to_list()
                    vals_array = np.array(vals)
                    average = np.nanmean(vals_array)
                    result_dic[AI][period][mode_] = average


        pprint(result_dic)

        fig, axs = plt.subplots(2, 3, figsize=(12, 4))
        flag=0


        ##plot stack bar as function of AI and period
        for AI in AI_classfication:
            ax=axs.ravel()[flag]

            result_dic_AI = result_dic[AI]
            df_new = pd.DataFrame(result_dic_AI)
            df_new = df_new.T
            df_new.plot(kind='bar', stacked=False, color=wet_color_list,  width=0.8, ax=ax)
            x_ticks = df_new.index.to_list()
            ax.set_xticklabels(x_ticks, rotation=45)
            ax.set_ylabel('Frequency events (%)')
            ax.set_title(AI)
            ax.set_ylim(0, 20)


            flag+=1
        plt.tight_layout()

        plt.show()
        exit()


    def plot_bar_frequency_dry_regional_one_mode(self):


        period_list = ['1982-1990', '1991-2000', '2001-2010', '2011-2020']
        period_list=['1982-1989', '1990-1997', '1998-2005', '2006-2013', '2014-2020']

        dry_color_list=['peachpuff','orange','darkorange','chocolate','saddlebrown'] ## reverse

        wet_color_list = [ 'deepskyblue', 'dodgerblue', 'navy']

        dff=rf'D:\Project3\Result\\asymmetry_response\\frequency_wet_CRU\frequency_wet_CRU.df'

        df = T.load_df(dff)
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['MODIS_LUCC'] != 12]
        # df = df[df['partial_corr_GPCC'] < 0.5]
        df = df.dropna()


        AI_classfication = ['Africa', 'Australia', 'Asia','North_America', 'South_America','global']
        result_dic = {}
        for AI in AI_classfication:
            result_dic[AI] = {}
            if AI=='global':
                df_AI=df
            else:

                df_AI=df[df['continent']==AI]

            for period in period_list:

                lower_period = int(period.split('-')[0])
                upper_period = int(period.split('-')[1])

                vals = df_AI[f'frequency_{lower_period}-{upper_period}_wet'].to_list()
                vals_array = np.array(vals)
                vals_array[vals_array<-9999]=np.nan
                vals_array[vals_array>9999]=np.nan

                average = np.nanmean(vals_array)
                result_dic[AI][period] = average


        pprint(result_dic)

        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        flag=0


        ##plot stack bar as function of AI and period
        for AI in AI_classfication:
            ax=axs.ravel()[flag]

            result_dic_AI = result_dic[AI]
            df_new = pd.DataFrame(result_dic_AI,index=[0])
            df_new = df_new.T
            df_new.plot(kind='bar', stacked=True, color=wet_color_list,  width=0.8, ax=ax,legend=False)
            x_ticks = df_new.index.to_list()
            ax.set_xticklabels(x_ticks, rotation=45)
            ax.set_ylabel('Frequency of events (%)')
            ax.set_title(AI)
            ax.set_ylim(0, 40)



            flag+=1
        plt.tight_layout()

        plt.show()
        exit()


    def calculate_frequency_wet_dry_anomaly(self):  ##based on pixel and then calculate the frequency of the wet and dry year
        f_precipitation = result_root+rf'zscore\CRU.npy'
        dic_precip = T.load_npy(f_precipitation)
        mode='wet'


        period_list = ['1982-2000', '1992-2010', '2002-2020']



        for period in period_list:
            period_upper = int(period.split('-')[1])+1
            period_lower = int(period.split('-')[0])
            year_list = range(period_lower, period_upper)
            year_list = np.array(year_list)

            result_dic = {}

            for pix in tqdm(dic_precip):


                precip_val = dic_precip[pix][period_lower-1982:period_upper-1982]
                    ### wet
                # mild=precip_val[(precip_val>0.5) & (precip_val<1)]
                # moderate=precip_val[(precip_val>1) & (precip_val<2)]
                # extreme=precip_val[(precip_val>2)]
                # random=precip_val[(precip_val>0) & (precip_val<0.5)]

                ##dry

                wet = precip_val[(precip_val > 0.5) & (precip_val < 3)]
                # dry = precip_val[(precip_val <-3) & (precip_val > -0.5)]


                # print(len(mild),len(moderate),len(extreme),len(random))
                # exit()
                wet_frenquency = len(wet)/len(year_list)*100

                result_dic[pix] = {f'{mode}':wet_frenquency}

            outdir=join(result_root,'asymmetry_response',f'frequency_{mode}_CRU')
            T.mk_dir(outdir)
            outf=join(outdir,f'frequency_{period}.npy')
            np.save(outf,result_dic)

class multi_regression():

    def __init__(self):
        pass
    def run(self):
        ## 1) build the dataframe for the multi-regression and calculate

        period_list=['1982-2000','1992-2010','2002-2020']
        for period in period_list:
            self.period=period
            df=self.build_df()
            self.cal_multi_regression_beta_pixel(df)

        # self.cal_multi_regression_beta_regional()
        # self.cal_partial_correlation()

        ### 2) plot spatial resulat
        #




    def build_df(self):
        period=self.period
        fdir_X = result_root + rf'asymmetry_response\\wet_relative_change_CRU\\'

        f_LAI = result_root + rf'\asymmetry_response\\wet_relative_change_CRU\\asymmetry_response_LAI_{period}.npy'
        dic_y = T.load_npy(f_LAI)

        x_var_list = ['precip', 'VPD', ]

        df = pd.DataFrame()

        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue


            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        for xvar in x_var_list:

            # print(var_name)
            x_val_list = []
            filex = fdir_X + f'asymmetry_response_{xvar}_{period}.npy'

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
    def clean_df(self, df):

        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df=df[df['Aridity']<0.65]

        df = df[df['MODIS_LUCC'] != 12]
        df=df.dropna()
        # print(len(df))


            ## remove the outlier

            # df = df[df['GPCC'] > -2.5]
            # df = df[df['GPCC'] < 2.5]
            # df=df[df['LAI4g_zscore']>-3]
            # df=df[df['LAI4g_zscore']<3]
            # #
            # print(len(df))
            # exit()


            # df=df[df['continent']=='Australia']
            # df = df[df['continent'] == 'North_America']
            # df = df[df['continent'] == 'South_America']
            # df = df[df['continent'] == 'Africa']
            # df = df[df['continent'] == 'Asia']



        return df



    def cal_multi_regression_beta_categroy(self, ):
        f=rf'D:\Project3\Result\asymmetry_response\wet_anomaly_GPCC\\asymmetry_response_precip.df'


        df = T.load_df(f)
        df=self.clean_df(df)

        x_var_list = ['precip', 'vpd',]

        ### loop wet and dry
        multi_derivative_period = {}

        mode_list=['wet','dry']
        # mode_list = ['mild','moderate','extreme',]

        period_list = ['1982-1990', '1991-2000', '2001-2010', '2011-2020']
        for period in period_list:
            df_period = df[df['year']>=int(period.split('-')[0])]
            df_period = df_period[df_period['year']<=int(period.split('-')[1])]
            multi_derivative_period[period] = {}
            multi_derivative_mode = {}



            for mode in mode_list:
                df_mode = df_period[df_period['mode']==mode]
                y_values=df_mode['lai'].to_list()
                y_mean = df_mode['growing_season_LAI_mean'].mean()

                    ## interpolate the nan value
                if len(y_values) == 0:
                    continue
                # y_values = T.interp_nan(y_values)
                #
                # y_vals=signal.detrend(y_values)

                df_new = pd.DataFrame()
                x_var_list_valid = []

                for x in x_var_list:

                    x_vals = df_mode[x].to_list()
                    x_vals = T.interp_nan(x_vals)


                    if len(x_vals) == 0:
                        continue

                    if np.isnan(np.nanmean(x_vals)):
                        continue


                    if len(x_vals) != len(y_values):
                        continue
                    # print(x_vals)
                    if x_vals[0] == None:
                        continue
                    # x_vals_detrend = signal.detrend(x_vals) #detrend
                    df_new[x] = x_vals

                    x_var_list_valid.append(x)
                    if len(df_new) <= 3:
                        continue
                df_new['y'] = y_values


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
                coef_ = np.array(linear_model.coef_)/y_mean*100 *100 ## *100% * 100mm
                coef_dic = dict(zip(x_var_list_valid_new, coef_))

                # print(df_new['y'])
                # exit()
                multi_derivative_mode[mode] = coef_dic
            multi_derivative_period[period] = multi_derivative_mode


        wet_color_list = ['cyan', 'deepskyblue', 'dodgerblue', 'navy']

        for period in period_list:

            for mode in mode_list:
                multi_derivative_period[period][mode] = multi_derivative_period[period][mode]['precip']
        ##extract precipitation and creat df
        df_new_plot = pd.DataFrame(multi_derivative_period)
        df_new_plot = df_new_plot.T
        df_new_plot.plot(kind='bar', color=wet_color_list, width=0.8)
        plt.xticks(rotation=0)
        plt.ylabel('(%/100mm)')
        plt.show()

    def cal_multi_regression_beta_regional(self, ):

        ###creat
        f = rf'D:\Project3\Result\asymmetry_response\\anomaly\nodetrend\\\wet_event_CRU\asymmetry_response_precip.df'

        df = T.load_df(f)
        df = self.clean_df(df)
        # precip=df['precip'].tolist()
        # plt.hist(precip,bins=100)
        # plt.show()
        # exit()


        x_var_list = ['precip', 'tmax', ]
        sample_number_list = []
        ### loop wet and dry
        multi_derivative_region = {}


        AI_classfication= ['Global','Africa', 'Asia','Australia','North_America', 'South_America',]

        mode_list = [ 'extreme', 'moderate', 'mild', ]
        # mode_list = ['dry']
        # period_list = ['1982-1997', '1983-1998','1984-1999','1985-2000','1986-2001','1987-2002','1988-2003','1989-2004','1990-2005','1991-2006','1992-2007','1993-2008','1994-2009','1995-2010','1996-2011','1997-2012','1998-2013','1999-2014','2000-2015','2001-2016','2002-2017','2003-2018','2004-2019','2005-2020']

        # period_list = ['1982-2000', '2001-2020']
        period_list=['1982-1990','1991-2000','2001-2010','2011-2020']



        for AI in AI_classfication:
            # df_AI = df[df['AI_classfication'] == AI]
            if AI == 'Global':
                df_AI = df
            else:
                df_AI = df[df['continent'] == AI]

            multi_derivative_period = {}

            for period in period_list:


                df_period = df_AI[df_AI['year'] >= int(period.split('-')[0])]
                df_period = df_period[df_period['year'] <= int(period.split('-')[1])]

                multi_derivative_mode = {}

                for mode in mode_list:


                    df_mode = df_period[df_period['mode'] == mode]
                    y_values = df_mode['lai'].to_list()

                    y_mean = df_mode['growing_season_LAI_mean'].mean()
                    sample_number_list.append(len(y_values))

                    #####plot scatter plot between precip and lai
                    # plt.scatter(df_mode['precip'],df_mode['lai'])
                    # pix_list = df_mode['pix'].to_list()
                    # spatial_dict_test = {}
                    # for pix in pix_list:
                    #     spatial_dict_test[pix] = 1
                    # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_test)
                    # background = rf"D:\Project3\Result\asymmetry_response\frequency_dry\frequency_1982-1990_extreme.tif"
                    # arr_i = ToRaster().raster2array(background)[0]
                    # arr_i[arr_i<-9999] = np.nan
                    # arr_i[arr_i > -9999] = 1
                    # plt.imshow(arr_i,interpolation='nearest',zorder=-99,cmap='gray')
                    # # plt.show()
                    # # DIC_and_TIF(pixelsize=.25).plot_back_ground_arr(background)
                    # plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=2)
                    # plt.show()
                    ##text pix
                    # for i in range(len(df_mode)):
                    #     plt.text(df_mode['precip'].iloc[i],df_mode['lai'].iloc[i],df_mode['pix'].iloc[i])
                    # plt.show()
                    # exit()

                    ## interpolate the nan value
                    if len(y_values) == 0:
                        continue
                    # y_values = T.interp_nan(y_values)
                    #
                    # detrended_y = signal.detrend(y_values)

                    df_new = pd.DataFrame()
                    x_var_list_valid = []

                    for x in x_var_list:

                        x_vals = df_mode[x].to_list()
                        # x_vals = T.interp_nan(x_vals)

                        if len(x_vals) == 0:
                            continue

                        if np.isnan(np.nanmean(x_vals)):
                            continue

                        if len(x_vals) != len(y_values):
                            continue
                        # print(x_vals)
                        if x_vals[0] == None:
                            continue
                        # x_vals_detrend = signal.detrend(x_vals)  # detrend
                        df_new[x] = x_vals

                        x_var_list_valid.append(x)
                        if len(df_new) <= 3:
                            continue
                    df_new['y'] = y_values

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
                    # print(len(df_new))
                    # key = f'{AI}_{period}_{mode}'
                    # if key == 'Arid_2001-2010_extreme':
                    #     T.print_head_n(df_new)
                    # df_new = df_new[df_new['precip'] > -230]
                        # exit()
                    ##partial correlation


                    linear_model = LinearRegression()

                    linear_model.fit(df_new[x_var_list_valid_new], df_new['y'])


                    # plt.hist(df_new['y'],bins=100)

                    # plt.hist(df_new['precip'],bins=100)
                    # # plt.hist(df_new['y'],bins=100)
                    # plt.title(f'{AI}_{period}_{mode}')
                    #
                    #
                    # plt.show()

                    coef_ = np.array(linear_model.coef_)/y_mean *100*100# *100% * 100mm
                    coef_dic = dict(zip(x_var_list_valid_new, coef_))


                    multi_derivative_mode[mode] = coef_dic['precip']
                multi_derivative_period[period] = multi_derivative_mode
                ##plot hist for each period


            multi_derivative_region[AI] = multi_derivative_period


        wet_color_list = ['cyan', 'deepskyblue', 'dodgerblue', 'navy']


        dry_color_list = ['peachpuff', 'orange', 'darkorange', 'brown']
        dry_color_list = [ 'darkorange', 'orange', 'peachpuff']

        fig, axs = plt.subplots(2, 3, figsize=(10, 5))
        flag = 0

        ##plot stack bar as function of AI and period

        for AI in AI_classfication:
            ax=axs.ravel()[flag]
            # ax = axs[flag]

            result_dic_AI = multi_derivative_region[AI]
            df_new = pd.DataFrame(result_dic_AI)
            df_new = df_new.T

            df_new.plot(kind='bar', stacked=False, color=wet_color_list, width=0.8, ax=ax,legend=True)
            ##add add number of sample on the bar
            # for i in range(len(df_new)):
            #         ax.text(i, df_new['wet'].iloc[i] + 0.1, f'{sample_number_list[i]}', ha='center', va='bottom')

            x_ticks = df_new.index.to_list()
            ax.set_xticklabels(x_ticks, rotation=45)
            ax.set_ylabel('(%/100mm)')
            ax.set_title(AI)
            # ax.set_ylim(-5,50)
            # ax.set_yticks([0,5,10,15,20,25,30])

            flag += 1
        plt.tight_layout()

        plt.show()
        exit()

    def cal_partial_correlation(self, ):

        ###creat
        f = rf'D:\Project3\Result\asymmetry_response\wet_event_CRU\asymmetry_response_precip.df'

        df = T.load_df(f)
        df = self.clean_df(df)
        # precip=df['precip'].tolist()
        # plt.hist(precip,bins=100)
        # plt.show()
        # exit()

        x_var_list = ['precip', 'vpd', ]
        sample_number_list = []
        ### loop wet and dry
        multi_derivative_region = {}
        period_derivative_region = {}

        AI_classfication = ['Africa', 'Asia', 'Australia', 'North_America', 'South_America', 'Global']

        mode_list = ['extreme', 'moderate', 'mild']
        # mode_list = ['dry']
        period_list = ['1982-1990', '1991-2000', '2001-2010', '2011-2020']
        # period_list = ['1982-1990', '1991-2000', '2001-2010', '2011-2020']

        for AI in AI_classfication:
            # df_AI = df[df['AI_classfication'] == AI]
            if AI == 'Global':
                df_AI = df
            else:
                df_AI = df[df['continent'] == AI]


            partial_correlation_period = {}
            partial_p_value_period = {}

            for period in period_list:

                df_period = df_AI[df_AI['year'] >= int(period.split('-')[0])]
                df_period = df_period[df_period['year'] <= int(period.split('-')[1])]


                partial_correlation_mode = {}
                partial_p_value_mode = {}


                for mode in mode_list:

                    df_mode = df_period[df_period['mode'] == mode]
                    y_values = df_mode['lai'].to_list()

                    y_mean = df_mode['growing_season_LAI_mean'].mean()
                    sample_number_list.append(len(y_values))

                    #####plot scatter plot between precip and lai
                    # plt.scatter(df_mode['precip'],df_mode['lai'])
                    # pix_list = df_mode['pix'].to_list()
                    # spatial_dict_test = {}
                    # for pix in pix_list:
                    #     spatial_dict_test[pix] = 1
                    # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict_test)
                    # background = rf"D:\Project3\Result\asymmetry_response\frequency_dry\frequency_1982-1990_extreme.tif"
                    # arr_i = ToRaster().raster2array(background)[0]
                    # arr_i[arr_i<-9999] = np.nan
                    # arr_i[arr_i > -9999] = 1
                    # plt.imshow(arr_i,interpolation='nearest',zorder=-99,cmap='gray')
                    # # plt.show()
                    # # DIC_and_TIF(pixelsize=.25).plot_back_ground_arr(background)
                    # plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=2)
                    # plt.show()
                    ##text pix
                    # for i in range(len(df_mode)):
                    #     plt.text(df_mode['precip'].iloc[i],df_mode['lai'].iloc[i],df_mode['pix'].iloc[i])
                    # plt.show()
                    # exit()

                    ## interpolate the nan value
                    if len(y_values) == 0:
                        continue
                    # y_values = T.interp_nan(y_values)
                    #
                    # detrended_y = signal.detrend(y_values)

                    df_new = pd.DataFrame()
                    x_var_list_valid = []

                    for x in x_var_list:

                        x_vals = df_mode[x].to_list()
                        # x_vals = T.interp_nan(x_vals)

                        if len(x_vals) == 0:
                            continue

                        if np.isnan(np.nanmean(x_vals)):
                            continue

                        if len(x_vals) != len(y_values):
                            continue
                        # print(x_vals)
                        if x_vals[0] == None:
                            continue
                        # x_vals_detrend = signal.detrend(x_vals)  # detrend
                        df_new[x] = x_vals

                        x_var_list_valid.append(x)
                        if len(df_new) <= 3:
                            continue
                    df_new['y'] = y_values

                    df_new = df_new.dropna(axis=1, how='all')

                    # T.print_head_n(df_new)
                    partial_correlation = {}
                    partial_correlation_p_value = {}
                    for x in x_var_list_valid:

                        x_var_list_valid_new_cov = copy.copy(x_var_list_valid)
                        x_var_list_valid_new_cov.remove(x)
                        r, p = self.partial_corr(df_new, x, 'y', x_var_list_valid_new_cov)
                        partial_correlation[x] = r
                        partial_correlation_p_value[x] = p

                    partial_correlation_mode[mode] = partial_correlation['precip']
                    partial_p_value_mode[mode] = partial_correlation_p_value['precip']



                    partial_correlation_period[period] = partial_correlation_mode
                    partial_correlation_period[period] = partial_p_value_mode

                ##plot hist for each period

                multi_derivative_region[AI] = partial_correlation_period

        wet_color_list = ['cyan', 'deepskyblue', 'dodgerblue', 'navy']

        dry_color_list = ['peachpuff', 'orange', 'darkorange', 'brown']

        fig, axs = plt.subplots(2, 3, figsize=(10, 5))
        flag = 0

        ##plot stack bar as function of AI and period

        for AI in AI_classfication:
            ax = axs.ravel()[flag]
            # ax = axs[flag]

            result_dic_AI = multi_derivative_region[AI]
            df_new = pd.DataFrame(result_dic_AI)
            df_new = df_new.T

            df_new.plot(kind='bar', stacked=False, color=wet_color_list, width=0.8, ax=ax, legend=True)
            ##add add number of sample on the bar
            # for i in range(len(df_new)):
            #         ax.text(i, df_new['wet'].iloc[i] + 0.1, f'{sample_number_list[i]}', ha='center', va='bottom')

            x_ticks = df_new.index.to_list()
            ax.set_xticklabels(x_ticks, rotation=45)
            ax.set_ylabel('(%/100mm)')
            ax.set_title(AI)
            ax.set_ylim(-0.5,0.5)
            ax.set_yticks([-0.5,0,0.5])

            flag += 1
        plt.tight_layout()

        plt.show()
        exit()

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


















































































    def plt_multi_regression_result(self, ):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        fdir=rf'D:\Project3\Result\asymmetry_response\multi_derivative\\'

        for f in os.listdir(fdir):
            outf=fdir+f



            dic = T.load_npy(fdir+f)
            var_list = []
            for pix in dic:

                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue

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
                arr = arr * array_mask
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, outf.replace('.npy', f'_{var_i}.tif'))
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

class monte_carlo:
    def __init__(self):
        pass
    def run(self):
        # self.cal_monte_carlo_wet()
        # self.cal_monte_carlo_wet_sperate_period()
        # self.cal_monte_carlo_dry()
        # self.cal_monte_carlo_dry_sperate_period()
        # self.rename_result()
        # self.check_result()
        # self.difference()

        self.bar_plot()
        # self.testrobinson()
        pass
    def cal_monte_carlo_wet(self):
        outdir = join(result_root, 'monte_carlo')
        T.mk_dir(outdir)
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends


        f_preciptation_zscore = result_root+rf'zscore\\GPCC.npy' ##to define the wet and dry year

        dic_precip_zscore = T.load_npy(f_preciptation_zscore)

        f_lai = rf'D:\Project3\Result\relative_change\OBS_LAI_extend\\LAI4g.npy'


        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []
        wet_threshold_list = [0.5, 1, 2, 3]
        wet_threshold_list = [0.5,3]

        for wet_thre_i in range(len(wet_threshold_list)):
            if wet_thre_i + 1 >= len(wet_threshold_list):
                continue
            wet_threshold_upper = wet_threshold_list[wet_thre_i + 1]
            wet_threshold_lower = wet_threshold_list[wet_thre_i]

            outf = join(outdir, f'monte_carlo_relative_change_{wet_threshold_list[wet_thre_i]}_{wet_threshold_list[wet_thre_i+1]}.npy')


            for pix in tqdm(dic_lai):
                if not pix in dic_precip_zscore:
                    continue

                lai = dic_lai[pix]

                precipitation_zscore=dic_precip_zscore[pix]



                picked_wet_index = (precipitation_zscore >= wet_threshold_lower) & (precipitation_zscore < wet_threshold_upper)


                wet_year = year_list[picked_wet_index]
                params = (pix,  lai, year_list, wet_year)
                params_list.append(params)
            result = MULTIPROCESS(self.kernel_cal_monte_carlo_wet, params_list).run(process=14)
            # print(result)
            pickle.dump(result, open(outf, 'wb'))
            # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
            # plt.plot(range(len(lai_copy)), lai_copy,label='random')
                # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
                # plt.plot(range(len(lai)), lai, label='origin')
                # plt.scatter(range(len(lai)), lai, label='origin')
                # plt.legend()
                # plt.show()
            # plt.hist(slope_list,bins=30)
            # plt.show()
            ## calculate the mean and std of the slope


            # result_dict_slope[pix] = mean
            # result_dict_slope_std[pix] = std

    def cal_monte_carlo_wet_sperate_period(self):
        
        
        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends


        f_preciptation_zscore = result_root+rf'zscore\\GPCC.npy' ##to define the wet and dry year

        dic_precip_zscore = T.load_npy(f_preciptation_zscore)

        f_lai = rf'D:\Project3\Result\relative_change\OBS_LAI_extend\\LAI4g.npy'


        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2001)
        year_list = range(2001, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []
        # wet_threshold_list = [0.5, 1, 2, 3]
        wet_threshold_list = [0.5,3]
        outdir = join(result_root, 'monte_carlo', 'wet', '2001_2020')
        T.mk_dir(outdir)

        for wet_thre_i in range(len(wet_threshold_list)):
            if wet_thre_i + 1 >= len(wet_threshold_list):
                continue
            wet_threshold_upper = wet_threshold_list[wet_thre_i + 1]
            wet_threshold_lower = wet_threshold_list[wet_thre_i]

            outf = join(outdir, f'monte_carlo_relative_change_{wet_threshold_list[wet_thre_i]}_{wet_threshold_list[wet_thre_i+1]}.npy')


            for pix in tqdm(dic_lai):

                if not pix in dic_precip_zscore:
                    continue
                lai = dic_lai[pix]

                # precipitation_zscore = dic_precip_zscore[pix][0:19] ## 1982-2000

                precipitation_zscore=dic_precip_zscore[pix][19:] ## 2001-2020

                # plt.plot(range(len(lai)), lai)

                # a1, b1, r1, p1 = T.nan_line_fit(range(len(lai)), lai)
                ##calculate slope
                # k1=stats.linregress(year_list, lai)
                #
                # k2=stats.linregress(range(len(lai)), lai)
                # print(k1.slope)
                # exit()
                #

                # print(pix, k, )
                # plt.show()
                ## define the different percentile of the precipitation and then find the extreme wet year and dry year


                ## based on threshold to find the extreme wet year and dry year
                # wet_index = precip > wet_threshold  ## return True, False
                # dry_index = precip < wet_threshold


                picked_wet_index = (precipitation_zscore >= wet_threshold_lower) & (precipitation_zscore < wet_threshold_upper)


                wet_year = year_list[picked_wet_index]
                params = (pix,  lai, year_list, wet_year)
                params_list.append(params)
            result = MULTIPROCESS(self.kernel_cal_monte_carlo_wet, params_list).run(process=14)
            # print(result)
            pickle.dump(result, open(outf, 'wb'))
            # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
            # plt.plot(range(len(lai_copy)), lai_copy,label='random')
                # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
                # plt.plot(range(len(lai)), lai, label='origin')
                # plt.scatter(range(len(lai)), lai, label='origin')
                # plt.legend()
                # plt.show()
            # plt.hist(slope_list,bins=30)
            # plt.show()
            ## calculate the mean and std of the slope


            # result_dict_slope[pix] = mean
            # result_dict_slope_std[pix] = std

    def cal_monte_carlo_dry(self):

        # if os.path.isfile(outf):
        #     return
        ## load LAI npy load preciptation npy
        ## for each pixel, has dry year and wet year index and using this index to extract LAI and then using mean or random value to subtitute the value
        ## then calculate trends of LAI
        ## using monte carlo to repeat 1000 times and then calculate the mean and std of the trends

        f_preciptation_zscore = result_root + rf'zscore\\GPCC.npy'  ##to define the wet and dry year

        dic_precip_zscore = T.load_npy(f_preciptation_zscore)

        f_lai = rf'D:\Project3\Result\relative_change\OBS_LAI_extend\\LAI4g.npy'



        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []
        wet_threshold_list = [-3, -2, -1, -0.5,0.5]
        wet_threshold_list = [-3,-0.5]
        outdir=join(result_root, 'monte_carlo','dry','1982_2020')
        T.mk_dir(outdir, force=True)

        for wet_thre_i in range(len(wet_threshold_list)):
            if wet_thre_i + 1 >= len(wet_threshold_list):
                continue
            wet_threshold_upper = wet_threshold_list[wet_thre_i + 1]
            wet_threshold_lower = wet_threshold_list[wet_thre_i]

            outf = join(outdir,
                        f'monte_carlo_relative_change_{wet_threshold_list[wet_thre_i]}_{wet_threshold_list[wet_thre_i + 1]}.npy')

            for pix in tqdm(dic_lai):
                if not pix in dic_precip_zscore:
                    continue

                if not pix in dic_lai:
                    continue
                lai = dic_lai[pix]

                precipitation_zscore = dic_precip_zscore[pix]

                picked_wet_index = (precipitation_zscore >= wet_threshold_lower) & (
                            precipitation_zscore < wet_threshold_upper)

                wet_year = year_list[picked_wet_index]
                params = (pix, lai, year_list, wet_year)
                params_list.append(params)
            result = MULTIPROCESS(self.kernel_cal_monte_carlo_wet, params_list).run(process=14)
            # print(result)
            pickle.dump(result, open(outf, 'wb'))
            # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
            # plt.plot(range(len(lai_copy)), lai_copy,label='random')
            # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
            # plt.plot(range(len(lai)), lai, label='origin')
            # plt.scatter(range(len(lai)), lai, label='origin')
            # plt.legend()
            # plt.show()
            # plt.hist(slope_list,bins=30)
            # plt.show()
            ## calculate the mean and std of the slope

            # result_dict_slope[pix] = mean
            # result_dict_slope_std[pix] = std
    def cal_monte_carlo_dry_sperate_period(self):


        f_preciptation_zscore = result_root+rf'zscore\\GPCC.npy' ##to define the wet and dry year

        dic_precip_zscore = T.load_npy(f_preciptation_zscore)

        f_lai = rf'D:\Project3\Result\relative_change\OBS_LAI_extend\\LAI4g.npy'


        dic_lai = T.load_npy(f_lai)
        year_list = range(1982, 2001)
        # year_list = range(2001, 2021)
        year_list = np.array(year_list)

        random.seed(1)
        params_list = []
        wet_threshold_list = [-3, -2, -1, -0.5,0.5]
        wet_threshold_list = [-3,-0.5]

        outdir = join(result_root, 'monte_carlo', 'dry', '1982_2000')
        T.mk_dir(outdir, force=True)

        for wet_thre_i in range(len(wet_threshold_list)):
            if wet_thre_i + 1 >= len(wet_threshold_list):
                continue
            wet_threshold_upper = wet_threshold_list[wet_thre_i + 1]
            wet_threshold_lower = wet_threshold_list[wet_thre_i]

            outf = join(outdir, f'monte_carlo_relative_change_{wet_threshold_list[wet_thre_i]}_{wet_threshold_list[wet_thre_i+1]}.npy')


            for pix in tqdm(dic_lai):

                if not pix in dic_precip_zscore:
                    continue
                lai = dic_lai[pix]

                # precipitation_zscore = dic_precip_zscore[pix][0:19] ## 1982-2000

                precipitation_zscore=dic_precip_zscore[pix][0:19] ## 1982-2000

                # plt.plot(range(len(lai)), lai)

                # a1, b1, r1, p1 = T.nan_line_fit(range(len(lai)), lai)
                ##calculate slope
                # k1=stats.linregress(year_list, lai)
                #
                # k2=stats.linregress(range(len(lai)), lai)
                # print(k1.slope)
                # exit()
                #

                # print(pix, k, )
                # plt.show()
                ## define the different percentile of the precipitation and then find the extreme wet year and dry year


                ## based on threshold to find the extreme wet year and dry year
                # wet_index = precip > wet_threshold  ## return True, False
                # dry_index = precip < wet_threshold


                picked_wet_index = (precipitation_zscore >= wet_threshold_lower) & (precipitation_zscore < wet_threshold_upper)


                wet_year = year_list[picked_wet_index]
                params = (pix,  lai, year_list, wet_year)
                params_list.append(params)
            result = MULTIPROCESS(self.kernel_cal_monte_carlo_wet, params_list).run(process=14)
            # print(result)
            pickle.dump(result, open(outf, 'wb'))
            # T.save_npy(result_dict_slope_std, outf.replace('.npy', '_std.npy'))
            # plt.plot(range(len(lai_copy)), lai_copy,label='random')
                # plt.scatter(range(len(lai_copy)), lai_copy,label='random')
                # plt.plot(range(len(lai)), lai, label='origin')
                # plt.scatter(range(len(lai)), lai, label='origin')
                # plt.legend()
                # plt.show()
            # plt.hist(slope_list,bins=30)
            # plt.show()
            ## calculate the mean and std of the slope


            # result_dict_slope[pix] = mean
            # result_dict_slope_std[pix] = std


    def kernel_cal_monte_carlo_wet(self,params):
            n = 100
            pix, lai, year_list, wet_year = params

            random_value_list = []
            for t in range(n):
                randm_value = random.gauss(np.nanmean(lai), np.nanstd(lai))
                random_value_list.append(randm_value)

            slope_list = []

            for t in range(n):
                lai_copy = copy.copy(lai)

                # for dr in dry_year:
                #     ### here using the random value to substitute the LAI value in dry year
                #     lai_copy[dr - 1982] = np.random.choice(random_value_list)

                for wet in wet_year:

                    lai_copy[wet - 1982] = np.random.choice(random_value_list)


                # result_i=stats.linregress(range(len(lai_copy)), lai_copy)
                # slope_list.append(result_i.slope)
                a,b,r,p = T.nan_line_fit(range(len(lai_copy)), lai_copy)
                slope_list.append(a)



            mean = np.nanmean(slope_list)
            std = np.nanstd(slope_list)
            return (pix, mean, std)

            pass

    def kernel_cal_monte_carlo_dry(self,params):
        n = 100
        pix,  lai, year_list, dry_year = params
        random_value_list = []
        for t in range(n):
            randm_value = random.gauss(np.nanmean(lai), np.nanstd(lai))
            random_value_list.append(randm_value)

        slope_list = []

        for t in range(n):
            lai_copy = copy.copy(lai)

            for dr in dry_year:
                ### here using the random value to substitute the LAI value in dry year
                lai_copy[dr - 1982] = np.random.choice(random_value_list)



            # result_i=stats.linregress(range(len(lai_copy)), lai_copy)
            # slope_list.append(result_i.slope)
            a,b,r,p = T.nan_line_fit(range(len(lai_copy)), lai_copy)
            slope_list.append(a)



        mean = np.nanmean(slope_list)
        std = np.nanstd(slope_list)
        return (pix, mean, std)

        pass

    def check_result(self):
        period_list=['1982_2000','2001_2020','1982_2020']
        for period in period_list:
            fdir=rf'D:\Project3\Result\monte_carlo\\wet\{period}\\'
            for f in os.listdir(fdir):
                if not 'npy' in f:
                    continue
                if 'slope' in f:
                    continue
                if 'std' in f:
                    continue



                fpath=join(fdir,f)
                result_dict = np.load(fpath, allow_pickle=True)
                spatial_dict1 = {}
                spatial_dict2 = {}
                # for i in result_dict:
                #     print(i)
                #     exit()
                for pix,slope,std in result_dict:
                    # if np.isnan(slope):
                    #     continue
                    # print(pix,slope,std)

                    spatial_dict1[pix] = slope
                    spatial_dict2[pix] = std
                arr1 = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict1)
                arr2 = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict2)
                # plt.imshow(arr1)
                # plt.colorbar()
                # plt.show()
                ##save
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr1, fpath.replace('.npy', '_slope.tif'))
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr2, fpath.replace('.npy', '_std.tif'))
                np.save(fpath.replace('.npy', '_slope.npy'), spatial_dict1)
                np.save(fpath.replace('.npy', '_std.npy'), spatial_dict2)

    def difference(self): ### calculate the difference of the slope the real and scenario
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        period_list=['1982_2000','2001_2020','1982_2020']

        for period in period_list:
            f_real_trend = rf'D:\Project3\Result\trend_analysis\split_relative_change\OBS_LAI_extend\\LAI4g_{period}_trend.tif'
            array_real, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_real_trend)
            fdir=rf'D:\Project3\Result\monte_carlo\\dry\{period}\\'
            for f in os.listdir(fdir):
                if not 'slope' in f:
                    continue
                if not 'tif' in f:
                    continue
                fpath=join(fdir,f)
                array_scenario, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                array_diff = array_real - array_scenario
                array_diff[array_diff==0]=np.nan
                fname=f.replace('slope','diff')

                outf=join(fdir,fname)
                dic_arr_difference = DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(array_diff)
                spatial_dict = {}
                for pix in tqdm(dic_arr_difference):
                    r,c = pix
                    if r<120:
                        continue

                    if not pix in dic_modis_mask:
                        continue
                    if dic_modis_mask[pix] == 12:
                        continue
                    landcover_value = crop_mask[pix]
                    if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                        continue
                    spatial_dict[pix] = dic_arr_difference[pix]
                array_diff_new = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dict)
                array_diff_new = array_diff_new * array_mask
                array_diff_new[array_diff_new > 99] = np.nan
                array_diff_new[array_diff_new < -99] = np.nan



                DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_diff_new, outf)




    def calcualte_contributing_of_wet_dry(self):  ##spatial
        ## calculate the contributing of wet and dry year to the LAI trend
        ## for each pixel,
        dic_LAI = T.load_npy(rf'D:\Project3\Result\extract_GS\OBS_LAI_extend\\LAI4g.npy')
        fdir_wet = rf'D:\Project3\Result\monte_carlo\\wet\\'
        for f in os.listdir(fdir_wet):
            fpath = join(fdir_wet, f)
            result_dict = np.load(fpath, allow_pickle=True)
            scenario_dict = {}
            for pix, slope, std in result_dict:
                if not pix in dic_LAI:
                    continue
                lai_raw = dic_LAI[pix]

                scenario_dict[pix]=slope
                pass

    def rename_result(self):
        period_list=['1982_2000','2001_2020','1982_2020']

        for period in period_list:

            fdir=rf'D:\Project3\Result\monte_carlo\\wet\\{period}\\'
            name_dict = {'monte_carlo_relative_change_all_diff.tif':'wet_year_trend.tif',
                         'monte_carlo_relative_change_extreme_diff.tif':'wet_year_trend_extreme.tif',
                            'monte_carlo_relative_change_moderate_diff.tif':'wet_year_trend_moderate.tif',
                            'monte_carlo_relative_change_mild_diff.tif':'wet_year_trend_mild.tif',
                            'monte_carlo_relative_change_normal_diff.tif':'wet_year_trend_normal.tif',



                         }


            for f in os.listdir(fdir):
                if not 'tif' in f:
                    continue
                if not f in name_dict:
                    continue
                fpath=join(fdir,f)
                new_name=name_dict[f]
                new_path=join(fdir,new_name)
                os.rename(fpath,new_path)
                print(fpath,new_path)
                # exit()



    def bar_plot(self):
        dff=rf'D:\Project3\Result\monte_carlo\Dataframe\monte_carlo.df'
        df = T.load_df(dff)
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['MODIS_LUCC'] != 12]
        # df=df[df['continent']=='Australia']
        # df=df[df['LAI4g_p_value']<0.05]
        # df=df[df['LAI4g_trend']>0]
        df=df.dropna()
        dry_color_list=['peachpuff', 'orange', 'darkorange', 'chocolate', 'saddlebrown']

        wet_color_list=['lightblue', 'cyan', 'deepskyblue', 'dodgerblue', 'navy']
        color_list=wet_color_list+dry_color_list[::-1]
        # df=df[df['partial_corr_GPCC']<0.4]
        df = df.dropna()
        # print(df.columns)
        # exit()
        #### plt. bar plot for the wet and dry year all the period



        result_period_dic={}
        mode_list=['extreme','moderate','mild',]
        patterns=['wet','dry']

        period_list=['1982_2000','2001_2020','1982_2020']

        for period in period_list:

            lai_raw = df[f'LAI4g_{period}_trend'].to_list()
            lai_raw = np.array(lai_raw)
            lai_raw_mean = np.nanmean(lai_raw)
            lai_raw_std = np.nanstd(lai_raw)

            result_dic_pattern = {}
            for pattern in patterns:
                result_dic_mode = {}
                for mode in mode_list:
                    vals=df[f'{pattern}_year_trend_{mode}_{period}'].to_list()
                    vals_array = np.array(vals)
                    vals_mean = np.nanmean(vals_array)
                    vals_std = np.nanstd(vals_array)
                    result_dic_mode[mode] = (vals_mean,vals_std)
                ##add the mean of the raw data
                vals_mean = np.nanmean(lai_raw)
                vals_std = np.nanstd(lai_raw)
                result_dic_mode['raw'] = (vals_mean,vals_std)


                result_dic_pattern[pattern] = result_dic_mode
            result_period_dic[period] = result_dic_pattern

      ##plot
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        flag = 0
        for period in period_list:
            ax = axs.ravel()[flag]
            result_dic_pattern = result_period_dic[period]
            vals_mean_list = []
            vals_std_list = []
            for pattern in patterns:
                result_dic_mode = result_dic_pattern[pattern]

                for mode in mode_list:
                    vals_mean,vals_std = result_dic_mode[mode]
                    vals_mean_list.append(vals_mean)
                    vals_std_list.append(vals_std)
            mode_list_new = [f'{mode}_{pattern}' for pattern in patterns for mode in mode_list]

            ax.bar(mode_list_new,vals_mean_list,yerr=vals_std_list,color=color_list)
            ax.set_title(period)
            ax.set_ylim(-0.5,0.5)
            ax.set_yticks([-0.5,0,0.5])
            ax.set_xticklabels(mode_list_new,rotation=45)
            flag += 1
        plt.tight_layout()
        plt.show()


    def testrobinson(self):

        period_list=['1982_2000','2001_2020','1982_2020']
        # fpath_p_value=result_root+rf'trend_analysis\split_anomaly\\\\GPCC_p_value.tif'
        temp_root=r'trend_analysis\anomaly\\temp_root\\'
        T.mk_dir(temp_root,force=True)


        for period in period_list:
            fdir = result_root + rf'monte_carlo\dry\{period}\\'
            for f in os.listdir(fdir):
                if not f.endswith('.tif'):
                    continue
                if not 'trend' in f:
                    continue
                if 'extreme' in f:
                    continue
                if 'normal' in f:
                    continue
                if 'mild' in f:
                    continue
                if 'moderate' in f:
                    continue


                fname=f.split('.')[0]
                print(fname)


                m,ret=Plot().plot_Robinson(fdir+f, vmin=-0.5,vmax=0.5,is_discrete=False,colormap_n=11,)
                # Plot().plot_Robinson_significance_scatter(m,fpath_p_value,temp_root,0.05)


                fname=f.split('.')[0]
                plt.title(f'{period}_{fname} %')


                plt.show()



class multi_regression_new:
    def __init__(self):
        pass

    def run(self):
        #### 1 create spatial pixel for wet year and dry year for each pixel
        # self.create_spatial_pixel()
        ## 2 build df
        window = 15
        mode_list = ['wet', 'dry']
        year_list = range(1982, 2021)
        year_list = np.array(year_list)
        x_var_list = ['precip', 'vpd', ]
        for mode in mode_list:
            for w in range(window, len(year_list)):


                df=self.build_df(mode,w,window)
                ## run multi regression
                self.cal_multi_regression_beta(df,w, mode, x_var_list,window)
        # self.plt_multi_regression_result()
        # self.convert_files_to_time_series()
        # self.plot_bar_moving_window()

    def create_spatial_pixel(self):
        f_preciptation_zscore = result_root+rf'zscore\\GPCC.npy' ##to define the wet and dry year
        dic_precip_zscore = T.load_npy(f_preciptation_zscore)
        f_lai = result_root+rf'\\relative_change\OBS_LAI_extend\LAI4g.npy'
        f_precip_relative_change = result_root+rf'\\relative_change\OBS_LAI_extend\CRU.npy'
        f_vpd= result_root+rf'relative_change\\OBS_LAI_extend\VPD.npy'
        dic_lai = T.load_npy(f_lai)
        dic_precip = T.load_npy(f_precip_relative_change)
        dic_vpd = T.load_npy(f_vpd)
        year_list = range(1982, 2021)
        year_list = np.array(year_list)
        ###moving window to calculate the wet and dry year
        window=15

        for w in range(window,len(year_list)):
            ## format
            w_fname=f'{w:02d}'
            w_fname=f'{w-window:02d}'


            picked_year = year_list[w-window:w]
            picked_year = np.array(picked_year)
            result_wet_lai_dic={}
            result_dry_lai_dic={}
            result_wet_precip_dic={}
            result_dry_precip_dic={}
            result_wet_vpd_dic={}
            result_dry_vpd_dic={}
            for pix in dic_lai:
                if not pix in dic_precip_zscore:
                    continue
                if not pix in dic_precip:
                    continue
                if not pix in dic_vpd:
                    continue

                lai = dic_lai[pix][w-window:w]
                precip = dic_precip[pix][w-window:w]
                vpd = dic_vpd[pix][w-window:w]
                selected_year_list = year_list[w-window:w]
                precipitation_zscore=dic_precip_zscore[pix][w-window:w]
                picked_wet_index = (precipitation_zscore >= 0.5) & (precipitation_zscore < 3)
                picked_dry_index = (precipitation_zscore <= -0.5) & (precipitation_zscore > -3)



                wet_year = selected_year_list[picked_wet_index]
                dry_year = selected_year_list[picked_dry_index]
                wet_lai = lai[picked_wet_index]
                dry_lai = lai[picked_dry_index]
                wet_precip = precip[picked_wet_index]
                dry_precip = precip[picked_dry_index]
                wet_vpd = vpd[picked_wet_index]
                dry_vpd = vpd[picked_dry_index]
                result_wet_lai_dic[pix] = wet_lai
                result_dry_lai_dic[pix] = dry_lai
                result_wet_precip_dic[pix] = wet_precip
                result_dry_precip_dic[pix] = dry_precip
                result_wet_vpd_dic[pix] = wet_vpd
                result_dry_vpd_dic[pix] = dry_vpd
            outdir= result_root + rf'multi_regression_moving_window\events_trend\\events_extraction\\'
            T.mk_dir(outdir,force=True)
            T.save_npy(result_wet_lai_dic, outdir+f'wet_lai_{w_fname}.npy')
            T.save_npy(result_dry_lai_dic, outdir+f'dry_lai_{w_fname}.npy')
            T.save_npy(result_wet_precip_dic, outdir+f'wet_precip_{w_fname}.npy')
            T.save_npy(result_dry_precip_dic, outdir+f'dry_precip_{w_fname}.npy')
            T.save_npy(result_wet_vpd_dic, outdir+f'wet_vpd_{w_fname}.npy')
            T.save_npy(result_dry_vpd_dic, outdir+f'dry_vpd_{w_fname}.npy')




        pass

    def build_df(self,mode,w,window):

        fdir_X = result_root + rf'\multi_regression_moving_window\events_trend\\events_extraction\\'

        fdir_Y = result_root + rf'\multi_regression_moving_window\events_trend\\events_extraction\\'

        f_y=fdir_Y+f'{mode}_lai_{w-window:02d}.npy'
        print(f_y)

        dic_y = T.load_npy(f_y)



        df = pd.DataFrame()

        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue


            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x
        x_var_list = ['precip', 'vpd', ]

        for xvar in x_var_list:

            # print(var_name)
            x_val_list = []
            f_x = fdir_X + f'{mode}_{xvar}_{w-window:02d}.npy'
            print(f_x)

            dic_x = T.load_npy(f_x)
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

    def cal_multi_regression_beta(self, df, w, mode, x_var_list,window):
        outdir= result_root + rf'multi_regression_moving_window\events_trend\\events_multiregression\\'
        T.mk_dir(outdir,force=True)
        print(w)

        outf = outdir + f'multi_regression_{mode}_{w-window:02d}.npy'
        if os.path.isfile(outf):
            return

        multi_derivative = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix

            y_vals = row['y']
            # print(y_vals)
            # y_vals = T.remove_np_nan(y_vals)
            # y_vals = T.interp_nan(y_vals)
            if len(y_vals) == 0:
                continue
            if T.is_all_nan(y_vals):
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

    def plt_multi_regression_result(self):
        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)

        multi_regression_result_dir = result_root + rf'multi_regression_moving_window\events_trend\\events_multiregression\\'
        for f in os.listdir(multi_regression_result_dir):
            if not 'npy' in f:
                continue
            if 'tif' in f:
                continue



            dic = T.load_npy(multi_regression_result_dir + f)
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
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, f'{multi_regression_result_dir}\\{f}_{var_i}.tif')
                std = np.nanstd(arr)
                mean = np.nanmean(arr)
                vmin = mean - std
                vmax = mean + std
                plt.figure()
                # arr[arr > 0.1] = 1
                # plt.imshow(arr,vmin=-5,vmax=5)

                # plt.title(var_i)
                # plt.colorbar()

            # plt.show()

    def convert_files_to_time_series(self):


        fdir = rf'D:\Project3\Result\multi_regression_moving_window\events_trend\events_multiregression\\'
        variable_list = ['precip', ]

        for variable in variable_list:
            array_list = []

            for f in os.listdir(fdir):



                if not variable in f:
                    continue
                if not f.endswith('.tif'):
                    continue
                if not 'dry' in f:
                    continue

                print(f)

                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
                array = np.array(array)


                array_list.append(array)
            array_list=np.array(array_list)

            ## array_list to dic
            dic=DIC_and_TIF(pixelsize=0.25).void_spatial_dic()
            result_dic = {}
            for pix in dic:
                r,c=pix


                dic[pix]=array_list[:,r,c] ## extract time series



                time_series=dic[pix]

                time_series = np.array(time_series)
                time_series = time_series*100  ###  %/100mm
                time_series[time_series > 99] = np.nan
                time_series[time_series < -99] = np.nan



                result_dic[pix]=time_series


                # if np.nanmean(dic[pix])<=5:
                #     continue
                # # print(len(dic[pix]))
                # # exit()
            outdir=rf'D:\Project3\Result\multi_regression\events_multiregression\\time_series\\'
            print(outdir)
            # exit()
            T.mk_dir(outdir,force=True)
            outf=outdir+f'{variable}_dry.npy'
            T.save_npy(result_dic,outf)
    def plot_bar_moving_window(self):
        dff=result_root+rf'multi_regression_moving_window\events\Dataframe\multi_regression_moving_window.df'
        df = T.load_df(dff)
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['MODIS_LUCC'] != 12]
        window_list=np.arange(0,24)

        window_list=np.array(window_list)
        result_region_dic={}
        continent_list = ['Global','Australia', 'Africa', 'Asia', 'North America', 'South America']
        AI_class_list = ['Arid', 'Semi-Arid', 'Sub-Humid', ]
        for AI in AI_class_list:
            df_region= df[df['AI_classfication'] == AI]
            result_dic={}
            for window in window_list:
                vals=df_region[f'multi_regression_wet_{window:02d}'].to_list()
                vals_array = np.array(vals)*100
                vals_mean = np.nanmean(vals_array)
                vals_std = np.nanstd(vals_array)
                result_dic[window] = (vals_mean,vals_std)
            result_region_dic[AI] = result_dic
            ##plot
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        flag = 0

        for AI in AI_class_list:
            ax = axs.ravel()[flag]
            result_dic = result_region_dic[AI]
            vals_mean_list = []
            vals_std_list = []
            for window in window_list:
                vals_mean,vals_std = result_dic[window]
                vals_mean_list.append(vals_mean)
                vals_std_list.append(vals_std)

            ax.bar(window_list,vals_mean_list)
            ax.set_title(AI)
            ax.set_ylim(0,20)
            ax.set_yticks([0,5,10,15,20])
            ax.set_xticks(window_list)
            ax.set_xticklabels(window_list,rotation=45)
            ax.set_xlabel('Window')
            ax.set_ylabel('LAI response to wet year precipitation  (%/100mm)')
            flag += 1

        plt.tight_layout()
        plt.show()






        pass
class moving_window_trend:
    def __init__(self):
        pass
    def run(self):
        # self.pick_wet_year_anomaly_moving_window()
        # self.pick_dry_year_anomaly_moving_window()
        # self.pick_all_year_anomaly_moving_window()
        # self.moving_window_trend_calculation()
        self.plot_moving_window()
        # self.plot_scatter()
    def pick_wet_year_anomaly_moving_window(self):  ##based on pixel and then calculate the frequency of the wet and dry year for each window
        mode='wet'

        outdir = join(result_root, 'moving_window_extraction', f'{mode}_year_moving_window_extraction')

        T.mk_dir(outdir,force=True)


        f_precipitation_zscore = (result_root + rf'zscore\CRU.npy')
        dic_precip_zscore = T.load_npy(f_precipitation_zscore)


        f_precip = rf'D:\Project3\Result\\Detrend\\detrend_relative_change\extend\\\CRU.npy'
        f_lai = rf'D:\Project3\Result\\Detrend\\detrend_relative_change\extend\\LAI4g.npy'
        f_vpd = rf'D:\Project3\Result\\Detrend\\detrend_relative_change\extend\\VPD.npy'

        dic_precip = T.load_npy(f_precip)
        dic_lai = T.load_npy(f_lai)
        dic_vpd = T.load_npy(f_vpd)
        year_list = range(1982, 2021)

        year_list=np.array(year_list)
        year_list = np.array(year_list)

        ### find the extreme wet year and dry year

        result_dict_precip = {}
        result_dict_vpd = {}
        result_dict_lai = {}


        window=15


        for pix in tqdm(dic_precip):


            if not pix in dic_precip_zscore:
                continue
            if not pix in dic_vpd:
                continue
            if not pix in dic_lai:
                continue

            window_lai_list=[]
            window_vpd_list=[]
            window_precip_list=[]


            for w in range(window, len(year_list)):
                ## format

                picked_year = year_list[w - window:w]
                precip_zscore = dic_precip_zscore[pix][w - window:w]

                precip = dic_precip[pix][w - window:w]
                vpd = dic_vpd[pix][w - window:w]
                lai = dic_lai[pix][w - window:w]

            ## based on threshold to find the wet year

                picked_mild_dry_index = (precip_zscore > 0.5) & (precip_zscore <= 3)


                # mild_dry_year = year_list[picked_mild_dry_index]  ## extract the year index based on the wet_index
                precip_mild_dry = precip[picked_mild_dry_index]
                vpd_mild_dry = vpd[picked_mild_dry_index]
                lai_mild_dry = lai[picked_mild_dry_index]
                average_precip_mild_dry = np.nanmean(precip_mild_dry)
                average_vpd_mild_dry = np.nanmean(vpd_mild_dry)
                average_lai_mild_dry = np.nanmean(lai_mild_dry)
                window_lai_list.append(average_lai_mild_dry)
                window_vpd_list.append(average_vpd_mild_dry)
                window_precip_list.append(average_precip_mild_dry)
                ### calculate average

            result_dict_precip[pix] = window_precip_list
            result_dict_vpd[pix] = window_vpd_list
            result_dict_lai[pix] = window_lai_list
            ##save
        outf_precip = join(outdir, f'{mode}_year_moving_window_extraction_precip.npy')
        outf_vpd = join(outdir, f'{mode}_year_moving_window_extraction_vpd.npy')
        outf_lai = join(outdir, f'{mode}_year_moving_window_extraction_lai.npy')
        np.save(outf_precip, result_dict_precip)
        np.save(outf_vpd, result_dict_vpd)
        np.save(outf_lai, result_dict_lai)
        pass

    def pick_dry_year_anomaly_moving_window(self):  ##based on pixel and then calculate the frequency of the wet and dry year for each window


        outdir = join(result_root, 'moving_window_extraction', 'dry_year_moving_window_extraction')

        T.mk_dir(outdir,force=True)


        f_precipitation_zscore = (result_root + rf'zscore\CRU.npy')
        dic_precip_zscore = T.load_npy(f_precipitation_zscore)


        f_precip = rf'D:\Project3\Result\\Detrend\\detrend_relative_change\extend\\\CRU.npy'
        f_lai = rf'D:\Project3\Result\\Detrend\\detrend_relative_change\extend\\LAI4g.npy'
        f_vpd = rf'D:\Project3\Result\\Detrend\\detrend_relative_change\extend\\VPD.npy'

        dic_precip = T.load_npy(f_precip)
        dic_lai = T.load_npy(f_lai)
        dic_vpd = T.load_npy(f_vpd)
        year_list = range(1982, 2021)

        year_list=np.array(year_list)
        year_list = np.array(year_list)

        ### find the extreme wet year and dry year

        result_dict_precip = {}
        result_dict_vpd = {}
        result_dict_lai = {}


        window=15


        for pix in tqdm(dic_precip):


            if not pix in dic_precip_zscore:
                continue
            if not pix in dic_vpd:
                continue
            if not pix in dic_lai:
                continue

            window_lai_list=[]
            window_vpd_list=[]
            window_precip_list=[]


            for w in range(window, len(year_list)):
                ## format

                picked_year = year_list[w - window:w]
                precip_zscore = dic_precip_zscore[pix][w - window:w]

                precip = dic_precip[pix][w - window:w]
                vpd = dic_vpd[pix][w - window:w]
                lai = dic_lai[pix][w - window:w]

            ## based on threshold to find the wet year

                picked_mild_dry_index = (precip_zscore <= -0.5) & (precip_zscore > -3)


                # mild_dry_year = year_list[picked_mild_dry_index]  ## extract the year index based on the wet_index
                precip_mild_dry = precip[picked_mild_dry_index]
                vpd_mild_dry = vpd[picked_mild_dry_index]
                lai_mild_dry = lai[picked_mild_dry_index]
                average_precip_mild_dry = np.nanmean(precip_mild_dry)
                average_vpd_mild_dry = np.nanmean(vpd_mild_dry)
                average_lai_mild_dry = np.nanmean(lai_mild_dry)
                window_lai_list.append(average_lai_mild_dry)
                window_vpd_list.append(average_vpd_mild_dry)
                window_precip_list.append(average_precip_mild_dry)
                ### calculate average

            result_dict_precip[pix] = window_precip_list
            result_dict_vpd[pix] = window_vpd_list
            result_dict_lai[pix] = window_lai_list
            ##save
        outf_precip = join(outdir, 'dry_year_moving_window_extraction_precip.npy')
        outf_vpd = join(outdir, 'dry_year_moving_window_extraction_vpd.npy')
        outf_lai = join(outdir, 'dry_year_moving_window_extraction_lai.npy')
        np.save(outf_precip, result_dict_precip)
        np.save(outf_vpd, result_dict_vpd)
        np.save(outf_lai, result_dict_lai)
        pass

    def pick_all_year_anomaly_moving_window(self):  #window
        mode='all'

        outdir = join(result_root, 'moving_window_extraction', f'{mode}_year_moving_window_extraction')

        T.mk_dir(outdir,force=True)


        f_precip = rf'D:\Project3\Result\\Detrend\\detrend_relative_change\extend\\\CRU.npy'
        f_lai = rf'D:\Project3\Result\\Detrend\\detrend_relative_change\extend\\LAI4g.npy'
        f_vpd = rf'D:\Project3\Result\\Detrend\\detrend_relative_change\extend\\VPD.npy'

        dic_precip = T.load_npy(f_precip)
        dic_lai = T.load_npy(f_lai)
        dic_vpd = T.load_npy(f_vpd)
        year_list = range(1982, 2021)

        year_list=np.array(year_list)
        year_list = np.array(year_list)

        ### find the extreme wet year and dry year

        result_dict_precip = {}
        result_dict_vpd = {}
        result_dict_lai = {}


        window=15


        for pix in tqdm(dic_precip):


            if not pix in dic_vpd:
                continue
            if not pix in dic_lai:
                continue

            window_lai_list=[]
            window_vpd_list=[]
            window_precip_list=[]


            for w in range(window, len(year_list)):
                ## format


                precip = dic_precip[pix][w - window:w]
                vpd = dic_vpd[pix][w - window:w]
                lai = dic_lai[pix][w - window:w]


                average_precip_mild_dry = np.nanmean(precip)
                average_vpd_mild_dry = np.nanmean(vpd)
                average_lai_mild_dry = np.nanmean(lai)
                window_lai_list.append(average_lai_mild_dry)
                window_vpd_list.append(average_vpd_mild_dry)
                window_precip_list.append(average_precip_mild_dry)
                ### calculate average

            result_dict_precip[pix] = window_precip_list
            result_dict_vpd[pix] = window_vpd_list
            result_dict_lai[pix] = window_lai_list
            ##save
        outf_precip = join(outdir, f'{mode}_year_moving_window_extraction_precip.npy')
        outf_vpd = join(outdir, f'{mode}_year_moving_window_extraction_vpd.npy')
        outf_lai = join(outdir, f'{mode}_year_moving_window_extraction_lai.npy')
        np.save(outf_precip, result_dict_precip)
        np.save(outf_vpd, result_dict_vpd)
        np.save(outf_lai, result_dict_lai)
        pass


    def moving_window_trend_calculation(self):

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        mode='dry'
        fdir = join(result_root, 'moving_window_extraction', f'{mode}_year_moving_window_extraction')
        T.mk_dir(fdir,force=True)



        year_list = range(1982, 2021)
        year_list = np.array(year_list)
        variable_list = ['precip', 'vpd', 'lai', ]


        outdir = join(result_root, 'moving_window_extraction', f'{mode}_year_moving_window_trend')
        T.mk_dir(outdir,force=True)

        for variable in variable_list:
            f = join(fdir, f'{mode}_year_moving_window_extraction_{variable}.npy')


            dic_variable = T.load_npy(f)
            spatial_trend_dic = {}
            spatial_p_value_dic = {}

            for pix in tqdm(dic_variable):

                r, c = pix
                if r < 120:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                vals = dic_variable[pix]
                print(vals)

                if len(vals) < 24:
                    continue
                # print(np.nanstd(vals))
                if np.round(np.nanstd(vals))==0:
                    continue



                a, b, r, p = T.nan_line_fit(range(len(vals)), vals)
                spatial_trend_dic[pix] = a
                spatial_p_value_dic[pix] = p
            outf_trend = join(outdir, f'{mode}_year_moving_window_extraction_{variable}_trend.npy')
            outf_p_value = join(outdir, f'{mode}_year_moving_window_extraction_{variable}_p_value.npy')
            np.save(outf_trend, spatial_trend_dic)
            np.save(outf_p_value, spatial_p_value_dic)
            array_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_trend_dic)
            array_p_value = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_p_value_dic)
            array_trend=array_trend*array_mask
            array_p_value=array_p_value*array_mask
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_trend, outf_trend.replace('.npy', '.tif'))
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_p_value, outf_p_value.replace('.npy', '.tif'))


        pass

    def plot_moving_window(self):

        ##plot time series
        dff=result_root+rf'moving_window_extraction\trend\\Dataframe\\moving_window.df'
        df = T.load_df(dff)
        color_list=['k','b','r']


        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['MODIS_LUCC'] != 12]
        # region_list= ['Global', 'Australia', 'Africa', 'Asia', 'North_America', 'South_America']
        region_list = ['Global', 'Arid', 'Semi-Arid', 'Sub-Humid']
        variable_list = ['precip', 'vpd', 'lai', ]
        mode_list=['all','wet','dry']
        fig = plt.figure()
        i = 1
        for region in region_list:
            ax = fig.add_subplot(2, 2, i)
            if region == 'Global':
                df_region = df
            else:
                df_region = df[df['AI_classfication'] == region]
                # df_region = df[df['continent'] == region]

            for variable in variable_list:
                if not 'precip' in variable:
                    continue
                vals_nonnan = []

                for mode in mode_list:
                    vals=df_region[f'{mode}_year_moving_window_extraction_{variable}'].to_list()


                    for val in vals:

                        if type(val) == float:  ## only screening
                            continue
                        if len(val) == 0:
                            continue


                        if not len(val) == 39:
                            ## add nan to the end of the list
                            for j in range(1):
                                val = np.append(val, np.nan)
                            # print(val)
                            # print(len(val))

                        vals_nonnan.append(list(val))

                        # exit()
                        # print(type(val))
                        # print(len(val))
                        # print(vals)

                    ###### calculate mean
                    vals_mean = np.array(vals_nonnan)  ## axis=0, mean of each row  竖着加
                    vals_mean = np.nanmean(vals_mean, axis=0)

                    val_std = np.nanstd(vals_mean, axis=0)


                    plt.plot(vals_mean, label=mode, color=color_list[mode_list.index(mode)],linewidth=2)
                    # plt.fill_between(range(len(vals_mean)),vals_mean-val_std,vals_mean+val_std,alpha=0.3,color=color_list[self.product_list.index(product)])

                i = i + 1

                ax.set_xticks(range(0,len(vals_mean),3))
                # ax.set_xticklabels()
                # plt.ylim(-0.2, 0.2)
                # plt.ylim(-1,1)

                plt.xlabel('window(year)')

                plt.ylabel(f'relative change(%/year)')
                plt.legend()

                plt.title(f'{region}_{variable}')
                plt.grid(which='major', alpha=0.5)
                # plt.legend()
        plt.show()

    def plot_scatter(self):
        ## plot scatter between precipitation and lai
        dff=result_root+rf'moving_window_extraction\Dataframe\\moving_window.df'
        df = T.load_df(dff)
        color_list=['b','r']
        df = df[df['landcover_classfication'] != 'Cropland']

        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['MODIS_LUCC'] != 12]

        region_list= ['Global', 'Australia', 'Africa', 'Asia', 'North_America', 'South_America']
        # region_list = ['Global', 'Arid', 'Semi-Arid', 'Sub-Humid']
        variable_list = ['precip', 'vpd', 'lai', ]
        mode_list=['wet','dry']
        fig = plt.figure()
        i = 1
        for region in region_list:
            ax = fig.add_subplot(2, 3, i)
            if region == 'Global':
                df_region = df
            else:
                # df_region = df[df['AI_classfication'] == region]
                df_region = df[df['continent'] == region]

            for mode in mode_list:
                ##extract the data 50>lai>-50 and 50>precip>-50

                vals_lai=df_region[f'{mode}_year_moving_window_extraction_lai'].to_list()
                vals_precip=df_region[f'{mode}_year_moving_window_extraction_precip'].to_list()
                vals_vpd=df_region[f'{mode}_year_moving_window_extraction_vpd'].to_list()
            ##plot scatter
                vals_lai_array=np.array(vals_lai)
                vals_precip_array=np.array(vals_precip)
                vals_vpd_array=np.array(vals_vpd)

                ##flatten
                vals_precip_array=vals_precip_array.flatten()
                vals_lai_array=vals_lai_array.flatten()


                vals_precip_array[vals_precip_array<-100]=np.nan
                vals_precip_array[vals_precip_array>100]=np.nan
                vals_lai_array[vals_lai_array<-100]=np.nan
                vals_lai_array[vals_lai_array>100]=np.nan

                plt.scatter(vals_precip_array,vals_lai_array,label='precip',color=color_list[mode_list.index(mode)])

                # plt.scatter(vals_vpd_array,vals_lai_array,label='vpd',color='b')
            plt.xlabel('precipitation')
            plt.ylabel('LAI')
            plt.legend()

            plt.title(f'{region}_scatter')
            i = i + 1
        plt.show()
class data_preprocess:
    def __init__(self):
        pass
    def run(self):
        self.df_to_spatial_arr()
    def df_to_spatial_arr(self):
        dff=result_root+rf'\ERA_precip_CV_trend\cv_trend\\trend.df'
        df = T.load_df(dff)
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['row'] > 120]
        df = df[df['LC_max'] < 20]
        df = df[df['MODIS_LUCC'] != 12]



        spatial_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = (row['pix'])
            spatial_dic[pix] = row['slope']*100
        outf = result_root + rf'ERA_precip_CV_trend\cv_trend\\trend.tif'
        arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr,outf)
        pass


def main():

    # aysmetry_response().run()
    # multi_regression().run()
    # monte_carlo().run()
    # multi_regression_new().run()
    moving_window_trend().run()


    # Dataframe_func().run()
    # data_preprocess().run()


if __name__ == '__main__':
    main()




