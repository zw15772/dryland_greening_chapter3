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


this_root = 'D:\Project3\\Result/Nov/Flux_validation\\'
data_root = 'D:/Project3/Result/Nov/Flux_validation//data/'
result_root = 'D:/Project3/Result/Nov/Flux_validation/result/'
T.mkdir(result_root)
class preprocessing_flux_validation():
    def __init__(self):
        pass
    def run(self):
        # self.preprocessing()
        # self.plot_GPP_growing_season()

        pass
    def preprocessing(self):
        fdir=rf'D:\Project3\Result\Nov\Flux_validation\monthly\\'
        dic_growing_season = {
            'US-SRG': [4, 5, 6, 7, 8, 9],
            'US-SRM': [4, 5, 6, 7, 8, 9],
            'US-Wkg': [4, 5, 6, 7, 8, 9],
            'US-Vcp':[4, 5, 6, 7, 8, 9],
            'US-Vcm':[4, 5, 6, 7, 8, 9],
            'AU-ASM': [11, 12, 1, 2, 3, 4],
            'US-Seg': [4, 5, 6, 7, 8, 9],
            'US-Ses': [4, 5, 6, 7, 8, 9],
            'US-Mpj': [4, 5, 6, 7, 8, 9],
            'US-Wjs': [4, 5, 6, 7, 8, 9],
            # 'CN-Aro':[4, 5, 6, 7, 8, 9],
            #
            # 'CN-Dsh':[4, 5, 6, 7, 8, 9],
            # 'CN-Xi2':[4, 5, 6, 7, 8, 9],


            'AU-Dry': [11, 12, 1, 2, 3, 4],
        }



        variable = 'GPP_NT_VUT_REF'

        all_result = []

        for f in T.listdir(fdir):

            site = f.split('_')[1]
            if not site in dic_growing_season:
                continue

            if site not in dic_growing_season:
                print('skip:', site, f)
                continue

            df = pd.read_csv(join(fdir, f))

            df[variable] = df[variable].where(df[variable] > -999, np.nan)

            df_new = pd.DataFrame({
                'date': pd.to_datetime(df['TIMESTAMP'].astype(str), format='%Y%m'),
                'GPP': df[variable]
            })

            # ===== 时间信息 =====
            df_new['year'] = df_new['date'].dt.year

            df_new['month'] = df_new['date'].dt.month
            df_new['days'] = df_new['date'].dt.days_in_month


            growing_months = dic_growing_season[site]

            df_gs = df_new[df_new['month'].isin(growing_months)].copy()

            # AU sites: Nov-Dec belong to next growing-season year
            if 11 in growing_months and 12 in growing_months and 1 in growing_months:
                df_gs.loc[df_gs['month'].isin([11, 12]), 'year'] += 1
            # df_gs = df_gs[df_gs['year'] < 2022]

            # gC m-2 d-1 -> gC m-2 month-1
            df_gs['GPP_monthly'] = df_gs['GPP'] * df_gs['days']

            df_result = (
                df_gs.groupby('year')['GPP_monthly']
                .sum()
                .reset_index()
                .rename(columns={'GPP_monthly': f'GPP_growing_season'})
            )

            df_result['site'] = site

            all_result.append(df_result)

        # ===== 合并所有站点 =====
        df_all = pd.concat(all_result, ignore_index=True)

        # ===== 转成宽表 =====
        df_wide = df_all.pivot(index='year', columns='site', values='GPP_growing_season')

        df_wide = df_wide.reset_index()

        T.save_df(df_wide, join(result_root, 'GPP_growing_season_VUT.df'))
        T.df_to_excel(df_wide, join(result_root, 'GPP_growing_season_VUT.xlsx'))
        pass

    def plot_GPP_growing_season(self):
        df = T.load_df(join(result_root, 'GPP_growing_season_VUT.df'))

        plt.figure(figsize=(10, 6))

        for col in df.columns:
            if col == 'year':
                continue
            df[col] = df[col].replace(0, np.nan)

            plt.plot(df['year'], df[col], label=col)

            plt.xlabel('Year')
            plt.ylabel('GPP Growing Season (gC m⁻²)')
            plt.title('Growing Season GPP')
            plt.legend(ncol=2)
            plt.grid(alpha=0.3)



            plt.show()



    def detrend_data(self):
        dff=join(result_root,'GPP_growing_season_VUT.df')
        df=T.load_df(dff)

        df_detrend = pd.DataFrame()
        df_detrend['year'] = df['year']

        for col in df.columns:
            if col == 'year':
                continue
            df[col] = df[col].replace(0, np.nan)

            vals = np.array(df[col], dtype=float)

            # detrend
            vals_detrend = T.detrend_vals(vals)

            # 保存 detrend 后结果
            df_detrend[col] = vals_detrend

            # plot check
            # plt.figure()
            # plt.plot(df['year'], vals, label=col)
            # plt.plot(df['year'], vals_detrend, label=col + '_detrend')
            # plt.legend()
            # plt.title(col)
            # plt.show()

        outf = join(result_root, 'GPP_growing_season_VUT_detrend.df')
        T.save_df(df_detrend, outf)
        T.df_to_excel(df_detrend, outf.replace('.df', '.xlsx'))

        return df_detrend





    def extract_shp_from_tif(self):

        shp_f = this_root + '/zip/All_flux_points.shp'
        f = rf'D:\Project3\Result\Nov\Composite_LAI\CV\trend_analysis\composite_LAI_mean_detrend_CV_trend.tif'
        outdir = result_root + '//extract_shp_from_tif/'
        T.mk_dir(outdir, force=True)

        pass
class preprocessing_flux_data:
    def __init__(self):
        pass

    def run(self):
        # self.unzip_data()
        self.extract_meta_data()

    def unzip_data(self):
        fdir=data_root+rf'\shuttle\\'


        outdir =data_root+rf'\shuttle_unzip\\'
        T.mk_dir(outdir)

        T.unzip(fdir, outdir)
        pass

    def extract_meta_data(self):
        ##1 data start/ end and range
        ## if all -9999 label probelm
        meta_data_file=this_root+rf'meta\\dryland_flux_sites_attribution.csv'
        fdir_all=data_root+rf'\shuttle_unzip\\'
        variable = 'GPP_NT_VUT_REF'
        df_meta=pd.read_csv(meta_data_file)


        # save info
        info_list = []

        for fdir in T.listdir(fdir_all):

            if 'FLUXNET' in fdir:

                time_col = 'TIMESTAMP'
                skiprows = 0
                dataset_type = 'FLUXNET'

            elif 'BASE' in fdir:

                time_col = 'TIMESTAMP_START'
                skiprows = 2
                dataset_type = 'BASE'

            else:
                continue

            for f in T.listdir(os.path.join(fdir_all, fdir)):

                if not f.endswith('.csv'):
                    continue

                # ===== FLUXNET =====
                if dataset_type == 'FLUXNET':

                    if ('FLUXMET' not in f) or ('MM' not in f):
                        continue

                # ===== BASE =====
                elif dataset_type == 'BASE':

                    if 'BASE_HH' not in f:
                        continue

                print(f)

                fpath = os.path.join(fdir_all, fdir, f)

                try:

                    df = pd.read_csv(
                        fpath,
                        skiprows=skiprows,

                    )
                    # print(df[time_col].head());exit()



                    # ===== site name =====
                    # example:
                    # FLX_US-XXX_FLUXMET_MM_....
                    site = f.split('_')[1]

                    # ===== time column =====


                    df[time_col] = df[time_col].astype(str)

                    start_time = df[time_col].min()
                    end_time = df[time_col].max()
                    start_year = int(start_time[:4])
                    end_year = int(end_time[:4])

                    total_years = end_year - start_year + 1


                    # ===== valid data =====



                    # print(valid_len, len(vals));exit()

                    # ===== append =====
                    info_list.append({
                        'site_id': site,
                        'start_time': start_year,
                        'end_time': end_year,
                        'total_years': total_years,
                    })

                except Exception as e:

                    print(f'ERROR: {f}')
                    print(e)

    # ===================================
        # build dataframe
        # ===================================
        df_info = pd.DataFrame(info_list)

        # ===================================
        # merge
        # ===================================
        dff_new = df_meta.merge(
            df_info,
            on='site_id',
            how='left'
        )

        # ===================================
        # save
        # ===================================
        outf = this_root + rf'\meta\\dryland_flux_sites_attribution_new2.csv'

        dff_new.to_csv(outf, index=False)

        print(dff_new.head())


        pass




def main():

    # preprocessing_flux_validation().run()
    preprocessing_flux_data().run()




if __name__ == '__main__':
    main()



