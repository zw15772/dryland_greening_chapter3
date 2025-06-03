# coding='utf-8'
from pygeodes import Geodes
from lytools import *
from pprint import pprint
from pygeodes.utils.formatting import format_collections
from pygeodes.utils.query import get_requestable_args
import urllib3
from pygeodes import Config
from pygeodes.utils.formatting import format_items

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
T=Tools()

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

class GEODES_download():
    def run (self):
        # self.download()
        # self.unzip()
        # self.hdf_to_tif()
        self.average_tif()




    def download(self):
        urllib3.disable_warnings()
        T = Tools()
        conf = Config.from_file(r'C:\Users\wenzhang1\PycharmProjects\dryland_greening_chapter3\pygeodes-config.json')
        # print(conf);exit()
        geodes = Geodes(conf=conf)

        pprint(get_requestable_args())
        # exit()
        query = { "start_datetime":{"gte":"1982-01-01T00:00:00Z"},
            "end_datetime":{"lte":"2021-01-01T23:59:59Z"},
                  "Platform":{"eq":"AVHRR"},
                  }


        items, dataframe = geodes.search_items(query=query, collections=['POSTEL_VEGETATION_LAI'],get_all=True)
        # items, dataframe = geodes.search_items(query=query, collections=['POSTEL_VEGETATION_LAI'],get_all=False)
        new_dataframe = format_items(
            dataframe, columns_to_add={"mission"}
        )
        pprint(new_dataframe)
        for item in tqdm(items):
            mission=item.properties["mission"]
            # geodes.download_item_archive(item)
            if not mission == "GEOV2-GCM":
                continue
            fname = item.properties['identifier']
            outf = "D:\\Project3\\Data\\GEODES_AVHRR_LAI\\" + fname+'.h5.gz'

            if isfile(outf):
                continue
            geodes.download_item_archive(item)

    def unzip(self):
        fdir='D:\Project3\Data\GEODES_AVHRR_LAI\\zip\\'
        outdir='D:\Project3\Data\GEODES_AVHRR_LAI\\unzip\\'
        Tools().mk_dir(outdir, force=True)

        for f in os.listdir(fdir):
            if not f.endswith('.h5.gz'):
                continue
            outf=outdir+f[:-3]
            if isfile(outf):
                continue

            self.unzip_gz_file(fdir+f, outf)
        pass



    def unzip_gz_file(self,input_path, output_path):
        import gzip
        import shutil
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


    def hdf_to_tif(self):
        import h5py
        fdir='D:\Project3\Data\GEODES_AVHRR_LAI\\unzip\\'
        outdir='D:\Project3\Data\GEODES_AVHRR_LAI\\tif\\'

        Tools().mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.h5'):
                continue
            outf=outdir+f[:-3]+'.tif'
            # read h5 file
            fr=h5py.File(fdir+f,'r')

            data=fr['LAI-MEAN'][:]
            data = np.array(data, dtype=np.float32)
            data[data>200] = np.nan
            data = data/30.
            data[data==0] = np.nan
            outf=outdir+f[:-3]+'.tif'
            longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.5, -0.5,
            ToRaster().array2raster(outf, longitude_start, latitude_start,
                                    pixelWidth, pixelHeight, data, ndv=-999999)
            # plt.imshow(data,interpolation='nearest',cmap='jet',vmin=0,vmax=7)
            # plt.colorbar()
            # plt.title(f)
            # plt.show()

    def average_tif(self):
        fdir='D:\Project3\Data\GEODES_AVHRR_LAI\\tif\\'
        outdir='D:\Project3\Data\GEODES_AVHRR_LAI\\tif_average\\'
        T.mk_dir(outdir, force=True)
        ## THEIA_GEOV2-GCM_R02_AVHRR_LAI_20170505
        year_list=['2017','2018',]

        month_list=['01','02','03','04','05','06','07','08','09','10','11','12']
        day_list=['05','15','25',]
        for year in year_list:
            for month in month_list:
                for day in day_list:
                    data_list=[]

                    for f in tqdm(os.listdir(fdir)):
                        if not f.endswith('.tif'):
                            continue
                        year_f=f.split('_')[-1][:4]
                        if not year_f == year:
                            continue
                        month_f=f.split('_')[-1][4:6]
                        if not month_f == month:
                            continue
                        day_f=f.split('_')[-1][6:8]
                        if not day_f == day:
                            continue
                        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
                        array = np.array(array, dtype=float)
                        array[array < -99] = np.nan
                        data_list.append(array)
                    data_list=np.array(data_list)
                    print(data_list.shape)
                    array_average=np.nanmean(data_list, axis=0)

                    outf=outdir+'THEIA_GEOV2-GCM_R02_AVHRR_LAI_'+year+month+day+'.tif'
                    ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, array_average)












def main():
    GEODES_download().run()





if __name__ == '__main__':
    main()