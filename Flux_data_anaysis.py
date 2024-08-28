# -*- coding: utf-8 -*  =

from matplotlib import pyplot as plt
import matplotlib.image as image
import matplotlib
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
from netCDF4 import Dataset
import glob
import xarray as xr
import datetime
import pandas as pd

def processing_flux(df):

    file = "AMF_US-YK1_BASE-BADM_1-5/AMF_US-YK1_BASE_HH_1-5.csv"

    df_YK1 = pd.read_csv(file, skiprows = 2)
    df_YK1["TIMESTAMP_START"] = pd.to_datetime(df_YK1["TIMESTAMP_START"], format='%Y%m%d%H%M')
    df_YK1["TIMESTAMP_END"] = pd.to_datetime(df_YK1["TIMESTAMP_END"], format='%Y%m%d%H%M')
    var_list = [ 'TIMESTAMP_START',
                'TIMESTAMP_END',
                'SWC_1_1_1',
                'SWC_2_1_1',
                'SWC_3_1_1',
                'TS_1_1_1',
                'TS_2_1_1',
                'TS_3_1_1',
                'GPP_PI_F',
                'RECO_PI_F',
                'FCH4_PI_F']

    df_YK1 = df_YK1[var_list]
    df_YK1 = df_YK1[df_YK1.SWC_1_1_1 > -9999]
    df_YK1 = df_YK1[df_YK1.TS_1_1_1 > -9999]
    df_YK1 = df_YK1[df_YK1.GPP_PI_F > -9999]

    # Set the 'datetime' column as the index
    df_YK1.set_index('TIMESTAMP_START', inplace=True)


    # Resample the data by day and apply filtering for days with more than 40 samples
    df_YK1_day = df_YK1.resample('D').apply(lambda x: x.mean() if len(x) > 47 else np.nan).dropna().reset_index()

