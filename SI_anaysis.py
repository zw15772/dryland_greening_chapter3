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
D = DIC_and_TIF(pixelsize=0.5)
centimeter_factor = 1/2.54


this_root = 'E:\Project3\\'
data_root = 'E:/Project3/Data/'
result_root = 'E:/Project3/Result/'

class greening_analysis():

    def __init__(self):

        pass
    def run(self):
        pass
    def greening_products_basemap(self):
        ## three products 3 time periods comparison
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np

        # Create synthetic data (replace with actual data as needed)
        nrows, ncols = 100, 100
        data_dict = {
            'LAI': [np.random.rand(nrows, ncols) for _ in range(3)],
            'NDVI': [np.random.rand(nrows, ncols) for _ in range(3)],
            'NIRv': [np.random.rand(nrows, ncols) for _ in range(3)],
        }

        products = ['LAI', 'NDVI', 'NIRv']
        periods = ['Period 1', 'Period 2', 'Period 3']

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # Loop through products and periods to create subplots
        for i, product in enumerate(products):
            for j, period in enumerate(periods):
                ax = axes[i, j]
                m = Basemap(projection='cyl', resolution='l', ax=ax)
                m.drawcoastlines()
                m.drawcountries()
                m.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0])
                m.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1])

                # Get the data for this subplot
                data = data_dict[product][j]
                lon = np.linspace(-180, 180, ncols)
                lat = np.linspace(-90, 90, nrows)
                lon, lat = np.meshgrid(lon, lat)

                # Plot the data using pcolormesh
                im = m.pcolormesh(lon, lat, data, shading='auto', cmap='viridis')

                # Add title and colorbar
                ax.set_title(f'{product} - {period}')
                cbar = m.colorbar(im, location='right', pad='5%')

        plt.suptitle('3x3 Subplot of LAI, NDVI, and NIRv for Three Periods')
        plt.show()

        pass

class Rainfall_product_comparison():
    pass

class TRENDY_trend():
    pass

class TRENDY_CV():
    pass

def main():
    # Data_processing_2().run()
    # Phenology().run()
    # build_dataframe().run()
    # build_moving_window_dataframe().run()
    # CO2_processing().run()
    # greening_analysis().run()
    # TRENDY_trend().run()
    # TRENDY_CV().run()
    # multi_regression_window().run()
    # bivariate_analysis().run()

    # visualize_SHAP().run()
    # PLOT_dataframe().run()
    # Plot_Robinson().robinson_template()
    plt.show()



    pass

if __name__ == '__main__':
    main()