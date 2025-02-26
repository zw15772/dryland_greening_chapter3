# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lytools import *
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pprint import pprint
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

T = Tools()
results_root = rf'E:\Project3\Result\\'
# data_root = rf'E:\Project3\Data\ERA5_daily\dict\\'


class SHAP:

    def __init__(self):

        self.this_class_png = results_root + 'SHAP\\png\\raw\\'
        self.dff =rf'E:\Project3\Result\Dataframe\moving_window\moving_window.df'
        self.variable_list_rt()


        ##----------------------------------

        self.y_variable = 'LAI4g_relative_change'


        ####################

        self.x_variable_list = self.x_variable_list
        self.x_variable_range_dict = self.x_variable_range_dict_global


        pass

    def run(self):
        # self.check_df_attributes()

        # self.check_variables_ranges()
        # self.show_colinear()

        # self.pdp_shap()
        self.plot_pdp_shap()
        # self.pdp_shap_trend()

        pass
    def check_df_attributes(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print(df.columns.tolist())
        print(len(df))
        # exit()
        pass
    def check_variables_ranges(self):

        dff = self.dff
        df = T.load_df(dff)
        df=self.df_clean(df)
        df=self.plot_hist(df)
        df = self.valid_range_df(df)
        # df = self.__select_extreme(df)
        # T.print_head_n(df)
        # exit()

        x_variable_list = self.x_variable_list
        print(len(x_variable_list))
        # exit()
        flag = 1
        for var in x_variable_list:
            print(flag,var)
            vals = df[var].tolist()
            plt.subplot(4,4,flag)
            flag += 1
            plt.hist(vals,bins=100)
            plt.title(var)
        plt.tight_layout()
        plt.show()

        pass


    def variable_list_rt(self):
        self.x_variable_list = [
            'CO2_relative_change',
            'GPCC_relative_change',
            'VPD_relative_change',
            'Aridity',
            # 'CO2_raw',
            # 'CRU_raw',
            # 'VPD_raw',


                                ]
        self.x_variable_range_dict_global = {
            # 'CO2_raw': [340, 450],
            # 'VPD_raw': [0.5, 3],
            # 'CRU_raw': [0, 1000],
            # 'tmax_raw': [0,40],

            'CO2_relative_change': [-10, 10],
            'GPCC_relative_change': [-100, 200],
            'VPD_relative_change': [-30, 30],
            'Aridity': [0, 0.65],




        }
        self.x_variable_range_dict_Africa = {
            # 'CO2_relative_change_detrended': [-1, 1],
            # 'VPD_relative_change_detrended': [-20, 20],
            # 'GPCC_relative_change_detrended': [-100, 150],
            # 'Noy_relative_change_detrended': [-10, 10],

            'CO2_relative_change': [-10, 10],

            'VPD_relative_change': [-20, 20],
            'GPCC_relative_change': [-100, 150],
            'Noy_relative_change': [-40, 40],

            'Noy': [100, 500],

            'Nhx': [50, 400],
            'GMST': [0, 1],

            'VPD': [0.5, 3],
            'GPCP_precip': [0, 125],
            'GPCP_precip_pre': [0, 100],

            'CO2': [350., 450.],

            'average_dry_spell': [0, 20],
            'maximum_dry_spell': [0, 60],
            'CV_rainfall': [0, 6],
            'total_rainfall': [0, 2000],

            'frequency_wet': [0, 50],
            'frequency_dry': [0, 50],
            # 'tmax_CV': [0, 10],
            # 'tmin_CV': [0, 10],

            'frequency_heat_event': [1, 50],
            'average_anomaly_heat_event': [2, 10],

            'silt': [0, 50],
            'rooting_depth': [0, 10],
            'SOC_sum': [0, 1500],
            'ENSO_index_average': [-2, 1],
            'ENSO_index_average_lagged': [-2, 2],
            'ENSO_index_distance': [0.5, 2],
            'ENSO_index_distance_lagged': [0.5, 2],
            'ENSO_index_DFJ': [-2, 2],

            'ENSO_index_binary': [-2, 2],
            'GPCC_trend': [-2, 2],
            'Aridity': [0.2, 0.65],
            'ENSO_index_average_lagged_whole_year': [-1, 1],
            'maximun_heat_spell': [0, 150],
            'average_heat_spell': [0, 30],
            'maximun_cold_spell': [0, 150],
            'average_cold_spell': [0, 30],
        }

        self.x_variable_range_dict_north_america = {
            'CO2_relative_change_detrended': [-1, 1],
            'VPD_relative_change_detrended': [-20, 20],
            'GPCC_relative_change_detrended': [-100, 100],
            'Noy_relative_change_detrended': [-5, 10],


            'ozone': [280, 320],
            'GMST': [0, 1],
            'CO2': [350, 450],
            'Noy': [100, 300],
            'Nhx': [25, 250],

            'VPD': [0, 3],

            'average_dry_spell': [0, 25],
            'maximum_dry_spell': [0, 60],
            'CV_rainfall': [0, 6],
            'total_rainfall': [0, 2000],

            'frequency_wet': [0, 50],
            'frequency_dry': [0, 50],
            'tmax_CV': [0, 10],
            'tmin_CV': [0, 10],

            'frequency_heat_event': [1, 50],
            'average_anomaly_heat_event': [2, 10],

            'GPCP_precip': [0, 100],
            'GPCP_precip_pre': [0, 100],
            'GLEAM_SMroot': [-50, 50],
            # 'tmin': [-10, 10],
            # 'tmax': [-7.5, 4],

            'silt': [0, 60],
            'rooting_depth': [0, 20],
            'SOC_sum': [0, 2000],
            'ENSO_index_average': [-1.5, 1.5],
            'ENSO_index_average_lagged': [-2, 2],
            'ENSO_index_distance': [0.5, 2],
            'ENSO_index_distance_lagged': [0.5, 2],
            'ENSO_index_DFJ': [-2, 2],

            'ENSO_index_binary': [-2, 2],
            'GPCC_trend': [-2, 2],
            'Aridity': [0.2, 0.65],
            'ENSO_index_average_lagged_whole_year': [-1, 1],
            'maximun_heat_spell': [0, 150],
            'average_heat_spell': [0, 30],
            'maximun_cold_spell': [0, 150],
            'average_cold_spell': [0, 30],

        }

        self.x_variable_range_dict_Asia = {

            'CO2_anomaly': [-30, 30],
            'VPD_anomaly': [-0.5, 0.5],
            'GPCC_anomaly': [-50, 50],
            'noy_anomaly': [-100, 100],
            'tmax_anomaly': [-3, 3],
            'fire_burned_area': [0,1 * 10 ** 7],
            'ENSO_index_average': [-2, 2],
            'rooting_depth': [0, 20],
            'silt': [0, 50],
            'CV_rainfall': [0, 8],
            'frequency_wet': [0, 60],
            'average_dry_spell': [0, 50],

            'frequency_heat_event': [0, 60],
            'average_anomaly_heat_event': [2, 11],


############################################3
            'CO2_relative_change_detrended': [-1, 1],
            'VPD_relative_change_detrended': [-20, 20],
            'GPCC_relative_change_detrended': [-100, 150],
            'Noy_relative_change_detrended': [-1, 1],


            'Noy': [10, 500],

            'GMST': [0, 1],

            'Nhx': [0, 1000],

            'CO2': [340, 450],

            'VPD': [0.5, 3],

            'maximum_dry_spell': [0, 150],

            'total_rainfall': [0, 2000],


            'frequency_dry': [0, 50],


            'average_anomaly_cold_event': [-12, -2],

            'GPCP_precip': [0, 200],
            'GPCP_precip_pre': [0, 100],
            'GLEAM_SMroot': [-50, 50],
            'tmin': [-10, 10],
            'tmax': [-7.5, 4],


            'ENSO_index_average_lagged': [-2, 2],
            'ENSO_index_distance': [0.5, 2],
            'ENSO_index_distance_lagged': [0.5, 2],
            'ENSO_index_DFJ': [-2, 2],

            'ENSO_index_binary': [-2, 2],
            'GPCC_trend': [-2, 2],
            'Aridity': [0.2, 0.65],
            'ENSO_index_average_lagged_whole_year': [-1, 1],
            'maximun_heat_spell': [0, 150],
            'average_heat_spell': [0, 30],
            'maximun_cold_spell': [0, 150],
            'average_cold_spell': [0, 40],
        }

        self.x_variable_range_dict_South_America = {
            'CO2_relative_change_detrended': [-1, 1],
            'VPD_relative_change_detrended': [-20, 20],
            'GPCC_relative_change_detrended': [-100, 150],
            'Noy_relative_change_detrended': [-10, 15],

            'Noy': [0, 200],
            'Nhx': [0, 600],
            'GMST': [0, 1],

            'CO2': [340, 450],

            'VPD': [0.5, 3],
            'average_dry_spell': [0, 25],
            'maximum_dry_spell': [0, 75],
            'CV_rainfall': [0, 6],
            'total_rainfall': [0, 2000],

            'frequency_wet': [0, 50],
            'frequency_dry': [0, 50],

            'frequency_heat_event': [1, 40],
            'average_anomaly_heat_event': [2, 10],
            'average_anomaly_cold_event': [-12, -2],

            'GPCP_precip': [0, 200],
            'GPCP_precip_pre': [0, 100],

            'silt': [0, 50],
            'rooting_depth': [0, 20],
            'ENSO_index_average': [-2, 2],
            'ENSO_index_average_lagged': [-2, 2],
            'ENSO_index_distance': [0.5, 2],
            'ENSO_index_distance_lagged': [0.5, 2],
            'ENSO_index_DFJ': [-2, 2],

            'ENSO_index_binary': [-2, 2],
            'GPCC_trend': [-2, 2],
            'Aridity': [0.2, 0.65],
            'ENSO_index_average_lagged_whole_year': [-1, 1],
            'maximun_heat_spell': [0, 150],
            'average_heat_spell': [0, 30],
            'maximun_cold_spell': [0, 150],
            'average_cold_spell': [0, 40],

        }

        self.x_variable_range_dict_AUS = {

            'CO2_anomaly': [-30, 30],
            'VPD_anomaly': [-0.4, 0.4],
            'GPCC_anomaly': [-50, 50],
            'noy_anomaly': [-100, 100],
            'tmax_anomaly': [-3, 3],
            'fire_burned_area': [0, 2 * 10 ** 7],
            'ENSO_index_average': [-2, 2],
            'rooting_depth': [0, 40],
            'silt': [0, 50],
            'CV_rainfall': [0, 8],
            'frequency_wet': [0, 80],
            'average_dry_spell': [0, 50],

            'frequency_heat_event': [0, 100],
            'average_anomaly_heat_event': [2, 11],


        }

    def variable_list_rt_detrend(self):
        self.x_variable_list_detrend = [


            'ENSO_index_average',

            # 'Noy_detrend',

            'rooting_depth',
            # 'SOC_sum',
            'silt',

            'VPD_detrend',

            'GPCC_detrend',
            # 'GPCP_precip_pre',

            'average_dry_spell',
            # 'maximum_dry_spell',

            'CV_rainfall',

            'frequency_wet',

            'frequency_heat_event',
            'average_anomaly_heat_event',

            # 'Aridity',

            # 'GPCC_trend',
        ]




        self.x_variable_range_dict_Africa_detrend = {

        'Noy': [100, 500],

        'Nhx': [50, 400],
        'GMST': [0, 1],

        'VPD': [0.5, 3],
        'GPCC': [0, 125],
        'GPCP_precip_pre': [0, 100],

        'CO2': [350., 450.],

        'average_dry_spell': [0, 20],
        'maximum_dry_spell': [0, 60],
        'CV_rainfall': [0, 6],
        'total_rainfall': [0, 2000],

        'frequency_wet': [0, 50],
        'frequency_dry': [0, 50],
        # 'tmax_CV': [0, 10],
        # 'tmin_CV': [0, 10],

        'frequency_heat_event': [1, 50],
        'average_anomaly_heat_event': [2, 10],

        'silt': [0, 50],
        'rooting_depth': [0, 10],
        'SOC_sum': [0, 1500],
        'ENSO_index_average': [-2, 1],
        'ENSO_index_average_lagged': [-2, 2],
        'ENSO_index_distance': [0.5, 2],
        'ENSO_index_distance_lagged': [0.5, 2],
        'ENSO_index_DFJ': [-2, 2],

        'ENSO_index_binary': [-2, 2],
        'GPCC_trend': [-2, 2],
        'Aridity': [0.2, 0.65],
        'ENSO_index_average_lagged_whole_year': [-1, 1],
        'maximun_heat_spell': [0, 150],
        'average_heat_spell': [0, 30],
        'maximun_cold_spell': [0, 150],
        'average_cold_spell': [0, 30],
    }

        self.x_variable_range_dict_north_america_detrend = {

        'ozone': [280, 320],
        'GMST': [0, 1],
        'CO2': [350, 450],
        'Noy': [100, 300],
        'Nhx': [25, 250],

        'VPD': [0, 3],

        'average_dry_spell': [0, 20],
        'maximum_dry_spell': [0, 60],
        'CV_rainfall': [0, 6],
        'total_rainfall': [0, 2000],

        'frequency_wet': [0, 50],
        'frequency_dry': [0, 50],
        'tmax_CV': [0, 10],
        'tmin_CV': [0, 10],

        'frequency_heat_event': [1, 50],
        'average_anomaly_heat_event': [2, 10],

        'GPCC': [0, 100],
        'GPCP_precip_pre': [0, 100],
        'GLEAM_SMroot': [-50, 50],
        # 'tmin': [-10, 10],
        # 'tmax': [-7.5, 4],

        'silt': [0, 60],
        'rooting_depth': [0, 20],
        'SOC_sum': [0, 2000],
        'ENSO_index_average': [-1.5, 1.5],
        'ENSO_index_average_lagged': [-2, 2],
        'ENSO_index_distance': [0.5, 2],
        'ENSO_index_distance_lagged': [0.5, 2],
        'ENSO_index_DFJ': [-2, 2],

        'ENSO_index_binary': [-2, 2],
        'GPCC_trend': [-2, 2],
        'Aridity': [0.2, 0.65],
        'ENSO_index_average_lagged_whole_year': [-1, 1],
        'maximun_heat_spell': [0, 150],
        'average_heat_spell': [0, 30],
        'maximun_cold_spell': [0, 150],
        'average_cold_spell': [0, 30],
    }

        self.x_variable_range_dict_Asia_detrend = {

        'Noy': [10, 500],

        'GMST': [0, 1],

        'Nhx': [0, 1000],

        'CO2': [340, 450],

        'VPD': [0.5, 3],
        'average_dry_spell': [0, 25],
        'maximum_dry_spell': [0, 150],
        'CV_rainfall': [0, 6],
        'total_rainfall': [0, 2000],

        'frequency_wet': [0, 50],
        'frequency_dry': [0, 50],

        'frequency_heat_event': [1, 40],
        'average_anomaly_heat_event': [2, 10],
        'average_anomaly_cold_event': [-12, -2],

        'GPCC': [0, 200],
        'GPCP_precip_pre': [0, 100],
        'GLEAM_SMroot': [-50, 50],
        'tmin': [-10, 10],
        'tmax': [-7.5, 4],

        'silt': [0, 50],
        'rooting_depth': [0, 10],
        'ENSO_index_average': [-2, 2],
        'ENSO_index_average_lagged': [-2, 2],
        'ENSO_index_distance': [0.5, 2],
        'ENSO_index_distance_lagged': [0.5, 2],
        'ENSO_index_DFJ': [-2, 2],

        'ENSO_index_binary': [-2, 2],
        'GPCC_trend': [-2, 2],
        'Aridity': [0.2, 0.65],
        'ENSO_index_average_lagged_whole_year': [-1, 1],
        'maximun_heat_spell': [0, 150],
        'average_heat_spell': [0, 30],
        'maximun_cold_spell': [0, 150],
        'average_cold_spell': [0, 40],
    }

        self.x_variable_range_dict_South_America_detrend = {

        'Noy': [0, 200],
        'Nhx': [0, 600],
        'GMST': [0, 1],

        'CO2': [340, 450],

        'VPD': [0.5, 3],
        'average_dry_spell': [0, 25],
        'maximum_dry_spell': [0, 75],
        'CV_rainfall': [0, 6],
        'total_rainfall': [0, 2000],

        'frequency_wet': [0, 50],
        'frequency_dry': [0, 50],

        'frequency_heat_event': [1, 40],
        'average_anomaly_heat_event': [2, 10],
        'average_anomaly_cold_event': [-12, -2],

        'GPCC': [0, 200],
        'GPCP_precip_pre': [0, 100],

        'silt': [0, 50],
        'rooting_depth': [0, 20],
        'ENSO_index_average': [-2, 2],
        'ENSO_index_average_lagged': [-2, 2],
        'ENSO_index_distance': [0.5, 2],
        'ENSO_index_distance_lagged': [0.5, 2],
        'ENSO_index_DFJ': [-2, 2],

        'ENSO_index_binary': [-2, 2],
        'GPCC_trend': [-2, 2],
        'Aridity': [0.2, 0.65],
        'ENSO_index_average_lagged_whole_year': [-1, 1],
        'maximun_heat_spell': [0, 150],
        'average_heat_spell': [0, 30],
        'maximun_cold_spell': [0, 150],
        'average_cold_spell': [0, 40],

    }

        self.x_variable_range_dict_AUS_detrend = {

        'noy_detrend': [20, 150],


        'VPD_detrend': [0.5, 3],
        'average_dry_spell': [0, 20],
        'maximum_dry_spell': [0, 100],
        'CV_rainfall': [0, 6],
        'total_rainfall': [0, 2000],

        'frequency_wet': [0, 40],
        'frequency_dry': [0, 50],

        'frequency_cold_event': [1, 40],
        'average_anomaly_cold_event': [-7, -3],
        'frequency_heat_event': [1, 50],
        'average_anomaly_heat_event': [4, 12],

        'GPCC_detrend': [0, 200],
        'GPCP_precip_pre': [0, 100],
        'GLEAM_SMroot': [-50, 50],

        'silt': [0, 50],

        'rooting_depth': [0, 20],
        'SOC_sum': [0, 4000],
        'ENSO_index_average': [-2, 2],

        'GPCC_trend': [-1, 1],
        'Aridity': [0.2, 0.65],

        'maximun_heat_spell': [0, 100],
        'average_heat_spell': [0, 25],
        'maximun_cold_spell': [25, 75],
        'average_cold_spell': [0, 25],

    }
    def show_colinear(self,):
        dff=self.dff
        df = T.load_df(dff)
        vars_list = self.x_variable_list
        df = df[vars_list]
        ## add LAI4g_raw
        df['LAI4g_raw'] = T.load_df(dff)['LAI4g_raw']
        ## plot heat map to show the colinear variables
        import seaborn as sns
        plt.figure(figsize=(10, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f")
        plt.show()




    def discard_vif_vars(self,df, x_vars_list):
        ##################实时计算#####################
        vars_list_copy = copy.copy(x_vars_list)

        X = df[vars_list_copy]
        X = X.dropna()
        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif.round(1)
        selected_vif_list = []
        for i in range(len(vif)):
            feature = vif['features'][i]
            VIF_val = vif['VIF Factor'][i]
            if VIF_val < 5.:
                selected_vif_list.append(feature)
        return selected_vif_list

        pass
    def plot_hist(self,df):
        # T.print_head_n(df)
        # exit()
        x_variable_list = self.x_variable_list
        # print(x_variable_list)
        # exit()
        for var in x_variable_list:


            vals = df[var].tolist()
            vals = np.array(vals)
            # vals[vals<-500] = np.nan
            # vals[vals>500] = np.nan
            # vals = vals[~np.isnan(vals)]
            plt.hist(vals,bins=100)
            plt.title(var)
            plt.show()
        exit()
        return df
    def valid_range_df(self,df):

        print('original len(df):',len(df))
        for var in self.x_variable_list:


            if not var in df.columns:
                print(var,'not in df')
                continue
            min,max = self.x_variable_range_dict[var]
            df = df[(df[var]>=min)&(df[var]<=max)]
        print('filtered len(df):',len(df))
        return df
    def df_clean(self,df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]
        df=df[df['LC_max']<20]

        df=df[df['MODIS_LUCC']!=12]
        df=df[df['LAI4g_relative_change']<50]
        df=df[df['LAI4g_relative_change']>-50]
        # df=df[df['LAI4g_p_value']<0.05]
        # df=df[df['LAI4g_trend']<0]
        # df=df[df['LAI4g_p_value']<0.05]
        # df=df[df['LAI4g_trend']>0]



        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        df = df[df['landcover_classfication'] != 'Cropland']



        return df


    def pdp_shap(self):

        dff = self.dff
        outdir = join(self.this_class_png, 'pdp_shap_second_decades')

        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list


        y_variable = self.y_variable
        # plt.hist(T.load_df(dff)[y_variable].tolist(),bins=100)
        # plt.show()
        df = T.load_df(dff)
        print(df['year'].unique())

        df=self.df_clean(df)

        T.print_head_n(df)


        df = self.valid_range_df(df)


        ## get the first decade

        df=df[df['year']>2000]

        print(len(df))
        T.print_head_n(df)
        print('-'*50)
        # exit()
        # model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        # all_vars_vif = self.discard_vif_vars(df, x_variable_list)
        # all_vars_vif.append('CV_rainfall')
        # print('all_vars_vif:',all_vars_vif)
        # exit()
        all_vars = copy.copy(x_variable_list)
        # all_vars=copy.copy(all_vars_vif)

        all_vars.append(y_variable)  # add the y variable to the list
        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna()  # drop rows with missing values

        X = all_vars_df[x_variable_list]


        Y = all_vars_df[y_variable]

        model,y,y_pred = self.__train_model(X, Y)  # train a Random Forests model
        imp_dict_xgboost = {}
        for i in range(len(x_variable_list)):
            imp_dict_xgboost[x_variable_list[i]] = model.feature_importances_[i]
        sorted_imp = sorted(imp_dict_xgboost.items(), key=lambda x: x[1], reverse=True)
        x_ = []
        y_ = []
        for key, value in sorted_imp:
            x_.append(key)
            y_.append(value)
        plt.figure()
        plt.bar(x_, y_)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.title('xgboost')
        # plt.show()
        plt.figure()

        explainer = shap.TreeExplainer(model)
        # R2=stats.pearsonr(y,y_pred)[0]**2
        # ### round R2
        # R2 = round(R2,2)
        # print('shaply_R2:',R2)
        x_variable_range_dict = self.x_variable_range_dict
        y_base = explainer.expected_value
        print('y_base', y_base)
        print('y_mean', np.mean(y))
        # shap_values = explainer.shap_values(X)
        outf_shap = join(outdir, self.y_variable + '.shap')
        ## how to resever X and Y before the shap


        shap_values = explainer(X)

        T.save_dict_to_binary(shap_values,outf_shap)
        exit()

    def plot_pdp_shap(self):
        x_variable_list = self.x_variable_list

        inf_shap = join(self.this_class_png, 'pdp_shap_first_decades', self.y_variable + '.shap.pkl')
        shap_values = T.load_dict_from_binary(inf_shap)
        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list=[]
        y_list=[]
        for key in imp_dict.keys():
            x_list.append(key)
            y_list.append(imp_dict[key])
        plt.bar(x_list,y_list)
        print(x_list)
        # plt.title(f'R2_{R2}')
        plt.xticks(rotation=90)
        plt.title('shap')

        plt.tight_layout()


        # plt.show()


        flag = 1
        centimeter_factor = 1 / 2.54
        plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        for x_var in x_list:
            shap_values_mat = shap_values[:,x_var]
            data_i = shap_values_mat.data
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var:data_i,'shap_v':value_i})
            df_i_random = df_i.sample(n=len(df_i)//2)
            df_i = df_i_random

            # x_variable_range_dict = self.x_variable_range_dict
            start,end = self.x_variable_range_dict[x_var]
            bins = np.linspace(start,end,50)
            df_group, bins_list_str = T.df_bin(df_i,x_var,bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            scatter_x_list = df_i[x_var].tolist()
            scatter_y_list = df_i['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)
            #     err,_,_ = self.uncertainty_err(SM)
            # print(df_i)
            # exit()
            plt.subplot(2,3,flag)
            # plt.scatter(scatter_x_list, scatter_y_list, alpha=0.2,c='gray',marker='.',s=1,zorder=-1)
            # print(data_i[0])
            # exit()
            # interp_model = interpolate.interp1d(x_mean_list, y_mean_list, kind='cubic')
            # y_interp = interp_model(x_mean_list)
            y_mean_list = SMOOTH().smooth_convolve(y_mean_list,window_len=7)
            plt.plot(x_mean_list,y_mean_list,c='red',alpha=1)

            # exit()
            # plt.fill_between(x_mean_list, np.array(y_mean_list) - np.array(y_err_list), np.array(y_mean_list) + np.array(y_err_list), alpha=0.3,color='red')
            #### rename x_label remove

            plt.xlabel(x_var)
            flag += 1
            plt.ylim(-10,10)

        plt.suptitle(self.y_variable)
        plt.tight_layout()
        # plt.show()
        outf=join(self.this_class_png,'pdp_shap_first_decades.pdf')
        plt.savefig(outf,dpi=300)
        plt.close()

    def pdp_shap_trend(self):
        dff = self.dff
        outdir = join(self.this_class_png, 'pdp_shap')
        outf = join(outdir, self.y_variable + '.png')
        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list

        y_variable = self.y_variable
        df = T.load_df(dff)
        df = self.df_clean(df)
        # df = self.__select_extreme(df)
        # df = self.valid_range_df(df)
        print(df.columns.tolist())
        print(len(df))
        T.print_head_n(df)
        print('-' * 50)
        # exit()
        # model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        all_vars = copy.copy(x_variable_list)  # copy the x variables
        all_vars.append(y_variable)  # add the y variable to the list
        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        # T.print_head_n(all_vars_df)
        # exit()
        all_vars_df = all_vars_df.dropna()  # drop rows with missing values
        X = all_vars_df[x_variable_list]
        Y = all_vars_df[y_variable]
        model, y, y_pred = self.__train_model(X, Y)  # train a Random Forests model
        # explainer = shap.Explainer(model)
        explainer = shap.TreeExplainer(model)
        x_variable_range_dict = self.x_variable_range_dict
        y_base = explainer.expected_value
        print('y_base', y_base)
        print('y_mean', np.mean(y))
        # shap_values = explainer.shap_values(X)
        shap_values = explainer(X)
        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)
            y_list.append(imp_dict[key])
        plt.barh(x_list, y_list)
        plt.tight_layout()
        # plt.show()
        # plt.figure(figsize=(8, 8))

        flag = 1
        centimeter_factor = 1 / 2.54
        plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        for x_var in x_variable_list:
            shap_values_mat = shap_values[:, x_var]
            data_i = shap_values_mat.data
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # x_variable_range_dict = self.x_variable_range_dict
            start, end = x_variable_range_dict[x_var]
            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                mean = np.nanmean(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)
            #     err,_,_ = self.uncertainty_err(SM)
            # print(df_i)
            # exit()
            plt.subplot(5, 4, flag)
            plt.scatter(data_i, value_i, alpha=0.9, c='gray', marker='.', s=4, zorder=-1)
            # print(data_i[0])
            # exit()
            # interp_model = interpolate.interp1d(x_mean_list, y_mean_list, kind='cubic')
            # y_interp = interp_model(x_mean_list)
            plt.plot(x_mean_list, y_mean_list, c='red', alpha=1)

            # exit()
            # plt.fill_between(x_mean_list, np.array(y_mean_list) - np.array(y_err_list), np.array(y_mean_list) + np.array(y_err_list), alpha=0.3,color='red')
            plt.xlabel(x_var)
            flag += 1
            plt.ylim(-1, 1)

        plt.suptitle(y_variable)
        plt.tight_layout()
        plt.show()
        # plt.savefig(outf,dpi=300)
        # plt.close()

    def feature_importances_shap_values(self,shap_values, features):
        '''
        Prints the feature importances based on SHAP values in an ordered way
        shap_values -> The SHAP values calculated from a shap.Explainer object
        features -> The name of the features, on the order presented to the explainer
        '''
        # Calculates the feature importance (mean absolute shap value) for each feature
        importances = []
        for i in range(shap_values.values.shape[1]):
            importances.append(np.mean(np.abs(shap_values.values[:, i])))
        # Calculates the normalized version
        # importances_norm = softmax(importances)
        # Organize the importances and columns in a dictionary
        feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        # feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
        # Sorts the dictionary
        feature_importances = {k: v for k, v in
                               sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
        # feature_importances_norm = {k: v for k, v in
        #                             sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
        # Prints the feature importances
        # for k, v in feature_importances.items():
        #     print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

        return feature_importances
        # return feature_importances_norm

    def __select_extreme(self,df):
        df = df[df['T_max'] > 1]
        df = df[df['intensity'] < -2]
        return df
    def __train_model(self,X,y):
        from sklearn.model_selection import train_test_split
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.3) # split the data into training and testing
        # model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=7,) # build a random forest model
        # rf.fit(X_train, y_train) # train the model
        # r2 = rf.score(X_test,y_test)
        model = xgb.XGBRegressor(objective="reg:squarederror",booster='gbtree',n_estimators=100,
                                 max_depth=13,eta=0.05,random_state=42,n_jobs=12)
        # model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=12)
        model.fit(X_train, y_train)
        # model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X_test)
        plt.scatter(y_test, y_pred)
        plt.show()
        r = stats.pearsonr(y_test, y_pred)
        r2 = r[0] ** 2
        print('r2:', r2)
        # exit()

        return model,y,y_pred

    def __train_model_RF(self,X,y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, random_state=1, test_size=0.) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=7) # build a random forest model
        rf.fit(X, y) # train the model
        coef = rf.feature_importances_
        imp_dict = {}
        for i in range(len(coef)):
            imp_dict[self.x_variable_list[i]] = coef[i]

        return imp_dict

    def benchmark_model(self,y,y_pred):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        plt.scatter(y, y_pred)
        plt.plot([0.6, 1.2], [0.6, 1.2], color='r', linestyle='-', linewidth=2)
        plt.ylabel('Predicted', size=20)
        plt.xlabel('Actual', size=20)
        plt.xlim(0.6, 1.2)
        plt.ylim(0.6, 1.2)
        plt.show()

class SHAP_pixel:
    def __init__(self):
        self.outdir = rf'E:\Project5\Result\SHAP_pixel\\'
        T.mk_dir(self.outdir, force=True)

        self.dff = rf'E:\Project5\Result\Dataframe\raw_data.df'
        self.y_variable_list = ['LAI4g_raw']
        # self.x_variable_list = ['CO2_raw','CRU_raw','tmax_raw']
        self.x_variable_list = ['CO2_raw', 'CRU_raw']
        self.x_variable_range_dict= {
            'CO2_raw': [340, 450],
            'CRU_raw': [0,1000],
            'tmax_raw': [0, 40],
            'ENSO_average': [-2, 2],
        }

    def run(self):
        # self.check_variables_ranges()
        self.shape_pixel()
        # self.plot_pixel_shapely()

        # self.plot_pixel_shapely_fixed()


        pass

    def check_variables_ranges(self):

        dff = self.dff
        df = T.load_df(dff)
        df=self.df_clean(df)
        df=self.plot_hist(df)

        # df = self.__select_extreme(df)
        # T.print_head_n(df)
        # exit()

        x_variable_list = self.x_variable_list
        print(len(x_variable_list))
        # exit()
        flag = 1
        for var in x_variable_list:
            print(flag,var)
            vals = df[var].tolist()
            plt.subplot(4,4,flag)
            flag += 1
            plt.hist(vals,bins=100)
            plt.title(var)
        plt.tight_layout()
        plt.show()


    def plot_hist(self,df):
        # T.print_head_n(df)
        # exit()
        x_variable_list = self.x_variable_list
        # print(x_variable_list)
        # exit()
        for var in x_variable_list:


            vals = df[var].tolist()
            vals = np.array(vals)
            # vals[vals<-500] = np.nan
            # vals[vals>500] = np.nan
            # vals = vals[~np.isnan(vals)]
            plt.hist(vals,bins=100)
            plt.title(var)
            plt.show()
        exit()
        return df
    def shape_pixel(self):

        dff = self.dff
        df = T.load_df(dff)
        df = self.df_clean(df)
        pix_list = T.get_df_unique_val_list(df, 'pix')
        spatial_dict = {}
        for pix in pix_list:
            spatial_dict[pix] = 1
        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.show()
        # ## plot spatial df
        # T.print_head_n(df)
        #
        group_dic = T.df_groupby(df, 'pix')
        # spatial_dict = {}
        # for pix in group_dic:
        #     df_pix = group_dic[pix]
        #     spatial_dict[pix] = len(df_pix)
        # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.show()



        outdir = join(self.outdir, 'raw_shapely_for_each_pixel_ENSO')
        T.mk_dir(outdir, force=True)

        for y_var in self.y_variable_list:
            shap_values_dict = {}
            model_dict = {}

            for pix in tqdm(group_dic):
                df_pix = group_dic[pix][0:19]



                x_variable_list = self.x_variable_list
                ## extract the data[1:]
                df_new = df_pix.dropna(subset=[y_var] + self.x_variable_list, how='any')
                if len(df_new) < 20:
                    continue
                X = df_new[x_variable_list]
                Y = df_new[y_var]
                # T.print_head_n(df_new)
                model,y,y_pred = self.__train_model(X, Y)

                explainer = shap.TreeExplainer(model)

                y_base = explainer.expected_value
                shap_values = explainer(X)
                shap_values_dict[pix] = shap_values
                model_dict[pix] = model
            T.save_dict_to_binary(shap_values_dict, join(outdir, y_var + '.shap'))
            # T.save_dict_to_binary(model_dict, join(outdir, y_var + '.model'))

    def plot_pixel_shapely(self):
        fdir=rf'E:\Project5\Result\SHAP_pixel\raw_shapely_for_each_pixel_ENSO\\'

        f=join(fdir,'LAI4g_raw.shap.pkl')
        shap_values_dict = T.load_dict_from_binary(f)
        R2_f = rf'E:\Project5\Result\RF_pix\raw_importance_for_each_pixel_ENSO\\LAI4g_raw_R2.tif'
        array_R, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(R2_f)
        dic_r2 = DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(array_R)
        f_cluster = rf'E:\Project5\Result\RF_pix\\spatial_distribution.tif'
        arr_cluster, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_cluster)
        dic_cluster = DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(arr_cluster)
        for pix in shap_values_dict:
            if not dic_cluster[pix] == 1:
                continue
            R2=dic_r2[pix]
            if R2<0.4:
                continue
            fig = plt.figure(figsize=(15, 5))
            flag = 1
            for x_var in self.x_variable_list:
                ax = fig.add_subplot(1, 3, flag)
                shap_values = shap_values_dict[pix]
                shap_values_mat = shap_values[:, x_var]
                data_i = shap_values_mat.data
                value_i = shap_values_mat.values
                df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})


                ax = df_i.plot.scatter(x=x_var, y='shap_v', c='b', ax=ax,)
                ax=df_i.plot.scatter(x=x_var, y='shap_v', c='r', ax=ax,alpha=.3)



                flag = flag + 1


            plt.suptitle(rf'{pix} R2={R2:.2f}')

            plt.show()
    def plot_pixel_shapely_fixed(self):
        fdir=rf'E:\Project5\Result\SHAP_pixel\raw_shapely_for_each_pixel_ENSO\\'

        f=join(fdir,'LAI4g_raw.shap.pkl')
        shap_values_dict = T.load_dict_from_binary(f)
        R2_f = rf'E:\Project5\Result\RF_pix\raw_importance_for_each_pixel_ENSO\\LAI4g_raw_R2.tif'
        array_R, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(R2_f)
        dic_r2 = DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(array_R)
        f_cluster = rf'E:\Project5\Result\RF_pix\\spatial_distribution.tif'
        arr_cluster, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_cluster)
        dic_cluster = DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(arr_cluster)

        for pix in shap_values_dict:
            if not dic_cluster[pix] == 1:
                continue
            R2=dic_r2[pix]
            if R2<0.4:
                continue
            flag = 1
            shap_dict = {}
            fig = plt.figure(figsize=(15, 5))
            for x_var in self.x_variable_list:

                shap_values = shap_values_dict[pix]
                shap_values_mat = shap_values[:, x_var]
                data_i = shap_values_mat.data
                value_i = shap_values_mat.values
                shap_dict[x_var] = data_i
                shap_dict[f'{x_var}_shap'] = value_i
                # df_i = pd.DataFrame({x_var: data_i, f'{x_var}_shap': value_i})

                # ax = df_ii.plot.scatter(x=x_var, y='shap_v', c='b', ax=ax,)
            df_i = pd.DataFrame(shap_dict)
            T.print_head_n(df_i)
            df_ii = df_i[df_i['CO2_raw'] > 385]
            df_ii = df_ii[df_ii['CO2_raw'] < 395]
            for x_var in self.x_variable_list:

                ax = fig.add_subplot(1, 3, flag)
                ax=df_i.plot.scatter(x=x_var, y=rf'{x_var}_shap', c='b', ax=ax,alpha=.3)
                ax=df_ii.plot.scatter(x=x_var, y=rf'{x_var}_shap', c='r', ax=ax,)
                flag=flag+1

            plt.suptitle(rf'{pix} R2={R2:.2f}')
            plt.show()








    def __train_model(self,X,y):
        from sklearn.model_selection import train_test_split
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        
        
        '''

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.3) # split the data into training and testing
        model = RandomForestRegressor(n_estimators=100, random_state=42,) # build a random forest model
        # rf.fit(X_train, y_train) # train the model
        # r2 = rf.score(X_test,y_test)

        model.fit(X_train, y_train)

        # Get predictions
        y_pred = model.predict(X_test)
        # plt.scatter(y_test, y_pred)
        # plt.show()
        r = stats.pearsonr(y_test, y_pred)
        r2 = r[0] ** 2
        # print('r2:', r2)
        # exit()

        return model,y,y_pred

    def df_clean(self,df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]
        df=df[df['LC_max']<20]
        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['MODIS_LUCC'] != 12]

        return df




class Partial_Dependence_Plots:

    def __init__(self):
        self.this_class_png = data_root + 'SHAP\\png\\'
        self.this_class_arr = data_root + 'SHAP\\arr\\'

        self.dff = rf'D:\Project3\Data\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        self.df = T.load_df(self.dff)
        self.xvars = ['CO2_relative_change_detrended',
            'VPD_relative_change_detrended',
            'GPCC_relative_change_detrended',
            'Noy_relative_change_detrended',
            # 'CO2_relative_change',
            # 'VPD_relative_change',
            # 'GPCC_relative_change',
            # # 'Noy_relative_change',


            # 'average_heat_spell',
            # 'CO2',
            'ENSO_index_average',


            'rooting_depth',
            # 'SOC_sum',
              'silt',



            'average_dry_spell',
                                 # 'maximum_dry_spell',


                                'frequency_wet',

                                'average_anomaly_heat_event',]

        self.yvar = 'LAI4g_relative_change_detrended'



        # self.variable_list_rt()
        #
        # ##----------------------------------
        #
        # self.y_variable = 'LAI4g_relative_change_detrended'
        # ####################
        #
        # self.x_variable_list = self.x_variable_list
        # self.x_variable_range_dict = self.x_variable_range_dict_Africa

        pass

    def run(self):
        # self.importance()
        # self.run_pdp()
        self.plot_pdp()
        pass

    def importance(self):
        df = self.df_clean(self.df)
        # X = df[self.xvars]
        # Y = df[self.yvar]

        all_vars = copy.copy(self.xvars)  # copy the x variables
        all_vars.append(self.yvar)  # add the y variable to the list
        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna()  # drop rows with missing values
        X = all_vars_df[self.xvars]
        Y = all_vars_df[self.yvar]

        model, r2, importance_dict = self.__train_model(X, Y)
        pprint(importance_dict)

    def run_pdp(self):
        outdir = join(self.this_class_arr, 'pdp_shap')
        T.mk_dir(outdir, force=True)
        df = self.df_clean(self.df)
        # T.print_head_n(df)
        # exit()
        pdp_result = self.partial_dependence_plots(df, self.xvars, self.yvar)
        outf = join(outdir, 'pdp_result')
        T.save_npy(pdp_result, outf)



    def df_clean(self,df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]
        # df=df[df['LAI4g_p_value']<0.05]
        # df=df[df['LAI4g_trend']>0]
        # df=df[df['continent']=='Australia']


        df = df[df['continent'] == 'Africa']
        # df=df[df['continent']=='Asia']

        # df=df[df['continent']=='South_America']


        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        df = df[df['landcover_classfication'] != 'Cropland']



        return df
    def partial_dependence_plots(self,df,x_vars,y_var):
        '''
        :param df: a dataframe
        :param x_vars: a list of x variables
        :param y_var: a y variable
        :return:
        '''
        all_vars = copy.copy(x_vars) # copy the x variables
        all_vars.append(y_var) # add the y variable to the list
        all_vars_df = df[all_vars] # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna() # drop rows with missing values
        X = all_vars_df[x_vars]
        Y = all_vars_df[y_var]
        model, r2,importance_dict = self.__train_model(X, Y) # train a Random Forests model
        flag = 0
        result_dic = {}
        for var in x_vars:
            flag += 1
            df_PDP = self.__get_PDPvalues(var, X, model) # get the partial dependence plot values
            ppx = df_PDP[var]
            ppy = df_PDP['PDs']
            ppy_std = df_PDP['PDs_std']
            result_dic[var] = {'x':ppx,
                               'y':ppy,
                               'y_std':ppy_std,
                               'r2':r2}
        return result_dic

    def plot_pdp(self):
        from pprint import pprint
        result_dict_f = join(self.this_class_arr, 'pdp_shap\\pdp_result.npy')
        result_dict = T.load_npy(result_dict_f)
        # print(len(result_dict))
        # exit()
        flag = 1
        for key in result_dict:
            print(key)
            result_i = result_dict[key]
            pprint(result_i)
            exit()
            x = result_i['x']
            y = result_i['y']
            plt.subplot(3,4,flag)
            flag += 1
            plt.plot(x,y)
            # pprint(result_i)
            # exit()
            plt.title(key)
            plt.ylim(-4,13)
        plt.show()

    def __train_model(self,X,y):
        from sklearn.model_selection import train_test_split
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.2) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=10) # build a random forest model
        rf.fit(X_train, y_train) # train the model
        r2 = rf.score(X_test,y_test)
        coef = rf.feature_importances_
        imp_dict = {}
        for i in range(len(coef)):
            imp_dict[self.xvars[i]] = coef[i]
        return rf,r2,imp_dict


    def __get_PDPvalues(self, col_name, data, model, grid_resolution=50):
        '''
        :param col_name: a variable
        :param data: a dataframe of x variables
        :param model: a random forest model
        :param grid_resolution: the number of points in the partial dependence plot
        :return: a dataframe of the partial dependence plot values
        '''
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution) # create a sequence of values
        Y_pdp = []
        Y_pdp_std = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            Y_pdp.append(np.mean(Y_temp))
            Y_pdp_std.append(np.std(Y_temp))
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp, 'PDs_std': Y_pdp_std})





class multi_regression():
    def __init__(self):

        self.fdirX = rf'E:\Data\ERA5_daily\dict\extract_rainfall_annual\moving_window_average_anaysis\\'
        self.fdirY = rf'E:\Data\ERA5_daily\dict\extract_rainfall_annual\moving_window_average_anaysis\\'



        self.y_var = ['LAI4g_CV']
        self.xvar = [ 'maxmum_dry_spell', 'GPCC', 'CV_rainfall', 'wet_frequency_90th','peak_rainfall_timing']

        self.multi_regression_result_dir = rf'E:\Data\ERA5_daily\dict\extract_rainfall_annual\multi_regression\\'
        T.mk_dir(self.multi_regression_result_dir, force=True)

        self.multi_regression_result_f = rf'E:\Data\ERA5_daily\dict\extract_rainfall_annual\multi_regression\multi_regression_result.npy'

        pass

    def run(self):

        # step 1 build dataframe

        # df=self.build_df(self.fdirX, self.fdirY,self.xvar,self.y_var)

        # # # step 2 cal correlation
        # self.cal_multi_regression_beta()

        # step 3 plot
        self.plt_multi_regression_result(self.multi_regression_result_f,self.y_var[0])

        ## step 4 convert m2/m2/ppm to %/100ppm
        # self.convert_CO2_sensitivity_unit()

        # step 5
        # self.calculate_trend_contribution()

        pass

    def build_df(self, fdir_X, fdir_Y, fx_list, fy):

        df = pd.DataFrame()

        filey = fdir_Y + fy[0] + '.npy'
        print(filey)

        dic_y = T.load_npy(filey)
        # array=np.load(filey)
        # dic_y=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        y_val_list = []

        for pix in dic_y:
            yvals = dic_y[pix]

            if len(yvals) == 0:
                continue
            yvals = T.interp_nan(yvals)
            yvals = np.array(yvals)
            if yvals[0] == None:
                continue

            pix_list.append(pix)
            y_val_list.append(yvals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        # build x

        for xvar in fx_list:

            # print(var_name)
            x_val_list = []
            filex = fdir_X + xvar + '.npy'
            # filex = fdir_X + xvar + f'_{period}.npy'

            # print(filex)
            # exit()
            # x_arr = T.load_npy(filex)
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
        ## save df
        T.save_df(df, self.multi_regression_result_dir + fy[0] + '.df')
        T.df_to_excel(df, self.multi_regression_result_dir + fy[0] + '.xlsx')

        return df

    def __linearfit(self, x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
        for i in range(0, int(N)):
            sx += x[i]
            sy += y[i]
            sxx += x[i] * x[i]
            syy += y[i] * y[i]
            sxy += x[i] * y[i]
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
        b = (sy - a * sx) / N
        r = -(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
        return a, b, r

    def cal_multi_regression_beta(self):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd
        import joblib

        df = T.load_df(self.multi_regression_result_dir + 'LAI4g_CV.df')

        x_var_list = self.xvar

        outf = self.multi_regression_result_f

        multi_derivative = {}

        for i, row in tqdm(df.iterrows(), total=len(df)):
            # print(row);exit()
            pix = row.pix

            y_vals = row['y'][0:-1]
            # y_vals = T.remove_np_nan(y_vals)
            # y_vals = T.interp_nan(y_vals)
            if len(y_vals) == 0:
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

                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue

                df_new[x] = x_vals

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue
            if len(x_var_list_valid) < 2:
                continue
            # T.print_head_n(df_new)

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
            ## build multiregression model and consider interactioon

            model = smf.ols(formula='y ~ ' + '+'.join(x_var_list_valid_new), data=df_new).fit()

            coef_ = np.array(model.params)
            coef_dic = dict(zip(x_var_list_valid_new, coef_))
            # print(df_new['y'])
            # exit()
            multi_derivative[pix] = coef_dic
        T.save_npy(multi_derivative, outf)

    pass

    def plt_multi_regression_result(self, multi_regression_result_dir, y_var):

        NDVI_mask_f = rf'D:/Project3/Data/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = rf'D:/Project3/Data//Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f =  rf'D:/Project3/Data//Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        f = self.multi_regression_result_f

        dic = T.load_npy(f)
        var_list = []
        for pix in dic:


            vals = dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            # print(var_i)
            spatial_dic = {}
            for pix in dic:
                r, c = pix
                if r < 120:
                    continue

                landcover_value = crop_mask[pix]

                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                dic_i = dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            arr = arr * array_mask
            print(var_i)
            # plt.imshow(arr)
            # plt.colorbar()
            # plt.show()
            outf = self.multi_regression_result_dir + rf'\\{var_i}.tif'

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, outf)




    def convert_CO2_sensitivity_unit(self):
        period_list = ['1982_2020']
        for period in period_list:
            CO2_sensitivity_f = result_root + rf'multi_regression\\anomaly\\{period}\\CO2_LAI4g_{period}.tif'
            average_LAI4g_f = result_root + rf'\state_variables\\\\LAI4g_{period}.npy'
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(CO2_sensitivity_f)
            arr[arr < -99] = np.nan
            dic_CO2_sensitivity = DIC_and_TIF().spatial_arr_to_dic(arr)

            dic_LAI4g_average = T.load_npy(average_LAI4g_f)

            for pix in dic_CO2_sensitivity:
                CO2_sensitivity = dic_CO2_sensitivity[pix]
                CO2_sensitivity = np.array(CO2_sensitivity, dtype=float)
                if np.isnan(CO2_sensitivity):
                    continue
                if not pix in dic_LAI4g_average:
                    continue
                LAI_average = dic_LAI4g_average[pix]
                LAI_average = np.array(LAI_average, dtype=float)

                if np.isnan(LAI_average):
                    continue
                CO2_sensitivity = CO2_sensitivity / LAI_average * 100
                if CO2_sensitivity < -99999:
                    continue
                dic_CO2_sensitivity[pix] = CO2_sensitivity
            arr_new = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(dic_CO2_sensitivity)
            arr_new[arr_new < -99] = np.nan
            arr_new[arr_new > 99] = np.nan

            # plt.imshow(arr_new)
            # plt.colorbar()
            # plt.show()

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_new, f'{CO2_sensitivity_f.replace(".tif", "_scale.tif")}')
            # DIC_and_TIF(pixelsize=0.25).pix_dic_to_tif(dic_CO2_sensitivity, f'{CO2_sensitivity_f.replace(".tif","_new.tif")}')
            # T.save_npy(dic_CO2_sensitivity, CO2_sensitivity_f.replace('.tif', '.npy'))

    def calculate_trend_contribution(self):
        ## here I would like to calculate the trend contribution of each variable
        ## the trend contribution is defined as the slope of the linear regression between the variable and the target variable mutiplied by trends of the variable
        ## load the trend of each variable
        ## load the trend of the target variable
        ## load multi regression result
        ## calculate the trend contribution
        trend_dir = result_root + rf'\trend_analysis\anomaly\OBS_extend\\'

        selected_vairables_list = [
            'CRU_trend',
            'CO2_trend',
            'tmax_trend',
            'VPD_trend',
        ]

        trend_dict = {}
        for variable in selected_vairables_list:
            fpath = join(trend_dir, f'{variable}.npy')
            array = np.load(fpath, allow_pickle=True)
            array[array < -9999] = np.nan
            spatial_dict = D.spatial_arr_to_dic(array)
            for pix in tqdm(spatial_dict, desc=variable):
                r, c = pix
                if r < 120:
                    continue
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue
                if not pix in trend_dict:
                    trend_dict[pix] = {}
                key = variable.replace('_trend', '')
                trend_dict[pix][key] = spatial_dict[pix]

        f = self.multi_regression_result_f
        print(f)
        print(isfile(f))
        # exit()
        dic_multiregression = T.load_npy(f)
        var_list = []
        for pix in dic_multiregression:

            # landcover_value = crop_mask[pix]
            # if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
            #     continue

            vals = dic_multiregression[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        # print(var_list)
        # exit()
        for var_i in var_list:
            spatial_dic = {}
            for pix in dic_multiregression:
                if not pix in trend_dict:
                    continue

                dic_i = dic_multiregression[pix]
                if not var_i in dic_i:
                    continue
                val_multireg = dic_i[var_i]
                val_trend = trend_dict[pix][var_i]
                val_contrib = val_multireg * val_trend
                spatial_dic[pix] = val_contrib
            arr_contrib = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr_contrib, cmap='RdBu', interpolation='nearest')
            plt.colorbar()
            plt.title(var_i)
            plt.show()
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_contrib,
                                                   f'{self.multi_regression_result_dir}\\{var_i}_trend_contribution.tif')



class Partial_correlation:
    def __init__(self):
        self.data_root = 'D:/Project3/Data/'

        pass
    def run(self):
        # self.partial_correlation()
        # self.plot_partial_correlation()
        self.cal_max_correlation()

        pass
    def partial_correlation(self):


        f_y = rf'E:\Data\ERA5_daily\dict\\moving_window_average_anaysis\\detrended_annual_LAI4g_CV.npy'
        x_var_list=self.xvar = [ 'maxmum_dry_spell', 'GPCC', 'wet_frequency_90th','peak_rainfall_timing']

        f_x_dir = rf'E:\Data\ERA5_daily\dict\moving_window_average_anaysis\\'
        spatial_dic = {}

        dic_y=T.load_npy(f_y)

        for xvar in x_var_list:

            f_x = f_x_dir + f'{xvar}.npy'
            dic_x=T.load_npy(f_x)

            spatial_dic[xvar] = dic_x
        spatial_dic['y'] = dic_y
        ## spatial to df
        df=T.spatial_dics_to_df(spatial_dic)
        T.print_head_n(df)
        df=df.dropna(axis=0,subset=['y'])

        partial_correlation_dic={}
        partial_p_value_dic={}



        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            y_vals = row['y'][0:23]
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:

                x_vals = row[x][0:23]

                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue

                if len(x_vals) != len(y_vals):
                    continue
                # print(x_vals)
                if x_vals[0] == None:
                    continue

                df_new[x] = x_vals

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue
            if len(x_var_list_valid) < 2:
                continue
            # T.print_head_n(df_new)

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

            partial_correlation = {}
            partial_correlation_p_value = {}
            for x in x_var_list_valid_new:
                x_var_list_valid_new_cov = copy.copy(x_var_list_valid_new)
                x_var_list_valid_new_cov.remove(x)
                r, p = self.partial_corr(df_new, x, 'y', x_var_list_valid_new_cov)
                partial_correlation[x] = r
                partial_correlation_p_value[x] = p

            partial_correlation_dic[pix] = partial_correlation
            partial_p_value_dic[pix] = partial_correlation_p_value
        outdir=rf'E:\Data\ERA5_daily\dict\partial_correlation\\'
        T.mk_dir(outdir,force=True)
        outf_corr = outdir + 'partial_correlation.npy'
        outf_pvalue = outdir + 'partial_correlation_pvalue.npy'


        T.save_npy(partial_correlation_dic, outf_corr)
        T.save_npy(partial_p_value_dic, outf_pvalue)

    def partial_corr(self, df, x, y, cov):
        import pingouin as pg
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

    def plot_partial_correlation(self):

        landcover_f = self.data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = self.data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = rf'E:\Data\ERA5_daily\dict\partial_correlation\\'
        f_partial = fdir + 'partial_correlation.npy'
        f_pvalue = fdir + 'partial_correlation_pvalue.npy'


        partial_correlation_dic = np.load(f_partial, allow_pickle=True, encoding='latin1').item()
        partial_correlation_p_value_dic = np.load(f_pvalue, allow_pickle=True, encoding='latin1').item()


        var_list = []
        for pix in partial_correlation_dic:

            r, c = pix


            if r < 120:
                continue
            landcover_value = crop_mask[pix]
            if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                continue
            if dic_modis_mask[pix] == 12:
                continue

            vals = partial_correlation_dic[pix]
            for var_i in vals:
                var_list.append(var_i)
        var_list = list(set(var_list))
        for var_i in var_list:
            spatial_dic = {}
            for pix in partial_correlation_dic:

                dic_i = partial_correlation_dic[pix]
                if not var_i in dic_i:
                    continue
                val = dic_i[var_i]
                spatial_dic[pix] = val
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            # arr = arr * array_mask
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr, fdir + f'partial_corr_{var_i}.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr, vmin=-1, vmax=1)

            plt.title(var_i)
            plt.colorbar()

        plt.show()

    def cal_max_correlation(self):
        fdir = rf'E:\Data\ERA5_daily\dict\partial_correlation\\'
        for f in os.listdir(fdir):
            if not f.endswith('partial_correlation.npy'):
                continue

            f_partial = fdir + f
            partial_correlation_dic = np.load(f_partial, allow_pickle=True, encoding='latin1').item()

            result_dic_val = {}
            result_dic_var = {}
            for pix in partial_correlation_dic:
                r,c=pix
                if r<120:
                    continue
                val_list = []
                vals = partial_correlation_dic[pix]
                for var_i in vals:
                    val_list.append(abs(vals[var_i]))

                max_val = max(val_list)
                max_var = val_list.index(max_val)
                result_dic_val[pix] = max_val
                result_dic_var[pix] = max_var

            np.save(fdir + 'max_correlation.npy', result_dic_val)
            np.save(fdir + 'max_correlation_var.npy', result_dic_var)
            arr_val=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic_val)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_val,fdir+'max_correlation.tif')
            arr_var = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic_var)
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr_var, fdir + 'max_correlation_var.tif')




        pass


class SHAP_CV():

    def __init__(self):
        self.y_variable = 'GIMMS_plus_NDVI_detrend_CV'

        # self.this_class_png = results_root + 'ERA5\\SHAP\\png\\'
        self.threshold = '3mm'
        self.this_class_png = results_root + rf'\{self.threshold}\SHAP\\RF_{self.y_variable}\\'

        # self.dff = rf'E:\Project3\Result\3mm\ERA5\Dataframe\moving_window\\moving_window.df'
        self.dff = results_root+rf'{self.threshold}\Dataframe\moving_window_CV\\moving_window_CV_new.df'
        self.variable_list_rt()
        self.variables_list = ['LAI4g', 'NDVI','CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',
                          'SDGVM_S2_lai',
                          'YIBs_S2_Monthly_lai']

        ##----------------------------------



        ####################

        self.x_variable_list = self.x_variable_list_CRU
        self.x_variable_range_dict = self.x_variable_range_dict_global_CRU

        pass

    def run(self):
        # self.check_df_attributes()

        # self.check_variables_ranges()
        # self.show_colinear()
        # self.check_spatial_plot()
        self.pdp_shap()
        # self.plot_pdp_shap()
        # self.plot_pdp_shap_density_cloud()
        # self.plot_pdp_shap_density_cloud_individual()  ## for paper use
        # self.plot_relative_importance()
        # self.plot_pdp_shap_all_models_SI()
        # self.plot_pdp_shap_all_models_main()
        # self.plot_heatmap_ranking()
        # self.plot_interaction_manual()
        # self.spatial_shapely()   ### spatial plot
        # self.variable_contributions()
        # self.plot_dominant_factors_bar()
        # self.max_contributions()
        # self.plot_pdp_shap_normalized()
        # self.pdp_shap_trend()

        pass

    def check_df_attributes(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print(df.columns.tolist())
        print(len(df))
        # exit()
        pass

    def check_variables_ranges(self):

        dff = self.dff
        df = T.load_df(dff)
        df = self.df_clean(df)

        df = self.plot_hist(df)
        df = self.valid_range_df(df)
        # df = self.__select_extreme(df)
        # T.print_head_n(df)
        # exit()

        x_variable_list = self.x_variable_list
        print(len(x_variable_list))
        # exit()
        flag = 1

        for var in x_variable_list:
            print(flag, var)
            vals = df[var].tolist()
            plt.subplot(4, 4, flag)
            flag += 1
            plt.hist(vals, bins=100)
            plt.title(var)
        plt.tight_layout()
        plt.show()


        pass

    def AIC_criteria(self):

        import statsmodels.api as sm

        pass
    def variable_list_rt(self):
        self.x_variable_list = [

            # 'detrended_average_annual_tmax',
            # 'heavy_rainfall_days',

            # 'rainfall_frequency',
            'rainfall_intensity',
            'maxmum_dry_spell',
            'rainfall_seasonality_all_year',
            # 'rainfall_seasonality',
            'detrended_sum_rainfall_interannual_CV',
            # 'CV_rainfall_interseasonal',
            # 'Aridity',
            # 'CO2_gridded',

            'heat_event_frequency',
            'silt',
            # 'rooting_depth',



            # # 'heavy_rainfall_days',


        ]


        self.x_variable_list_CRU = [
            # 'nitrogen_0-5cm_mean_5000_05',

            # 'nitrogen',
            # 'zroot_cwd80_05',
            'cwdx80_05',
            # 'cec',
            'sand',
             # 'soc',
            # 'S_CEC_CLAY',
            # 'AWC_CLASS',
            'rainfall_intensity',
            # 'sum_rainfall',

            'rainfall_frenquency',

            'rainfall_seasonality_all_year',

            'detrended_sum_rainfall_CV',

                # 'Aridity',
                # 'CO2',
                'heat_event_frenquency',
                 # 'Tmax',
            # 'VPD',
            # 'SOC',
            # 'S_SAND',
            #  'T_SAND',
            # 'T_CLAY',
            # 'rooting_depth',


            ]
        self.x_variable_range_dict_global = {
            'CO2_ecosystem_year': [350, 410],
            'detrended_average_annual_tmax': [-10, 40],
            'detrended_sum_rainfall_growing_season_CV_ecosystem_year': [0, 70],

            'detrended_sum_rainfall_std': [0, 250],
            'detrended_sum_rainfall': [0, 1000],
            'CV_rainfall_interseasonal': [100, 600],
            'detrended_sum_rainfall_interannual_CV': [0, 70],


            'rainfall_seasonality': [0, 10],  # rainfall_seasonality


            'sum_rainfall': [0, 1500],
            'CO2_gridded': [350, 410],
            'CO2': [350, 410],
            'Aridity': [0, 1],

            'heat_event_frenquency_growing_season': [0, 6],




            'maxmum_dry_spell': [0, 200],  # maxmum_dry_spell
            'rainfall_frequency': [0, 200],  # rainfall_frequency
            'rainfall_intensity': [0, 5],  # rainfall_intensity
            'rainfall_seasonality_all_year': [0, 25],  #
            'heavy_rainfall_days': [0, 50],
            'T_sand': [20, 90],
            'rooting_depth': [0, 30],

        }

        self.x_variable_range_dict_global_CRU = {
            'nitrogen': [0, 500],
            'zroot_cwd80_05': [0, 25000],
            'cwdx80_05': [0, 1000],
            'cec': [0, 400],
            'sand': [0, 900],
            'soc': [0, 600],



            'CO2': [350, 410],
            'sum_rainfall': [0, 1500],
            'S_CEC_CLAY': [0, 100],
            'AWC_CLASS': [0, 7],
            'dry_spell': [0, 20],
            'rainfall_intensity': [0, 25],
            'rainfall_frenquency': [0, 100],
    'rainfall_seasonality_all_year': [15, 80],

            'detrended_sum_rainfall_CV':[0,60],

    'heat_event_frenquency': [0, 3],
            'S_SAND': [0, 100],
            'T_SAND': [20, 90],
            'T_CLAY': [0, 60],
            'Tmax': [0, 40],
            'VPD': [0, 4],
            'SOC': [0, 1],
            'rooting_depth': [0, 25],

    }

    def show_colinear(self, ):
        dff = self.dff
        df = T.load_df(dff)
        vars_list = self.x_variable_list
        df = df[vars_list]
        ## add LAI4g_raw
        df['LAI4g_detrend_CV'] = T.load_df(dff)['LAI4g_detrend_CV']
        ## plot heat map to show the colinear variables

        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': 'Rainfall frequency (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': 'Heat event frequency (events/year)',
                    'cwdx80_05': 'Rooting zone water storage capacity (mm)',
                    'sand': 'Sand (g/kg)',

                    }

        import seaborn as sns
        fig, ax=plt.subplots(figsize=(8, 5))
        ### x tick label rotate
        vmin = -1
        vmax = 1

        sns.heatmap(df.corr(), annot=True, fmt=".2f",vmin=vmin, vmax=vmax,cmap="RdBu")
        plt.xticks(rotation=45)
        ax.set_yticks(np.arange(len(vars_list)) + 0.5)
        # ax.set_yticklabels(model_list[::-1], rotation=0, va='center')
        ##get name from dic
        # ax.set_yticklabels([name_dic[x] for x in vars_list], rotation=0, va='center')
        #
        # ax.set_xticks(np.arange(len(vars_list)) + 0.5)
        # ax.set_xticklabels([name_dic[x] for x in vars_list], rotation=45, ha='center')
        # ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

    def discard_vif_vars(self, df, x_vars_list):
        ##################实时计算#####################
        vars_list_copy = copy.copy(x_vars_list)

        X = df[vars_list_copy]
        X = X.dropna()
        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif.round(1)
        selected_vif_list = []
        for i in range(len(vif)):
            feature = vif['features'][i]
            VIF_val = vif['VIF Factor'][i]
            if VIF_val < 5.:
                selected_vif_list.append(feature)
        return selected_vif_list

        pass

    def plot_hist(self, df):
        # T.print_head_n(df)
        # exit()
        x_variable_list = self.x_variable_list
        ## combine x and y
        all_list = copy.copy(x_variable_list)
        all_list.append(self.y_variable)
        # print(all_list)
        # exit()
        for var in all_list:
            vals = df[var].tolist()
            vals = np.array(vals)
            # vals[vals<-500] = np.nan
            # vals[vals>500] = np.nan
            # vals = vals[~np.isnan(vals)]
            plt.hist(vals, bins=100)
            plt.title(var)
            plt.show()
        exit()
        return df

    def valid_range_df(self, df):

        print('original len(df):', len(df))
        for var in self.x_variable_list_CRU:

            if not var in df.columns:
                print(var, 'not in df')
                continue
            min, max = self.x_variable_range_dict[var]
            df = df[(df[var] >= min) & (df[var] <= max)]
        print('filtered len(df):', len(df))
        return df

    def df_clean(self, df):
        # T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        # df = df[df['row'] > 120]
        df = df[df['Aridity'] < 0.65]
        df=df[df['LC_max']<20]
        print(len(df))
        # df = df[df['LAI4g_detrend_CV_p_value'] < 0.05]
        # print(len(df))
        # exit()

        df = df[df['MODIS_LUCC'] != 12]

        # #
        # df = df[df['lon'] > -125]
        # df = df[df['lon'] < -105]
        # df = df[df['lat'] > 0]
        # df = df[df['lat'] < 45]
        # print(len(df))

        df = df[df['landcover_classfication'] != 'Cropland']

        return df
    def check_spatial_plot(self):

        dff = self.dff
        df=T.load_df(dff)
        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        region_arr = DIC_and_TIF(pixelsize=.5).pix_dic_to_spatial_arr(unique_pix_list)
        plt.imshow(region_arr, cmap='jet', vmin=1, vmax=3,interpolation='nearest')
        plt.colorbar()
        plt.show()

    def pdp_shap(self):

        dff = self.dff
        outdir = join(self.this_class_png, 'pdp_shap_CV')

        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list_CRU

        y_variable = self.y_variable
        # plt.hist(T.load_df(dff)[y_variable].tolist(),bins=100)
        # plt.show()
        df = T.load_df(dff)
        df = self.df_clean(df)
        # df = self.valid_range_df(df)

        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic={}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='jet',interpolation='nearest')
        # plt.colorbar()
        # plt.show()



        T.print_head_n(df)
        print(len(df))
        T.print_head_n(df)
        print('-' * 50)
        ## text select df the first 1000

        # df = df[0:1000]
        # exit()
        # model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        # all_vars_vif = self.discard_vif_vars(df, x_variable_list)
        # all_vars_vif.append('CV_rainfall')
        # print('all_vars_vif:',all_vars_vif)
        # exit()
        # df = self.valid_range_df(df)
        all_vars = copy.copy(x_variable_list)


        all_vars.append(y_variable)  # add the y variable to the list
        all_vars.append('pix')


        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna(subset=x_variable_list, how='any')
        all_vars_df = all_vars_df.dropna(subset=self.y_variable, how='any')

        ## for plot use not training
        ## I want to add CO2 into new df but using all_vars_df to selected from df
        ## so that all_vars_df can be used for future ploting
        # all_vars_df_CO2 = copy.copy(all_vars_df)
        # merged = pd.merge(all_vars_df_CO2, df[["pix", "Aridity"]], on="pix", how="left")
        # T.save_df(merged, join(outdir, 'all_vars_df_aridity.df'))
        # exit()



        ######


        pix_list = all_vars_df['pix'].tolist()
        # print(len(pix_list));exit()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}
        #
        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()


        X = all_vars_df[x_variable_list]

        Y = all_vars_df[y_variable]
        train_data_X_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.X.df')
        train_data_y_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.y.df')

        # exit()

        ## save selected df for future ploting
        T.print_head_n(X)
        # X = X.dropna()
        # print(len(X));exit()



        model, y, y_pred = self.__train_model(X, Y)  # train a Random Forests model
        imp_dict_xgboost = {}
        for i in range(len(x_variable_list)):
            imp_dict_xgboost[x_variable_list[i]] = model.feature_importances_[i]
        #     plt.barh(x_variable_list[i], model.feature_importances_[i])
        # plt.show()
        sorted_imp = sorted(imp_dict_xgboost.items(), key=lambda x: x[1], reverse=True)

        x_ = []
        y_ = []
        for key, value in sorted_imp:
            x_.append(key)
            y_.append(value)
        print(x_)
        plt.figure()
        plt.bar(x_, y_)
        plt.xticks(rotation=45)
        # plt.tight_layout()
        plt.title('RF')
        plt.show()
        # exit()
        # plt.figure()

        ## random sample

        sample_indices = np.random.choice(X.shape[0], 40000, replace=False)
        X_sample = X.iloc[sample_indices]
        explainer = shap.TreeExplainer(model)


        # ### round R2

        # # x_variable_range_dict = self.x_variable_range_dict
        # y_base = explainer.expected_value
        # print('y_base', y_base)
        # print('y_mean', np.mean(y))
        # shap_values = explainer.shap_values(X) ##### not use!!!
        shap_values = explainer(X_sample)
        outf_shap = join(outdir, self.y_variable + '.shap')
        # ## how to resever X and Y before the shap
        #


        T.save_dict_to_binary(shap_values, outf_shap)
        ## save model

        T.save_dict_to_binary(model, join(outdir, self.y_variable + '.model'))
        # exit()
    def pdp_shap_significant(self):
        ##here

        dff = self.dff
        outdir = join(self.this_class_png, 'pdp_shap_CV')

        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list_CRU

        y_variable = self.y_variable
        # plt.hist(T.load_df(dff)[y_variable].tolist(),bins=100)
        # plt.show()
        df = T.load_df(dff)
        df = self.df_clean(df)
        df = self.valid_range_df(df)

        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic={}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='jet',interpolation='nearest')
        # plt.colorbar()
        # plt.show()



        T.print_head_n(df)
        print(len(df))
        T.print_head_n(df)
        print('-' * 50)
        ## text select df the first 1000

        # df = df[0:1000]
        # exit()
        # model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        # all_vars_vif = self.discard_vif_vars(df, x_variable_list)
        # all_vars_vif.append('CV_rainfall')
        # print('all_vars_vif:',all_vars_vif)
        # exit()
        # df = self.valid_range_df(df)
        all_vars = copy.copy(x_variable_list)


        all_vars.append(y_variable)  # add the y variable to the list
        all_vars.append('pix')


        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna(subset=x_variable_list, how='any')

        ## for plot use not training
        ## I want to add CO2 into new df but using all_vars_df to selected from df
        ## so that all_vars_df can be used for future ploting
        # all_vars_df_CO2 = copy.copy(all_vars_df)
        # merged = pd.merge(all_vars_df_CO2, df[["pix", "Aridity"]], on="pix", how="left")
        # T.save_df(merged, join(outdir, 'all_vars_df_aridity.df'))
        # exit()



        ######


        pix_list = all_vars_df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}
        #
        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()


        X = all_vars_df[x_variable_list]

        Y = all_vars_df[y_variable]
        train_data_X_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.X.df')
        train_data_y_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.y.df')

        # exit()

        ## save selected df for future ploting
        T.print_head_n(X)
        # X = X.dropna()
        # print(len(X));exit()


        model, y, y_pred = self.__train_model(X, Y)  # train a Random Forests model
        imp_dict_xgboost = {}
        for i in range(len(x_variable_list)):
            imp_dict_xgboost[x_variable_list[i]] = model.feature_importances_[i]
        #     plt.barh(x_variable_list[i], model.feature_importances_[i])
        # plt.show()
        sorted_imp = sorted(imp_dict_xgboost.items(), key=lambda x: x[1], reverse=True)

        x_ = []
        y_ = []
        for key, value in sorted_imp:
            x_.append(key)
            y_.append(value)
        print(x_)
        plt.figure()
        plt.bar(x_, y_)
        plt.xticks(rotation=45)
        # plt.tight_layout()
        plt.title('RF')
        plt.show()
        # exit()
        # plt.figure()

        ## random sample

        sample_indices = np.random.choice(X.shape[0], 40000, replace=False)
        X_sample = X.iloc[sample_indices]
        explainer = shap.TreeExplainer(model)


        # ### round R2

        # # x_variable_range_dict = self.x_variable_range_dict
        # y_base = explainer.expected_value
        # print('y_base', y_base)
        # print('y_mean', np.mean(y))
        # # shap_values = explainer.shap_values(X)
        shap_values = explainer(X_sample)
        outf_shap = join(outdir, self.y_variable + '.shap')
        # ## how to resever X and Y before the shap
        #


        T.save_dict_to_binary(shap_values, outf_shap)
        ## save model

        T.save_dict_to_binary(model, join(outdir, self.y_variable + '.model'))
        # exit()

    def plot_pdp_shap(self):
        x_variable_list = self.x_variable_list

        inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')
        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)

        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)
            y_list.append(imp_dict[key])
        plt.barh(x_list[::-1], y_list[::-1],color='grey',alpha=0.5)
        print(x_list)
        # plt.title(f'R2_{R2}')
        # plt.xticks(rotation=45, )
        ## set xlabel font size
        plt.xticks(fontsize=12)

        plt.title('shap')

        plt.tight_layout()

        plt.show()

        flag = 1
        centimeter_factor = 1 / 2.54
        plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))

        for x_var in x_list:
            shap_values_mat = shap_values[:, x_var]
            data_i = shap_values_mat.data
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # pprint(df_i);exit()
            df_i_random = df_i.sample(n=len(df_i) )
            df_i = df_i_random

            x_variable_range_dict = self.x_variable_range_dict
            ## redefine start, end
            start, end = self.x_variable_range_dict[x_var]



            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            df_i_copy = copy.copy(df_i)
            df_i_copy = df_i_copy[df_i_copy[x_var]>start]
            df_i_copy = df_i_copy[df_i_copy[x_var]<end]
            scatter_x_list = df_i_copy[x_var].tolist()
            scatter_y_list = df_i_copy['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)
            #     err,_,_ = self.uncertainty_err(SM)
            # print(df_i)
            # exit()
            plt.subplot(4, 3, flag)
            plt.scatter(scatter_x_list, scatter_y_list, alpha=0.2, c='gray', marker='.', s=1, zorder=-1)
            # print(data_i[0])
            # exit()
            # interp_model = interpolate.interp1d(x_mean_list, y_mean_list, kind='cubic')
            # y_interp = interp_model(x_mean_list)
            y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
            plt.plot(x_mean_list, y_mean_list, c='red', alpha=1)

            # exit()
            # plt.fill_between(x_mean_list, np.array(y_mean_list) - np.array(y_err_list), np.array(y_mean_list) + np.array(y_err_list), alpha=0.3,color='red')
            #### rename x_label remove

            plt.xlabel(x_var, fontsize=12)

            flag += 1
            plt.ylim(-5, 5)

        plt.suptitle(self.y_variable)

        plt.tight_layout()
        plt.show()
        # plt.savefig(outf,dpi=300)
        # plt.close()

    def plot_pdp_shap_density_cloud(self):
        x_variable_list = self.x_variable_list

        name_dic={'rainfall_intensity':'Rainfall intensity (mm/events)',
                  'rainfall_frenquency':'Rainfall frequency (events/year)',
                  'rainfall_seasonality_all_year':'Rainfall seasonality (unitless)',
                  'detrended_sum_rainfall_CV':r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                  'heat_event_frenquency':'Heat event frequency (events/year)',
                  'cwdx80_05':'Rooting zone water storage capacity (mm)',

                  'sand':'Sand (g/kg)',

        }

        # inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')
        inf_shap = rf'E:\Project3\Result\3mm\SHAP\RF_LAI4g_selected_samples_detrend_CV_\pdp_shap_CV\\LAI4g_detrend_CV_.shap.pkl'

        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)

        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)
            y_list.append(imp_dict[key])

        flag = 1
        centimeter_factor = 1 / 2.54
        # plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        fig, axs = plt.subplots(4, 2,
                                figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        # print(axs);exit()
        axs = axs.flatten()
        for x_var in x_list:
            shap_values_mat = shap_values[:, x_var]
            data_i = shap_values_mat.data
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # pprint(df_i);exit()
            df_i_random = df_i.sample(n=len(df_i) )
            df_i = df_i_random

            ## redefine start, end
            start, end = self.x_variable_range_dict[x_var]

            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            df_i_copy = copy.copy(df_i)
            df_i_copy = df_i_copy[df_i_copy[x_var]>start]
            df_i_copy = df_i_copy[df_i_copy[x_var]<end]
            scatter_x_list = df_i_copy[x_var].tolist()
            scatter_y_list = df_i_copy['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)

            percentiles = [5, 95]
            ## datapoints percentile
            percentile_values = np.percentile(scatter_x_list, percentiles)
            print(percentile_values)

            # plt.subplot(4, 3, flag)
            ax = axs[flag-1]
            ax.vlines(percentile_values, -5, 5, color='gray', linestyle='--', alpha=1)

            # ax2 = ax.twiny()  # Create a twin x-axis
            # ax2.set_xlim(ax.get_xlim())  # Match the limits with the main axis
            # ax2.set_xticks(percentile_values)  # Set percentile values as ticks
            # ax2.set_xticklabels([f'{p}%' for p in percentiles])  # Label with percentiles


            KDE_plot().plot_scatter(scatter_x_list, scatter_y_list,ax=ax )

            y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
            ax.plot(x_mean_list, y_mean_list, c='red', alpha=1)

            # ax.set_title(name_dic[x_var], fontsize=12)
            ax.set_xlabel(name_dic[x_var], fontsize=12)
            ax.set_ylabel(r'CV$_{\mathrm{LAI}}$ (%/year)', fontsize=12)

            flag += 1
            ax.set_ylim(-5, 5)
            # plt.show()


        plt.suptitle(self.y_variable)

        plt.tight_layout()
        plt.show()
        # plt.savefig(outf,dpi=300)
        # plt.close()

    def plot_pdp_shap_density_cloud_individual(self,line=False    ,scatter=True   ):
        x_variable_list = self.x_variable_list

        name_dic={'rainfall_intensity':'Rainfall intensity (mm/events)',
                  'rainfall_frenquency':'Rainfall frequency (events/year)',
                  'rainfall_seasonality_all_year':'Rainfall seasonality (unitless)',
                  'detrended_sum_rainfall_CV':r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                  'heat_event_frenquency':'Heat event frequency (events/year)',
                  'cwdx80_05':'Rooting zone water storage capacity (mm)',

                  'sand':'Sand (g/kg)',

        }

        inf_shap = rf'E:\Project3\Result\3mm\SHAP\RF_LAI4g_selected_samples_detrend_CV_\pdp_shap_CV\\LAI4g_detrend_CV_.shap.pkl'
        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)

        imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
        x_list = []
        y_list = []
        for key in imp_dict.keys():
            x_list.append(key)
            y_list.append(imp_dict[key])

        flag = 1
        centimeter_factor = 1 / 2.54
        # plt.figure(figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        # fig, axs = plt.subplots(4, 2,
        #                         figsize=(18 * centimeter_factor, 14 * centimeter_factor))
        # print(axs);exit()
        # axs = axs.flatten()
        for x_var in x_list:
            shap_values_mat = shap_values[:, x_var]
            data_i = shap_values_mat.data
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # pprint(df_i);exit()
            df_i_random = df_i.sample(n=len(df_i) )
            df_i = df_i_random

            ## redefine start, end
            start, end = self.x_variable_range_dict[x_var]

            bins = np.linspace(start, end, 50)
            df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            df_i_copy = copy.copy(df_i)
            df_i_copy = df_i_copy[df_i_copy[x_var]>start]
            df_i_copy = df_i_copy[df_i_copy[x_var]<end]
            scatter_x_list = df_i_copy[x_var].tolist()
            scatter_y_list = df_i_copy['shap_v'].tolist()
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                # mean = np.nanmean(vals)
                mean = np.nanmedian(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)

            percentiles = [5, 95]
            ## datapoints percentile
            percentile_values = np.percentile(scatter_x_list, percentiles)
            print(percentile_values)

            # plt.subplot(4, 3, flag)
            # ax = axs[flag-1]
            # fig = plt.figure(figsize=(5*centimeter_factor,3*centimeter_factor))
            fig,ax = plt.subplots(1,1,figsize=(8*centimeter_factor,5*centimeter_factor))
            # ax.vlines(percentile_values, -5, 5, color='gray', linestyle='--', alpha=1)

            y_lims = {
                "rainfall_intensity": [-7, 7],
                "rainfall_frenquency": [-6, 6],
                'detrended_sum_rainfall_CV': [-3, 3],
                "heat_event_frenquency": [-2, 2],
                "rainfall_seasonality_all_year": [-1, 3],
                "sand": [-3, 3],
                "cwdx80_05": [-2, 2],
            }

            if scatter:
                KDE_plot().plot_scatter(scatter_x_list, scatter_y_list,ax=ax )
                plt.axis('off')

            if line:
                y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
                ax.plot(x_mean_list, y_mean_list, c='red', alpha=1)

                # ax.set_title(name_dic[x_var], fontsize=12)
                ax.set_xlabel(name_dic[x_var], fontsize=12)
                ax.set_ylabel(r'CV$_{\mathrm{LAI}}$ (%/year)', fontsize=12)

            # flag += 1
            ax.set_ylim(y_lims[x_var])
            ## add line when y=0
            # ax.axhline(0, c='black', linestyle='-', alpha=1)
            # plt.show()

            outf = rf'E:\Project3\Result\3mm\SHAP\RF_LAI4g_selected_samples_detrend_CV_\pdp_shap_CV\\pdf\\{x_var}.png'
            plt.savefig(outf,dpi=300)
            plt.close()



        # plt.tight_layout()
        # plt.show()




    def plot_pdp_shap_all_models_main(self): ### plot all models in main
        fdir_all=results_root+rf'\3mm\SHAP\\'

        all_model_results = {}
        model_list = ['LAI4g',  'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai','LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',
                          'SDGVM_S2_lai',
                          'YIBs_S2_Monthly_lai']

        for model in model_list:

            fdir = join(fdir_all, rf'RF_{model}_detrend_CV_')

            for fdir_ii in T.listdir(fdir):



                for f in T.listdir(join(fdir, fdir_ii)):

                    if not '.shap.pkl' in f:
                        continue

                    inf_shap = join(fdir, fdir_ii, f)

                    shap_values = T.load_dict_from_binary(inf_shap)
                    print(shap_values)
                    x_list=['rainfall_intensity','rainfall_frenquency','detrended_sum_rainfall_CV','heat_event_frenquency', 'rainfall_seasonality_all_year',
                            'sand','cwdx80_05',]

                    # imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)
                    # x_list = []
                    # y_list = []
                    # for key in imp_dict.keys():
                    #     x_list.append(key)
                    #     y_list.append(imp_dict[key])
                    result_dic_X = {}
                    result_dic_Y = {}
                    result_dic_err = {}
                    for x_var in x_list:
                        shap_values_mat = shap_values[:, x_var]
                        data_i = shap_values_mat.data
                        value_i = shap_values_mat.values
                        df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
                        # pprint(df_i);exit()
                        df_i_random = df_i.sample(n=len(df_i) )
                        df_i = df_i_random


                        start, end = self.x_variable_range_dict[x_var]

                        bins = np.linspace(start, end, 50)
                        df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
                        y_mean_list = []
                        x_mean_list = []
                        y_err_list = []
                        df_i_copy = copy.copy(df_i)
                        df_i_copy = df_i_copy[df_i_copy[x_var]>start]
                        df_i_copy = df_i_copy[df_i_copy[x_var]<end]

                        for name, df_group_i in df_group:
                            x_i = name[0].left

                            vals = df_group_i['shap_v'].tolist()

                            if len(vals) == 0:
                                continue
                            # mean = np.nanmean(vals)
                            mean = np.nanmedian(vals)
                            err = np.nanstd(vals)
                            y_mean_list.append(mean)
                            x_mean_list.append(x_i)
                            y_err_list.append(err)

                        result_dic_X[x_var] = x_mean_list
                        result_dic_Y[x_var] = y_mean_list
                        result_dic_err[x_var] = y_err_list
                    all_model_results[f]=result_dic_X,result_dic_Y,result_dic_err

            ### plot all models



        flag = 1
        centimeter_factor = 1 / 2.54
        rows=2
        cols=4
        color_list=['black','red','blue','green','orange','greenyellow','gray',
                    'yellow','pink','brown','cyan','magenta','lime','teal','lavender','maroon','navy','olive','silver','aqua','fuchsia','lime','teal','lavender','maroon','navy','olive','silver','aqua','fuchsia']
        y_scale_list = [1,1,1,1,1]

        linewidth_list=[2]
        linewidth_list.extend([1]*20)
        alpha_list=[1]
        alpha_list.extend([0.6]*20)

        name_dic = {'rainfall_intensity': 'Rainfall intensity (mm/events)',
                    'rainfall_frenquency': 'Rainfall frequency (events/year)',
                    'rainfall_seasonality_all_year': 'Rainfall seasonality (unitless)',
                    'detrended_sum_rainfall_CV': r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                    'heat_event_frenquency': 'Heat event frequency (events/year)',
                    'cwdx80_05': 'Rooting zone water storage capacity (mm)',

                    'sand': 'Sand (g/kg)',

                    }



        plt.figure(figsize=(cols * 8 * centimeter_factor, rows * 6 * centimeter_factor))
        y_lims = {
            "rainfall_intensity": [-15, 10],
            "rainfall_frenquency": [-15, 20],
            'detrended_sum_rainfall_CV': [-15, 40],
            "heat_event_frenquency": [-5, 8],
            "rainfall_seasonality_all_year": [-2, 10],
            "sand": [-10, 20],
            "cwdx80_05": [-5, 15],
        }


        for x_var in x_list:
            color_flag = 1
            plt.subplot(rows, cols, flag)

            for f in all_model_results.keys():

                result_dic_X,result_dic_Y,result_dic_err = all_model_results[f]

                x_mean_list = result_dic_X[x_var]
                y_mean_list = result_dic_Y[x_var]
                y_err_list = result_dic_err[x_var]

                y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)

                zorder_list=[1]
                zorder_list.extend([0]*20)

                plt.plot(x_mean_list, y_mean_list, c= color_list[color_flag-1], linewidth=linewidth_list[color_flag-1],zorder=zorder_list[color_flag-1])
                plt.xlabel(name_dic[x_var], fontsize=12)
                ## y_lims
                plt.ylim(y_lims[x_var])
                color_flag+=1
            flag += 1

    # plt.suptitle(self.y_variable)
            plt.tight_layout()
        plt.show()


    def plot_pdp_shap_all_models_SI(self): ### plot all models in SI
        fdir_all=results_root+rf'\3mm\SHAP\\'

        x_variable_list = self.x_variable_list
        all_model_results = {}

        dic_color={}
        model_list = ['LAI4g_selected_samples', 'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                      'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                      'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                      'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                      'ORCHIDEE_S2_lai',
                      'SDGVM_S2_lai',
                      'YIBs_S2_Monthly_lai']

        for model in model_list:

            fdir = join(fdir_all, rf'RF_{model}_detrend_CV_')

            for fdir_ii in T.listdir(fdir):

                for f in T.listdir(join(fdir, fdir_ii)):

                    if not '.shap.pkl' in f:
                        continue


                    inf_shap = join(fdir, fdir_ii, f)

                    shap_values = T.load_dict_from_binary(inf_shap)
                    print(shap_values)
                    x_list=['rainfall_intensity','rainfall_frenquency','sand','detrended_sum_rainfall_CV','cwdx80_05','heat_event_frenquency', 'rainfall_seasonality_all_year']

                    imp_dict = self.feature_importances_shap_values(shap_values, x_variable_list)

                    imp_dict_sort = sorted(imp_dict.items(), key=lambda x: x[1])
                    ## color list
                    cmap = plt.cm.turbo_r  # Use a color map (e.g., Reds)
                    norm = plt.Normalize(1, len(imp_dict_sort))  # Norm
                    #colors = {var: cmap(norm(rank)) for var, rank in imp_dict_sort}
                    ## color is attributed to the ranking of the variable
                    colors = {var: cmap(norm(i)) for i, (var, _) in enumerate(imp_dict_sort)}
                    dic_color[f] = colors

                    result_dic_X = {}
                    result_dic_Y = {}
                    result_dic_err = {}
                    for x_var in x_list:
                        shap_values_mat = shap_values[:, x_var]
                        data_i = shap_values_mat.data
                        value_i = shap_values_mat.values
                        df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
                        # pprint(df_i);exit()
                        df_i_random = df_i.sample(n=len(df_i) )
                        df_i = df_i_random


                        start, end = self.x_variable_range_dict[x_var]

                        bins = np.linspace(start, end, 50)
                        df_group, bins_list_str = T.df_bin(df_i, x_var, bins)
                        y_mean_list = []
                        x_mean_list = []
                        y_err_list = []
                        df_i_copy = copy.copy(df_i)
                        df_i_copy = df_i_copy[df_i_copy[x_var]>start]
                        df_i_copy = df_i_copy[df_i_copy[x_var]<end]

                        for name, df_group_i in df_group:
                            x_i = name[0].left

                            vals = df_group_i['shap_v'].tolist()

                            if len(vals) == 0:
                                continue
                            # mean = np.nanmean(vals)
                            mean = np.nanmedian(vals)
                            err = np.nanstd(vals)
                            y_mean_list.append(mean)
                            x_mean_list.append(x_i)
                            y_err_list.append(err)

                        result_dic_X[x_var] = x_mean_list
                        result_dic_Y[x_var] = y_mean_list
                        result_dic_err[x_var] = y_err_list
                    all_model_results[f]=result_dic_X,result_dic_Y,result_dic_err

            ### plot all models
        num_subplots = len(T.listdir(fdir_all)) * len(x_list)
        cols = 7  # Define the number of columns you want in the layout
        rows = int(np.ceil(num_subplots / cols))

        y_lims = {
            "rainfall_intensity": [-15, 10],
            "rainfall_frenquency": [-15, 20],
            'detrended_sum_rainfall_CV': [-15, 40],
            "heat_event_frenquency": [-5, 20],
            "rainfall_seasonality_all_year": [-5, 55],
            "sand": [-10, 20],
            "cwdx80_05": [-5, 15],
        }


        flag = 1
        centimeter_factor = 1 / 2.54

        plt.figure(figsize=(cols * 4 * centimeter_factor, rows * 2 * centimeter_factor))

        for f in all_model_results.keys():
            for x_var in x_list:

                result_dic_X,result_dic_Y,result_dic_err = all_model_results[f]
                plt.subplot(rows, cols, flag)
                x_mean_list = result_dic_X[x_var]
                y_mean_list = result_dic_Y[x_var]
                y_err_list = result_dic_err[x_var]
                color= dic_color[f][x_var]

                y_mean_list = SMOOTH().smooth_convolve(y_mean_list, window_len=7)
                plt.plot(x_mean_list, y_mean_list, c=color, alpha=1)
                ## non xticks
                # plt.xticks([])
                plt.gca().axes.xaxis.set_ticklabels([])
                plt.ylim(y_lims[x_var])

                flag += 1
                # plt.show()
            # plt.title(f)
            # plt.show()



    # plt.suptitle(self.y_variable)
            plt.tight_layout()
        outf = rf'E:\Project3\Result\3mm\SHAP\RF_LAI4g_selected_samples_detrend_CV_\pdp_shap_CV\\all_models_ticks.pdf'
        plt.savefig(outf)
        T.open_path_and_file(rf'E:\Project3\Result\3mm\SHAP\RF_LAI4g_selected_samples_detrend_CV_\pdp_shap_CV')




    def plot_relative_importance(self):  ## bar plot

        ## here plot relative importance of each variable
        x_variable_list = self.x_variable_list


        name_dic={'rainfall_intensity':'Rainfall intensity (mm/events)',
                  'rainfall_frenquency':'Rainfall frequency (events/year)',
                  'rainfall_seasonality_all_year':'Rainfall seasonality (unitless)',
                  'detrended_sum_rainfall_CV':r'CV$_{\mathrm{interannual\ rainfall}}$ (%)',
                  'heat_event_frenquency':'Heat event frequency (events/year)',
                  'cwdx80_05':'Rooting zone water storage capacity (mm)',

                  'sand':'Sand (g/kg)',

        }

        inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')
        # print(isfile(inf_shap));exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values)
        total_sum_list = []
        sum_abs_shap_dic = {}
        for i in range(shap_values.values.shape[1]):
            sum_abs_shap_dic[i]=(np.sum(np.abs(shap_values.values[:, i])))

            total_sum_list.append(np.sum(np.abs(shap_values.values[:, i])))
        total_sum_list=np.array(total_sum_list)
        total_sum=np.sum(total_sum_list, axis=0)
        relative_importance={}

        for key in sum_abs_shap_dic.keys():
            relative_importance[key]=sum_abs_shap_dic[key]/total_sum*100


        x_list = []
        y_list = []
        imp_dict = {}
        for key in relative_importance.keys():
            x_list.append(key)
            y_list.append(relative_importance[key])
            imp_dict[key]=relative_importance[key]
        imp_dict_sort = sorted(imp_dict.items(), key=lambda x: x[1])
        x_list_sort = [x_variable_list[x[0]] for x in imp_dict_sort]
        ## use new name from dictionary
        x_list_sort = [name_dic[x] for x in x_list_sort]
        y_list_sort = [x[1] for x in imp_dict_sort]
        print(y_list_sort);exit()
        # pprint(imp_dict_sort);exit()
        # plt.barh(x_variable_list[::-1], y_list[::-1], color='grey', alpha=0.5)
        plt.barh(x_list_sort, y_list_sort, color='grey', alpha=0.5,edgecolor='black')
        print(x_list)

        plt.xticks(fontsize=12)
        plt.xlabel('Importance (%)', fontsize=12)
        ## add text R2=0.89 in (0.5, 0.5)
        plt.text(15, 0.1, 'R2=0.86', fontsize=12)





        plt.tight_layout()

        plt.show()

        pass

    def spatial_shapely(self):  #### spatial plot

        dff = self.dff
        outdir = rf'E:\Project3\Result\3mm\SHAP\RF_LAI4g_detrend_CV_\pdp_shap_CV\\spatial_shapely\\'
        T.mk_dir(outdir, force=True)
        # T.open_path_and_file(outdir)
        # exit()

        x_variable_list = self.x_variable_list

        y_variable = self.y_variable
        # plt.hist(T.load_df(dff)[y_variable].tolist(),bins=100)
        # plt.show()
        df_origin = T.load_df(dff)
        df_origin = self.df_clean(df_origin)
        # df_origin = self.valid_range_df(df_origin)


        # all_vars=copy.copy(all_vars_vif)
        pix_list = T.get_df_unique_val_list(df_origin, 'pix')
        spatial_dict = {}
        for pix in pix_list:
            spatial_dict[pix] = 1
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr, interpolation='nearest', cmap='jet')
        # plt.colorbar()
        # plt.show()

        all_vars = copy.copy(x_variable_list)

        all_vars.append(y_variable)  # add the y variable to the list
        all_vars.append('pix')

        all_vars_df = df_origin[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna(subset=x_variable_list, how='any')
        all_vars_df = all_vars_df.dropna(subset=self.y_variable, how='any')

        print(len(df_origin))
        x_variable_list = self.x_variable_list
        inf_shap = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.shap.pkl')
        # print(inf_shap);exit()
        shap_values = T.load_dict_from_binary(inf_shap)
        print(shap_values.shape)
        # T.print_head_n(df_origin);exit()
        i=0
        for x_var in x_variable_list:
            print(x_var)

            # shap_values_mat = shap_values[:, x_var]

            col_name = f'{x_var}_shap'

            all_vars_df[col_name]= shap_values[:, i]

            i+=1
        all_vars_df = all_vars_df.dropna(subset=x_variable_list, how='all')
            # df_i = pd.DataFrame({x_var: data_i, 'shap_v': value_i})
            # arr = T.
        # T.print_head_n(df_origin)
        df_pix_dict = T.df_groupby(all_vars_df, 'pix')

        for xvar in x_variable_list:
            col_name = f'{xvar}_shap'
            spatial_dict = {}
            for pix in df_pix_dict:
                df_pix = df_pix_dict[pix]
                vals = df_pix[col_name].tolist()
                # vals = np.array(vals)
                vals_abs = np.abs(vals)

                vals_abs_sum = np.sum(vals_abs)
                vals_abs_sum_mean = vals_abs_sum / len(vals)
                spatial_dict[pix] = vals_abs_sum_mean
            outf = join(outdir, col_name + '.tif')
            DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(spatial_dict, outf)

        T.open_path_and_file(outdir)
        # exit()

    def variable_contributions(self):  ## each variable contribution and the max one
        r2 = .86
        fdir = join(self.this_class_png, 'pdp_shap_CV', 'spatial_shapely')
        outdir = join(self.this_class_png,'pdp_shap_CV', 'variable_contributions')
        T.mk_dir(outdir, force=True)
        all_spatial_dict = {}
        keys = []
        for f in T.listdir(fdir):
            if 'sand' in f:
                continue
            if 'cwdx' in f:
                continue
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            spatial_dict = DIC_and_TIF(pixelsize=.5).spatial_tif_to_dic(fpath)
            key = f.split('.')[0]
            all_spatial_dict[key] = spatial_dict
            keys.append(key)
        df = T.spatial_dics_to_df(all_spatial_dict)
        sum_val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            val_list = []
            for key in keys:
                val = row[key]
                # print(val)
                val_list.append(val)
            sum_val = np.sum(val_list)
            sum_val_list.append(sum_val)
        df['sum'] = sum_val_list
        new_key_dict = {}
        flag = 1
        for key in keys:
            df[key + '_contrib'] = df[key] / df['sum'] * 100 * r2
            new_key_dict[key + '_contrib'] = flag
            flag += 1
        pprint(new_key_dict)
        T.print_head_n(df)
        result_dict = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            dict_i = {}
            for new_key in new_key_dict:
                val = row[new_key]
                dict_i[new_key] = val
            pix = row['pix']
            max_key = T.get_max_key_from_dict(dict_i)
            max_key_flag = new_key_dict[max_key]
            max_val = dict_i[max_key]
            result_dict[pix] = {'max_key': max_key, 'max_key_flag': max_key_flag, 'max_val': max_val}
        result_df = T.dic_to_df(result_dict, 'pix')
        outf_max_val = join(outdir, 'max_val_climate_only.tif')
        outf_max_flag = join(outdir, 'max_flag_climate_only.tif')
        max_val_dict = T.df_to_spatial_dic(result_df, 'max_val')
        DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(max_val_dict, outf_max_val)
        max_flag_dict = T.df_to_spatial_dic(result_df, 'max_key_flag')
        DIC_and_TIF(pixelsize=.5).pix_dic_to_tif(max_flag_dict, outf_max_flag)

        legend_f = join(outdir, 'legend.txt')
        fw = open(legend_f, 'w')
        fw.write(str(new_key_dict))
        fw.close()

        T.open_path_and_file(outdir)

    def max_contributions(self):   #### no use
        fdir = join(self.this_class_png, 'pdp_shap_CV', 'variable_contributions')
        outdir = join(self.this_class_png,'pdp_shap_CV','variable_contributions')

        T.mk_dir(outdir, force=True)
        array_list = []
        variable_dict = {}
        flag = 0
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if 'max' in f:
                continue
            variable = f.split('.')[0]
            variable_dict[flag] = variable
            flag += 1

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir, f))
            array[array < -99] = np.nan

            array_list.append(array)
        pprint(variable_dict)
        # exit()



        array_list = np.array(array_list)
        max_index_matrix= []
        for r in tqdm(range(len(array_list[0]))):
            max_index_matrix_i = []
            for c in range(len(array_list[0][0])):
                vals_list = []
                for arr in array_list:
                    val = arr[r][c]
                    vals_list.append(val)
                if T.is_all_nan(vals_list):
                    max_index_matrix_i.append(np.nan)
                    continue
                max_index = np.argmax(vals_list)
                max_index_matrix_i.append(max_index)
            max_index_matrix.append(max_index_matrix_i)
        max_index = np.array(max_index_matrix)
        # max_index = np.nanargmax(array_list, axis=0)
        # max_index = np.array(max_index, dtype=float)



        # plt.imshow(max_index, interpolation='nearest',vmin=0,vmax=10)
        # plt.colorbar()
        # plt.show()
        # outf = join(outdir, 'max_variable.tif')
        # DIC_and_TIF(pixelsize=.5).arr_to_tif(max_index, outf)

    def plot_dominant_factors_bar(self):  ### insert bar plot
        dff=rf'E:\Project3\Result\3mm\Dataframe\dominant_factors\\dominant_factors.df'

        df=pd.read_pickle(dff)
        df=self.df_clean(df)

        val_list=[1,2,3,4,5,]
        dic_name={1:'detrended_sum_rainfall',2:'heat event',
                  3:'rainfall frequency',4:'rainfall intensity',
                  5:'rainfall seasonality '}
        percetage_dict={}
        for val in val_list:
            val=df[df['max_flag_climate_only']==val]
            count=len(val)
            percetage=count/len(df)*100
            print(dic_name[val['max_flag_climate_only'].values[0]],percetage)
            percetage_dict[dic_name[val['max_flag_climate_only'].values[0]]]=percetage
        df_new=pd.DataFrame.from_dict(percetage_dict,orient='index')
        ## plot

        df_new.plot.bar()
        plt.show()




    def feature_importances_shap_values(self, shap_values, features):
        '''
        Prints the feature importances based on SHAP values in an ordered way
        shap_values -> The SHAP values calculated from a shap.Explainer object
        features -> The name of the features, on the order presented to the explainer
        '''
        # Calculates the feature importance (mean absolute shap value) for each feature
        importances = []
        # for i in range(len(shap_values)):
        #     importances.append(np.abs(shap_values[i]).mean())
        for i in range(shap_values.values.shape[1]):
            importances.append(np.mean(np.abs(shap_values.values[:, i])))


        # Calculates the normalized version
        # importances_norm = softmax(importances)
        # Organize the importances and columns in a dictionary
        feature_importances = {fea: imp for imp, fea in zip(importances, features)}
        # feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
        # Sorts the dictionary
        feature_importances = {k: v for k, v in
                               sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
        # feature_importances_norm = {k: v for k, v in
        #                             sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
        # Prints the feature importances
        # for k, v in feature_importances.items():
        #     print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

        return feature_importances
        # return feature_importances_norm

    def __select_extreme(self, df):
        df = df[df['T_max'] > 1]
        df = df[df['intensity'] < -2]
        return df

    def __train_model(self, X, y):
        from sklearn.model_selection import train_test_split
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.3)  # split the data into training and testing
        # model = RandomForestRegressor(n_estimators=50, random_state=42,n_jobs=-1,) # build a random forest model
        # rf.fit(X_train, y_train) # train the model
        # r2 = rf.score(X_test,y_test)
        # model = xgb.XGBRegressor(objective="reg:squarederror", booster='gbtree', n_estimators=100,
        #                        max_depth=7, eta=0.1, random_state=42, n_jobs=14,  )
        # model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=12,max_depth=7)
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=12, max_depth=7)

        model.fit(X_train, y_train)
        # model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X_test)
        # print(len(y_pred))
        # plt.scatter(y_test, y_pred)
        # plt.show()
        r = stats.pearsonr(y_test, y_pred)
        r2 = r[0] ** 2
        print('r2:', r2)
        # exit()

        return model, y, y_pred

    def __train_model_RF(self, X, y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, random_state=1, test_size=0.) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # build a random forest model
        rf.fit(X, y)  # train the model
        coef = rf.feature_importances_
        imp_dict = {}
        for i in range(len(coef)):
            imp_dict[self.x_variable_list[i]] = coef[i]

        return imp_dict

    def benchmark_model(self, y, y_pred):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        plt.scatter(y, y_pred)
        plt.plot([0.6, 1.2], [0.6, 1.2], color='r', linestyle='-', linewidth=2)
        plt.ylabel('Predicted', size=20)
        plt.xlabel('Actual', size=20)
        plt.xlim(0.6, 1.2)
        plt.ylim(0.6, 1.2)
        plt.show()




class Random_Forests:

    def __init__(self):
        self.this_class_arr = data_root

        self.dff = rf'E:\Data\ERA5_daily\dict\Dataframe\\moving_window.df'

        self.variables_list()

        ##----------------------------------

        self.y_variable = 'detrended_annual_LAI4g_CV'
        ####################

        self.x_variable_list = self.x_variable_list
        # self.x_variable_range_dict = self.x_variable_range_dict_AUS
    #
    #     pass

    def run(self):
        # self.copy_df()
        df = self.__gen_df_init()
        # self.check_df_attributes()
        # exit()

        # self.check_variables_valid_ranges()
        # self.show_colinear()

        self.run_important_for_each_pixel()

        # self.plot_importance_result_for_each_pixel()
        # self.plot_most_important_factor_for_each_pixel()
        # # self.plot_three_demension()
        # self.plot_two_demension()

        pass

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff


    def copy_df(self):
        outdir = join(self.this_class_arr,'RF')
        T.mk_dir(outdir,force=True)
        outf = self.dff
        dff_origin =  'D:\Project3\Result\Dataframe\RF\RF.df'
        dff_origin_xlsx = 'D:\Project3\Result\Dataframe\RF\RF.xlsx'
        shutil.copy(dff_origin,outf)
        shutil.copy(dff_origin_xlsx,join(outdir,'RF.df.xlsx'))
        pass


    def check_variables_valid_ranges(self):
        dff = self.dff
        df = T.load_df(dff)
        plt.figure(figsize=(6,6))
        flag = 1
        x_variable_list = self.x_variable_list
        for x_var in x_variable_list:
            plt.subplot(3,3,flag)
            flag += 1
            vals = df[x_var].tolist()
            vals = np.array(vals)
            # vals[vals>1000] = np.nan
            # vals[vals<-1000] = np.nan
            plt.hist(vals,bins=100)
            plt.xlabel(x_var)
        plt.tight_layout()
        plt.show()

    def show_colinear(self,):
        dff=self.dff
        df = T.load_df(dff)
        vars_list = self.x_variable_list
        df = df[vars_list]
        ## add LAI4g_raw
        df['detrended_annual_LAI4g_CV'] = T.load_df(dff)['detrended_annual_LAI4g_CV']
        ## plot heat map to show the colinear variables
        import seaborn as sns
        plt.figure(figsize=(10, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f")
        plt.show()



    def check_df_attributes(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print(df.columns.tolist())
        print(len(df))
        # exit()
        pass
    def clean_dataframe(self,df):
        x_variable_range_dict = self.x_variable_range_dict
        for x_var in x_variable_range_dict:
            x_var_range = x_variable_range_dict[x_var]
            df = df[df[x_var] >= x_var_range[0]]
            df = df[df[x_var] <= x_var_range[1]]

        return df

    def run_important_for_each_pixel(self):



        dff = self.dff
        df = T.load_df(dff)
        df=self.df_clean(df)

        pix_list = T.get_df_unique_val_list(df,'pix')
        spatial_dict = {}
        for pix in pix_list:
            spatial_dict[pix] = 1
        arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.show()
        # ## plot spatial df
        # T.print_head_n(df)


        group_dic = T.df_groupby(df,'pix')
        # spatial_dict = {}
        # for pix in group_dic:
        #     df_pix = group_dic[pix]
        #     spatial_dict[pix] = len(df_pix)
        # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.show()
        ## plot spatial df
        # spatial_dic = T.df_to_spatial_dic(df,'pix')
        # array= DIC_and_TIF().spatial_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()


        outdir= join(self.this_class_arr,'importance_for_each_pixel_CV')
        outdir_distribution= join(outdir,'separate_folder')
        T.mk_dir(outdir,force=True)
        T.mk_dir(outdir_distribution,force=True)

        for y_var in self.y_variable_list:
            importance_spatial_dict = {}
            spatial_model_dic = {}

            for pix in tqdm(group_dic):
                df_pix = group_dic[pix]

                ### to extract 1983-2020
                # vals_list=[]
                # name_list=[]
                #
                # for col in self.x_variable_list:
                #     vals = df_pix[col].tolist()
                #     vals=vals[1:]
                #     name=col
                #     vals_list.append(vals)
                #     name_list.append(name)
                # y_vals = df_pix[y_var].tolist()
                # y_vals = y_vals[1:]
                # vals_list.append(y_vals)
                # name_list.append(y_var)
                # dic_new = dict(zip(name_list,vals_list))
                # df_new = pd.DataFrame(dic_new)
                #
                #
                # T.print_head_n(df_new)

                x_variable_list = self.x_variable_list
                ## extract the data[1:]
                df_new = df_pix.dropna(subset=[y_var] + self.x_variable_list, how='any')
                if len(df_new) < 20:
                    continue
                X=df_new[x_variable_list]
                ### normalized the data to [0,1]
                X_normalized=(X-X.min())/(X.max()-X.min())

                Y=df_new[y_var]
                Y_normalized=(Y-Y.min())/(Y.max()-Y.min())
                # print(Y_normalized)

                # T.print_head_n(df_new)

                X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y_normalized, test_size=0.2,
                                                                    random_state=42)
                # split the data into training and testing
                # clf=xgb.XGBRegressor(objective="reg:squarederror",booster='gbtree',n_estimators=100,
                #                  max_depth=11,eta=0.05,random_state=42,n_jobs=12)
                clf = RandomForestRegressor(n_estimators=100,random_state=42,)  # build a random forest model
                clf.fit(X_train, Y_train)  # train the model
                # R2= clf.score(X_test, Y_test)  # calculate the R2
                R = stats.pearsonr(clf.predict(X_test),Y_test)[0]
                R2 = R**2

                importance=clf.feature_importances_
                # print(importance)
                importance_dic=dict(zip(x_variable_list,importance))
                importance_dic['R2']=R2

                # print(importance_dic)
                importance_spatial_dict[pix]=importance_dic
                spatial_model_dic[pix]=clf
            importance_dateframe = T.dic_to_df(importance_spatial_dict, 'pix')
            T.print_head_n(importance_dateframe)
            outf = join(outdir, f'{y_var}.df')
            T.save_df(importance_dateframe, outf)
            outf_xlsx = outf + '.xlsx'
            T.df_to_excel(importance_dateframe, outf_xlsx)

            # outf = join(outdir, f'{y_var}.pkl')
            # T.save_dict_to_binary(spatial_model_dic, outf)
            self.save_distributed_perpix_dic(spatial_model_dic, outdir_distribution, n=10000)

    def save_distributed_perpix_dic(self, dic, outdir, n=10000):
        '''
        :param dic:
        :param outdir:
        :param n: save to each file every n sample
        :return:
        '''
        flag = 0
        temp_dic = {}
        for key in tqdm(dic, 'saving...'):
            flag += 1
            arr = dic[key]
            # arr = np.array(arr)
            temp_dic[key] = arr
            if flag % n == 0:
                outf = outdir + '/per_pix_dic_%03d.pkl' % (flag / n)
                T.save_dict_to_binary(temp_dic, outf)
                # np.save(outdir + '/per_pix_dic_%03d' % (flag / n), temp_dic)
                temp_dic = {}
        # np.save(outdir + '/per_pix_dic_%03d' % 0, temp_dic)
        outf_0 = outdir + '/per_pix_dic_%03d.pkl' % 0
        T.save_dict_to_binary(temp_dic, outf_0)

        pass


    def plot_importance_result_for_each_pixel(self):
        keys=list(range(len(self.x_variable_list)))
        x_variable_dict=dict(zip(self.x_variable_list, keys))
        print(x_variable_dict)
        # exit()

        fdir = rf'E:\Data\ERA5_daily\dict\\\importance_for_each_pixel_CV\\'
        for f in os.listdir(fdir):

            if not f.endswith('.df'):
                continue
            fpath=join(fdir,f)
            fname=f.split('.')[0]


            df = T.load_df(fpath)

            T.print_head_n(df)
            spatial_dic={}
            sptial_R2_dic={}
            x_variable_list = self.x_variable_list
            for x_var in x_variable_list:


            ## plot individual importance
                for i, row in df.iterrows():
                    pix = row['pix']
                    importance_dic = row.to_dict()
                    # print(importance_dic)
                    print(importance_dic)

                    spatial_dic[pix] = importance_dic[x_var]
                arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)

                plt.imshow(arr,vmin=0,vmax=0.5,interpolation='nearest',cmap='RdYlGn')

                plt.colorbar()
                plt.title(f'{fname}_{x_var}')
                plt.show()
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr,join(fdir,f'{fname}_{x_var}.tif'))



    def plot_most_important_factor_for_each_pixel(self):  ## second important factor/ third important factor
        keys=list(range(len(self.x_variable_list)))
        x_variable_dict=dict(zip(self.x_variable_list, keys))
        print(x_variable_dict)
        # exit()

        fdir = rf'E:\Data\ERA5_daily\dict\\\importance_for_each_pixel_CV\\'
        for f in os.listdir(fdir):

            if not f.endswith('.df'):
                continue
            fpath=join(fdir,f)
            fname=f.split('.')[0]


            df = T.load_df(fpath)

            # df_R2_05=df[df['R2']>0.4]
            # percent = len(df_R2_05) / len(df) * 100
            # # print(percent)
            # # exit()

            T.print_head_n(df)
            spatial_dic={}
            sptial_R2_dic={}
            for i, row in df.iterrows():
                pix = row['pix']
                importance_dic = row.to_dict()
                # print(importance_dic)
                x_variable_list = self.x_variable_list
                importance_dici = {}
                for x_var in x_variable_list:
                    importance_dici[x_var] = importance_dic[x_var]
                    # print(importance_dici)
                max_var = max(importance_dici, key=importance_dici.get)
                ## second important factor
                # max_var = sorted(importance_dici, key=importance_dici.get, reverse=True)[1]
                ## third important factor
                # max_var = sorted(importance_dici, key=importance_dici.get, reverse=True)[2]
                max_var_val=x_variable_dict[max_var]
                spatial_dic[pix]=max_var_val

                # print(max_var_val)
                # print(max_var)
                importance_dici['R2'] = importance_dic['R2']
                R2 = importance_dic['R2']
                sptial_R2_dic[pix]=R2
            #### print average R2
            # R2_list = list(sptial_R2_dic.values())
            # R2_mean = np.nanmean(R2_list)
            # print(f'{fname} R2 mean: {R2_mean}')
            # exit()

                ### plot R2
            arrR2 = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(sptial_R2_dic)
            arrR2[arrR2<0]=np.nan
            arrR2_flat=arrR2.flatten()

            # plt.hist(arrR2_flat, bins=100)
            # plt.title(rf'R2_hist_{percent:.2f}%')
            # plt.show()

            plt.imshow(arrR2,vmin=0,vmax=0.5,interpolation='nearest',cmap='RdYlGn')
            plt.colorbar()
            plt.title(f'{fname}_R2')
            plt.show()

            outtif_R2 = join(fdir, f'{fname}_R2.tif')

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arrR2, outtif_R2)


            ### plot importance
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)

            plt.imshow(arr,vmin=0,vmax=12,interpolation='nearest')
            plt.colorbar()
            plt.title(fname)
            plt.show()
            outtif=join(fdir,f'{fname}_most.tif')

            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr,outtif)

            pass


    def plot_three_demension(self):
        fdir = join(self.this_class_arr, 'raw_importance_for_each_pixel_ENSO','separate_folder')
        for f in os.listdir(fdir):

            model_dict = T.load_dict_from_binary(join(fdir,f))
            R2_f = rf'E:\Project5\Result\RF_pix\raw_importance_for_each_pixel_ENSO\\LAI4g_raw_R2.tif'
            array_R, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(R2_f)
            dic_r2 = DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(array_R)
            f_cluster = rf'E:\Project5\Result\RF_pix\\spatial_distribution.tif'
            arr_cluster, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_cluster)
            dic_cluster = DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(arr_cluster)
            for pix in model_dict:
                model = model_dict[pix]
                R2=dic_r2[pix]
                if R2<0.4:
                    continue

                cluster = dic_cluster[pix]
                if not cluster==1:
                    continue

                CO2_values = np.linspace(350, 450, 50)  # CO2 from 300 to 500 ppm
                ENSO_values = np.linspace(-2, 2, 0.5)  # ENSO from -0.5 to 0.5
                # CO2_values = np.array([405,])
                # print(CO2_values);exit()
                # precip_values = np.linspace(0, 1000, 50)  # Precipitation from 0 to 1000 mm
                # tmax_values = np.linspace(0, 40, 50)
                # tmax_values = np.array([40,])

                # Create a meshgrid for CO2 and precipitation
                CO2_grid, tmax_grid, precip_grid = np.meshgrid(CO2_values,tmax_values, precip_values)
                precip_grid1, CO2_grid1, = np.meshgrid(precip_values, CO2_values)


                # Prepare data for prediction
                input_data = np.c_[CO2_grid.ravel(),precip_grid.ravel(), tmax_grid.ravel()]
                # print(input_data);exit()

                # Predict LAI using the RF model

                predicted_LAI = model.predict(input_data)
                # print(predicted_LAI.shape);exit()

                # Reshape the predictions back to the grid shape
                predicted_LAI_grid = predicted_LAI.reshape(CO2_grid1.shape)

                # Now, plot the data
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # 3D Surface Plot
                surf = ax.plot_surface(CO2_grid1, precip_grid1, predicted_LAI_grid, cmap='viridis')

                # Add labels and title
                ax.set_xlabel('CO2 (ppm)')
                # ax.set_xlabel('Tmax (C)')
                ax.set_ylabel('Precipitation (mm)')
                ax.set_zlabel('LAI')
                ax.set_title(rf'{pix}_R2_{R2:.2f}')
                ## auto rotate and show






                plt.show()

                pass
    def __train_model(self,X,y):
        from sklearn.model_selection import train_test_split
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.3) # split the data into training and testing
        # model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=7,) # build a random forest model
        # rf.fit(X_train, y_train) # train the model
        # r2 = rf.score(X_test,y_test)
        model = xgb.XGBRegressor(objective="reg:squarederror",booster='gbtree',n_estimators=100,
                                 max_depth=13,eta=0.05,random_state=42,n_jobs=12)
        # model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=12)
        model.fit(X_train, y_train)
        # model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X_test)
        plt.scatter(y_test, y_pred)
        plt.show()
        r = stats.pearsonr(y_test, y_pred)
        r2 = r[0] ** 2
        print('r2:', r2)
        # exit()

        return model,y,y_pred
    def plot_two_demension(self):
        fdir = join(self.this_class_arr, 'raw_importance_for_each_pixel_ENSO','separate_folder')
        for f in os.listdir(fdir):

            model_dict = T.load_dict_from_binary(join(fdir,f))
            R2_f = rf'E:\Project5\Result\RF_pix\raw_importance_for_each_pixel_ENSO\\LAI4g_raw_R2.tif'
            array_R, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(R2_f)
            dic_r2 = DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(array_R)
            f_cluster = rf'E:\Project5\Result\RF_pix\\spatial_distribution.tif'
            arr_cluster, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_cluster)
            dic_cluster = DIC_and_TIF(pixelsize=0.25).spatial_arr_to_dic(arr_cluster)
            for pix in model_dict:
                model = model_dict[pix]
                R2=dic_r2[pix]
                if R2<0.4:
                    continue

                cluster = dic_cluster[pix]
                if not cluster==1:
                    continue

                CO2_values = np.linspace(350, 450, 50)  # CO2 from 300 to 500 ppm
                ENSO_values = np.linspace(-2, 2, 0.5)  # ENSO from -0.5 to 0.5


                # Create a meshgrid for CO2 and precipitation
                CO2_grid, ENSO_grid = np.meshgrid(CO2_values,ENSO_values)



                # Prepare data for prediction
                input_data = np.c_[CO2_grid.ravel(),ENSO_grid.ravel()]
                # print(input_data);exit()

                # Predict LAI using the RF model

                predicted_LAI = model.predict(input_data)
                # print(predicted_LAI.shape);exit()

                # Reshape the predictions back to the grid shape
                predicted_LAI_grid = predicted_LAI.reshape(CO2_grid.shape)

                # Now, plot the data
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # 3D Surface Plot
                surf = ax.plot_surface(CO2_grid, ENSO_grid, predicted_LAI_grid, cmap='viridis')

                # Add labels and title
                ax.set_xlabel('CO2 (ppm)')
                # ax.set_xlabel('Tmax (C)')
                ax.set_ylabel('ENSO (mm)')
                ax.set_zlabel('LAI')
                ax.set_title(rf'{pix}_R2_{R2:.2f}')
                ## auto rotate and show






                plt.show()

                pass

    def df_clean(self,df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df=df[df['row']>120]
        df=df[df['Aridity']<0.65]
        df=df[df['LC_max']<20]

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def variables_list(self):
        self.x_variable_list = [
            'maxmum_dry_spell',
            # 'CO2',
            'GPCC',


            'wet_frequency_90th',
            'peak_rainfall_timing',

            # 'rainfall_intensity',


        ]


        self.y_variable_list = [
            'detrended_annual_LAI4g_CV',


        ]
        pass

        self.x_variable_range_dict = {
            'CO2_raw': [350, 450],

            'CRU_raw': [0, 1000],
            'tmax_raw': [0, 40],

        'CO2': [350, 450],
        'GPCC': [0, 1000],
        'CV_rainfall': [0,600],
        'maxmum_dry_spell': [0, 200],
        'wet_frequency_95th': [3,7],
        'wet_frequency_90th': [10, 22],
        'rainfall_intensity': [0, 5],
            'peak_rainfall_timing': [0, 300],


        }


class single_correlation():
    def __init__(self):
        self.this_root = 'D:\Project3\\'
        self.data_root = 'D:/Project3/Data/'
        self.result_root = 'D:/Project3/Result/'
        pass
    def run(self):

        self.cal_single_correlation()



        pass
    def cal_single_correlation(self):

        NDVI_mask_f = self.data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        landcover_f = self.data_root + rf'/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = self.data_root + rf'/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)


        fdir_Y = self.result_root + rf'\extract_window\extract_original_window\extract_average\\'
        fdir_X = self.result_root + rf'extract_window\extract_detrend_original_window_CV\\'
        outdir = self.result_root + rf'result_new\\\single_correlation\\'
        T.mk_dir(outdir, force=True)

        for fx in os.listdir(fdir_X):
            print(fx)
            if not  'LAI4g' in fx:
                continue


            if not fx.endswith('.npy'):
                continue
            for fy in os.listdir(fdir_Y):
                if not 'LAI4g' in fy:
                    continue
                if 'trend' in fy:
                    continue

                if not fy.endswith('.npy'):
                    continue
                outf=outdir + f'{fx.split(".")[0]}_{fy.split(".")[0]}.npy'

                dic_monthly_data_X = np.load(fdir_X + fx, allow_pickle=True, encoding='latin1').item()
                dic_monthly_data_Y = np.load(fdir_Y + fy, allow_pickle=True, encoding='latin1').item()
                result_dic = {}
                result_dic_pvalue = {}
                for pix in tqdm(dic_monthly_data_X, desc=fy):
                    if not pix in dic_monthly_data_Y:
                        continue

                    r, c = pix
                    if r < 120:
                        continue
                    landcover_value = crop_mask[pix]
                    if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                        continue
                    if dic_modis_mask[pix] == 12:
                        continue

                    time_series_X = dic_monthly_data_X[pix]
                    if np.isnan(time_series_X).any():
                        continue

                    time_series_Y = dic_monthly_data_Y[pix]
                    # print(time_series_X)
                    # print(time_series_Y)
                    ## remove ()



                    time_series_X = np.array(time_series_X)
                    time_series_Y = np.array(time_series_Y)
                    # print(time_series_X.shape, time_series_Y.shape)
                    # print(time_series_X, time_series_Y)
                    # exit()
                    time_series_X = time_series_X[~np.isnan(time_series_X)]
                    time_series_Y = time_series_Y[~np.isnan(time_series_Y)]


                    if len(time_series_X) != len(time_series_Y):
                        continue
                    if len(time_series_X) <= 3:
                        continue
                    if len(time_series_Y) <= 3:
                        continue



                    r, p = stats.pearsonr(time_series_X, time_series_Y)
                    result_dic[pix] = r
                    result_dic_pvalue[pix] = p


                np.save(outf, result_dic)

                array_trend = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic)
                array_trend=array_trend*array_mask
                array_p_value = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(result_dic_pvalue)
                array_p_value = array_p_value * array_mask
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_trend, outf.replace('.npy', '.tif'))
                DIC_and_TIF(pixelsize=0.25).arr_to_tif(array_p_value, outf.replace('.npy', '_pvalue.tif'))

    def plot_single_correlation(self):
        fdir = result_root + rf'extract_GS_return_monthly_data\single_correlation\\'
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            dic = np.load(fdir + f, allow_pickle=True, encoding='latin1').item()
            spatial_dic = {}
            for pix in dic:
                spatial_dic[pix] = dic[pix]
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr, interpolation='nearest', cmap='jet')
            plt.colorbar()
            plt.title(f)
            plt.show()

class simple_linear_regression():
    def __init__(self):

        self.f_y = rf'E:\Data\ERA5_daily\dict\\moving_window_average_anaysis\\annual_LAI4g_trend.npy'
        self.f_x = rf'E:\Data\ERA5_daily\dict\\moving_window_average_anaysis\\detrended_annual_LAI4g_std.npy'
        self.outdir = rf'E:\Data\ERA5_daily\dict\simple_linear_regression\\'
        T.mk_dir(self.outdir, force=True)
        pass

    def run(self):
        # df=self.build_df(self.f_x, self.f_y)
        # self.cal_simple_linear_regression(df)
        self.plt_single_regression_result(self.outdir)
        pass
    def build_df(self, f_x, f_y, ):

        df = pd.DataFrame()
        dic_y=T.load_npy(f_y)
        pix_list = []
        y_val_list=[]

        for pix in dic_y:
            r,c= pix

            if len(dic_y[pix]) == 0:
                continue
            vals = dic_y[pix]
            # print(vals)
            # exit()
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            vals[vals>999] = np.nan
            vals[vals<-999] = np.nan

            pix_list.append(pix)
            y_val_list.append(vals)
        df['pix'] = pix_list
        df['LAI4g_trend'] = y_val_list

        # build x

        dic_x = T.load_npy(f_x)
        x_val_list=[]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if not pix in dic_x:
                x_val_list.append([])
                continue
            # print(len(x_arr[pix]))

            vals_x = dic_x[pix]
            vals_x = np.array(vals_x)
            vals_x[vals_x > 999] = np.nan
            vals_x[vals_x < -999] = np.nan
            if len(vals_x) == 0:

                x_val_list.append([])

                continue

            x_val_list.append(vals_x)

        df['LAI4g_CV'] = x_val_list

        return df

    def cal_simple_linear_regression(self, df):
        import numpy as np
        from sklearn.linear_model import LinearRegression

        simple_derivative = {}

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            r, c = pix
            y_vals = row['LAI4g_trend']
            y_vals=np.array(y_vals)
            y_vals[y_vals < -999] = np.nan
            y_vals = T.remove_np_nan(y_vals)
            if len(y_vals) == 0:
                continue

            x_vals = row['LAI4g_CV']

            x_vals = np.array(x_vals)
            x_vals[x_vals < -999] = np.nan
            x_vals = T.remove_np_nan(x_vals)
            if len(x_vals) == 0:
                continue
            x_vals = np.array(x_vals)
            if len(x_vals) != len(y_vals):
                continue

            df_new = pd.DataFrame()
            df_new['LAI4g_CV'] = x_vals
            df_new['LAI4g_trend'] = y_vals
            df_new = df_new.dropna(axis=1, how='all')


            # Create a linear regression model
            model = LinearRegression()

            # Fit the model to the data
            model.fit(df_new['LAI4g_CV'].values.reshape(-1, 1), df_new['LAI4g_trend'].values.reshape(-1, 1))
            # model.fit(df_new['LAI4g'], df_new['LAI4g_CV'])

            # Get the coefficient (slope) and intercept
            coefficient = model.coef_[0]
            # print(coefficient)
            simple_derivative[pix] = coefficient

            intercept = model.intercept_
        outf=self.outdir+ 'linear_regression_result_std_raw.npy'
            ## save the result
        np.save(outf, simple_derivative)

        # Print the results

        pass

    def plt_single_regression_result(self, outdir):

        landcover_f = rf'D:\Project3\Data/Base_data/glc_025\\glc2000_025.tif'
        crop_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(landcover_f)
        MODIS_mask_f = rf'D:\Project3\Data/Base_data/MODIS_LUCC\\MODIS_LUCC_resample.tif'
        MODIS_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(MODIS_mask_f)
        dic_modis_mask = DIC_and_TIF().spatial_arr_to_dic(MODIS_mask)

        fdir = outdir
        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue


            dic = T.load_npy(fdir+f)
            spatial_dic = {}

            for pix in dic:
                r, c = pix
                if r<120:
                    continue

                r, c = pix
                if r < 120:
                    continue
                landcover_value = crop_mask[pix]
                if landcover_value == 16 or landcover_value == 17 or landcover_value == 18:
                    continue
                if dic_modis_mask[pix] == 12:
                    continue

                # print(pix)
                vals = dic[pix][0]

                spatial_dic[pix] = vals
            arr = DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(spatial_dic)
            outf=fdir+f.replace('.npy','.tif')
            DIC_and_TIF(pixelsize=0.25).arr_to_tif(arr,outf)
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
            # plt.figure()
            # arr[arr > 0.1] = 1
            plt.imshow(arr,vmin=-0.5,vmax=0.5)


            plt.colorbar()

            plt.show()


def main():
    # RT_RS().run()
    # SHAP().run()
    # SHAP_pixel().run()

    # multi_regression_window().run()
    # multi_regression().run()
    # Random_Forests().run()
    SHAP_CV().run()
    # SHAP_rainfall_seasonality().run()
    # simple_linear_regression().run()
    # Partial_correlation().run()
    # bivariate_analysis().run()
    # single_correlation().run()
    # Partial_Dependence_Plots().run()
    pass

if __name__ == '__main__':
    main()