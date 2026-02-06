import copy

import matplotlib.pyplot as plt
import numpy as np
from lytools import *
import xymap
from pprint import pprint

from openpyxl.styles.builtins import percent
# from requests.packages import target
from setuptools.command.rotate import rotate

T = Tools()
D = DIC_and_TIF(pixelsize=0.5)
centimeter_factor = 1/2.54

this_root = 'D:\Project3\\'
data_root = 'D:/Project3/Data/'
result_root = 'D:/Project3/Result/Nov/'

class Delta_regression:

    def __init__(self):
        self.xvar = ['Precip_sum_detrend_CV_zscore',
                     'CV_daily_rainfall_average_zscore',  ]

        # self.model_list = ['composite_LAI_median','composite_LAI_mean']

        self.model_list = ['composite_LAI_mean','composite_LAI_median', 'SNU_LAI', 'GLOBMAP_LAI', 'LAI4g' ]

        # self.outdir = rf'D:\Project3\Result\Nov\Multiregression_contribution\Obs\\result_new\\1mm\\'
        # T.mkdir(self.outdir, True)

        pass


    def run(self):

        ## step 1 zscore
        # self.zscore()
        # # step 2 build dataframe manually
        # df=self.build_df()
        # self.append_attributes(df)





        ##### step 1

        for model in self.model_list:
            x_list=self.xvar+[model+'_sensitivity_zscore']
            # self.do_multi_regression(model, x_list)
            ## not using below function
            # self.do_multi_regression_control_experiment(model,x_list) ## not use this but the result is the same


            # self.calculate_trend_contribution(model,x_list)

        ## step 2
        ### before calculating contribution, build dataframe
        self.statistic_contribution_no_residual()  ## use this


        ###########################################3


        # self.maximum_contribution() not use
        # self.dominant_region_trends() not use


        # self.sensitivity_vs_climate_factors()  ## not used
        # self.sensitivity_vs_climate_factors()  ## not used
        # self.sensitivity_vs_climate_factors_2()  ## not used


        # self.normalized_partial_corr()  ## not used

        # self.heatmap2() ## not used
        # self.calculate_mean()  ## not used

        pass

    def zscore(self):
        NDVI_mask_f = data_root + rf'/Base_data/aridity_index_05/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        dic_dryland_mask = DIC_and_TIF().spatial_arr_to_dic(array_mask)

        fdir = result_root + rf'\Multiregression_contribution\Obs\input\X\\'
        outdir = result_root + rf'\Multiregression_contribution\Obs\input\Y\\zscore\\'
        T.mk_dir(outdir, force=True)
        Tools().mk_dir(outdir, force=True)
        for f in os.listdir(fdir):
            if not '5mm' in f:
                continue


            dic = T.load_npy(fdir + f)
            outf = outdir + f.split('.')[0] + '_zscore.npy'

            zscore_dic = {}

            for pix in tqdm(dic):

                if pix not in dic_dryland_mask:
                    continue

                # time_series = dic[pix]['intersensitivity_precip_val']
                time_series = dic[pix]

                # # 检查 time_series 是否为 list 或 array（防止是 float/NaN）

                if not isinstance(time_series, (list, np.ndarray)):
                    print(f"{pix}: invalid time_series (not iterable): {time_series}")
                    continue

                time_series = np.array(time_series, dtype=float)
                # time_series = time_series[3:37]

                print(len(time_series))
                ## exclude nan

                if np.isnan(np.nanmean(time_series)):
                    continue
                # if np.nanmean(time_series) >999:
                #     continue
                if np.nanmean(time_series) < -999:
                    continue
                time_series = time_series
                mean = np.nanmean(time_series)
                zscore = (time_series - mean) / np.nanstd(time_series)

                zscore_dic[pix] = zscore

                # plt.plot(time_series)
                # plt.legend(['raw'])
                # # plt.show()
                #
                # #
                # plt.plot(zscore)
                # plt.legend(['zscore'])
                # # # plt.legend(['raw','zscore'])
                # plt.show()

                ## save
            np.save(outf, zscore_dic)

    def build_df(self,):

        fdir = result_root+rf'\Multiregression_contribution\Obs\input\X\zscore\\'
        all_dic = {}

        for f in os.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            if not 'zscore' in f:
                continue

            fname = f.split('.')[0]

            fpath = fdir + f

            dic = T.load_npy(fpath)
            key_name = fname

            all_dic[key_name] = dic
        # print(all_dic.keys())
        df = T.spatial_dics_to_df(all_dic)
        T.print_head_n(df)


        return df

    def append_attributes(self, df):  ## add attributes
        fdir = result_root+rf'\Multiregression_contribution\Obs\input\Y\zscore\\'

        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.npy'):
                continue


            # array=np.load(fdir+f)
            # dic = DIC_and_TIF().spatial_arr_to_dic(array)
            dic = T.load_npy(fdir + f)
            key_name = f.split('.')[0]
            # if not key_name in var_list:
            #     continue
            print(key_name)

            # df[key_name] = df['pix'].map(dic)
            # T.print_head_n(df)
            df = T.add_spatial_dic_to_df(df, dic, key_name)
        T.save_df(df, result_root + rf'\Multiregression_contribution\Obs\Dataframe\Dataframe_new.df')
        T.df_to_excel(df, result_root + rf'\Multiregression_contribution\Obs\Dataframe\Dataframe_new.xlsx')



        return df


    # def append_value(self, df):  ##补齐
    #
    #
    #
    #     for col in df.columns:
    #
    #         vals_new = []
    #
    #         for i, row in tqdm(df.iterrows(), total=len(df), desc=f'append {col}'):
    #             pix = row['pix']
    #             r, c = pix
    #             if r<60:
    #                 continue
    #             vals = row[col]
    #             print(vals)
    #             if type(vals) == float:
    #                 vals_new.append(np.nan)
    #                 continue
    #             vals = np.array(vals)
    #             print(len(vals))
    #
    #             if len(vals) == 25:
    #
    #                 vals = np.append(vals, np.nan)
    #                 vals_new.append(vals)
    #
    #             vals_new.append(vals)
    #
    #             # exit()
    #         df[col] = vals_new

        # T.save_df(df, result_root + rf'\Multiregression_contribution\Obs\Dataframe\Dataframe.df')
        # T.df_to_excel(df, result_root + rf'\Multiregression_contribution\Obs\Dataframe\Dataframe.xlsx')

    def do_multi_regression(self,mode_name,x_list):
        outdir = self.outdir + f'{mode_name}\\'
        T.mk_dir(outdir, force=True)

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        import statsmodels.api as sm
        df = self.load_df()
        T.print_head_n(df)

        var_list = x_list  + [mode_name+'_detrend_CV_zscore']
        spatial_dict = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']

            # ---------- 1. 收集所有变量 ----------
            series_dict = {}
            valid = True

            for var in var_list:
                val = row[var]
                if not isinstance(val, (list, np.ndarray)) or len(val) < 10:
                    valid = False
                    break
                series_dict[var] = np.array(val)

            if not valid:
                continue

            # ---------- 2. 对齐公共长度（关键） ----------
            min_len = min(len(v) for v in series_dict.values())

            for var in series_dict:
                series_dict[var] = series_dict[var][-min_len:]

            # ---------- 3. 构造 DataFrame（一次性） ----------
            df_i = pd.DataFrame(series_dict)

            # ---------- 4. 标准化（和 Methods 一致） ----------


            # ---------- 5. 回归 ----------
            X = df_i[x_list]
            y = df_i[mode_name + '_detrend_CV_zscore']

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            # ---------- 6. 提取 β ----------
            beta_dict = {var: model.params[var] for var in x_list}
            spatial_dict[pix] = beta_dict
        df_beta = T.dic_to_df(spatial_dict, 'pix')


        for x_var in x_list:
            spatial_dict_i = T.df_to_spatial_dic(df_beta,x_var)
            outf = join(outdir,f'{x_var}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_i,outf)
        T.open_path_and_file(self.outdir)

        pass

    def do_multi_regression_control_experiment(self,mode_name,x_list):
        outdir = self.outdir+f'{mode_name}\\'
        T.mk_dir(outdir,force=True)

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        import statsmodels.api as sm
        df = self.load_df()
        T.print_head_n(df)

        var_list = x_list  + [mode_name+'_detrend_CV_zscore']
        spatial_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']

            # === 检查变量是否都存在并获取长度 ===
            length_dict = {}
            valid = True
            for var_i in var_list:
                val = row[var_i]
                if isinstance(val, float) or len(val) == 0:
                    valid = False
                    break
                length_dict[var_i] = len(val)

            # === 如果长度不匹配，则跳过该像素 ===
            if len(set(length_dict.values())) > 1:
                print(f"Length mismatch at pixel {row['pix']}:")
                for k, v in length_dict.items():
                    print(f"   {k}: length={v}")
                continue


            df_i = pd.DataFrame()
            success = 0

            for var_i in var_list:
                if type(row[var_i]) == float:
                    success = 0
                    break
                if len(row[var_i]) == 0:
                    success = 0
                    break
                else:
                    success = 1
                df_i[var_i] = row[var_i]

            if not success:
                continue
            print(df_i[x_list])
            X = df_i[x_list]

            y = df_i[mode_name+'_detrend_CV_zscore']

            X=sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            y_pred = model.predict(X)
            # delta_y = y - y_pred

            dict_i = {}

            for x_var in x_list:
                X_constant = copy.copy(X)


                X_constant[x_var] = X_constant[x_var][0]
                y_pred_i = model.predict(X_constant)
                delta_y = y_pred - y_pred_i
                delta_i = X[x_var] - X[x_var][0]
                model_i = sm.OLS(delta_y, sm.add_constant(delta_i)).fit()
                beta = model_i.params[1]
                dict_i[x_var] = beta
            spatial_dict[pix] = dict_i
        df_beta = T.dic_to_df(spatial_dict,'pix')


        for x_var in x_list:
            spatial_dict_i = T.df_to_spatial_dic(df_beta,x_var)
            outf = join(outdir,f'{x_var}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_i,outf)
        T.open_path_and_file(outdir)

    def calculate_trend_contribution(self, y_variable, x_list):
        """
        Calculate the trend contribution of each variable:
        contribution = slope(x, y) * trend(x) / trend(y) * 100
        """

        trend_dir = result_root + rf'\Multiregression_contribution\Obs\input\X\zscore\trend\\'
        trend_dict = {}

        # === Load trend for each X variable ===
        for variable in x_list:
            fpath = join(trend_dir, f'{variable}_trend.tif')


            array, _, _, _, _ = ToRaster().raster2array(fpath)
            array[array < -9999] = np.nan
            spatial_dict = DIC_and_TIF().spatial_arr_to_dic(array)

            for pix, val in tqdm(spatial_dict.items(), desc=f'Loading {variable} trend'):
                if np.isnan(val):
                    continue
                if pix[0] < 60:  # skip high latitude
                    continue
                if pix not in trend_dict:
                    trend_dict[pix] = {}
                trend_dict[pix][variable] = val

        # === Load multiregression slopes ===
        fdir_slope = self.outdir + f'\\{y_variable}\\'

        multiregression_dic = {}
        for f in os.listdir(fdir_slope):
            if not f.endswith('.tif') or 'contrib' in f:
                continue
            arr, _, _, _, _ = ToRaster().raster2array(join(fdir_slope, f))
            multiregression_dic[f.split('.')[0]] = DIC_and_TIF().spatial_arr_to_dic(arr)


        # === Load Y trend and p-value ===
        fdir_Y = result_root + rf'\Multiregression_contribution\Obs\input\Y\zscore\trend\\'
        fy_trend = join(fdir_Y, f'{y_variable}_detrend_CV_zscore_trend.tif')
        fy_pval = join(fdir_Y, f'{y_variable}_detrend_CV_zscore_p_value.tif')

        arr_y_trend, _, _, _, _ = ToRaster().raster2array(fy_trend)
        arr_y_pval, _, _, _, _ = ToRaster().raster2array(fy_pval)

        dic_y_trend = DIC_and_TIF().spatial_arr_to_dic(arr_y_trend)
        dic_y_pval = DIC_and_TIF().spatial_arr_to_dic(arr_y_pval)

        # === Calculate contribution ===
        for var_i in x_list:
            if var_i not in multiregression_dic:
                print(f"Missing slope for {var_i}")
                continue

            spatial_dic = {}
            for pix in tqdm(multiregression_dic[var_i], desc=f'Calculating {var_i} contribution'):
                if pix not in trend_dict or var_i not in trend_dict[pix]:
                    continue
                if pix not in dic_y_trend or pix not in dic_y_pval:
                    continue

                trend_y = dic_y_trend[pix]
                p_value = dic_y_pval[pix]
                if np.isnan(trend_y) or np.isnan(p_value):
                    continue
                if p_value > 0.05 or abs(trend_y) < 1e-6:
                    continue
                if trend_y < 0:
                    continue

                val_multireg = multiregression_dic[var_i][pix]
                if np.isnan(val_multireg) or val_multireg < -9999:
                    continue

                val_trend = trend_dict[pix][var_i]
                val_contrib = val_multireg * val_trend / trend_y * 100
                spatial_dic[pix] = val_contrib

            # === Output contribution map ===
            arr_contrib = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            outpath = join(fdir_slope, f'{var_i}_trend_contrib.tif')
            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_contrib, outpath)



    def normalized_contribution(self):



        fdir_all=self.outdir


        for model in self.model_list:
            spatial_dicts = {}
            variables_list = []


            outdir=join(fdir_all,model)

            for f in os.listdir(join(fdir_all,model)):
                if not f.endswith('.tif'):
                    continue
                if  'contrib' in f:
                    continue
                if 'norm' in f:
                    continue

                if 'Ternary_plot' in f:
                    continue
                if 'color' in f:
                    continue

                print(f)
                fpath = join(fdir_all, model, f)
                fname=f.split('.')[0]
                spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
                spatial_dicts[fname] = spatial_dict_i
                variables_list.append(fname)

            df = T.spatial_dics_to_df(spatial_dicts)
            df = df.dropna(subset=variables_list, how='any')
            # T.print_head_n(df);exit()
            df_abs = pd.DataFrame()
            df_abs['pix'] = df['pix'].tolist()
            for var_i in variables_list:
                abs_vals = np.array(df[var_i].tolist())
                abs_vals = np.abs(abs_vals)
                df_abs[var_i] = abs_vals
            # T.print_head_n(df_abs);exit()

            norm_dict = {}
            # T.add_dic_to_df()

            for i, row in tqdm(df_abs.iterrows(), total=len(df_abs)):
                # print(row[variables_list])
                sum_vals = row[variables_list].sum()
                # print(sum_vals)
                # if sum_vals == 0:
                #     sum_vals = np.nan
                pix = row['pix']
                norm_dict_i = {}
                for var_i in variables_list:
                    var_i_norm = row[var_i] / sum_vals
                    norm_dict_i[f'{var_i}_norm'] = var_i_norm
                norm_dict[pix] = norm_dict_i

            df_abs = T.add_dic_to_df(df_abs, norm_dict, 'pix')

            for var_i in variables_list:
                dic_norm = T.df_to_spatial_dic(df_abs, f'{var_i}_norm', )
                DIC_and_TIF().pix_dic_to_tif(dic_norm, join(outdir, f'{var_i}_norm.tif'))
            ######T.save_df(df_abs,join(fdir,'df_normalized.df'));exit()

            ## df to dic

            # T.print_head_n(df_abs);exit()


            rgb_arr = np.zeros((360, 720, 4))
            # Ter = xymap.Ternary_plot()
            Ter = xymap.Ternary_plot(
                top_color=(67, 198, 219),
                left_color=(255, 165, 00),
                # left_color=(119,0,188),
                right_color=(230, 0, 230),
                # center_color=(85,85,85),
                center_color=(230, 230, 230),
                # center_color=(255,255,255),
            )

            for i, row in df_abs.iterrows():
                pix = row['pix']
                r, c = pix
                CV_IAV_norm = row[f'detrended_sum_rainfall_growing_season_zscore_norm']
                rainfall_frequency_norm = row[f'rainfall_frenquency_zscore_norm']
                composite_LAI_beta_mean_norm = row[f'{model}_sensitivity_zscore_norm']
                x, y, z = CV_IAV_norm, rainfall_frequency_norm, composite_LAI_beta_mean_norm
                color = Ter.get_color(x, y, z)
                color = color * 255
                color = np.array(color, dtype=np.uint8)
                alpha = 255
                color = np.append(color, alpha)
                # print(color);exit()

                rgb_arr[r][c] = color
            # xymap.GDAL_func().ar
            rgb_arr = np.array(rgb_arr, dtype=np.uint8)
            ### - 蓝绿色（上）： 主导
            # - 橙黄色（左下）： 主导
            # - 粉紫色（右下）：LAI_sensitivity（植被敏感性）主导
            outtif = join(outdir, 'Ternary_plot.tif')
            tif_template = join(fdir_all, model, f'rainfall_frenquency_zscore.tif')
            print(rgb_arr)

            xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
            grid_triangle_legend = Ter.grid_triangle_legend()
            plt.imshow(grid_triangle_legend)
            plt.show()
            # # T.open_path_and_file(fdir)
            # exit()



    def normalized_partial_corr(self):


        dff=result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_fire_zscore\Dataframe\\\\Dataframe.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        df.dropna(inplace=True)

        fdir=self.outdir

        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr,vmin=-0.5,vmax=0.5,cmap='jet',interpolation='nearest')
        plt.colorbar()
        plt.show()
        variables_list = []
        for var in self.xvar:
            var_new = f'{var}_contrib'
            variables_list.append(var_new)

        df_abs = pd.DataFrame()
        df_abs['pix'] = df['pix'].tolist()
        for var_i in variables_list:
            abs_vals = np.array(df[var_i].tolist())
            abs_vals = np.abs(abs_vals)
            df_abs[var_i] = abs_vals

        norm_dict = {}


        for i,row in tqdm(df_abs.iterrows(),total=len(df_abs)):

            sum_vals = row[variables_list].sum()

            pix = row['pix']
            norm_dict_i = {}
            for var_i in variables_list:
                var_i_norm = row[var_i] / sum_vals
                norm_dict_i[f'{var_i}_norm'] = var_i_norm
            norm_dict[pix] = norm_dict_i
        df = T.add_dic_to_df(df, norm_dict, 'pix')
        # T.print_head_n(df);exit()
        # for var_i in variables_list:
        #
        #     dic_norm=T.df_to_spatial_dic(df,f'{var_i}_norm',)
        #     DIC_and_TIF().pix_dic_to_tif(dic_norm,join(fdir,f'{var_i}_norm.tif'))
        # T.save_df(df_abs,join(fdir,'df_normalized.df'));exit()

        climate_weights_list = []
        for i,row in df.iterrows():

            detrended_sum_rainfall_CV = row['detrended_sum_rainfall_CV_contrib_norm']
            CV_intraannual_rainfall_ecosystem_year = row['CV_intraannual_rainfall_ecosystem_year_contrib_norm']
            climate_sum =  detrended_sum_rainfall_CV + CV_intraannual_rainfall_ecosystem_year
            climate_weights_list.append(climate_sum)
        df['climate_variability_norm']=climate_weights_list
        rgb_arr = np.zeros((360, 720, 4))
        # Ter = xymap.Ternary_plot()
        Ter = xymap.Ternary_plot(
            top_color=(67, 198, 219),
            left_color=(255, 165, 00),
            # left_color=(119,0,188),
            right_color=(230, 0, 230),
            # center_color=(85,85,85),
            center_color=(230, 230, 230),
            # center_color=(255,255,255),
        )

        for i, row in df.iterrows():
                pix = row['pix']
                r,c = pix
                climate_norm = row['climate_variability_norm']
                Fire_sum_max_norm = row['sum_rainfall_contrib_norm']
                composite_LAI_beta_mean_norm = row[f'composite_LAI_beta_mean_contrib_norm']
                x,y,z = climate_norm, Fire_sum_max_norm, composite_LAI_beta_mean_norm
                color = Ter.get_color(x,y,z)
                color = color * 255
                color = np.array(color,dtype=np.uint8)
                alpha = 255
                color = np.append(color, alpha)
                # print(color);exit()

                rgb_arr[r][c] = color
        # xymap.GDAL_func().ar
        rgb_arr = np.array(rgb_arr, dtype=np.uint8)
        outtif = join(fdir, 'Ternary_plot.tif')
        tif_template = join(fdir,os.listdir(fdir)[0])
        print(rgb_arr)

        xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
        grid_triangle_legend = Ter.grid_triangle_legend()
        plt.imshow(grid_triangle_legend)
        plt.show()
        T.open_path_and_file(fdir)
        exit()

    def max_correlation_with_sign(self):

        dff = result_root + rf'3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_5\\statistics\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        model_list = self.model_list
        # for col in df.columns:
        #     print(col)
        # exit()

        var_list = ['detrended_sum_rainfall_ecosystem_year_CV_zscore',
                    'CV_intraannual_rainfall_ecosystem_year_zscore',
                    'sensitivity_zscore', ]

        for model in tqdm(model_list):

            outdir = self.outdir + model + '\\'
            T.mk_dir(outdir, force=True)
            var_list_contrib = [f'{model}_' + i + '_contrib' for i in var_list]

            var_list_sens = [f'{model}_' + i for i in var_list]

            max_var_list = []
            max_var_sign_list = []
            color_list = []
            df_continent = df
            for i, row in df_continent.iterrows():
                vals_contrib = row[var_list_contrib].tolist()
                vals_contrib = np.array(vals_contrib)
                vals_contrib[vals_contrib < -10] = np.nan
                vals_contrib[vals_contrib > 10] = np.nan

                vals_sens = row[var_list_sens].tolist()
                vals_sens = np.array(vals_sens)
                vals_sens[vals_sens < -10] = np.nan
                vals_sens[vals_sens > 10] = np.nan

                if True in np.isnan(vals_contrib):
                    max_var_list.append(np.nan)
                    max_var_sign_list.append(np.nan)
                    color_list.append(np.nan)
                    continue
                vals_contri_abs = np.abs(vals_contrib)
                vals_contrib_dict = T.dict_zip(var_list_contrib, vals_contri_abs)
                vals_sens_dic = T.dict_zip(var_list_sens, vals_sens)

                max_var = T.get_max_key_from_dict(vals_contrib_dict)
                max_var_new_dict = {f'{model}_detrended_sum_rainfall_ecosystem_year_CV_zscore_contrib':
                                        f'{model}_detrended_sum_rainfall_ecosystem_year_CV_zscore',
                                    f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore_contrib':
                                        f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore',
                                    f'{model}_sensitivity_zscore_contrib':
                                        f'{model}_sensitivity_zscore',
                                    }

                max_val = vals_sens_dic[max_var_new_dict[max_var]]

                if max_val > 0:
                    max_var_sign = '+'
                else:
                    max_var_sign = '-'

                if 'sensitivity' in max_var:
                    if max_var_sign == '-':
                        color = 1
                    else:
                        color = 6
                elif 'CV_intraannual_rainfall_ecosystem_year_zscore' in max_var:
                    if max_var_sign == '-':
                        color = 2
                    else:
                        color = 5
                elif 'detrended_sum_rainfall_ecosystem_year_CV_zscore' in max_var:
                    if max_var_sign == '-':
                        color = 3
                    else:
                        color = 4
                else:

                    continue
                max_var_list.append(max_var)
                max_var_sign_list.append(max_var_sign)
                color_list.append(color)

            df_continent['max_var'] = max_var_list

            df_continent['max_var_sign'] = max_var_sign_list
            df_continent['color'] = color_list

            ## to tiff
            spatial_dic = T.df_to_spatial_dic(df, 'color')
            outtif = join(outdir, 'color_map.tif')
            array = DIC_and_TIF().pix_dic_to_tif(spatial_dic, outtif)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(arr, interpolation='nearest')
            # plt.colorbar()
            # plt.show()


    def statistic_contribution_no_residual(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        dff = result_root + rf'\Multiregression_contribution\Obs\Dataframe\\1mm\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)


        # === 配色方案 ===
        color_list = ['#a577ad', 'yellowgreen', 'Pink', '#f599a1']
        dark_colors = ['#774685', 'Olive', 'Salmon', '#c3646f']  # 可以改为你自定义的 darken_color 函数


        for model in self.model_list:
            if not 'median' in model:
                continue
            # === 变量名 ===
            fixed_order = [
                f'{model}_sensitivity_zscore_trend_contrib',
                f'{model}_Precip_sum_detrend_CV_zscore_trend_contrib',
                f'{model}_CV_daily_rainfall_average_zscore_trend_contrib'
                # f'{model}_CV_daily_rainfall_3mm_average_zscore_trend_contrib'
            ]

            label_map = {
                f'{model}_sensitivity_zscore_trend_contrib': 'γ',
                f'{model}_Precip_sum_detrend_CV_zscore_trend_contrib': 'CV_inter',
                # f'{model}_CV_daily_rainfall_average_zscore_trend_contrib': 'CV_intra'
                f'{model}_CV_daily_rainfall_average_zscore_trend_contrib':'CV_intra'

            }

            means, sems, labels = [], [], []
            print(len(df))



            df = df[df[f'{model}_detrend_CV_zscore_trend'] > 0]
            df = df[df[f'{model}_detrend_CV_zscore_p_value'] < 0.05]
            #
            print(len(df));exit()

            # === 计算平均值和标准误差 ===
            for var in fixed_order:
                if var not in df.columns:
                    continue
                vals = np.array(df[var].values, dtype=float)
                vals[(vals > 99) | (vals < -99)] = np.nan
                vals = vals[~np.isnan(vals)]
                # print(vals);exit()
                if len(vals) == 0:
                    continue

                mean_val = np.nanmean(vals)
                # print(np.std(vals));exit()
                sem_val = np.nanstd(vals) / np.sqrt(len(vals))  # 标准误差

                means.append(mean_val)
                sems.append(sem_val)
                labels.append(label_map[var])
            print(sems)
            print(means);exit()
            # print(f'{model}:', means)

            # === 绘图 ===
            fig, ax = plt.subplots(figsize=(4, 3))
            x = np.arange(len(labels))
            colors = color_list
            edges = dark_colors

            bars = ax.bar(
                x, means, width=0.4,
                color=colors, edgecolor=edges, linewidth=1.2, zorder=2
            )

            # 误差线
            for xi, mean, sem, edge in zip(x, means, sems, edges):
                ax.errorbar(
                    xi, mean, yerr=sem,
                    fmt='none', ecolor=edge, elinewidth=1.2, capsize=4, zorder=3
                )

            # 美化
            ax.axhline(0, color='gray', linestyle='--', lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=12)
            ax.set_ylabel('Attribution of CVLAI (%)', fontsize=12)
            ax.set_yticklabels(ax.get_yticks(), fontsize=12)
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['left'].set_linewidth(1)
            # ax.spines['bottom'].set_linewidth(1)
            ax.tick_params(axis='y', width=1, length=3)
            ax.tick_params(axis='x', width=1, length=0)
            # plt.tight_layout()

            # === 输出保存 ===
            outdir =result_root + rf'\FIGURE\\SI\\'
            print(outdir)
            #
            Tools().mk_dir(outdir, force=True)
            outf = os.path.join(outdir, f'{model}_relative_contribution_median_1mm.pdf')
            plt.savefig(outf, bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close()

    def statistic_contribution(self):

        model_list = self.model_list


        dff = result_root + rf'\Multiregression_contribution\Obs\Dataframe\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df = df[df['composite_LAI_detrend_CV_median_trend'] > 0]
        df = df[df['composite_LAI_detrend_CV_median_p_value'] < 0.05]


        # 设置变量名

        result_stat_dict = {}
        x_list = []

        for model in model_list:
            for variable in self.xvar:
                x_list.append(f'{model}_{variable}_contrib')
            x_list.append(f'{model}_sensitivity_zscore_contrib')


            labels, means, sems, = [], [], []


            for variable in x_list:



                values = df[variable].values
                values = np.array(values)
                # plt.hist(values, bins=20)
                # plt.show()
                values[values > 99] = np.nan
                values[values < -99] = np.nan
                values = values[~np.isnan(values)]
                values_100 = values


                vals_CV_obs = df[f'{model}_detrend_CV_zscore_trend'].values
                vals_CV_obs_mean = np.nanmean(vals_CV_obs)
                values_average = np.nanmean(values)


                valus_relative_contribution = (values_average)
                standard_error = np.nanstd(values_100)
                result_stat_dict[f'{model}_{variable}']=[valus_relative_contribution,sems]

                labels.append(variable)
                means.append(valus_relative_contribution)
                sems.append(standard_error)


            residual = df[f'{model}_residual_contrib'].values
            residual_100 = residual * 100
            result_stat_dict['residual_contrib'] = np.nanmean(residual) / vals_CV_obs_mean * 100
            labels.append('residual')
            means.append(result_stat_dict['residual_contrib'])
            sems.append(np.nanstd(residual_100))

            result_stat_dict[model] = {
                "labels": labels,
                "means": np.array(means),
                "sems": np.array(sems),

            }


        for model in model_list:
            labels = result_stat_dict[model]["labels"]


            means = result_stat_dict[model]["means"]
            sems = result_stat_dict[model]["sems"]
            means = np.array(means)
            sems = np.array(sems)

            order = np.argsort(means)[::-1]  # 从大到小排序
            labels = [labels[i] for i in order]
            means = means[order]
            sems = sems[order]
            print(labels)
            print(means)


            label_map = {
                f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore_contrib': 'CV intra-rainfall',
                f'{model}_detrended_sum_rainfall_ecosystem_year_CV_zscore_contrib': 'CV inter-rainfall',
                'residual': 'Residual',


            }

            xtick_labels = []
            for lab in labels:
                if 'sensitivity' in lab:
                    xtick_labels.append('γ')
                else:
                    xtick_labels.append(label_map.get(lab, lab))




            x = np.arange(len(xtick_labels))

            plt.figure(figsize=(4, 3))




            # 1) 柱状 + 误差线
            bar_width = 0.5
            offset = 0.22  # 往右挪一点画 violin/散点

            # 1) 柱子（居中）
            ## err bar color
            color_list=['#a577ad','#9fd79e', '#73c79e', '#f599a1',  ]

            dark_colors = [self.darken_color(c, 0.8) for c in color_list]

            bars = plt.bar(
                x, means, width=bar_width,
                color=color_list, edgecolor=dark_colors, linewidth=1, zorder=1
            )

            # 再逐个加误差线
            for xi, mean, sem, dc in zip(x, means, sems, dark_colors):
                plt.errorbar(
                    xi, mean, yerr=sem,
                    fmt='none', ecolor=dc, elinewidth=1, capsize=5, zorder=2
                )

            plt.xticks(x, xtick_labels, rotation=0, ha='center', fontsize=10)
            plt.ylabel("Relative contribution (%)", fontsize=10)
            # plt.tight_layout()

            # #
            plt.show()
            outdir=result_root+rf'\3mm\FIGURE\Figure3_attribution\\'
            outf=join(outdir,f'{model}_relative_contribution.pdf')
            plt.savefig(outf,bbox_inches='tight',dpi=300)
            plt.close()

    def darken_color(self,color, amount=0.7):
        """
        给颜色加深，amount 越小越深 (0~1之间)
        """
        import matplotlib.colors as mcolors
        c = mcolors.to_rgb(color)
        return tuple([max(0, x * amount) for x in c])


    def statistic_contribution_area(self):
        dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_5\statistics\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df = df.dropna(subset=['composite_LAI_color_map'])

        percentage_list=[]
        sum=0


        for ii in [1,2,3,4,5,6]:
            df_ii=df[df['composite_LAI_median_color_map']==ii]
            percent=len(df_ii)/len(df)*100
            sum=sum+percent
            percentage_list.append(percent)
        print(percentage_list)
        print(sum)

        ## plot

        color_list = [

            '#f599a1', '#fcd590',
            '#e73618', '#dae67a',
            '#9fd7e9', '#a577ad',

        ]



        plt.figure(figsize=(4,3))
        plt.bar([1,2,3,4,5,6], percentage_list, color=color_list)
        ## save fig
        plt.ylabel('Area precentage (%)')

        # plt.tight_layout()
        outdir=result_root + (rf'\3mm\FIGURE\\Robinson\\')
        T.mk_dir(outdir,force=True)
        plt.show()

        # plt.savefig(outdir+'Area_precentage_composite_LAI_mean.pdf',dpi=300)





        pass

    def sensitivity_vs_climate_factors(self):
        dff=result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_2\Dataframe\\statistic.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        df.dropna(inplace=True)


        # 设置变量名
        target_var_list=[
                         'detrended_sum_rainfall_growing_season_zscore_sensitivity',
                         'rainfall_frenquency_zscore_sensitivity']


        color_list=['#dd492c','#c05f77','#b5869a',
'#bfa8b1','#d5c1ca','#e7dce1',
        ]

        sand_color_list= ['#ffffe5', '#fffaca', '#fff0ae', '#fee391', '#fece65',
    '#feb642','#fe9929',  '#f27e1b',  '#e1640e', '#cc4c02', '#aa3c03', '#882f05', '#662506']
        aridity_color_list=['#d73027','#ea6f44', '#f48e52', '#fec279', '#fed690',
                            '#ffffbf', '#f7fccd','#f0f9dc', '#e8f6ea', '#e0f3f8']
        root_depth_list=[
            '#f1e0b6',
            '#e6c981',
            '#d9b15a',
            '#c89b43',
            '#a88432',
            '#7f6d28',
            '#4f7f3b',
            '#2e7031',
            '#00441b'  # 深根
        ]
        short_vegetation_cover=[ '#ffffcc',  # 极低覆盖
    '#d9f0a3',
    '#addd8e',
    '#78c679',
    '#41ab5d',
    '#238443',
    '#006837',
    '#004529'   ]



        for target_var in target_var_list:


            bin_var = 'Burn_area_sum'
            # bin_var = 'sand'
            # bin_var = 'dry_freq'
            # bin_var='soc'
            # bin_var = 'sum_rainfall_mean'
            # bin_var = 'Tree cover_mean'
            # bin_var='Non_tree_vegetation_mean'
            # bin_var = 'rooting_depth_05'
            plt.hist(df[bin_var])
            plt.show()
            # bin_edges = np.arange(200,1201,100)
            # bin_edges = np.arange(0,501,50)
            bin_edges = np.arange(0,5001,500)
            # bin_edges=np.arange(10,91,10)
            # bin_edges = np.arange(0.2, 0.66, 0.05)
            # bin_edges = np.arange(150,850,50)
            # bin_edges = np.arange(0,25,3)
            # bin_edges = np.quantile(df[bin_var], np.linspace(0, 0.66, 11))
            bin_labels = [f'{round(bin_edges[i], 2)}-{round(bin_edges[i + 1], 2)}' for i in range(len(bin_edges) - 1)]
            # bin_labels = [f'{round(bin_edges[i ], 2)}' for i in range(len(bin_edges) - 1)]

            df['bin'] = pd.cut(df[bin_var], bins=bin_edges, labels=bin_labels, include_lowest=True)

            # 初始化结果字典
            result_dic = {}
            count_list=[]

            for label in bin_labels:
                df_bin = df[df['bin'] == label][[target_var]].dropna()

                if len(df_bin) == 0:
                    result_dic[label] = [0, 0, 0, 0]
                    continue

                mean_val = np.nanmean(df_bin[target_var])
                std_err = np.nanstd(df_bin[target_var]) / np.sqrt(len(df_bin))  # 标准误差
                result_dic[label] = [mean_val, std_err]
                count_list.append(len(df_bin))

            # 构造 DataFrame
            result_df = pd.DataFrame(result_dic).T
            result_df.columns = ['mean', 'std_err']
            result_df.index = bin_labels

            # 画图





            ax = result_df['mean'].plot(
                kind='bar',
                yerr=result_df['std_err'],
                figsize=(4, 3),
                color=aridity_color_list,
                capsize=3,
                error_kw={'elinewidth': 1, 'ecolor': 'gray'},
                edgecolor='gray',
            )

## add count

            for i, bar in enumerate(ax.patches):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        height+0.02, # 微调高度

                        f'{count_list[i]}',
                        ha='center', va='bottom', fontsize=10,rotation=90)


            plt.axhline(y=0, color='gray', linestyle='-')
            ## xtick every 100 label
            xticks = ax.get_xticks()
            # xticklabels = [label.get_text() for label in ax.get_xticklabels()]
            # new_labels = [label if i % 2 == 0 else '' for i, label in enumerate(xticklabels)]
            # ax.set_xticklabels(new_labels, rotation=0)




            if target_var=='detrended_sum_rainfall_CV_zscore_sensitivity':
                plt.ylabel('CV Interannual Rainfall (zscore)')


            elif target_var=='rainfall_frenquency_zscore_sensitivity':
                plt.ylabel('Fq Rainfall(zscore)')




            if target_var=='composite_LAI_beta_mean_zscore_contrib':
                plt.xticks([])
            elif target_var=='rainfall_frenquency_zscore_contrib':
                plt.xticks([])
            else:
                plt.xticks(rotation=45)


            #
            plt.tight_layout()
            plt.show()
            # ## save pdf
            # fig = ax.get_figure()
            # outdir=result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_2\figure\\'
            # T.mk_dir(outdir, force=True)
            # fig.savefig(outdir + f'{target_var}_{bin_var}.pdf', dpi=300, bbox_inches='tight')
            # plt.close(fig)









    def maximum_contribution(self):
        fdir = self.outdir
        array_dic_all = {}
        array_arg = {}

        var_name_list = []
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if  not 'contrib' in f:
                continue
            if 'max_label' in f:
                continue
            if 'Ternary_plot' in f:
                continue
            var_name = f.split('.')[0]
            var_name_list.append(var_name)
            print(f)
            fpath = join(fdir, f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            array_dic_all[var_name] = spatial_dict

        spatial_df = T.spatial_dics_to_df(array_dic_all)
        max_key_list = []
        max_val_list = []
        for i, row in spatial_df.iterrows():
            vals = row[var_name_list].tolist()
            vals = np.array(vals)
            var_name_list_array = np.array(var_name_list)
            vals_no_nan = vals[~np.isnan(vals)]
            var_name_list_array_no_nan = var_name_list_array[~np.isnan(vals)]
            vals_dict = T.dict_zip(var_name_list_array_no_nan, vals_no_nan)
            # if True in np.isnan(vals):
            # max_key_list.append(np.nan)
            # max_val_list.append(np.nan)
            # continue
            max_key = T.get_max_key_from_dict(vals_dict)
            max_val = vals_dict[max_key]
            max_key_list.append(max_key)
            max_val_list.append(max_val)
            # print(vals_dict)
            # print(max_key)
            # print(max_val)
            # exit()
        spatial_df['max_key'] = max_key_list
        spatial_df['max_val'] = max_val_list
        T.print_head_n(spatial_df)
        spatial_df.dropna()
        ## df to tif
        dic_label = {'CV_intraannual_rainfall_ecosystem_year_contrib': 2,
                     'detrended_sum_rainfall_CV_contrib': 3,
                     'composite_LAI_beta_mean_contrib':1,

                     }

        spatial_df['max_label'] = spatial_df['max_key'].map(dic_label)
        # # ## calculate _percentage
        # for ii in range(1, 5):
        #     percent=spatial_df[spatial_df['max_label']==ii].shape[0]/spatial_df.shape[0]*100
        #     percent=round(percent,2)
        #     print(ii,percent)
        #
        #     plt.bar(ii,percent)
        # # plt.show()
        #
        #
        spatial_dict = T.df_to_spatial_dic(spatial_df, 'max_label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(spatial_dict, self.outdir + 'max_label.tif')

        dff_new=result_root+ rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\Dataframe\\Dataframe.df'
        df=T.load_df(dff_new)
        df=self.df_clean(df)
        df=df.dropna()

        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()

        for ii in range(1, 4):
            percent = df[df['max_label'] == ii].shape[0] / df.shape[0] * 100
            percent = round(percent, 2)
            print(ii, percent)

            plt.bar(ii, percent)
        plt.show()











    def heatmap2(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        dff=result_root + rf'3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_2\Dataframe\\statistic.df'
        df=T.load_df(dff)
        df=self.df_clean(df)
        df.dropna(inplace=True)
        ###df =color map ==3 and 4
        df = df[df['color_map'].isin([2, 5])]



        # df=df.dropna()
        T.print_head_n(df)
        x_var = 'sand'

        y_var = 'rainfall_frenquency_mean'
        plt.hist(df[y_var])
        plt.show()
        plt.hist(df[x_var])
        plt.show()
        z_var = 'rainfall_frenquency_zscore_sensitivity'
        # z_var='Fire_sum_average'

        bin_x= np.linspace(300,800,7)
        bin_y = np.linspace(40,90,7)

        # percentile_list=np.linspace(0,100,9)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure()


        matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)
        self.plot_df_bin_2d_matrix(matrix_dict, -.7, .7, x_ticks_list, y_ticks_list, cmap='Viridis',
                                is_only_return_matrix=False)
        plt.colorbar()
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.show()

        # plt.figure()

        matrix_dict, x_ticks_list, y_ticks_list = self.df_bin_2d_sample_size(df, val_col_name=z_var,
                                                                             col_name_x=x_var,
                                                                             col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        T.plot_df_bin_2d_matrix(matrix_dict, 0, 100, x_ticks_list, y_ticks_list, cmap='RdBu',
                                is_only_return_matrix=False)



        plt.colorbar()
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.show()

    def plot_df_bin_2d_matrix(self,matrix_dict,vmin,vmax,x_ticks_list,y_ticks_list,cmap='RdBu',
                              is_only_return_matrix=False):
        keys = list(matrix_dict.keys())
        r_list = []
        c_list = []
        for r, c in keys:
            r_list.append(r)
            c_list.append(c)
        r_list = set(r_list)
        c_list = set(c_list)

        row = len(r_list)
        col = len(c_list)
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in matrix_dict:
                    val_pix = matrix_dict[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        matrix = np.array(spatial, dtype=float)
        matrix = matrix[::-1]
        if is_only_return_matrix:
            return matrix
        plt.imshow(matrix,cmap='RdBu',vmin=vmin,vmax=vmax)
        plt.xticks(range(len(c_list)), x_ticks_list)
        plt.yticks(range(len(r_list)), y_ticks_list[::-1])
    def df_bin_2d_sample_size(self,df,val_col_name,col_name_x,col_name_y,bin_x,bin_y,round_x=2,round_y=2):
        df_group_y, _ = T.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = T.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                rt_mean = len(vals)
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict,x_ticks_list,y_ticks_list







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


    def load_df(self):
        dff=result_root+'\Multiregression_contribution\Obs\Dataframe\\Dataframe_new.df'

        df = T.load_df(dff)
        # exit()
        # start_year = 0
        # end_year = 21
        # variable_list = self.xvar + self.y_var
        # df = Dataframe_per_value_transform(df, variable_list, start_year, end_year).df
        # T.print_head_n(df)
        return df


class Delta_regression_TRENDY:

    def __init__(self):
        self.map_width = 8.2 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        self.xvar = ['CV_intraannual_rainfall_ecosystem_year_zscore',
                     'detrended_sum_rainfall_ecosystem_year_CV_zscore', ]

        self.model_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                           'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                           'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                           'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                           'ORCHIDEE_S2_lai',

                           'YIBs_S2_Monthly_lai',


                           ]
        #
        # self.model_list = ['TRENDY_ensemble_median',
        #
        #                    ]

        self.model_list = ['composite_LAI_median','LAI4g',
                           'GLOBMAP_LAI','SNU_LAI',

                           'TRENDY_ensemble_median','CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                           'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                           'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                           'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                           'ORCHIDEE_S2_lai',

                           'YIBs_S2_Monthly_lai',

                           ]




        self.outdir = rf'D:\Project3\Result\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg_3\\'
        T.mkdir(self.outdir, force=True)

        pass

    def run(self):
        df = self.load_df()
        # ##self.do_multi_regression()

        # for model in self.model_list:
        #     x_list=self.xvar+[model+'_sensitivity_zscore']
        #
        #     self.do_multi_regression_control_experiment(model,x_list) ## not use this but the result is the same
        # # #
        # # # #
        #     self.calculate_trend_contribution(model,x_list)
        # self.statistic_contribution()
        # self.ensemble_trend_contribution()
        # self.normalized_contribution()
        # self.Ternary_plot()
        # self.plot_pdf()

        # self.max_correlation_with_sign()
        # self.statistic_contribution_area_heatmap()
        # self.statistic_contribution_area_barplot()

        # self.TRENDY_barplot()
        # self.TRENDY_barplot2()

        # self.maximum_contribution()
        # self.dominant_region_trends()
        # self.statistic_max_correlation()
        # self.Figure2_robinson()
        self.statistic_contribution_area()

        # self.sensitivity_vs_climate_factors()
        # self.statistic_contribution_area_barplot()


        # self.percentage_pft()
        # self.sensitivity_vs_climate_factors()
        # self.sensitivity_vs_climate_factors_2()

        # self.normalized_partial_corr()

        # self.heatmap2()
        # self.calculate_mean()

        pass

    def do_multi_regression(self):
        self.outdir = self.outdir

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        import statsmodels.api as sm
        df = self.load_df()
        T.print_head_n(df)

        var_list = self.xvar + y_var
        spatial_dict = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            df_i = pd.DataFrame()

            # 构造时序 DataFrame
            valid = True
            for var in var_list:
                val = row[var]
                if not isinstance(val, (list, np.ndarray)) or len(val) == 0:
                    valid = False
                    break
                df_i[var] = val

            if not valid:
                continue

            X = df_i[self.xvar]
            y = df_i[self.y_var[0]]

            # 标准多元回归
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            # 提取 β 系数（剔除常数项）
            beta_dict = {var: model.params[var] for var in self.xvar}
            spatial_dict[pix] = beta_dict

        df_beta = T.dic_to_df(spatial_dict, 'pix')

        for x_var in self.xvar:
            spatial_dict_i = T.df_to_spatial_dic(df_beta, x_var)
            outf = join(self.outdir, f'{x_var}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_i, outf)
        T.open_path_and_file(self.outdir)

        pass

    def do_multi_regression_control_experiment(self, mode_name, x_list):
        outdir = self.outdir + f'{mode_name}\\'
        T.mk_dir(outdir, force=True)

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        import statsmodels.api as sm
        df = self.load_df()
        T.print_head_n(df)

        var_list = x_list + [mode_name + '_detrend_CV_zscore']
        spatial_dict = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            df_i = pd.DataFrame()
            success = 0
            for var_i in var_list:
                if type(row[var_i]) == float:
                    success = 0
                    break
                if len(row[var_i]) == 0:
                    success = 0
                    break
                else:
                    success = 1
                if len(row[var_i]) !=24:
                    success = 0
                    break

                df_i[var_i] = row[var_i]
            if not success:
                continue
            X = df_i[x_list]
            ### add interaction terms

            y = df_i[mode_name + '_detrend_CV_zscore']

            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            y_pred = model.predict(X)
            # delta_y = y - y_pred

            dict_i = {}

            for x_var in x_list:
                X_constant = copy.copy(X)

                X_constant[x_var] = X_constant[x_var][0]
                y_pred_i = model.predict(X_constant)
                delta_y = y_pred - y_pred_i
                delta_i = X[x_var] - X[x_var][0]
                model_i = sm.OLS(delta_y, sm.add_constant(delta_i)).fit()
                beta = model_i.params[1]
                dict_i[x_var] = beta
            spatial_dict[pix] = dict_i
        df_beta = T.dic_to_df(spatial_dict, 'pix')

        for x_var in x_list:
            spatial_dict_i = T.df_to_spatial_dic(df_beta, x_var)
            outf = join(outdir, f'{x_var}.tif')
            DIC_and_TIF().pix_dic_to_tif(spatial_dict_i, outf)
        T.open_path_and_file(outdir)

    def calculate_trend_contribution(self, y_variable, x_list):
        ## here I would like to calculate the trend contribution of each variable
        ## the trend contribution is defined as the slope of the linear regression between the variable and the target variable mutiplied by trends of the variable
        ## load the trend of each variable
        ## load the trend of the target variable
        ## load multi regression result
        ## calculate the trend contribution
        trend_dir = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\Input\X_new\trend\\'

        selected_vairables_list = x_list

        trend_dict = {}
        for variable in selected_vairables_list:
            fpath = join(trend_dir, f'{variable}_trend.tif')
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array[array < -9999] = np.nan
            spatial_dict = D.spatial_arr_to_dic(array)
            for pix in tqdm(spatial_dict, desc=variable):
                r, c = pix
                if r < 60:
                    continue
                val = spatial_dict[pix]
                if np.isnan(val):
                    continue
                if not pix in trend_dict:
                    trend_dict[pix] = {}
                key = variable
                trend_dict[pix][key] = spatial_dict[pix]

        outdir = self.outdir + f'\\{y_variable}\\'

        fdir_slope = outdir

        multiregression_dic = {}
        for f in os.listdir(fdir_slope):
            if not f.endswith('.tif'):
                continue
            if 'contrib' in f:
                continue


            arr_multiregression, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
                join(fdir_slope, f))
            dic_multiregression = DIC_and_TIF().spatial_arr_to_dic(arr_multiregression)
            multiregression_dic[f.split('.')[0]] = dic_multiregression

            # exit()
        for var_i in x_list:
            spatial_dic = {}
            for pix in tqdm(dic_multiregression, desc=var_i):
                if not pix in trend_dict:
                    continue

                vals = multiregression_dic[var_i][pix]
                if vals < -9999:
                    continue

                val_multireg = vals
                if var_i not in trend_dict[pix]:
                    continue

                val_trend = trend_dict[pix][var_i]
                val_contrib = val_multireg * val_trend
                spatial_dic[pix] = val_contrib
            arr_contrib = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr_contrib, cmap='RdBu', interpolation='nearest')
            plt.colorbar()
            # plt.title(var_i)
            # plt.show()

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_contrib, join(outdir, f'{var_i}_contrib.tif'))

    def ensemble_trend_contribution(self):

        model_list=self.model_list

        fdir=result_root+rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\X\\'
        arr_list = []

        arr_sensitivity=[]

        for model in model_list:
            fdir_i=result_root+rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\X\\'+model+'\\'
            for f in os.listdir(fdir_i):
                if not f.endswith('.tif'):
                    continue
                if 'sensitivity_zscore' in f:
                    arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_i,f))
                    arr_sensitivity.append(arr)
        arr_sensitivity=np.nanmedian(arr_sensitivity, axis=0)
        arr_sensitivity[arr_sensitivity > 99] = np.nan
        arr_sensitivity[arr_sensitivity < -99] = np.nan
        plt.imshow(arr_sensitivity, cmap='RdYlGn')
        plt.colorbar()
        plt.show()
        outdir=result_root+rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg_2\\TRENDY_ensemble_median\\'
        T.mk_dir(outdir, force=True)
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_sensitivity, outdir + f'TRENDY_ensemble_sensitivity_zscore.tif')
        #




        for variable in self.xvar:
            for model in model_list:

                fpath = join(fdir,model,f'{variable}.tif')
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr[arr > 99] = np.nan
                arr[arr < -99] = np.nan

                arr_list.append(arr)

            arr_ensemble = np.nanmedian(arr_list, axis=0)
            arr_ensemble[arr_ensemble > 99] = np.nan
            arr_ensemble[arr_ensemble < -99] = np.nan
            plt.imshow(arr_ensemble, cmap='RdYlGn')
            plt.colorbar()
            plt.show()
            outdir=result_root+rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg_2\\TRENDY_ensemble_median\\'
            T.mk_dir(outdir, force=True)

            DIC_and_TIF(pixelsize=0.5).arr_to_tif(arr_ensemble,
                                              outdir + f'TRENDY_ensemble_{variable}.tif')






    def normalized_contribution(self):



        fdir_all=self.outdir


        for model in self.model_list:

            spatial_dicts = {}
            variables_list = []


            outdir=join(fdir_all,model)

            for f in os.listdir(join(fdir_all,model)):
                if not f.endswith('.tif'):
                    continue
                if  'contrib' in f:
                    continue
                if 'norm' in f:
                    continue


                if 'Ternary_plot' in f:
                    continue

                print(f)
                fpath = join(fdir_all, model, f)
                fname=f.split('.')[0]
                spatial_dict_i = DIC_and_TIF().spatial_tif_to_dic(fpath)
                spatial_dicts[fname] = spatial_dict_i
                variables_list.append(fname)

            df = T.spatial_dics_to_df(spatial_dicts)
            df = df.dropna(subset=variables_list, how='any')
            # T.print_head_n(df);exit()
            df_abs = pd.DataFrame()
            df_abs['pix'] = df['pix'].tolist()
            for var_i in variables_list:
                abs_vals = np.array(df[var_i].tolist())
                abs_vals = np.abs(abs_vals)
                df_abs[var_i] = abs_vals
            # T.print_head_n(df_abs);exit()

            norm_dict = {}
            # T.add_dic_to_df()

            for i, row in tqdm(df_abs.iterrows(), total=len(df_abs)):
                # print(row[variables_list])
                sum_vals = row[variables_list].sum()
                # print(sum_vals)
                # if sum_vals == 0:
                #     sum_vals = np.nan
                pix = row['pix']
                norm_dict_i = {}
                for var_i in variables_list:
                    var_i_norm = row[var_i] / sum_vals
                    norm_dict_i[f'{var_i}_norm'] = var_i_norm
                norm_dict[pix] = norm_dict_i

            df_abs = T.add_dic_to_df(df_abs, norm_dict, 'pix')

            for var_i in variables_list:
                dic_norm = T.df_to_spatial_dic(df_abs, f'{var_i}_norm', )
                DIC_and_TIF().pix_dic_to_tif(dic_norm, join(outdir, f'{var_i}_norm.tif'))
            ######T.save_df(df_abs,join(fdir,'df_normalized.df'));exit()

            ## df to dic

            # T.print_head_n(df_abs);exit()


            rgb_arr = np.zeros((360, 720, 4))
            # Ter = xymap.Ternary_plot()
            Ter = xymap.Ternary_plot(
                top_color=(67, 198, 219),
                left_color=(255, 165, 00),
                # left_color=(119,0,188),
                right_color=(230, 0, 230),
                # center_color=(85,85,85),
                center_color=(230, 230, 230),
                # center_color=(255,255,255),
            )

            for i, row in df_abs.iterrows():
                pix = row['pix']
                r, c = pix
                CV_IAV_norm = row[f'detrended_sum_rainfall_growing_season_zscore_norm']
                rainfall_frequency_norm = row[f'rainfall_frenquency_zscore_norm']
                composite_LAI_beta_mean_norm = row[f'{model}_sensitivity_zscore_norm']
                x, y, z = CV_IAV_norm, rainfall_frequency_norm, composite_LAI_beta_mean_norm
                color = Ter.get_color(x, y, z)
                color = color * 255
                color = np.array(color, dtype=np.uint8)
                alpha = 255
                color = np.append(color, alpha)
                # print(color);exit()

                rgb_arr[r][c] = color
            # xymap.GDAL_func().ar
            rgb_arr = np.array(rgb_arr, dtype=np.uint8)
            ### - 蓝绿色（上）： 主导
            # - 橙黄色（左下）： 主导
            # - 粉紫色（右下）：LAI_sensitivity（植被敏感性）主导
            outtif = join(outdir, 'Ternary_plot.tif')
            # tif_template = join(fdir_all, model, f'rainfall_frenquency_zscore.tif')
            tif_template=result_root+rf'3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg\CABLE-POP_S2_lai\\rainfall_frenquency_zscore.tif'
            print(rgb_arr)

            xymap.GDAL_func().RGBA_to_tif(rgb_arr, outtif, tif_template)
            grid_triangle_legend = Ter.grid_triangle_legend()
            plt.imshow(grid_triangle_legend)
            plt.show()
            # # T.open_path_and_file(fdir)
            # exit()

    def max_correlation_with_sign(self):


        dff = result_root + rf'3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_5\\statistics\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        model_list=self.model_list
        # for col in df.columns:
        #     print(col)
        # exit()



        var_list = ['detrended_sum_rainfall_ecosystem_year_CV_zscore',
                    'CV_intraannual_rainfall_ecosystem_year_zscore',
                    'sensitivity_zscore', ]


        for model in tqdm(model_list):

            outdir=self.outdir+model+'\\'
            T.mk_dir(outdir,force=True)
            var_list_contrib = [ f'{model}_'+i +'_contrib' for i in var_list]

            var_list_sens =[f'{model}_'+i for i in var_list]


            max_var_list = []
            max_var_sign_list = []
            color_list = []
            df_continent = df
            for i, row in df_continent.iterrows():
                vals_contrib = row[var_list_contrib].tolist()
                vals_contrib = np.array(vals_contrib)
                vals_contrib[vals_contrib<-10]=np.nan
                vals_contrib[vals_contrib > 10] = np.nan


                vals_sens = row[var_list_sens].tolist()
                vals_sens = np.array(vals_sens)
                vals_sens[vals_sens<-10]=np.nan
                vals_sens[vals_sens > 10] = np.nan



                if True in np.isnan(vals_contrib):
                    max_var_list.append(np.nan)
                    max_var_sign_list.append(np.nan)
                    color_list.append(np.nan)
                    continue
                vals_contri_abs = np.abs(vals_contrib)
                vals_contrib_dict = T.dict_zip(var_list_contrib, vals_contri_abs)
                vals_sens_dic=T.dict_zip(var_list_sens,vals_sens)



                max_var = T.get_max_key_from_dict(vals_contrib_dict)
                max_var_new_dict={f'{model}_detrended_sum_rainfall_ecosystem_year_CV_zscore_contrib':
                                      f'{model}_detrended_sum_rainfall_ecosystem_year_CV_zscore',
                                      f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore_contrib':
                                      f'{model}_CV_intraannual_rainfall_ecosystem_year_zscore',
                                      f'{model}_sensitivity_zscore_contrib':
                                      f'{model}_sensitivity_zscore',
                }


                max_val = vals_sens_dic[max_var_new_dict[max_var]]


                if max_val > 0:
                    max_var_sign = '+'
                else:
                    max_var_sign = '-'

                if 'sensitivity' in max_var:
                    if max_var_sign == '-':
                        color = 1
                    else:
                        color = 6
                elif 'CV_intraannual_rainfall_ecosystem_year_zscore' in max_var:
                    if max_var_sign == '-':
                        color = 2
                    else:
                        color = 5
                elif 'detrended_sum_rainfall_ecosystem_year_CV_zscore' in max_var:
                    if max_var_sign == '-':
                        color = 3
                    else:
                        color = 4
                else:

                    continue
                max_var_list.append(max_var)
                max_var_sign_list.append(max_var_sign)
                color_list.append(color)

            df_continent['max_var'] = max_var_list

            df_continent['max_var_sign'] = max_var_sign_list
            df_continent['color'] = color_list

            ## to tiff
            spatial_dic = T.df_to_spatial_dic(df, 'color')
            outtif = join(outdir, 'color_map.tif')
            array = DIC_and_TIF().pix_dic_to_tif(spatial_dic, outtif)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(arr, interpolation='nearest')
            # plt.colorbar()
            # plt.show()

    def statistic_contribution_area_heatmap(self):
        dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_4\\Dataframe\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        for col in df.columns:
            print(col)
        # df = df.dropna(subset=['composite_LAI_color_map'])
        model_list=self.model_list

        heatmap_cols = ['TRENDY_ensemble']+model_list

        row_order = [1, 6, 2, 5, 3, 4]
        heatmap_data = pd.DataFrame(index=[f'Group {i}' for i in row_order], columns=heatmap_cols)

        for ii in [1,6,2,5,3,4]:
            percentage_list=[]

            for model in model_list:
                sum = 0

                df_mask = df.dropna(subset=['composite_LAI_color_map'])
                # df_mask=df_mask.dropna(subset=[f'{model}_color_map'])

                df_mask1 = df.dropna(subset=[f'{model}_color_map'])

                # tmp = df_mask[[f'{model}_color_map', 'composite_LAI_color_map']]
                # ### filter data

                df_ii=df_mask1[df_mask1[f'{model}_color_map']==ii]
                df_obs=df_mask[df_mask['composite_LAI_color_map']==ii]

                percent_ii=len(df_ii)/len(df_mask1)*100
                percent_obs=len(df_obs)/len(df_mask)*100
                percent_diff=percent_ii-percent_obs

                percentage_list.append(percent_diff)

            ensemble_mean = float(np.nanmean(percentage_list)) if len(percentage_list) else np.nan
            heatmap_data.loc[f'Group {ii}'] = [ensemble_mean] + percentage_list

        ## plot

        heatmap_data = heatmap_data.astype(float)

        # 设置颜色顺序（可选）


        plt.figure(figsize=(9, 4))

        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdBu", cbar_kws={'label': 'Models-obs difference (%)'},
                    linewidths=0.3, vmin=-15, vmax=15)

        dic_label_name = {

                        'TRENDY_ensemble': 'TRENDY ensemble',
                          'CABLE-POP_S2_lai': 'CABLE-POP',
                          'CLASSIC_S2_lai': 'CLASSIC',
                          'CLM5': 'CLM5',
                          'DLEM_S2_lai': 'DLEM',
                          'IBIS_S2_lai': 'IBIS',
                          'ISAM_S2_lai': 'ISAM',
                          'ISBA-CTRIP_S2_lai': 'ISBA-CTRIP',
                          'JSBACH_S2_lai': 'JSBACH',
                          'JULES_S2_lai': 'JULES',
                          'LPJ-GUESS_S2_lai': 'LPJ-GUESS',
                          'LPX-Bern_S2_lai': 'LPX-Bern',
                          'ORCHIDEE_S2_lai': 'ORCHIDEE',

                          'YIBs_S2_Monthly_lai': 'YIBs',

                          }
        dic_variable_name = {1: '$\gamma$-',
                             6: '$\gamma$+',
                             2: 'Fq of rainfall-',
                             5: 'Fq of rainfall+',


                             3: 'CV Interannual rainfall-',

                             4:'CV Interannual rainfall+',




                             }


        ax = plt.gca()
        ax.set_xticklabels([dic_label_name.get(k, k) for k in heatmap_data.columns],
                           rotation=90, fontsize=10, font='Arial')
        ax.set_yticklabels([dic_variable_name[i] for i in row_order],
                           rotation=0, fontsize=10, font='Arial')
        # plt.tight_layout()
        plt.show()
        # plt.savefig(result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_3\\heatmap_contribution_area.pdf')


    def statistic_contribution_area_barplot(self):
        dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_5\\statistics\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        for col in df.columns:
            print(col)

        model_list=self.model_list


        result_dic = {}

        # —— 统计：各模型在每个组 ii 的面积百分比（分母用各自非空的样本）——
        for ii in [1, 6, 2, 5, 3, 4]:
            percentage_list = []
            for model in model_list:
                col = f'{model}_color_map'
                df_mask = df.dropna(subset=[col])  # 不要改写 df 本体
                df_ii = df_mask[df_mask[col] == ii]
                percent_ii = len(df_ii) / len(df_mask) * 100.0
                percentage_list.append(percent_ii)
            result_dic[ii] = percentage_list
        pprint(result_dic)

        dic_variable_name = {1: '$\gamma$-',
                             6: '$\gamma$+',
                             2: 'CV rainfall intra-',
                             5: 'CV rainfall intra+',

                             3: 'CV rainfall inter-',

                             4: 'CV rainfall inter+',

                             }

        # 颜色：前四个为 obs，第五个（如 TRENDY ensemble）单独色，其余为统一色
        color_list = ['#ADC9E4', '#EBF0FC', '#EBF0FC', '#EBF0FC', '#dd736c'] \
                     + ['#F7DAD4'] * (len(model_list) - 5)

        # 用模型名作为行索引，便于对齐
        df_new = pd.DataFrame(result_dic, index=model_list)

        # —— 画图：每个 ii 一张图，obs 与 models 留间隔，第一根柱子的高度画虚线（只跨 models）——
        for ii in [1, 6, 2, 5, 3, 4]:
            vals = df_new[ii].values
            n_all = len(vals)
            n_obs = 4  # 前 4 个是 obs
            gap = 1.2  # obs 与 models 间的空隙（单位≈一个柱宽）

            # 构造 x 位置：models 整体右移形成间隔
            x = np.arange(n_all, dtype=float)
            x[n_obs:] += gap

            fig, ax = plt.subplots(figsize=(self.map_width, self.map_height))
            ax.bar(x, vals, color=color_list[:n_all], edgecolor='black', width=0.8)

            # 在第一个柱子的高度画虚线（只跨 models 区域）
            y_ref = vals[0]  # 第一个柱子的高度
            xmin = x[0] - 0.4  # 第一个柱子的左边缘
            xmax = x[-1] + 0.4  # 最后一个柱子的右边缘
            ax.hlines(y_ref, xmin, xmax, colors='k', linestyles='--', linewidth=1.1, zorder=5)

                # 可选：标出 obs/models 分界
            ax.axvline(x[n_obs] - 0.9, color='0.75', linestyle=':', linewidth=1)

            # plt.ylabel('Area percentage (%)')
            plt.xticks([])
            ax.text(0.02, 0.98, dic_variable_name[ii],
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=12, fontfamily='Arial',
                    bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=1.5))
            ax.set_ylim(0, 45)
            plt.grid(axis='y', alpha=0.25)


            plt.show()

            #
            # plt.savefig(result_root + rf'\3mm\FIGURE\Figure5_comparison\barplot\\barplot_{ii}.pdf', dpi=300, bbox_inches='tight')
            # plt.close()











    def Figure2_robinson(self):

        fdir_trend = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg_3\\TRENDY_ensemble_median\\'
        temp_root = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\delta_multi_reg_3\\TRENDY_ensemble_median'
        outdir = result_root + rf'3mm\FIGURE\Robinson\\'
        T.mk_dir(outdir, force=True)
        T.mk_dir(temp_root, force=True)

        for f in os.listdir(fdir_trend):

            if not f.endswith('.tif'):
                continue

            if not 'color_map' in f:
                continue
            fpath = fdir_trend + f

            plt.figure(figsize=(Plot_Robinson().map_width, Plot_Robinson().map_height))
            m, ret = Plot_Robinson().plot_Robinson(fpath, vmin=1, vmax=6, is_discrete=True, colormap_n=7, )

            # plt.show()
            outf = outdir + f + '.pdf'
            plt.savefig(outf)
            plt.close()

    def TRENDY_barplot(self):
        dff = result_root + rf'3mm\Multiregression\partial_correlation\Obs\obs_climate\Dataframe\\partial_correlation.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        # df=df.dropna(axis=0, how='any')

        variables_list = ['composite_LAI',
                           'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                          'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                          'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                          'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                          'ORCHIDEE_S2_lai',

                          'YIBs_S2_Monthly_lai']
        result_dic={}
        result_stats={}
        values_beta_list=[]
        CI_list=[]


        for variable in variables_list:
            values_beta=df[f'{variable}_detrended_sum_rainfall_growing_season_zscore'].values
            values_beta=np.array(values_beta)
            values_beta[values_beta>100]=np.nan
            values_beta[values_beta<-100]=np.nan

            n=len(values_beta)
            confidence=0.95
            std=np.nanstd(values_beta)
            t_critical=stats.t.ppf((1 + confidence) / 2., n - 1)
            margin_of_error=t_critical * std / np.sqrt(n)
            ci_lower=np.nanmean(values_beta)-margin_of_error
            ci_upper=np.nanmean(values_beta)+margin_of_error
            CI_list.append([ci_lower,ci_upper])


            values_beta_list.append(values_beta)
        CI_list=np.array(CI_list)
        CI_list_T=CI_list.T



        values_beta_list=np.array(values_beta_list)

        values_beta_list_mean=np.nanmean(values_beta_list,axis=1)
        values_beta_list_std=np.nanstd(values_beta_list,axis=1)

        # add legend
        df_new=pd.DataFrame(result_dic)

        fig, ax = plt.subplots(figsize=(self.map_width*1.5, self.map_height))
        dic_label_name = {'composite_LAI': 'Composite LAI',

                          'CABLE-POP_S2_lai': 'CABLE-POP',
                          'CLASSIC_S2_lai': 'CLASSIC',
                          'CLM5': 'CLM5',
                          'DLEM_S2_lai': 'DLEM',
                          'IBIS_S2_lai': 'IBIS',
                          'ISAM_S2_lai': 'ISAM',
                          'ISBA-CTRIP_S2_lai': 'ISBA-CTRIP',
                          'JSBACH_S2_lai': 'JSBACH',
                          'JULES_S2_lai': 'JULES',
                          'LPJ-GUESS_S2_lai': 'LPJ-GUESS',
                          'LPX-Bern_S2_lai': 'LPX-Bern',
                          'ORCHIDEE_S2_lai': 'ORCHIDEE',

                          'YIBs_S2_Monthly_lai': 'YIBs',

                          }

        ## plot volin
        plt.bar(variables_list,values_beta_list_mean,color='#96cccb',width=0.7,edgecolor='black',
                label='Trend in Beta',yerr=CI_list_T[1]-CI_list_T[0],capsize=3)
        ## CI bar



        plt.xticks(np.arange(len(variables_list)),variables_list,rotation=45)
        plt.ylim(-.5,.5)
        ## add y=0
        plt.hlines(0, -0.5, len(variables_list) - 0.5, colors='black', linestyles='dashed')
        plt.ylabel('Beta (%/100ppm/yr)')
        plt.axhline(y=0, color='grey', linestyle='-')
        ax.set_xticks(range(len(variables_list)))
        ax.set_xticklabels(dic_label_name.values(), rotation=90, fontsize=10, font='Arial')
        plt.tight_layout()
        plt.show()









    def Ternary_plot(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import mpltern

        dff=result_root + rf'3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\Dataframe\\statistic.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        # df=df.dropna(axis=0, how='any')
        obs_list=['composite_LAI', 'SNU_LAI', 'GLOBMAP_LAI', 'LAI4g']
        new_model_list=self.model_list+obs_list

        result = []
        for model in new_model_list:
            gamma = df[f'{model}_sensitivity_zscore_contrib'].abs()  # gamma
            cv_iav = df[f'{model}_rainfall_frenquency_zscore_contrib'].abs()  # CV_IAV rainfall
            rfq = df[f'{model}_detrended_sum_rainfall_growing_season_zscore_contrib'].abs()  # rainfall frequency

            gamma_mean = gamma.mean()  # gamma
            cv_iav_mean = cv_iav.mean()  # CV_IAV rainfall
            rfq_mean = rfq.mean()  # rainfall frequency
            result.append([model, gamma_mean, rfq_mean, cv_iav_mean])

        df_summary = pd.DataFrame(result, columns=["model", "Gamma", "rainfall_frenquency", "CV_IAV"])

        # ------- 取绝对值 -------
        cols = ["Gamma", "rainfall_frenquency", "CV_IAV"]
        df_abs = df_summary[cols].abs()

        # ------- 归一化 -------
        sums = df_abs.sum(axis=1).replace(0, np.nan)  # 避免除0
        df_summary["Gamma_normalize"] = df_abs["Gamma"] / sums
        df_summary["rainfall_frenquency_normalize"] = df_abs["rainfall_frenquency"] / sums
        df_summary["CV_IAV_normalize"] = df_abs["CV_IAV"] / sums

        # 把 sum=0 的情况填回 0
        df_summary[["Gamma_normalize", "rainfall_frenquency_normalize", "CV_IAV_normalize"]] = \
            df_summary[["Gamma_normalize", "rainfall_frenquency_normalize", "CV_IAV_normalize"]].fillna(0)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection="ternary")

        # 轴标签
        ax.set_tlabel("CV IAV rainfall")  # 顶边
        ax.set_llabel("rainfall frequency")  # 左边
        ax.set_rlabel("Gamma")  # 右边

        ax.grid(True, linestyle=":", alpha=0.5)

        # 按照 (t, l, r) 顺序传值
        ax.scatter(
            df_summary["CV_IAV_normalize"],  # t = 顶边
            df_summary["rainfall_frenquency_normalize"],  # l = 左边
            df_summary["Gamma_normalize"],  # r = 右边
            s=90, c="C0", edgecolor="k"
        )

        # 标注模型名
        for i, row in df_summary.iterrows():
            ax.text(
                row["CV_IAV_normalize"],
                row["rainfall_frenquency_normalize"],
                row["Gamma_normalize"],
                row["model"], fontsize=9,
                ha="center", va="center"
            )

        plt.show()

    def plot_pdf(self):
        dff = result_root + rf'3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\Dataframe\\statistic.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        # df=df.dropna(axis=0, how='any')
        obs_list = ['composite_LAI', 'SNU_LAI', 'GLOBMAP_LAI', 'LAI4g']
        new_model_list = self.model_list + obs_list
        flag=0



        fig, axes = plt.subplots(3, 6, figsize=(12, 18))  # Adjust figsize if too tight
        axes = axes.flatten()
        self.model_list = ['composite_LAI', 'GLOBMAP_LAI','LAI4g','SNU_LAI','TRENDY_ensemble',  'CABLE-POP_S2_lai', 'CLASSIC_S2_lai',
                           'CLM5', 'DLEM_S2_lai', 'IBIS_S2_lai', 'ISAM_S2_lai',
                           'ISBA-CTRIP_S2_lai', 'JSBACH_S2_lai',
                           'JULES_S2_lai', 'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai',
                           'ORCHIDEE_S2_lai', 'YIBs_S2_Monthly_lai',

                           ]



        for model in new_model_list:

            result={ 'gamma' : df[f'{model}_sensitivity_zscore_contrib'].to_list(),
            'cv_iav' :df[f'{model}_rainfall_frenquency_zscore_contrib'].to_list(),  # CV_IAV rainfall
            'rfq' : df[f'{model}_detrended_sum_rainfall_growing_season_zscore_contrib'].to_list(),  # rainfall frequency
            }




            ## all model plot in the same layout
            ax = axes[flag]


            for var_name, values in result.items():
                if flag >= len(axes):
                    break

                arr = np.array(values)
                arr[arr>99]=np.nan
                arr[arr<-99]=np.nan
                arr=arr*100
                arr = arr[~np.isnan(arr)]
                mean_val = np.mean(arr)
                # ax.axvline(mean_val, linestyle='--', linewidth=1, alpha=0.8)



                # sns.kdeplot(arr, fill=False, linewidth=2,label=var_name,ax=ax)
                sns.ecdfplot(arr, label=var_name, ax=ax, linewidth=2, )
            ax.set_xlim(-30, 30)
            ax.set_ylabel('')
            ax.grid(True)
            ax.set_title(model)
            ax.legend(fontsize=6)

            # plt.grid(True)

            flag=flag+1


            #
            #
        plt.legend()
        plt.show()

        pass





    def statistic_contribution_area(self):
        dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_5\\statistics\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)


        percentage_list = []
        sum = 0

        for ii in [1, 2, 3, 4, 5, 6]:

            # df = df.dropna(subset=['TRENDY_ensemble_median_color_map'])
            # df_ii = df[df['TRENDY_ensemble_median_color_map'] == ii]
            # df = df.dropna(subset=['composite_LAI_median_color_map'])
            # df_ii = df[df['composite_LAI_median_color_map'] == ii]



            percent = len(df_ii) / len(df) * 100
            sum = sum + percent
            percentage_list.append(percent)
        print(percentage_list)
        print(sum);
        # exit()

        ## plot

        color_list = [

            '#f599a1', '#fcd590',
            '#e73618', '#dae67a',
            '#9fd7e9', '#a577ad',

        ]

        plt.figure(figsize=(3, 3))
        plt.bar([1, 2, 3, 4, 5, 6], percentage_list, color=color_list)

        plt.ylabel('Area precentage (%)')
        # plt.show()
        outdir=result_root + rf'\3mm\FIGURE\Robinson\\'
        plt.savefig(outdir + '\\statistics_contribution_area_model.pdf', dpi=300)
        plt.close()

        pass



        pass

    def sensitivity_vs_climate_factors(self):
        dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_3\Dataframe\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        df.dropna(inplace=True)

        # 设置变量名
        target_var_list = [
            'composite_LAI_detrended_sum_rainfall_growing_season_zscore',
            'composite_LAI_rainfall_frenquency_zscore']

        color_list = ['#dd492c', '#c05f77', '#b5869a',
                      '#bfa8b1', '#d5c1ca', '#e7dce1',
                      ]

        sand_color_list = ['#ffffe5', '#fffaca', '#fff0ae', '#fee391', '#fece65',
                           '#feb642', '#fe9929', '#f27e1b', '#e1640e', '#cc4c02', '#aa3c03', '#882f05', '#662506']
        aridity_color_list = ['#d73027', '#ea6f44', '#f48e52', '#fec279', '#fed690',
                              '#ffffbf', '#f7fccd', '#f0f9dc', '#e8f6ea', '#e0f3f8']
        root_depth_list = [
            '#f1e0b6',
            '#e6c981',
            '#d9b15a',
            '#c89b43',
            '#a88432',
            '#7f6d28',
            '#4f7f3b',
            '#2e7031',
            '#00441b'  # 深根
        ]
        short_vegetation_cover = ['#ffffcc',  # 极低覆盖
                                  '#d9f0a3',
                                  '#addd8e',
                                  '#78c679',
                                  '#41ab5d',
                                  '#238443',
                                  '#006837',
                                  '#004529']

        for target_var in target_var_list:

            # bin_var = 'Burn_area_sum'
            # bin_var = 'S_SAND'

            bin_var='SOC'
            # bin_var = 'sum_rainfall_mean'
            # bin_var = 'Tree cover_mean'
            # bin_var='Non_tree_vegetation_mean'
            # bin_var = 'rooting_depth_05'
            plt.hist(df[bin_var])
            plt.show()
            # bin_edges = np.arange(0, 101, 10)
            # bin_edges = np.arange(200,1201,100)
            # bin_edges = np.arange(0,501,50)
            # bin_edges = np.arange(0, 4000, 500)
            # bin_edges=np.arange(10,91,10)
            # bin_edges = np.arange(0.2, 0.66, 0.05)
            # bin_edges = np.arange(150,850,50)
            bin_edges = np.arange(0,0.5,0.05)
            # bin_edges = np.quantile(df[bin_var], np.linspace(0, 0.66, 11))
            bin_labels = [f'{round(bin_edges[i], 2)}-{round(bin_edges[i + 1], 2)}' for i in range(len(bin_edges) - 1)]
            # bin_labels = [f'{round(bin_edges[i ], 2)}' for i in range(len(bin_edges) - 1)]

            df['bin'] = pd.cut(df[bin_var], bins=bin_edges, labels=bin_labels, include_lowest=True)

            # 初始化结果字典
            result_dic = {}
            count_list = []

            for label in bin_labels:
                df_bin = df[df['bin'] == label][[target_var]].dropna()

                if len(df_bin) == 0:
                    result_dic[label] = [0, 0, 0, 0]
                    continue

                mean_val = np.nanmean(df_bin[target_var])
                std_err = np.nanstd(df_bin[target_var]) / np.sqrt(len(df_bin))  # 标准误差
                result_dic[label] = [mean_val, std_err]
                count_list.append(len(df_bin))

            # 构造 DataFrame
            result_df = pd.DataFrame(result_dic).T
            result_df.columns = ['mean', 'std_err']
            result_df.index = bin_labels

            # 画图

            ax = result_df['mean'].plot(
                kind='bar',
                yerr=result_df['std_err'],
                figsize=(4, 3),
                color=aridity_color_list,
                capsize=3,
                error_kw={'elinewidth': 1, 'ecolor': 'gray'},
                edgecolor='gray',
            )

            ## add count

            for i, bar in enumerate(ax.patches):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        height + 0.02,  # 微调高度

                        f'{count_list[i]}',
                        ha='center', va='bottom', fontsize=10, rotation=90)

            plt.axhline(y=0, color='gray', linestyle='-')
            ## xtick every 100 label
            xticks = ax.get_xticks()
            # xticklabels = [label.get_text() for label in ax.get_xticklabels()]
            # new_labels = [label if i % 2 == 0 else '' for i, label in enumerate(xticklabels)]
            # ax.set_xticklabels(new_labels, rotation=0)

            if target_var == 'detrended_sum_rainfall_CV_zscore_sensitivity':
                plt.ylabel('CV Interannual Rainfall (zscore)')


            elif target_var == 'rainfall_frenquency_zscore_sensitivity':
                plt.ylabel('Fq Rainfall(zscore)')

            if target_var == 'composite_LAI_beta_mean_zscore_contrib':
                plt.xticks([])
            elif target_var == 'rainfall_frenquency_zscore_contrib':
                plt.xticks([])
            else:
                plt.xticks(rotation=45)

            #
            plt.tight_layout()
            plt.show()
            # ## save pdf
            # fig = ax.get_figure()
            # outdir=result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_2\figure\\'
            # T.mk_dir(outdir, force=True)
            # fig.savefig(outdir + f'{target_var}_{bin_var}.pdf', dpi=300, bbox_inches='tight')
            # plt.close(fig)

    def cohens_d(self,x, y):
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
        return (np.nanmean(x) - np.nanmean(y)) / pooled if pooled > 0 else np.nan

    def box_plot_test(self):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from scipy import stats

        # ---------- load & clean ----------
        dff = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_3\Dataframe\statistics.df'
        df0 = T.load_df(dff)
        df0 = self.df_clean(df0)

        # variables to compare across panels
        var_list = [
            'sum_rainfall_trend', 'SM_ecosystem_year_trend', 'pi_growing_season_trend',
            'sand_soil_grid', 'fire_ecosystem_year_sum', 'rooting_depth_05', ]
        # df_postive=df0[df0['composite_LAI_detrended_sum_rainfall_growing_season_zscore_correlation']>0]
        # df_negative=df0[df0['composite_LAI_detrended_sum_rainfall_growing_season_zscore_correlation']<0]

        df_postive = df0[df0['composite_LAI_rainfall_frenquency_zscore_correlation'] > 0]
        df_negative = df0[df0['composite_LAI_rainfall_frenquency_zscore_correlation'] < 0]
        ## ttest
        for var in var_list:
            vals_pos = df_postive[var].tolist()
            vals_pos_arr=np.array(vals_pos)
            vals_pos_arr=vals_pos_arr[~np.isnan(vals_pos_arr)]
            vals_neg = df_negative[var].tolist()
            vals_neg_arr=np.array(vals_neg)
            vals_neg_arr=vals_neg_arr[~np.isnan(vals_neg_arr)]
            t, p = stats.ttest_ind(vals_pos_arr, vals_neg_arr, equal_var=False)
            print(f'{var}: t={t:.3f}, p={p:.3f}')






    def filter_data(self, df):
        self.x_variable_range_dict_global_CRU = {
            'pi_growing_season_trend': [0, 7],
            'sum_rainfall_trend': [0, 7],
            'SM_ecosystem_year_trend': [0, 1500],
            'sand_soil_grid': [0, 800],
            'rooting_depth_05': [0, 25],
            'cwdx80_05_soil_grid': [0, 50],}





    def maximum_contribution(self):
        fdir = self.outdir
        array_dic_all = {}
        array_arg = {}

        var_name_list = []
        for f in os.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if not 'contrib' in f:
                continue
            if 'max_label' in f:
                continue
            if 'Ternary_plot' in f:
                continue
            var_name = f.split('.')[0]
            var_name_list.append(var_name)
            print(f)
            fpath = join(fdir, f)
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            array_dic_all[var_name] = spatial_dict

        spatial_df = T.spatial_dics_to_df(array_dic_all)
        max_key_list = []
        max_val_list = []
        for i, row in spatial_df.iterrows():
            vals = row[var_name_list].tolist()
            vals = np.array(vals)
            var_name_list_array = np.array(var_name_list)
            vals_no_nan = vals[~np.isnan(vals)]
            var_name_list_array_no_nan = var_name_list_array[~np.isnan(vals)]
            vals_dict = T.dict_zip(var_name_list_array_no_nan, vals_no_nan)
            # if True in np.isnan(vals):
            # max_key_list.append(np.nan)
            # max_val_list.append(np.nan)
            # continue
            max_key = T.get_max_key_from_dict(vals_dict)
            max_val = vals_dict[max_key]
            max_key_list.append(max_key)
            max_val_list.append(max_val)
            # print(vals_dict)
            # print(max_key)
            # print(max_val)
            # exit()
        spatial_df['max_key'] = max_key_list
        spatial_df['max_val'] = max_val_list
        T.print_head_n(spatial_df)
        spatial_df.dropna()
        ## df to tif
        dic_label = {'CV_intraannual_rainfall_ecosystem_year_contrib': 2,
                     'detrended_sum_rainfall_CV_contrib': 3,
                     'composite_LAI_beta_mean_contrib': 1,

                     }

        spatial_df['max_label'] = spatial_df['max_key'].map(dic_label)
        # # ## calculate _percentage
        # for ii in range(1, 5):
        #     percent=spatial_df[spatial_df['max_label']==ii].shape[0]/spatial_df.shape[0]*100
        #     percent=round(percent,2)
        #     print(ii,percent)
        #
        #     plt.bar(ii,percent)
        # # plt.show()
        #
        #
        spatial_dict = T.df_to_spatial_dic(spatial_df, 'max_label')
        DIC_and_TIF(pixelsize=0.5).pix_dic_to_tif(spatial_dict, self.outdir + 'max_label.tif')

        dff_new = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\Dataframe\\Dataframe.df'
        df = T.load_df(dff_new)
        df = self.df_clean(df)
        df = df.dropna()

        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()

        for ii in range(1, 4):
            percent = df[df['max_label'] == ii].shape[0] / df.shape[0] * 100
            percent = round(percent, 2)
            print(ii, percent)

            plt.bar(ii, percent)
        plt.show()

    def heatmap2(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        dff = result_root + rf'3mm\Multiregression\partial_correlation\Obs\Dataframe\\statistics.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        # df.dropna(inplace=True)
        ###df =color map ==3 and 4
        # df = df[df['color_map'].isin([2, 5])]

        # df=df.dropna()
        T.print_head_n(df)
        x_var = 'sum_rainfall_trend'

        y_var = 'SM_trend'
        plt.hist(df[y_var])
        plt.show()
        plt.hist(df[x_var])
        plt.show()
        z_var = 'composite_LAI_detrended_sum_rainfall_growing_season_CV_zscore'
        # z_var='Fire_sum_average'

        bin_x = np.linspace(-4, 4, 6)
        bin_y = np.linspace(-0.001, 0.001, 6)

        # percentile_list=np.linspace(0,100,9)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure()

        matrix_dict, x_ticks_list, y_ticks_list = T.df_bin_2d(df, val_col_name=z_var,
                                                              col_name_x=x_var,
                                                              col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        self.plot_df_bin_2d_matrix(matrix_dict, -.5, .5, x_ticks_list, y_ticks_list, cmap='Viridis',
                                   is_only_return_matrix=False)
        plt.colorbar()
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.show()

        # plt.figure()

        matrix_dict, x_ticks_list, y_ticks_list = self.df_bin_2d_sample_size(df, val_col_name=z_var,
                                                                             col_name_x=x_var,
                                                                             col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        T.plot_df_bin_2d_matrix(matrix_dict, 0, 100, x_ticks_list, y_ticks_list, cmap='RdBu',
                                is_only_return_matrix=False)

        plt.colorbar()
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.show()

    def plot_df_bin_2d_matrix(self, matrix_dict, vmin, vmax, x_ticks_list, y_ticks_list, cmap='RdBu',
                              is_only_return_matrix=False):
        keys = list(matrix_dict.keys())
        r_list = []
        c_list = []
        for r, c in keys:
            r_list.append(r)
            c_list.append(c)
        r_list = set(r_list)
        c_list = set(c_list)

        row = len(r_list)
        col = len(c_list)
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in matrix_dict:
                    val_pix = matrix_dict[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        matrix = np.array(spatial, dtype=float)
        matrix = matrix[::-1]
        if is_only_return_matrix:
            return matrix
        plt.imshow(matrix, cmap='RdBu', vmin=vmin, vmax=vmax)
        plt.xticks(range(len(c_list)), x_ticks_list)
        plt.yticks(range(len(r_list)), y_ticks_list[::-1])

    def df_bin_2d_sample_size(self, df, val_col_name, col_name_x, col_name_y, bin_x, bin_y, round_x=2, round_y=2):
        df_group_y, _ = T.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = T.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                rt_mean = len(vals)
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict, x_ticks_list, y_ticks_list

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

    def load_df(self):
        dff = rf'D:\Project3\Result\3mm\Multiregression\Multiregression_result_residual\OBS_zscore\Dataframe\Dataframe.df'
        df = T.load_df(dff)
        # exit()
        # start_year = 0
        # end_year = 21
        # variable_list = self.xvar + self.y_var
        # df = Dataframe_per_value_transform(df, variable_list, start_year, end_year).df
        # T.print_head_n(df)
        return df


class Plot_Robinson:
    def __init__(self):
        # plt.figure(figsize=(15.3 * centimeter_factor, 8.2 * centimeter_factor))
        self.map_width = 15.3 * centimeter_factor
        self.map_height = 8.2 * centimeter_factor
        pass

    def robinson_template(self):
        '''
                :param fpath: tif file
                :param is_reproj: if True, reproject file from 4326 to Robinson
                :param res: resolution, meter
                '''

        # Blue represents high values, and red represents low values.
        plt.figure(figsize=(self.map_width, self.map_height))
        m = Basemap(projection='robin', lon_0=0, lat_0=90., resolution='c')

        m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        # for obj in meridict:
        #     line = meridict[obj][0][0]
        coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        polys = m.fillcontinents(color='#FFFFFF', lake_color='#EFEFEF', zorder=90)
    def plot_Robinson_significance_scatter(self, m, fpath_p, temp_root, sig_level=0.05, ax=None, linewidths=0.5, s=20,
                                           c='k', marker='x',
                                           zorder=100, res=2):

        fpath_clip = fpath_p + 'clip.tif'
        fpath_spatial_dict = DIC_and_TIF(tif_template=fpath_p).spatial_tif_to_dic(fpath_p)
        D_clip = DIC_and_TIF(tif_template=fpath_p)
        D_clip_lon_lat_pix_dict = D_clip.spatial_tif_to_lon_lat_dic(temp_root)
        fpath_clip_spatial_dict_clipped = {}
        for pix in fpath_spatial_dict:
            lon, lat = D_clip_lon_lat_pix_dict[pix]
            fpath_clip_spatial_dict_clipped[pix] = fpath_spatial_dict[pix]
        DIC_and_TIF(tif_template=fpath_p).pix_dic_to_tif(fpath_clip_spatial_dict_clipped, fpath_clip)
        fpath_resample = fpath_clip + 'resample.tif'
        ToRaster().resample_reproj(fpath_clip, fpath_resample, res=res)
        fpath_resample_ortho = fpath_resample + 'Robinson.tif'
        self.Robinson_reproj(fpath_resample, fpath_resample_ortho, res=res * 100000)
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_resample_ortho)

        arr = Tools().mask_999999_arr(arr, warning=False)
        arr[arr > sig_level] = np.nan
        D_resample = DIC_and_TIF(tif_template=fpath_resample_ortho)
        #
        os.remove(fpath_clip)
        os.remove(fpath_resample_ortho)
        os.remove(fpath_resample)

        spatial_dict = D_resample.spatial_arr_to_dic(arr)
        lon_lat_pix_dict = D_resample.spatial_tif_to_lon_lat_dic(temp_root)

        lon_list = []
        lat_list = []
        for pix in spatial_dict:
            val = spatial_dict[pix]
            if np.isnan(val):
                continue
            lon, lat = lon_lat_pix_dict[pix]
            lon_list.append(lon)
            lat_list.append(lat)
        lon_list = np.array(lon_list)
        lat_list = np.array(lat_list)
        lon_list = lon_list - originX
        lat_list = lat_list + originY
        lon_list = lon_list + pixelWidth / 2
        lat_list = lat_list + pixelHeight / 2
        # m,ret = Plot().plot_ortho(fpath,vmin=-0.5,vmax=0.5)
        m.scatter(lon_list, lat_list, latlon=False, s=s, c=c, zorder=zorder, marker=marker, ax=ax,
                  linewidths=linewidths)

        return m


    def plot_Robinson(self, fpath, ax=None, cmap=None, vmin=None, vmax=None, is_plot_colorbar=True, is_reproj=True,
                      res=25000, is_discrete=False, colormap_n=11):
        '''
        :param fpath: tif file
        :param is_reproj: if True, reproject file from 4326 to Robinson
        :param res: resolution, meter
        ## trend color list
        '''

        color_list = [

            '#f599a1', '#fcd590',
            '#e73618', '#dae67a',
            '#9fd7e9', '#a577ad',

        ]

        # color_list = [
        #
        #     '#f599a1', '#fcd590',
        #     '#e73618', '#73c79e',
        #     '#9fd7e9', '#a577ad',
        #
        # ]



        # Blue represents high values, and red represents low values.
        if ax == None:
            # plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 1, 1)
        if cmap is None:
            cmap = Tools().cmap_blend(color_list)
        elif type(cmap) == str:
            cmap = plt.get_cmap(cmap)
        if not is_reproj:
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        else:
            fpath_robinson = self.Robinson_reproj(fpath, fpath + '_robinson-reproj.tif', res=res)
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_robinson)
            os.remove(fpath_robinson)
        originY1 = copy.copy(originY)
        arr = Tools().mask_999999_arr(arr, warning=False)
        arr_m = ma.masked_where(np.isnan(arr), arr)
        originX = 0
        originY = originY * 2
        lon_list = np.arange(originX, originX + pixelWidth * arr.shape[1], pixelWidth)
        lat_list = np.arange(originY, originY + pixelHeight * arr.shape[0], pixelHeight)
        lon_list, lat_list = np.meshgrid(lon_list, lat_list)
        m = Basemap(projection='robin', lon_0=0, lat_0=90., ax=ax, resolution='c')
        ret = m.pcolormesh(lon_list, lat_list, arr_m, cmap=cmap, zorder=99, vmin=vmin, vmax=vmax, )

        # m.drawparallels(np.arange(-60., 90., 30.), zorder=99, dashes=[8, 8], linewidth=.5)
        # m.drawparallels((-90., 90.), zorder=99, dashes=[1, 0], linewidth=2)
        # meridict = m.drawmeridians(np.arange(0., 420., 60.), zorder=100, latmax=90, dashes=[8, 8], linewidth=.5)
        # meridict = m.drawmeridians((-180,180), zorder=100, latmax=90, dashes=[1, 0], linewidth=2)
        # for obj in meridict:
        #     line = meridict[obj][0][0]
        # coastlines = m.drawcoastlines(zorder=100, linewidth=0.2)
        # polys = m.fillcontinents(color='#FFFFFF', lake_color='#EFEFEF', zorder=90)
        if is_plot_colorbar:
            if is_discrete:
                bounds = np.linspace(vmin, vmax, colormap_n)
                # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cax, kw = mpl.colorbar.make_axes(ax, location='bottom', pad=0.05, shrink=0.5)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds,
                                                 orientation='horizontal')
            else:
                cbar = plt.colorbar(ret, ax=ax, shrink=0.5, location='bottom', pad=0.05)
        return m, ret

    def Robinson_reproj(self, fpath, outf, res=50000):
        wkt = self.Robinson_wkt()
        srs = DIC_and_TIF().gen_srs_from_wkt(wkt)
        ToRaster().resample_reproj(fpath, outf, res, dstSRS=srs)
        return outf

    def Robinson_wkt(self):
        wkt = '''
        PROJCRS["Sphere_Robinson",
    BASEGEOGCRS["Unknown datum based upon the Authalic Sphere",
        DATUM["Not specified (based on Authalic Sphere)",
            ELLIPSOID["Sphere",6371000,0,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["Sphere_Robinson",
        METHOD["Robinson"],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Not known."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["ESRI",53030]]'''
        return wkt







    def heatmap(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        f_sensitivity_trend = result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year_SNU_LAI\npy_time_series\\sum_rainfall_detrend_trend.tif'
        f_rainfall_trend=result_root+rf'\3mm\CRU_JRA\extract_rainfall_phenology_year\moving_window_average_anaysis_trend\ecosystem_year\trend\\\detrended_sum_rainfall_CV_trend.tif'
        f_CVLAI=result_root + rf'3mm\extract_SNU_LAI_phenology_year\moving_window_extraction\trend\\detrended_SNU_LAI_CV_trend.tif'
        outf = result_root + rf'\3mm\heatmap\\heatmap_CVLAI.pdf'
        arr_LAI_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_CVLAI)

        arr_LAI_trend[arr_LAI_trend < -999] = np.nan

        arr_LAI_sensitivity_precip, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_sensitivity_trend)
        arr_precip_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_rainfall_trend)
        arr_LAI_sensitivity_precip[arr_LAI_sensitivity_precip < -999] = np.nan
        arr_precip_trend[arr_precip_trend < -999] = np.nan
        arr_LAI_trend=np.array(arr_LAI_trend)
        arr_LAI_sensitivity_precip=np.array(arr_LAI_sensitivity_precip)
        arr_precip_trend=np.array(arr_precip_trend)

        dic_LAI_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_trend)
        dic_arr_LAI_sensitivity_precip=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_sensitivity_precip)
        dic_precip_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_precip_trend)

        result_dic={
            'LAI_CV_trend':dic_LAI_trend,
            'LAI_sensitivity_precip_trend':dic_arr_LAI_sensitivity_precip,
            'interannnual_rainallCV_trend':dic_precip_trend
        }
        # plt.hist(result_dic['LAI_CV_trend'].values())
        # plt.show()
        # plt.hist(result_dic['LAI_sensitivity_precip_trend'].values())
        # plt.show()
        # plt.hist(result_dic['interannnual_rainallCV_trend'].values())
        # plt.show();exit()
        df=T.spatial_dics_to_df(result_dic)
        T.print_head_n(df)
        x_var = 'LAI_sensitivity_precip_trend'
        y_var = 'interannnual_rainallCV_trend'
        z_var = 'LAI_CV_trend'
        # bin_x = [ -0.6,-0.4,-0.2,0,0.2,0.4,0.6,]
        bin_x = np.linspace(-2.5, 2.5, 11)
        # bin_y = [ -4, -3, -2, -1, 0, 1, 2, 3, 4, ]
        bin_y = np.linspace(-1.5, 1.5, 11)
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure(figsize=(self.map_width, self.map_height))

        matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)

        my_cmap = T.cmap_blend(color_list = ['#000000','r', 'b'])
        my_cmap = 'GnBu'
        self.plot_df_bin_2d_matrix(matrix_dict,-0.8,0.8,x_ticks_list,y_ticks_list,cmap=my_cmap,
                              is_only_return_matrix=False)
        plt.colorbar()
        pprint(matrix_dict)


        matrix_dict_count, x_ticks_list, y_ticks_list = self.df_bin_2d_count(df, val_col_name=z_var,
                                                              col_name_x=x_var,
                                                              col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        pprint(matrix_dict_count)
        scatter_size_dict = {
            (1,20): 5,
            (20,50): 20,
            (50,100): 50,
            (100,np.inf): 100
        }
        matrix_dict_count_normalized = {}
        # Normalize counts for circle size
        for key in matrix_dict_count:
            num = matrix_dict_count[key]
            for key2 in scatter_size_dict:
                if num >= key2[0] and num < key2[1]:
                    matrix_dict_count_normalized[key] = scatter_size_dict[key2]
                    break
        pprint(matrix_dict_count_normalized)
        reverse_x = list(range(len(bin_y)-1))[::-1]
        reverse_x_dict = {}
        for i in range(len(bin_y)-1):
            reverse_x_dict[i] = reverse_x[i]
        # print(reverse_x_dict);exit()
        for x,y in matrix_dict_count_normalized:
            plt.scatter(y,reverse_x_dict[x],s=matrix_dict_count_normalized[(x,y)],c='gray',edgecolors='none',alpha=.5)
        for x,y in matrix_dict_count_normalized:
            plt.scatter(y,reverse_x_dict[x],s=matrix_dict_count_normalized[(x,y)],c='none',edgecolors='gray',alpha=1)

        plt.xlabel('beta')
        plt.ylabel('Trend in Rainfall (mm/yr)')

        plt.show()
        # plt.savefig(outf)
        # plt.close()




    #     plt.savefig(result_root + rf'Data_frame\\Frequency\\Trendy_{region}.pdf', dpi=300, )
    #     plt.close()


    def heatmap_count(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        f_sensitivity_trend = result_root + rf'3mm\moving_window_multi_regression\moving_window\multi_regression_result_detrend_ecosystem_year\npy_time_series\trend\\sum_rainfall_sensitivity_trend.tif'
        f_rainfall_trend=result_root+rf'3mm\CRU_JRA\extract_rainfall_phenology_year\extraction_rainfall_characteristic\ecosystem_year\trend\\\sum_rainfall_trend.tif'
        f_CVLAI=result_root + rf'\3mm\extract_LAI4g_phenology_year\dryland\moving_window_average_anaysis\trend_analysis\\LAI4g_detrend_CV_trend.tif'

        arr_LAI_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_CVLAI)

        arr_LAI_trend[arr_LAI_trend < -999] = np.nan

        arr_LAI_sensitivity_precip, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_sensitivity_trend)
        arr_precip_trend, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
            f_rainfall_trend)
        arr_LAI_sensitivity_precip[arr_LAI_sensitivity_precip < -999] = np.nan
        arr_precip_trend[arr_precip_trend < -999] = np.nan
        arr_LAI_trend=np.array(arr_LAI_trend)
        arr_LAI_sensitivity_precip=np.array(arr_LAI_sensitivity_precip)
        arr_precip_trend=np.array(arr_precip_trend)

        dic_LAI_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_trend)
        dic_arr_LAI_sensitivity_precip=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_LAI_sensitivity_precip)
        dic_precip_trend=DIC_and_TIF(pixelsize=0.5).spatial_arr_to_dic(arr_precip_trend)

        result_dic={
            'LAI_CV':dic_LAI_trend,
            'LAI_sensitivity_precip_trend':dic_arr_LAI_sensitivity_precip,
            'Preci_trend':dic_precip_trend
        }
        # plt.hist(result_dic['LAI_CV'].values())
        # plt.show()
        # plt.hist(result_dic['LAI_sensitivity_precip_trend'].values())
        # plt.show()
        # plt.hist(result_dic['Preci_trend'].values())
        # plt.show();exit()
        df=T.spatial_dics_to_df(result_dic)
        T.print_head_n(df)
        x_var = 'LAI_sensitivity_precip_trend'
        y_var = 'Preci_trend'
        z_var = 'LAI_CV'
        # bin_x = [ -0.6,-0.4,-0.2,0,0.2,0.4,0.6,]
        bin_x = np.linspace(-0.7, 1.5, 13)
        # bin_y = [ -4, -3, -2, -1, 0, 1, 2, 3, 4, ]
        bin_y = np.linspace(-3, 3, 13)
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure()

        matrix_dict,x_ticks_list,y_ticks_list = self.df_bin_2d_count(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y)

        self.plot_df_bin_2d_matrix(matrix_dict,0,200,x_ticks_list,y_ticks_list,
                              is_only_return_matrix=False)

        plt.xlabel('beta')
        plt.ylabel('Trend in Rainfall (mm/yr)')

        plt.colorbar()
        plt.show()


    def df_bin_2d_count(self,df,val_col_name,col_name_x,col_name_y,bin_x,bin_y,round_x=2,round_y=2):
        df_group_y, _ = self.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = self.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                rt_mean = len(vals)
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict,x_ticks_list,y_ticks_list

    def df_bin(self, df, col, bins):
        df_copy = df.copy()
        df_copy[f'{col}_bins'] = pd.cut(df[col], bins=bins)
        df_group = df_copy.groupby([f'{col}_bins'],observed=True)
        bins_name = df_group.groups.keys()
        bins_name_list = list(bins_name)
        bins_list_str = [str(i) for i in bins_name_list]
        # for name,df_group_i in df_group:
        #     vals = df_group_i[col].tolist()
        #     mean = np.nanmean(vals)
        #     err,_,_ = self.uncertainty_err(SM)
        #     # x_list.append(name)
        #     y_list.append(mean)
        #     err_list.append(err)
        return df_group, bins_list_str

    def plot_df_bin_2d_matrix(self,matrix_dict,vmin,vmax,x_ticks_list,y_ticks_list,cmap='RdBu',
                              is_only_return_matrix=False):
        keys = list(matrix_dict.keys())
        r_list = []
        c_list = []
        for r, c in keys:
            r_list.append(r)
            c_list.append(c)
        r_list = set(r_list)
        c_list = set(c_list)

        row = len(r_list)
        col = len(c_list)
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in matrix_dict:
                    val_pix = matrix_dict[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        matrix = np.array(spatial, dtype=float)
        matrix = matrix[::-1]
        if is_only_return_matrix:
            return matrix
        plt.imshow(matrix,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.xticks(range(len(c_list)), x_ticks_list)
        plt.yticks(range(len(r_list)), y_ticks_list[::-1])


class convert_to_Robinson():
    def __init__(self):
        pass
    def run (self):
        self.convert()
    pass

    def convert(self):  # convert figure to robinson


        fdir_trend_all = result_root + rf'3mm\Multiregression\Multiregression_result_residual\OBS_zscore\slope\delta_multi_reg_3\\'

        outdir = result_root + rf'\3mm\Multiregression\Multiregression_result_residual\TRENDY_zscore\slope\\Robinson\\'
        T.mk_dir(outdir, force=True)

        for fdir in os.listdir(fdir_trend_all):
            fdir_trend = join(fdir_trend_all, fdir)

            fpath = join(fdir_trend, 'Ternary_plot.tif')


            outf=outdir + fdir + '.tif'
            srcSRS=self.wkt_84()
            dstSRS=self.wkt_robinson()

            ToRaster().resample_reproj(fpath,outf, 50000, srcSRS=srcSRS, dstSRS=dstSRS)

            T.open_path_and_file(outdir)


    def wkt_robinson(self):
        wkt='''PROJCRS["World_Robinson",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["World_Robinson",
        METHOD["Robinson"],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Not known."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["ESRI",54030]]
        '''
        return wkt


    def wkt_84(self):
        wkt = '''GEOGCRS["WGS 84",
    ENSEMBLE["World Geodetic System 1984 ensemble",
        MEMBER["World Geodetic System 1984 (Transit)"],
        MEMBER["World Geodetic System 1984 (G730)"],
        MEMBER["World Geodetic System 1984 (G873)"],
        MEMBER["World Geodetic System 1984 (G1150)"],
        MEMBER["World Geodetic System 1984 (G1674)"],
        MEMBER["World Geodetic System 1984 (G1762)"],
        MEMBER["World Geodetic System 1984 (G2139)"],
        ELLIPSOID["WGS 84",6378137,298.257223563,
            LENGTHUNIT["metre",1]],
        ENSEMBLEACCURACY[2.0]],
    PRIMEM["Greenwich",0,
        ANGLEUNIT["degree",0.0174532925199433]],
    CS[ellipsoidal,2],
        AXIS["geodetic latitude (Lat)",north,
            ORDER[1],
            ANGLEUNIT["degree",0.0174532925199433]],
        AXIS["geodetic longitude (Lon)",east,
            ORDER[2],
            ANGLEUNIT["degree",0.0174532925199433]],
    USAGE[
        SCOPE["Horizontal component of 3D system."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["EPSG",4326]]'''
        return wkt
def main():
    Delta_regression().run()
    # Delta_regression_TRENDY().run()
    # convert_to_Robinson().run()

    pass


if __name__ == '__main__':
    main()