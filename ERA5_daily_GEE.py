# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import urllib3
from __init__ import *
import ee

import math
import pprint
# import geemap
# exit()

this_script_root = join(this_root, 'ERA5')

class ERA5_daily:

    def __init__(self):
        # self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
        #     'ERA5_daily',
        #     this_script_root, mode=2)
        ee.Initialize()

    def run(self):
        # for year in range(2013,2014):
        #     self.download_images(year)
        # self.download_images()
        # self.unzip()
        # self.tiff_to_dict()
        # self.reproj()
        # self.statistic()
        # self.transform_ERA()
        self.detrend_deseasonal()


        # self.check()
        # self.wet_spell_dry_spell()

        pass

    def download_images(self,year):
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\raw\\{year}\\'
        T.mk_dir(outdir,force=True)
        startDate = f'{year}-01-01'
        endDate = f'{year+1}-01-01'
        Collection = ee.ImageCollection('ECMWF/ERA5/DAILY')
        # l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
        Collection = Collection.filterDate(startDate, endDate)

        info_dict = Collection.getInfo()
        # pprint.pprint(info_dict)
        # print(len(info_dict['features']))
        # exit()
        # for key in info_dict:
        #     print(key)
        ids = info_dict['features']
        for i in tqdm(ids,desc=f'{year}'):
            dict_i = eval(str(i))
            # pprint.pprint(dict_i['id'])
            # exit()
            outf_name = dict_i['id'].split('/')[-1] + '.zip'
            out_path = join(outdir, outf_name)
            if isfile(out_path):
                continue
            # print(outf_name)
            # exit()
            # print(dict_i['id'])
            # l8 = l8.median()
            # l8_qa = l8.select(['QA_PIXEL'])
            # l8_i = ee.Image(dict_i['LANDSAT/LC08/C02/T1_L2/LC08_145037_20200712'])
            Image = ee.Image(dict_i['id'])
            # Image_product = Image.select('mean_2m_air_temperature')
            # Image_product = Image.select('maximum_2m_air_temperature')
            Image_product = Image.select('minimum_2m_air_temperature')
            exportOptions = {
                'scale': 27830,
                'maxPixels': 1e13,
                # 'region': region,
                # 'fileNamePrefix': 'exampleExport',
                # 'description': 'imageToAssetExample',
            }
            url = Image_product.getDownloadURL(exportOptions)

            try:
                self.download_i(url, out_path)
            except:
                print('download error', out_path)
                continue
        pass




    def download_i(self,url,outf):
        # try:
        http = urllib3.PoolManager()
        r = http.request('GET', url, preload_content=False)
        body = r.read()
        with open(outf, 'wb') as f:
            f.write(body)

    def unzip(self):
        fdir = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\raw\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\\min_temp\\unzip\\'
        T.mk_dir(outdir)
        for folder in T.listdir(fdir):
            print(folder)



            fdir_i = join(fdir,folder)



            outdir_i = join(outdir,folder)
            if isdir(outdir_i):
                continue
            T.unzip(fdir_i,outdir_i)
        pass

    def wkt(self):
        wkt = '''
        PROJCS["Sinusoidal",
    GEOGCS["GCS_Undefined",
        DATUM["Undefined",
            SPHEROID["User_Defined_Spheroid",6371007.181,0.0]],
        PRIMEM["Greenwich",0.0],
        UNIT["Degree",0.0174532925199433]],
    PROJECTION["Sinusoidal"],
    PARAMETER["False_Easting",0.0],
    PARAMETER["False_Northing",0.0],
    PARAMETER["Central_Meridian",0.0],
    UNIT["Meter",1.0]]'''
        return wkt

    def reproj(self):
        fdir = join(self.this_class_arr,'unzip')
        outdir = join(self.this_class_arr,'reproj')
        T.mk_dir(outdir)
        for site in T.listdir(fdir):
            fdir_i = join(fdir,site)
            outdir_i = join(outdir,site)
            T.mk_dir(outdir_i)
            for date in T.listdir(fdir_i):
                fdir_i_i = join(fdir_i,date)
                for f in T.listdir(fdir_i_i):
                    fpath = join(fdir_i_i,f)
                    outpath = join(outdir_i,date+'.tif')
                    SRS = DIC_and_TIF().gen_srs_from_wkt(self.wkt())
                    ToRaster().resample_reproj(fpath,outpath,.005,srcSRS=SRS, dstSRS='EPSG:4326')
    def tiff_to_dict(self):
        fdir_all=rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\unzip\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\dict\\'
        T.mk_dir(outdir,force=True)

        NDVI_mask_f = data_root + rf'/Base_data/dryland_mask.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan

        year_list = list(range(1982, 2021))


        # 作为筛选条件
        for fdir in os.listdir(fdir_all):

            outdir_i = outdir + fdir + '\\'

            if isdir(outdir_i):
                continue
            T.mk_dir(outdir_i, force=True)


            all_array = [] #### so important  it should be go with T.mk_dic

            for fdir_i in os.listdir(fdir_all + fdir):


                # f=fdir_all + fdir + '\\' + fdir_i+f'\\{fdir_i}.total_precipitation.tif'
                f=fdir_all + fdir + '\\' + fdir_i+f'\\{fdir_i}.maximum_2m_air_temperature.tif'
                # f = fdir_all + fdir + '\\' + fdir_i + f'\\{fdir_i}.minimum_2m_air_temperature.tif'

                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
                    f)
                array = np.array(array, dtype=float)

                array_unify = array[:720][:720,
                              :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

                # array_unify[array_unify < -999] = np.nan

                # array_unify[array_unify > 7] = np.nan
                # array[array ==0] = np.nan

                # array_unify[array_unify < 0] = np.nan
                # array_unify = array_unify * 1000  ## precipitation unit is m so we need to multiply 1000 to get mm

                array_unify = array_unify - 273.15  ## temperature unit is K so we need to minus 273.15 to get Celsius

                # plt.imshow(array_unify)
                # plt.show()
                array_mask = np.array(array_mask, dtype=float)
                # plt.imshow(array_mask)
                # plt.show()
                array_dryland = array_unify * array_mask
                # plt.imshow(array_dryland)
                # plt.show()

                all_array.append(array_dryland)

            row = len(all_array[0])
            col = len(all_array[0][0])
            key_list = []
            dic = {}

            for r in tqdm(range(row), desc='构造key'):  # 构造字典的键值，并且字典的键：值初始化
                for c in range(col):
                    dic[(r, c)] = []
                    key_list.append((r, c))
            # print(dic_key_list)

            for r in tqdm(range(row), desc='构造time series'):  # 构造time series
                for c in range(col):
                    for arr in all_array:
                        value = arr[r][c]
                        dic[(r, c)].append(value)
                    # print(dic)
            time_series = []
            flag = 0
            temp_dic = {}
            for key in tqdm(key_list, desc='output...'):  # 存数据
                flag = flag + 1
                time_series = dic[key]
                time_series = np.array(time_series)
                temp_dic[key] = time_series
                if flag % 10000 == 0:
                    # print(flag)
                    np.save(outdir_i + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                    temp_dic = {}
            np.save(outdir_i + 'per_pix_dic_%03d' % 0, temp_dic)

    def statistic(self):
        fdir = join(self.this_class_arr,'reproj')
        outdir = join(self.this_class_arr,'statistic')
        T.mk_dir(outdir)
        for site in T.listdir(fdir):
            fdir_i = join(fdir,site)
            mean_list = []
            date_list = []
            for f in T.listdir(fdir_i):
                if not f.endswith('.tif'):
                    continue
                date = f.split('.')[0]
                y,m,d = date.split('_')
                y = int(y)
                m = int(m)
                d = int(d)
                date_obj = datetime.datetime(y,m,d)
                fpath = join(fdir_i,f)
                arr = ToRaster().raster2array(fpath)[0]
                arr[arr<=0] = np.nan
                mean = np.nanmean(arr)
                mean_list.append(mean)
                date_list.append(f'{y}-{m:02d}-{d:02d}')
            df = pd.DataFrame({'date':date_list,'NDVI':mean_list})
            outf = join(outdir,site)
            T.df_to_excel(df,outf)

        pass

    def transform_ERA(self):

        fdir_all = rf'C:\Users\wenzhang1\Desktop\ERA5\\min_temp\\deseasonal\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\\\min_temp\\detrend\\'
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(106):
            data_list.append(i)

        for data in data_list:
            dic_all_list = []
            for fdir_i in os.listdir(fdir_all):

                for f in os.listdir(fdir_all + fdir_i + '\\'):
                    if not f.endswith('.npy'):
                        continue
                    if f.split('.')[0].split('_')[-1] != '%03d' % data:
                        continue

                    dict = np.load(fdir_all + fdir_i + '\\' + f, allow_pickle=True).item()
                    dic_all_list.append(dict)

            result_dic = {}


            for pix in tqdm(dic_all_list[0].keys()):
                result_list = []
                for i in range(len(dic_all_list)):
                    if pix not in dic_all_list[i].keys():
                        continue
                    else:
                        # print(dic_all_list[i][pix])
                        result_list.append(dic_all_list[i][pix][0:365])
                result_dic[pix] = result_list
            ## save
            np.save(outdir + f'per_pix_dic_%03d' % data, result_dic)
            # print(result_dic)
    def detrend_deseasonal(self):
        fdir_all = rf'C:\Users\wenzhang1\Desktop\ERA5\\max_temp\\deseasonal\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\\max_temp\\deseasonal_detrend\\'
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(106):


            data_list.append(i)

        for data in data_list:
            dic_all = np.load(fdir_all + f'per_pix_dic_%03d.npy' % data, allow_pickle=True).item()
            result_dic = {}
            outf=outdir + f'per_pix_dic_%03d' % data+'.npy'
            print(outf)
            if isfile(outf):
                continue
            for pix in tqdm(dic_all.keys()):
                vals = dic_all[pix]
                vals=np.array(vals)
                vals_flatten = vals.flatten()
                #
                if T.is_all_nan(vals_flatten):
                    continue

                anomaly=self.daily_climatology_anomaly(vals_flatten)
                anomaly_detrend=T.detrend_vals(vals)
                # plt.bar(range(len(anomaly)),anomaly)
                # plt.show()
                # #
                # plt.bar(range(len(anomaly_detrend)),anomaly_detrend)
                # plt.show()

                result_dic[pix] = anomaly_detrend
            np.save(outdir + f'per_pix_dic_%03d' % data, result_dic)

        pass

    def daily_climatology_anomaly(self, vals):
        '''
        juping
        :param vals: 40 * 365
        :return:
        '''
        pix_anomaly = []
        climatology_means = []
        for day in range(1, 366):
            one_day = []
            for i in range(len(vals)):
                d = i % 365 + 1
                if day == d:
                    one_day.append(vals[i])
            mean = np.nanmean(one_day)
            std = np.nanstd(one_day)
            climatology_means.append(mean)
        for i in range(len(vals)):
            d_ind = i % 365
            mean_ = climatology_means[d_ind]
            anomaly = vals[i] - mean_
            pix_anomaly.append(anomaly)
        pix_anomaly = np.array(pix_anomaly)
        return pix_anomaly

class extraction_extreme_event_rainfall_ENSO:
    def __init__(self):
        self.strong_El_Nino_list = [1982, 1983, 1987, 1988, 1991, 1992, 1997, 1998, 2015, 2016]

        self.strong_La_Nina_list = [1988, 1989, 1998, 1999, 2000, 2007, 2008, 2010, 2011]

    def  run(self):
        # self.define_quantile_extreme()
        # self.extract_extreme_ENSO_year()
        # self.extract_extreme_rainfall_event()
        # self.extract_rainfall_event_total()
        self.wet_spell_dry_spell()
        # self.LAI_ENSO_extraction()

        # self.check()

        pass
    def define_quantile_threshold(self):
        # 1) extract extreme wet event based on 90th percentile and calculate frequency and total duration
        # 2) extract extreme dry event based on 10th percentile and calculate frequency and total duration
        # 3) extract wet event intensity
        ## 4) extract dry event intensity
        ## extract VPD and calculate the frequency of VPD>2kpa
        fdir=data_root+rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir = data_root+rf'\ERA5\ERA5_daily\dict\\define_quantile_threshold\\'
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            spatial_dic = np.load(fdir+f,allow_pickle=True).item()
            result_dic={}
            for pix in tqdm(spatial_dic):

                vals=spatial_dic[pix]
                vals_flatten=[item for sublist in vals for item in sublist]
                vals_flatten = np.array(vals_flatten)

                if T.is_all_nan(vals_flatten):
                    continue
                # plt.bar(range(len(vals_flatten)),vals_flatten)
                # plt.show()

                val_90th= np.percentile(vals_flatten,90)
                val_10th = np.percentile(vals_flatten, 10)
                val_95th = np.percentile(vals_flatten, 95)
                val_5th = np.percentile(vals_flatten, 5)
                val_99th = np.percentile(vals_flatten, 99)
                val_1st = np.percentile(vals_flatten, 1)
                dic_i={
                    '90th':val_90th,
                    '10th':val_10th,
                    '95th':val_95th,
                    '5th':val_5th,
                    '99th':val_99th,
                    '1st':val_1st
                }
                result_dic[pix]=dic_i
            outf=outdir+f
            np.save(outf,result_dic)

    def extract_extreme_ENSO_year(self):


        strong_El_Nino_year=[[1982,1983],[1987,1988],[1991,1992],[1997,1998],[2015,2016]]

        strong_La_nina_year=[[1988,1989],[1998,1999],[2000,2001],[2007,2008],[2010,2011]]

        # fdir_threshold = data_root+rf'ERA5\ERA5_daily\dict\precip_transform_extreme\\'
        fdir_yearly_all=rf'D:\Project3\Data\ERA5\ERA5_daily\dict\precip_transform\\'
        outdir_LA_nina = data_root + rf'\ERA5\ERA5_daily\dict\\ENSO_year_extraction\\LA_nina\\'
        outdir_El_nino = data_root + rf'\ERA5\ERA5_daily\dict\\ENSO_year_extraction\\El_nino\\'

        T.mk_dir(outdir_LA_nina,force=True)
        T.mk_dir(outdir_El_nino,force=True)
        # spatial_threshold_dic=T.load_npy_dir(fdir_threshold)
        for f in T.listdir(fdir_yearly_all):


            spatial_dic=T.load_npy(fdir_yearly_all+f)

            result_dic_La_nina = {}
            result_dic_El_nino = {}
            for pix in tqdm(spatial_dic):

                vals = spatial_dic[pix]
                vals_reshape = np.array(vals).reshape(-1, 365)
                r,c = pix

                ### extract El Nino and La Nina year data
                ## transform daily data to yearly data

                ## reshape the data 38


                result_dic_i_El_nino = {}


                for year_range in strong_El_Nino_year:

                    start_year,end_year = year_range

                    key_EL_nino = f'{start_year}_{end_year}'


                    EL_Nino_index_list = []
                    for i in range(len(vals_reshape)):
                        year_i=1982+i
                        if year_i>=start_year and year_i<=end_year:
                            EL_Nino_index_list.append(vals_reshape[i])
                    EL_Nino_index_list = np.array(EL_Nino_index_list)
                    EL_Nino_index_list = EL_Nino_index_list.flatten()

                    ## extract 180 to 485
                    extract_vals_el_nino = EL_Nino_index_list[180:485]
                    result_dic_i_El_nino[key_EL_nino] = extract_vals_el_nino

                result_dic_i_La_nina = {}

                for year_range in strong_La_nina_year:
                    start_year,end_year = year_range
                    key_La_nina = f'{start_year}_{end_year}'
                    La_nina_index_list = []
                    for i in range(len(vals_reshape)):
                        year_i=1982+i
                        if year_i>=start_year and year_i<=end_year:
                            La_nina_index_list.append(vals_reshape[i])
                    La_nina_index_list = np.array(La_nina_index_list)
                    La_nina_index_list = La_nina_index_list.flatten()

                    extract_vals_la_nina = La_nina_index_list[180:485]
                    result_dic_i_La_nina[key_La_nina] = extract_vals_la_nina
                result_dic_La_nina[pix] = result_dic_i_La_nina
                result_dic_El_nino[pix] = result_dic_i_El_nino
            outf_La_nina = outdir_LA_nina + f'{f.split(".")[0]}_La_nina'
            outf_El_nino = outdir_El_nino + f'{f.split(".")[0]}_El_nino'
            np.save(outf_La_nina, result_dic_La_nina)
            np.save(outf_El_nino, result_dic_El_nino)


    def extract_extreme_rainfall_event(self):
        ENSO_type = 'La_nina'
        fdir_threshold = data_root+rf'ERA5\ERA5_daily\dict\define_quantile_threshold\\'
        fdir_yearly_all=rf'D:\Project3\Data\ERA5\ERA5_daily\dict\ENSO_year_extraction\\{ENSO_type}\\'
        outdir = data_root+rf'\ERA5\ERA5_daily\dict\\extreme_event_extraction\\{ENSO_type}\\'
        T.mk_dir(outdir,force=True)
        spatial_threshold_dic=T.load_npy_dir(fdir_threshold)
        result_dic = {}
        for f in T.listdir(fdir_yearly_all):
            spatial_dic = T.load_npy(fdir_yearly_all+f)
            for pix in tqdm(spatial_dic):
                if not pix in spatial_threshold_dic:
                    continue
                threshold_dic=spatial_threshold_dic[pix]

                val_90th = threshold_dic['90th']
                print(val_90th)
                val_10th = threshold_dic['10th']
                print(val_10th)
                EI_nino_dic= spatial_dic[pix]
                result_dic_i = {}
                for year_range in EI_nino_dic:

                    extreme_wet_event = []
                    extreme_dry_event = []
                    for val in EI_nino_dic[year_range]:
                        if val > val_90th:
                            extreme_wet_event.append(val)

                    ## calculate the frequency and average intensity of extreme wet event and extreme dry event
                    ## intensity
                    average_intensity_extreme_wet_event = np.nanmean(extreme_wet_event)

                    ## frequency
                    frequency_extreme_wet_event = len(extreme_wet_event)




                    result_dic_i[year_range] = {
                        f'{ENSO_type}_average_intensity_extreme_wet_event':average_intensity_extreme_wet_event,

                        f'{ENSO_type}_frequency_extreme_wet_event':frequency_extreme_wet_event,



                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)


    def check(self):
        f=rf'D:\Project3\Data\ERA5\ERA5_daily\dict\dry_spell\\per_pix_dic_045_El_nino.npy'
        spatial_dic = np.load(f,allow_pickle=True).item()
        for pix in spatial_dic:
            vals = spatial_dic[pix]
            print(vals)

    def extract_rainfall_event_total(self):
        ENSO_type = 'El_nino'
        fdir_yearly_all = rf'D:\Project3\Data\ERA5\ERA5_daily\dict\ENSO_year_extraction\\{ENSO_type}\\'
        outdir = data_root + rf'\ERA5\ERA5_daily\dict\\total_rainfall\\{ENSO_type}\\'
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir_yearly_all):

            fdir_yearly_data = fdir_yearly_all + f
            spatial_dic = T.load_npy(fdir_yearly_data)

            result_dic = {}
            for pix in tqdm(spatial_dic):

                vals = spatial_dic[pix]
                EI_nino_dic = spatial_dic[pix]
                result_dic_i = {}
                for year_range in EI_nino_dic:
                    vals = EI_nino_dic[year_range]
                    vals = np.array(vals)
                    total_rainfall = np.nansum(vals)
                    result_dic_i[year_range] = {
                        f'{ENSO_type}_total_rainfall': total_rainfall
                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def wet_spell_dry_spell(self):
        ENSO_type = 'El_nino'
        # ENSO_type = 'La_nina'

        fdir_yearly_all=rf'D:\Project3\Data\ERA5\ERA5_daily\dict\ENSO_year_extraction\\{ENSO_type}\\'
        outdir = data_root+rf'\ERA5\ERA5_daily\dict\\dry_spell\\{ENSO_type}\\'
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir_yearly_all):

            fdir_yearly_data = fdir_yearly_all+f
            spatial_dic=T.load_npy(fdir_yearly_data)

            result_dic = {}
            for pix in tqdm(spatial_dic):

                vals = spatial_dic[pix]
                EI_nino_dic = spatial_dic[pix]
                result_dic_i = {}
                for year_range in EI_nino_dic:
                    vals=EI_nino_dic[year_range]
                    vals = np.array(vals)
                    vals_wet = vals.copy()

                    vals_wet[vals_wet >= 1] = np.nan

                    wet_index = np.where(~np.isnan(vals_wet))
                    if len(wet_index[0])==0:
                        continue
                    wet_index = np.array(wet_index)
                    wet_index = wet_index.flatten()
                    wet_index_groups = T.group_consecutive_vals(wet_index)
                    # plt.bar(range(len(vals)), vals)
                    # plt.bar(range(len(vals)), vals_wet)
                    # print(wet_index_groups)
                    # plt.show()
                    ## calcuate average wet spell
                    wet_spell = []
                    for group in wet_index_groups:
                        wet_spell.append(len(group))
                    wet_spell = np.array(wet_spell)
                    average_wet_spell = np.nanmean(wet_spell)
                    maxmum_wet_spell = np.nanmax(wet_spell)
                    result_dic_i[year_range] = {
                        f'{ENSO_type}_average_dry_spell':average_wet_spell,
                        f'{ENSO_type}_maxmum_dry_spell':maxmum_wet_spell
                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def LAI_ENSO_extraction(self):


        fdir = results_root + rf'Detrend\detrend_original\\'
        ### extract EL Nino year of LAI
        strong_El_Nino_year = [[1982, 1983], [1987, 1988], [1991, 1992], [1997, 1998], [2015, 2016]]

        strong_La_nina_year = [[1988, 1989], [1998, 1999], [2000, 2001], [2007, 2008], [2010, 2011]]
        result_dic_El_nino = {}
        for f in T.listdir(fdir):
            if not 'GPCC' in f:
                continue
            spatial_dic = np.load(fdir + f, allow_pickle=True).item()
            for pix in tqdm(spatial_dic):
                vals = spatial_dic[pix]
                r, c = pix
                result_dic_i = {}
                for year_range in strong_El_Nino_year:
                    start_year, end_year = year_range
                    key_EL_nino = f'{start_year}_{end_year}'
                    EL_Nino_index_list = []
                    for i in range(len(vals)):
                        year_i = 1982 + i
                        if year_i >= start_year and year_i <= end_year:
                            EL_Nino_index_list.append(vals[i])
                    EL_Nino_index_list = np.array(EL_Nino_index_list)
                    EL_Nino_index_list = EL_Nino_index_list.flatten()
                    EL_Nino_mean = np.nanmean(EL_Nino_index_list)

                    result_dic_i[key_EL_nino] = EL_Nino_mean
                result_dic_El_nino[pix] = result_dic_i
        result_dic_La_nina = {}
        for f in T.listdir(fdir):
            if not 'GPCC' in f:
                continue
            spatial_dic = np.load(fdir + f, allow_pickle=True).item()

            for pix in tqdm(spatial_dic):
                vals = spatial_dic[pix]
                r, c = pix
                result_dic_i = {}
                for year_range in strong_La_nina_year:
                    start_year, end_year = year_range
                    key_La_nina = f'{start_year}_{end_year}'
                    La_nina_index_list = []
                    for i in range(len(vals)):
                        year_i = 1982 + i
                        if year_i >= start_year and year_i <= end_year:
                            La_nina_index_list.append(vals[i])
                    La_nina_index_list = np.array(La_nina_index_list)
                    La_nina_index_list = La_nina_index_list.flatten()
                    result_dic_i[key_La_nina] = np.nanmean(La_nina_index_list)
                result_dic_La_nina[pix] = result_dic_i

        outdir= data_root + rf'\ERA5\ERA5_daily\dict\\GPCC_ENSO_extraction\\'
        T.mk_dir(outdir,force=True)
        np.save(outdir+f'El_nino',result_dic_El_nino)
        np.save(outdir+f'La_nina',result_dic_La_nina)
        pass





    def foo(self):
        # coding=utf-8

        np.random.seed(42)
        # vals = np.random.randn(200)
        vals = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1]
        vals = np.array(vals, dtype=float)
        vals = abs(vals)
        vals = np.array(vals)
        # print(vals)
        threshold = 0.5
        vals_wet = vals.copy()
        vals_wet[vals_wet > threshold] = np.nan
        wet_index = np.where(~np.isnan(vals_wet))
        wet_index = np.array(wet_index)
        wet_index = wet_index.flatten()
        wet_index_groups = T.group_consecutive_vals(wet_index)
        for group in wet_index_groups:
            print(group)
        plt.bar(range(len(vals)), vals)
        plt.bar(range(len(vals)), vals_wet)
        plt.show()

        pass
class extration_extreme_event_temperature_ENSO:
    def __init__(self):
        self.strong_El_Nino_list = [1982, 1983, 1987, 1988, 1991, 1992, 1997, 1998, 2015, 2016]

        self.strong_La_Nina_list = [1988, 1989, 1998, 1999, 2000, 2007, 2008, 2010, 2011]

    pass

    def run(self):
        # self.define_quantile_threshold()
        # self.extract_extreme_ENSO_year()
        # self.extract_extreme_heat_event()
        # self.extract_extreme_cold_event()
        # self.heat_spell()
        self.cold_spell()

        # self.extract_temperature_event_total()
        # self.wet_spell_dry_spell()

        pass

    def define_quantile_threshold(self):
        # 1) extract extreme wet event based on 90th percentile and calculate frequency and total duration
        # 2) extract extreme dry event based on 10th percentile and calculate frequency and total duration
        # 3) extract wet event intensity
        ## 4) extract dry event intensity
        ## extract VPD and calculate the frequency of VPD>2kpa
        fdir=rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\deseasonal_detrend\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\\define_quantile_threshold\\'
        T.mk_dir(outdir,force=True)

        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue

            spatial_dic = np.load(fdir+f,allow_pickle=True).item()
            result_dic={}
            for pix in tqdm(spatial_dic):

                vals=spatial_dic[pix]


                if T.is_all_nan(vals):
                    continue
                # plt.bar(range(len(vals)),vals)
                # plt.show()

                val_90th= np.percentile(vals,90)
                val_10th = np.percentile(vals, 10)
                val_95th = np.percentile(vals, 95)
                val_5th = np.percentile(vals, 5)
                val_99th = np.percentile(vals, 99)
                val_1st = np.percentile(vals, 1)
                dic_i={
                    '90th':val_90th,
                    '10th':val_10th,
                    '95th':val_95th,
                    '5th':val_5th,
                    '99th':val_99th,
                    '1st':val_1st
                }
                result_dic[pix]=dic_i
            outf=outdir+f
            np.save(outf,result_dic)
    def extract_extreme_ENSO_year(self):


        strong_El_Nino_year=[[1982,1983],[1987,1988],[1991,1992],[1997,1998],[2015,2016]]

        strong_La_nina_year=[[1988,1989],[1998,1999],[2000,2001],[2007,2008],[2010,2011]]


        fdir_yearly_all=rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\\deseasonal_detrend\\'
        outdir_LA_nina = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\\ENSO_year_extraction\\LA_nina\\'
        outdir_El_nino = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\\ENSO_year_extraction\\El_nino\\'


        T.mk_dir(outdir_LA_nina,force=True)
        T.mk_dir(outdir_El_nino,force=True)
        # spatial_threshold_dic=T.load_npy_dir(fdir_threshold)
        for f in T.listdir(fdir_yearly_all):


            spatial_dic=T.load_npy(fdir_yearly_all+f)

            result_dic_La_nina = {}
            result_dic_El_nino = {}
            for pix in tqdm(spatial_dic):

                vals = spatial_dic[pix]
                vals_reshape = np.array(vals).reshape(-1, 365)
                r,c = pix

                ### extract El Nino and La Nina year data
                ## transform daily data to yearly data

                ## reshape the data 38


                result_dic_i_El_nino = {}


                for year_range in strong_El_Nino_year:

                    start_year,end_year = year_range

                    key_EL_nino = f'{start_year}_{end_year}'


                    EL_Nino_index_list = []
                    for i in range(len(vals_reshape)):
                        year_i=1982+i
                        if year_i>=start_year and year_i<=end_year:
                            EL_Nino_index_list.append(vals_reshape[i])
                    EL_Nino_index_list = np.array(EL_Nino_index_list)
                    EL_Nino_index_list = EL_Nino_index_list.flatten()

                    ## extract 180 to 485
                    extract_vals_el_nino = EL_Nino_index_list[180:485]
                    result_dic_i_El_nino[key_EL_nino] = extract_vals_el_nino

                result_dic_i_La_nina = {}

                for year_range in strong_La_nina_year:
                    start_year,end_year = year_range
                    key_La_nina = f'{start_year}_{end_year}'
                    La_nina_index_list = []
                    for i in range(len(vals_reshape)):
                        year_i=1982+i
                        if year_i>=start_year and year_i<=end_year:
                            La_nina_index_list.append(vals_reshape[i])
                    La_nina_index_list = np.array(La_nina_index_list)
                    La_nina_index_list = La_nina_index_list.flatten()

                    extract_vals_la_nina = La_nina_index_list[180:485]
                    result_dic_i_La_nina[key_La_nina] = extract_vals_la_nina
                result_dic_La_nina[pix] = result_dic_i_La_nina
                result_dic_El_nino[pix] = result_dic_i_El_nino
            outf_La_nina = outdir_LA_nina + f'{f.split(".")[0]}_La_nina'
            outf_El_nino = outdir_El_nino + f'{f.split(".")[0]}_El_nino'
            np.save(outf_La_nina, result_dic_La_nina)
            np.save(outf_El_nino, result_dic_El_nino)

    def extract_extreme_heat_event(self):

        # ENSO_type = 'El_nino'
        ENSO_type = 'La_nina'
        fdir_threshold = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\define_quantile_threshold\\'
        fdir_yearly_all = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\ENSO_year_extraction\\{ENSO_type}\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\extreme_heat_event_extraction\\{ENSO_type}\\'
        T.mk_dir(outdir, force=True)
        spatial_threshold_dic = T.load_npy_dir(fdir_threshold)
        result_dic = {}
        for f in T.listdir(fdir_yearly_all):
            spatial_dic = T.load_npy(fdir_yearly_all + f)
            for pix in tqdm(spatial_dic):
                if not pix in spatial_threshold_dic:
                    continue
                threshold_dic = spatial_threshold_dic[pix]

                val_90th = threshold_dic['90th']
                print(val_90th)

                EI_nino_dic = spatial_dic[pix]
                result_dic_i = {}
                for year_range in EI_nino_dic:

                    extreme_wet_event = []
                    extreme_dry_event = []
                    for val in EI_nino_dic[year_range]:
                        if val > val_90th:
                            extreme_wet_event.append(val)

                    ## calculate the frequency and average intensity of extreme wet event and extreme dry event
                    ## intensity
                    average_intensity_extreme_wet_event = np.nanmean(extreme_wet_event)

                    ## frequency
                    frequency_extreme_wet_event = len(extreme_wet_event)


                    result_dic_i[year_range] = {
                        f'{ENSO_type}_average_intensity_extreme_heat_event': average_intensity_extreme_wet_event,

                        f'{ENSO_type}_frequency_extreme_heat_event': frequency_extreme_wet_event,


                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def extract_extreme_cold_event(self):

        # ENSO_type = 'El_nino'
        ENSO_type = 'La_nina'
        fdir_threshold = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\define_quantile_threshold\\'
        fdir_yearly_all = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\ENSO_year_extraction\\{ENSO_type}\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\extreme_cold_event_extraction\\{ENSO_type}\\'
        T.mk_dir(outdir, force=True)
        spatial_threshold_dic = T.load_npy_dir(fdir_threshold)
        result_dic = {}
        for f in T.listdir(fdir_yearly_all):
            spatial_dic = T.load_npy(fdir_yearly_all + f)
            for pix in tqdm(spatial_dic):
                if not pix in spatial_threshold_dic:
                    continue
                threshold_dic = spatial_threshold_dic[pix]

                val_10th = threshold_dic['10th']
                print(val_10th)

                EI_nino_dic = spatial_dic[pix]
                result_dic_i = {}
                for year_range in EI_nino_dic:

                    extreme_wet_event = []
                    extreme_dry_event = []
                    for val in EI_nino_dic[year_range]:
                        if val < val_10th:
                            extreme_dry_event.append(val)

                    ## calculate the frequency and average intensity of extreme wet event and extreme dry event
                    ## intensity
                    average_intensity_extreme_dry_event = np.nanmean(extreme_dry_event)

                    ## frequency
                    frequency_extreme_dry_event = len(extreme_dry_event)

                    result_dic_i[year_range] = {
                        f'{ENSO_type}_average_intensity_extreme_cold_event': average_intensity_extreme_dry_event,

                        f'{ENSO_type}_frequency_extreme_cold_event': frequency_extreme_dry_event,
                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)
    def heat_spell(self):
        # ENSO_type = 'El_nino'
        ENSO_type = 'La_nina'
        fdir_threshold = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\define_quantile_threshold\\'
        dic_threshold = T.load_npy_dir(fdir_threshold)

        fdir_yearly_all=rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\ENSO_year_extraction\\{ENSO_type}\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\heat_spell\\{ENSO_type}\\'
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir_yearly_all):

            fdir_yearly_data = fdir_yearly_all+f
            spatial_dic=T.load_npy(fdir_yearly_data)

            result_dic = {}
            for pix in tqdm(spatial_dic):
                threshold= dic_threshold[pix]
                val_90th = threshold['90th']


                vals = spatial_dic[pix]
                EI_nino_dic = spatial_dic[pix]
                result_dic_i = {}
                for year_range in EI_nino_dic:
                    vals=EI_nino_dic[year_range]
                    vals = np.array(vals)
                    vals_wet = vals.copy()
                    ### calculat heat event and cold event duration
                    vals_wet[vals_wet < val_90th] = np.nan

                    wet_index = np.where(~np.isnan(vals_wet))
                    if len(wet_index[0])==0:
                        continue
                    wet_index = np.array(wet_index)
                    wet_index = wet_index.flatten()
                    wet_index_groups = T.group_consecutive_vals(wet_index)
                    # plt.bar(range(len(vals)), vals)
                    # plt.bar(range(len(vals)), vals_wet)
                    # print(wet_index_groups)
                    # plt.show()
                    # calcuate average wet spell
                    wet_spell = []
                    for group in wet_index_groups:
                        wet_spell.append(len(group))
                    wet_spell = np.array(wet_spell)
                    average_wet_spell = np.nanmean(wet_spell)
                    maxmum_wet_spell = np.nanmax(wet_spell)
                    result_dic_i[year_range] = {
                        f'{ENSO_type}_average_heat_event_duration':average_wet_spell,
                        f'{ENSO_type}_maximun_heat_event_duration':maxmum_wet_spell
                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def cold_spell(self):
        # ENSO_type = 'El_nino'
        ENSO_type = 'La_nina'
        fdir_threshold = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\define_quantile_threshold\\'
        dic_threshold = T.load_npy_dir(fdir_threshold)

        fdir_yearly_all = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\ENSO_year_extraction\\{ENSO_type}\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\cold_spell\\{ENSO_type}\\'
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir_yearly_all):

            fdir_yearly_data = fdir_yearly_all + f
            spatial_dic = T.load_npy(fdir_yearly_data)

            result_dic = {}
            for pix in tqdm(spatial_dic):
                threshold = dic_threshold[pix]

                val_10th = threshold['10th']
                print(val_10th)

                vals = spatial_dic[pix]
                EI_nino_dic = spatial_dic[pix]
                result_dic_i = {}
                for year_range in EI_nino_dic:
                    vals = EI_nino_dic[year_range]
                    vals = np.array(vals)
                    vals_wet = vals.copy()
                    ### calculat cold event duration
                    vals_wet[vals_wet < val_10th] = np.nan

                    wet_index = np.where(~np.isnan(vals_wet))
                    if len(wet_index[0]) == 0:
                        continue
                    wet_index = np.array(wet_index)
                    wet_index = wet_index.flatten()
                    wet_index_groups = T.group_consecutive_vals(wet_index)
                    # plt.bar(range(len(vals)), vals)
                    # plt.bar(range(len(vals)), vals_wet)
                    # print(wet_index_groups)
                    # plt.show()
                    # calcuate average wet spell
                    wet_spell = []
                    for group in wet_index_groups:
                        wet_spell.append(len(group))
                    wet_spell = np.array(wet_spell)
                    average_wet_spell = np.nanmean(wet_spell)
                    maxmum_wet_spell = np.nanmax(wet_spell)
                    result_dic_i[year_range] = {
                        f'{ENSO_type}_average_cold_event_duration': average_wet_spell,
                        f'{ENSO_type}_maximun_cold_event_duration': maxmum_wet_spell
                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def run(self):
        self.extract_lagged_precipitation()

        pass

class build_df:
    def run (self):
        # self.build_df()
        # self.add_attribution_rainfall()
        # self.add_attribution_rainfall_frequency()
        # self.add_attribution_dry_spell()
        # self.add_attribution_extreme_tmax_average_CV()
        # self.add_attribution_extreme_tmin_average_CV()
        # self.add_extreme_cold_frequency()
        # self.add_extreme_heat_frequency()
        self.add_attribution_heat_spell()
        # self.add_LAI_relative_change()
        # self.add_GPCP_lagged()


        # self.rename_columns()
        # self.delete_field()



        pass
    def build_df(self):
        fdir = data_root + rf'ERA5\ERA5_daily\dict\dry_spell\\'
        outdir = data_root + rf'\ERA5\ERA5_daily\SHAP\\RF_df\\'
        T.mk_dir(outdir, force=True)
        flag = 1981
        result_dic = {}
        for f in T.listdir(fdir):
            spatial_dic = np.load(fdir + f, allow_pickle=True).item()
            for pix in spatial_dic:
                dict_i = spatial_dic[pix]
                if len(dict_i) == 0:
                    continue
                for year_range in dict_i:
                    dict_result_i = dict_i[year_range]
                    dict_result_i['pix'] = pix
                    flag += 1
                    dict_result_i['year_range'] = year_range

                    result_dic[flag] = dict_result_i
        df = T.dic_to_df(result_dic, 'flag')
        T.print_head_n(df)
        outf = outdir + 'RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
    def add_attribution_rainfall(self): ## add dry spell, frequency, total rainfall, average intensity, wet spell, maxmum wet spell

        df= T.load_df(data_root+rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        fdir_all = data_root+rf'\ERA5\ERA5_daily\dict\\rainfall_CV_total\\'

        spatial_dic = T.load_npy_dir(fdir_all)
        for i, row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            df.loc[i,key] = dict_i_i[key]
        outf = data_root+rf'\ERA5\ERA5_daily\\SHAP\RF_df\\RF_df'
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def add_attribution_rainfall_frequency(self): ## add dry spell, frequency, total rainfall, average intensity, wet spell, maxmum wet spell

        df= T.load_df(data_root+rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        fdir_all = data_root+rf'\ERA5\ERA5_daily\dict\\rainfall_frequency\\'

        spatial_dic = T.load_npy_dir(fdir_all)
        for i, row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            df.loc[i,key] = dict_i_i[key]
        outf = data_root+rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def add_attribution_dry_spell(self):

        df= T.load_df(data_root+rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        fdir_all = data_root+rf'\ERA5\ERA5_daily\dict\\dry_spell\\'

        spatial_dic = T.load_npy_dir(fdir_all)
        for i, row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            df.loc[i,key] = dict_i_i[key]
        outf = data_root+rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def add_attribution_heat_spell(self):

        df= T.load_df(data_root+rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        fdir_all = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\cold_spell\\'

        spatial_dic = T.load_npy_dir(fdir_all)
        for i, row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            r,c=pix
            if r<120:
                continue
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            df.loc[i,key] = dict_i_i[key]
        outf = data_root+rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df,outf)
        T.df_to_excel(df,outf)


    def add_attribution_extreme_tmax_average_CV(self):
        df = T.load_df(data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        fdir_all =rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\extreme_tmax_average_CV\\'

        spatial_dic = T.load_npy_dir(fdir_all)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            df.loc[i, key] = dict_i_i[key]
        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

        pass

    def add_attribution_extreme_tmin_average_CV(self):
        df = T.load_df(data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        fdir_all = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\extreme_tmin_average_CV\\'

        spatial_dic = T.load_npy_dir(fdir_all)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            df.loc[i, key] = dict_i_i[key]
        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
    def add_extreme_cold_frequency(self):
        df = T.load_df(data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        fdir_all = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\extreme_cold_frequency\\'

        spatial_dic = T.load_npy_dir(fdir_all)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            df.loc[i, key] = dict_i_i[key]
        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def add_extreme_heat_frequency(self):
        df = T.load_df(data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        fdir_all = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\extreme_heat_frequency\\'

        spatial_dic = T.load_npy_dir(fdir_all)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r,c=pix
            if r<120:
                continue
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            df.loc[i, key] = dict_i_i[key]
        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

    def add_LAI_relative_change(self):
        df= T.load_df(data_root+rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'relative_change\OBS_LAI_extend\GPCC.npy'


        spatial_dic = T.load_npy(f)

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            year = row.year_range

            pix = row['pix']
            r, c = pix

            if not pix in spatial_dic:
                NDVI_list.append(np.nan)
                continue

            vals = spatial_dic[pix][0:38]


            v1 = vals[year]
            # v1=vals[year-1982]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)
        df['GPCP_precip'] = NDVI_list

        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass

    def add_GPCP_lagged(self): ##
        df = T.load_df(data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'relative_change\OBS_LAI_extend\GPCC.npy'

        spatial_dic = T.load_npy(f)

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            year = row.year_range

            pix = row['pix']
            r, c = pix

            if not pix in spatial_dic:
                NDVI_list.append(np.nan)
                continue

            vals = spatial_dic[pix][0:38]
            if year - 1 < 0:
                NDVI_list.append(np.nan)
                continue
            v1 = vals[year-1]
            # v1=vals[year-1982]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)
        df['GPCP_precip_pre'] = NDVI_list

        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass

    def rename_columns(self):
        df= T.load_df(data_root+rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        df = df.rename(columns={'CV':'CV_rainfall',

                                'total':'total_rainfall',



                            }

                               )



        outf = data_root+rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df,outf)
        T.df_to_excel(df,outf)
        pass
    def delete_field(self):
        df= T.load_df(data_root+rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        df = df.drop(columns=['total_rainfall',])
        outf = data_root+rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

        pass
class extract_rainfall:
    ## 1) extract rainfall CV
    ## 2) extract rainfall total
    ## 3) extract rainfall frequency
    ## extract dry frequency
    ## 4) extract rainfall intensity
    ## 5) extract rainfall wet spell
    ## 6) extract rainfall dry spell
    def run(self):
        # self.extract_growing_season()
        self.extract_rainfall_CV_total()
        # self.rainfall_frequency()
        # self.dry_spell()

        pass
    def extract_rainfall_CV_total(self):  ## extract total and CV of rainfall
        fdir = data_root+rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir_CV = data_root+rf'\ERA5\ERA5_daily\dict\\rainfall_CV_total\\'

        T.mk_dir(outdir_CV,force=True)

        for f in T.listdir(fdir):

            spatial_dic = np.load(fdir+f,allow_pickle=True).item()
            result_dic = {}

            for pix in tqdm(spatial_dic):
                ### ui==if northern hemisphere
                r,c = pix
                vals = spatial_dic[pix]
                vals_flatten = np.array(vals).flatten()


                result_dic_i = {}

                for i in range(38):

                    if 120<r<=240:  # Northern hemisphere
                        ### April to October is growing season

                        vals_growing_season = vals_flatten[i*365+120:(i+1)*365+304]

                    elif 240<r<480:### whole year is growing season

                        vals_growing_season = vals_flatten[i*365:(i+1)*365]


                    else: ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break


                        vals_growing_season = vals_flatten[i*365+304:(i+1)*365+120]

                    vals_growing_season = np.array(vals_growing_season)
                    if T.is_all_nan(vals_growing_season):
                        continue
                    CV = np.std(vals_growing_season)/np.mean(vals_growing_season)
                    total = np.nansum(vals_growing_season)
                    result_dic_i[i] = {f'CV_rainfall':CV,
                                       f'total_rainfall':total}
                result_dic[pix] = result_dic_i

            outf = outdir_CV+f

            np.save(outf,result_dic)

    def extract_growing_season(self):
        fdir = data_root + rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir = data_root + rf'\ERA5\ERA5_daily\dict\\growing_season_extraction\\'
        T.mk_dir(outdir, force=True)

        for f in T.listdir(fdir):
            spatial_dic = np.load(fdir + f, allow_pickle=True).item()
            result_dic = {}
            for pix in tqdm(spatial_dic):

                spatial_dic = np.load(fdir + f, allow_pickle=True).item()

                r, c = pix
                vals = spatial_dic[pix]
                vals_flatten = np.array(vals).flatten()

                for i in range(38):

                    if 120 < r <= 240:  # Northern hemisphere
                        ### April to October is growing season

                        vals_growing_season = vals_flatten[i * 365 + 120:(i + 1) * 365 + 304]

                    elif 240 < r < 480:  ### whole year is growing season

                        vals_growing_season = vals_flatten[i * 365:(i + 1) * 365]


                    else:  ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break

                        vals_growing_season = vals_flatten[i * 365 + 304:(i + 1) * 365 + 120]

                    vals_growing_season = np.array(vals_growing_season)
                    result_dic[pix] = vals_growing_season


            outf = outdir + f
            np.save(outf, result_dic)

    def rainfall_frequency(self):
        fdir = data_root + rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir= data_root + rf'\ERA5\ERA5_daily\dict\\rainfall_frequency\\'
        threshold_f= data_root + rf'\ERA5\ERA5_daily\dict\\define_quantile_threshold\\'
        dic_threshold = T.load_npy_dir(threshold_f)
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):

            spatial_dic = np.load(fdir + f, allow_pickle=True).item()
            result_dic = {}
            for pix in tqdm(spatial_dic):
                r,c=pix
                if not pix in dic_threshold:
                    continue
                vals = spatial_dic[pix]
                vals_flatten = np.array(vals).flatten()
                threshold = dic_threshold[pix]
                threshold_wet = threshold['90th']
                threshold_dry = threshold['10th']

                result_dic_i = {}
                for i in range(38):
                    if 120<r<=240:  # Northern hemisphere

                        vals_growing_season = vals_flatten[i * 365 + 120:(i + 1) * 365 + 304]
                        ## calculate days >threshold



                    elif 240 < r < 480:  ### whole year is growing season

                        vals_growing_season = vals_flatten[i * 365:(i + 1) * 365]


                    else:  ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break

                        vals_growing_season = vals_flatten[i * 365 + 304:(i + 1) * 365 + 120]
                    vals_growing_season = np.array(vals_growing_season)
                    if T.is_all_nan(vals_growing_season):
                        continue
                    frequency_wet = len(np.where(vals_growing_season > threshold_wet)[0])

                    result_dic_i[i] = {f'frequency_wet':frequency_wet,
                                        }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def dry_spell(self):

        fdir = data_root + rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir = data_root + rf'\ERA5\ERA5_daily\dict\\dry_spell\\'
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):

            spatial_dic = np.load(fdir + f, allow_pickle=True).item()

            result_dic = {}
            for pix in tqdm(spatial_dic):
                r,c=pix

                vals = spatial_dic[pix]
                vals_flatten = np.array(vals).flatten()

                result_dic_i = {}

                for i in range(38):
                    if 120<r<=240:  # Northern hemisphere

                        vals_growing_season = vals_flatten[i * 365 + 120:(i + 1) * 365 + 304]
                    if 240 < r < 480:  ### whole year is growing season

                            vals_growing_season = vals_flatten[i * 365:(i + 1) * 365]
                    else:  ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break
                        vals_growing_season = vals_flatten[i * 365 + 304:(i + 1) * 365 + 120]
                    vals_growing_season = np.array(vals_growing_season)
                    vals_wet = vals_growing_season.copy()

                    vals_wet[vals_wet >= 1] = np.nan

                    dry_index = np.where(~np.isnan(vals_wet))
                    if len(dry_index[0]) == 0:
                        continue
                    dry_index = np.array(dry_index)
                    dry_index = dry_index.flatten()
                    dry_index_groups = T.group_consecutive_vals(dry_index)

                    # plt.bar(range(len(vals_growing_season)), vals_growing_season)
                    # plt.bar(range(len(vals_growing_season)), vals_wet)
                    # print(dry_index_groups)
                    # plt.show()
                    ## calcuate average wet spell
                    dry_spell = []
                    for group in dry_index_groups:
                        dry_spell.append(len(group))
                    dry_spell = np.array(dry_spell)

                    average_wet_spell = np.nanmean(dry_spell)
                    maxmum_wet_spell = np.nanmax(dry_spell)
                    result_dic_i[i] = {
                        f'average_dry_spell': average_wet_spell,
                        f'maximun_dry_spell': maxmum_wet_spell
                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)


pass


class extract_temperature:
    def run(self):

        self.extract_extreme_heat_frequency()
        self.extract_extreme_cold_frequency()
        # self.extract_tmax_average_CV()
        # self.extract_tmin_average_CV()
        # self.cold_spell()
        # self.heat_spell()


        pass
    def extract_extreme_heat_frequency(self):
        threshold_f = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\define_quantile_threshold\\'
        fdir = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\\deseasonal_detrend\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\\extreme_heat_frequency\\'


        T.mk_dir(outdir, force=True)
        spatial_threshold_dic = T.load_npy_dir(threshold_f)
        result_dic = {}
        for f in T.listdir(fdir):


            spatial_dic = T.load_npy(fdir + f)
            for pix in tqdm(spatial_dic):
                r, c = pix
                if not pix in spatial_threshold_dic:
                    continue
                threshold_dic = spatial_threshold_dic[pix]

                val_90th = threshold_dic['90th']
                # print(val_90th)

                vals = spatial_dic[pix]


                result_dic_i = {}
                for i in range(38):
                    if 120 < r <= 240:  # Northern hemisphere

                        vals_growing_season = vals[i * 365 + 120:(i + 1) * 365 + 304]
                        ## calculate days >threshold

                    elif 240 < r < 480:  ### whole year is growing season

                        vals_growing_season = vals[i * 365:(i + 1) * 365]


                    else:  ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break

                        vals_growing_season = vals[i * 365 + 304:(i + 1) * 365 + 120]
                    vals_growing_season = np.array(vals_growing_season)
                    if T.is_all_nan(vals_growing_season):
                        continue
                    array_abovethrehold=np.where(vals_growing_season>val_90th,vals_growing_season,0)
                    ### calculate average anomaly of days >threshold
                    array_abovethrehold[array_abovethrehold==0]=np.nan
                    average_anomaly_heat = np.nanmean(array_abovethrehold)



                    frequency_heat = len(np.where(vals_growing_season > val_90th)[0])
                    ### average anomaly heat event = average anomaly of all days >threshold





                    result_dic_i[i] = {f'frequency_heat_event': frequency_heat,
                                        f'average_anomaly_heat_event': average_anomaly_heat}
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def extract_extreme_cold_frequency(self):
        threshold_f = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\define_quantile_threshold\\'
        fdir = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\\deseasonal_detrend\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\\extreme_cold_frequency\\'

        T.mk_dir(outdir, force=True)
        spatial_threshold_dic = T.load_npy_dir(threshold_f)
        result_dic = {}
        for f in T.listdir(fdir):

            spatial_dic = T.load_npy(fdir + f)
            for pix in tqdm(spatial_dic):
                r, c = pix
                if not pix in spatial_threshold_dic:
                    continue
                threshold_dic = spatial_threshold_dic[pix]

                val_10th = threshold_dic['10th']
                # print(val_90th)

                vals = spatial_dic[pix]

                result_dic_i = {}
                for i in range(38):
                    if 120 < r <= 240:  # Northern hemisphere

                        vals_growing_season = vals[i * 365 + 120:(i + 1) * 365 + 304]
                        ## calculate days >threshold

                    elif 240 < r < 480:  ### whole year is growing season

                        vals_growing_season = vals[i * 365:(i + 1) * 365]


                    else:  ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break

                        vals_growing_season = vals[i * 365 + 304:(i + 1) * 365 + 120]
                    vals_growing_season = np.array(vals_growing_season)
                    if T.is_all_nan(vals_growing_season):
                        continue
                    frequency_heat = len(np.where(vals_growing_season < val_10th)[0])
                    ### calculate average anomaly of days >threshold
                    array_abovethrehold = np.where(vals_growing_season < val_10th, vals_growing_season, 0)
                    array_abovethrehold[array_abovethrehold == 0] = np.nan
                    average_anomaly_heat = np.nanmean(array_abovethrehold)

                    result_dic_i[i] = {f'frequency_cold_event': frequency_heat,
                                       f'average_anomaly_cold_event': average_anomaly_heat}
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def extract_tmin_average_CV(self):  ## extract total and CV of temperature
        fdir = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\\deseasonal_detrend\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\\extreme_tmin_average_CV\\'


        T.mk_dir(outdir, force=True)

        for f in T.listdir(fdir):

            spatial_dic = np.load(fdir+f,allow_pickle=True).item()
            result_dic = {}

            for pix in tqdm(spatial_dic):
                ### ui==if northern hemisphere
                r,c = pix
                vals = spatial_dic[pix]



                result_dic_i = {}

                for i in range(38):

                    if 120<r<=240:  # Northern hemisphere
                        ### April to October is growing season

                        vals_growing_season = vals[i*365+120:(i+1)*365+304]

                    elif 240<r<480:### whole year is growing season

                        vals_growing_season = vals[i*365:(i+1)*365]


                    else: ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break


                        vals_growing_season = vals[i*365+304:(i+1)*365+120]

                    vals_growing_season = np.array(vals_growing_season)
                    if T.is_all_nan(vals_growing_season):
                        continue

                    CV= np.std(vals_growing_season)/np.mean(vals_growing_season)
                    result_dic_i[i] = {
                                       f'tmin_CV':CV}
                result_dic[pix] = result_dic_i

            outf = outdir+f

            np.save(outf,result_dic)

    def extract_tmax_average_CV(self):  ## extract total and CV of temperature
        fdir = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\\deseasonal_detrend\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\\extreme_tmax_average_CV\\'

        T.mk_dir(outdir, force=True)

        for f in T.listdir(fdir):

            spatial_dic = np.load(fdir + f, allow_pickle=True).item()
            result_dic = {}

            for pix in tqdm(spatial_dic):
                ### ui==if northern hemisphere
                r, c = pix
                vals = spatial_dic[pix]

                result_dic_i = {}

                for i in range(38):

                    if 120 < r <= 240:  # Northern hemisphere
                        ### April to October is growing season

                        vals_growing_season = vals[i * 365 + 120:(i + 1) * 365 + 304]

                    elif 240 < r < 480:  ### whole year is growing season

                        vals_growing_season = vals[i * 365:(i + 1) * 365]


                    else:  ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break

                        vals_growing_season = vals[i * 365 + 304:(i + 1) * 365 + 120]

                    vals_growing_season = np.array(vals_growing_season)
                    if T.is_all_nan(vals_growing_season):
                        continue

                    CV = np.std(vals_growing_season) / np.mean(vals_growing_season)

                    result_dic_i[i] = {
                                       f'tmax_CV': CV}
                result_dic[pix] = result_dic_i

            outf = outdir + f

            np.save(outf, result_dic)

    def heat_spell(self):
        fdir_threshold = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\define_quantile_threshold\\'
        dic_threshold = T.load_npy_dir(fdir_threshold)

        fdir_yearly_all = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\\deseasonal_detrend\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\heat_spell\\'
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir_yearly_all):

            fdir_yearly_data = fdir_yearly_all + f
            spatial_dic = T.load_npy(fdir_yearly_data)

            result_dic = {}
            for pix in tqdm(spatial_dic):
                r, c = pix
                threshold = dic_threshold[pix]
                val_90th = threshold['90th']

                vals = spatial_dic[pix]


                result_dic_i = {}

                for i in range(38):
                    if 120 < r <= 240:  # Northern hemisphere

                        vals_growing_season = vals[i * 365 + 120:(i + 1) * 365 + 304]
                    if 240 < r < 480:  ### whole year is growing season

                        vals_growing_season = vals[i * 365:(i + 1) * 365]
                    else:  ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break
                        vals_growing_season = vals[i * 365 + 304:(i + 1) * 365 + 120]
                    vals_growing_season = np.array(vals_growing_season)
                    vals_wet = vals_growing_season.copy()

                    vals_wet[vals_wet >= val_90th] = np.nan

                    dry_index = np.where(~np.isnan(vals_wet))
                    if len(dry_index[0]) == 0:
                        continue
                    dry_index = np.array(dry_index)
                    dry_index = dry_index.flatten()
                    dry_index_groups = T.group_consecutive_vals(dry_index)

                    # plt.bar(range(len(vals_growing_season)), vals_growing_season)
                    # plt.bar(range(len(vals_growing_season)), vals_wet)
                    # print(dry_index_groups)
                    # plt.show()
                    ## calcuate average wet spell
                    dry_spell = []
                    for group in dry_index_groups:
                        dry_spell.append(len(group))
                    dry_spell = np.array(dry_spell)

                    average_wet_spell = np.nanmean(dry_spell)
                    maxmum_wet_spell = np.nanmax(dry_spell)
                    result_dic_i[i] = {
                        f'average_heat_spell': average_wet_spell,
                        f'maximun_heat_spell': maxmum_wet_spell
                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

    def cold_spell(self):
        fdir_threshold = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\define_quantile_threshold\\'
        dic_threshold = T.load_npy_dir(fdir_threshold)

        fdir_yearly_all = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\\deseasonal_detrend\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5\min_temp\cold_spell\\'
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir_yearly_all):

            fdir_yearly_data = fdir_yearly_all + f
            spatial_dic = T.load_npy(fdir_yearly_data)

            result_dic = {}
            for pix in tqdm(spatial_dic):
                r, c = pix
                threshold = dic_threshold[pix]
                val_10th = threshold['10th']

                vals = spatial_dic[pix]

                result_dic_i = {}

                for i in range(38):
                    if 120 < r <= 240:  # Northern hemisphere

                        vals_growing_season = vals[i * 365 + 120:(i + 1) * 365 + 304]
                    if 240 < r < 480:  ### whole year is growing season

                        vals_growing_season = vals[i * 365:(i + 1) * 365]
                    else:  ## october to April is growing season  Southern hemisphere
                        if i > 37:
                            break
                        vals_growing_season = vals[i * 365 + 304:(i + 1) * 365 + 120]
                    vals_growing_season = np.array(vals_growing_season)
                    vals_wet = vals_growing_season.copy()

                    vals_wet[vals_wet <= val_10th] = np.nan

                    dry_index = np.where(~np.isnan(vals_wet))
                    if len(dry_index[0]) == 0:
                        continue
                    dry_index = np.array(dry_index)
                    dry_index = dry_index.flatten()
                    dry_index_groups = T.group_consecutive_vals(dry_index)

                    # plt.bar(range(len(vals_growing_season)), vals_growing_season)
                    # plt.bar(range(len(vals_growing_season)), vals_wet)
                    # print(dry_index_groups)
                    # plt.show()
                    ## calcuate average wet spell
                    dry_spell = []
                    for group in dry_index_groups:
                        dry_spell.append(len(group))
                    dry_spell = np.array(dry_spell)

                    average_wet_spell = np.nanmean(dry_spell)
                    maxmum_wet_spell = np.nanmax(dry_spell)
                    result_dic_i[i] = {
                        f'average_cold_spell': average_wet_spell,
                        f'maximun_cold_spell': maxmum_wet_spell
                    }
                result_dic[pix] = result_dic_i
            outf = outdir + f
            np.save(outf, result_dic)

class build_df_ENSO:
    def run (self):
        self.build_df()
        # self.add_EL_nino_attribution()
        # self.add_EL_nino_attribution_temperature()
        # self.rename_columns()


        pass
    def build_df(self):
        fdir = data_root+rf'ERA5\ERA5_daily\dict\dry_spell\\'
        outdir = data_root+rf'\ERA5\ERA5_daily\SHAP\\RF_df\\'
        T.mk_dir(outdir,force=True)
        flag= 1982
        result_dic= {}
        for f in T.listdir(fdir):
            spatial_dic = np.load(fdir+f,allow_pickle=True).item()
            for pix in spatial_dic:
                dict_i = spatial_dic[pix]
                if len(dict_i)==0:
                    continue
                for year_range in dict_i:
                    dict_result_i = dict_i[year_range]
                    dict_result_i['pix'] = pix
                    dict_result_i['year_range'] = year_range
                    flag+=1
                    result_dic[flag] = dict_result_i
        df = T.dic_to_df(result_dic,'flag')
        T.print_head_n(df)
        outf = outdir+'RF_df'
        T.save_df(df,outf)
        T.df_to_excel(df,outf)
    def add_EL_nino_attribution_rainfall(self): ## add dry spell, frequency, total rainfall, average intensity, wet spell, maxmum wet spell
        ENSO_type = 'El_nino'
        df= T.load_df(data_root+rf'\ERA5\ERA5_daily\dict\\RF_df_EL_nino\\RF_df_{ENSO_type}.df')

        fdir_all = data_root+rf'\ERA5\ERA5_daily\dict\\total_rainfall\\{ENSO_type}\\'

        spatial_dic = T.load_npy_dir(fdir_all)
        for i, row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            df.loc[i,key] = dict_i_i[key]
        outf = data_root+rf'\ERA5\ERA5_daily\dict\\RF_df_EL_nino\\RF_df_El_nino.df'
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def add_EL_nino_attribution_temperature(self): ## add dry spell, frequency, total rainfall, average intensity, wet spell, maxmum wet spell
        ENSO_type = 'El_nino'
        df= T.load_df(data_root+rf'\ERA5\ERA5_daily\dict\\RF_df_EL_nino\\RF_df_{ENSO_type}.df')

        fdir_all = rf'C:\Users\wenzhang1\Desktop\ERA5\max_temp\extreme_event_extraction\\{ENSO_type}\\'

        spatial_dic = T.load_npy_dir(fdir_all)
        for i, row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            print(key)
                            df.loc[i,key] = dict_i_i[key]
        outf = data_root+rf'\ERA5\ERA5_daily\dict\\RF_df_EL_nino\\RF_df_El_nino.df'
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def add_LAI_relative_change(self):
        df= T.load_df(data_root+rf'\ERA5\ERA5_daily\dict\\RF_df\\RF_df.df')
        fdir = data_root+rf'\ERA5\ERA5_daily\dict\\GPCC_ENSO_extraction\\'


        spatial_dic = T.load_npy_dir(fdir)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            year_range = row['year_range']
            if pix in spatial_dic:
                dict_i = spatial_dic[pix]
                for year_range_i in dict_i:
                    if year_range_i == year_range:
                        dict_i_i = dict_i[year_range_i]
                        for key in dict_i_i:
                            df.loc[i, key] = dict_i_i[key]
        outf = data_root + rf'\ERA5\ERA5_daily\dict\\RF_df\\RF_df.df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass
    def rename_columns(self):
        df= T.load_df(data_root+rf'ERA5\ERA5_daily\dict\RF_df_EL_nino\\RF_df_EL_nino.df')
        df = df.rename(columns={'El_nino_average_wet_spell':'El_nino_average_dry_spell',

                                'El_nino_maxmum_wet_spell':'El_nino_maxmum_dry_spell',



                            }

                               )



        return df

class ERA5_hourly:

    # def __init__(self):
    #     self.this_class_arr, self.this_class_tif, self.this_class_png = T.mk_class_dir(
    #         'ERA5_hourly',
    #         this_script_root, mode=2)
    #     # self.product = 'temperature_2m'
    #     self.product = 'total_precipitation'

    def run(self):
        # ee.Initialize()
        # date_range_list = self.gen_date_list(start_date = '2020-01-01',end_date = '2023-01-02')
        # outdir = join(self.this_class_arr,self.product)

        # for date_range in tqdm(date_range_list):
        #     startDate = date_range[0]
        #     endDate = date_range[1]
        #     self.download_images(startDate,endDate,outdir)
        self.unzip()
        # self.reproj()
        # self.statistic()
        pass

    def download_images(self,startDate,endDate,outdir):
        start_year = startDate.split('-')[0]
        # outdir_mean = join(outdir, 'mean',start_year)
        # outdir_max = join(outdir, 'max',start_year)
        # outdir_min = join(outdir, 'min',start_year)
        outdir_sum = join(outdir, 'sum',start_year)

        # T.mk_dir(outdir_mean,force=True)
        # T.mk_dir(outdir_max,force=True)
        # T.mk_dir(outdir_min,force=True)
        T.mk_dir(outdir_sum,force=True)
        # print(startDate)
        # exit()

        # out_path_mean = join(outdir_mean, f'{startDate.replace("-","")}.zip')
        # out_path_max = join(outdir_max, f'{startDate.replace("-","")}.zip')
        # out_path_min = join(outdir_min, f'{startDate.replace("-","")}.zip')
        out_path_sum = join(outdir_sum, f'{startDate.replace("-","")}.zip')
        T.mk_dir(outdir,force=True)
        # startDate = f'{year}-01-01'
        # endDate = f'{year+1}-01-01'
        # startDate = '2011-04-03'
        # endDate = '2011-04-04'
        Collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
        # l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
        Collection = Collection.filterDate(startDate, endDate)
        # Image = Collection.mean()
        # Image_product_mean = Collection.select(self.product).mean()
        # Image_product_max = Collection.select(self.product).max()
        # Image_product_min = Collection.select(self.product).min()
        Image_product_sum = Collection.select(self.product).sum()

        exportOptions = {
            'scale': 27830,
            # 'scale': 11132,
            'maxPixels': 1e13,
            # 'region': region,
            # 'fileNamePrefix': 'exampleExport',
            # 'description': 'imageToAssetExample',
        }
        # url_mean = Image_product_mean.getDownloadURL(exportOptions)
        # url_max = Image_product_max.getDownloadURL(exportOptions)
        # url_min = Image_product_min.getDownloadURL(exportOptions)
        url_sum = Image_product_sum.getDownloadURL(exportOptions)


        try:
            # self.download_i(url_mean, out_path_mean)
            # self.download_i(url_max, out_path_max)
            # self.download_i(url_min, out_path_min)
            self.download_i(url_sum, out_path_sum)
        except:
            # print('download error', out_path_mean, out_path_max, out_path_min)
            print('download error', out_path_sum)


    def gen_date_list(self,start_date = '1982-01-01',end_date = '2023-01-02'):

        days_count = T.count_days_of_two_dates(start_date, end_date)
        date_list = []
        base_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        for i in range(days_count):
            date = base_date + datetime.timedelta(days=i)
            date_list.append(date.strftime('%Y-%m-%d'))
        date_range_list = []
        for i in range(len(date_list) - 1):
            date_range_list.append([date_list[i], date_list[i + 1]])
        return date_range_list


    def download_i(self,url,outf):
        # try:
        http = urllib3.PoolManager()
        r = http.request('GET', url, preload_content=False)
        body = r.read()
        with open(outf, 'wb') as f:
            f.write(body)

    def unzip(self):
        fdir = data_root+rf'\ERA5\ERA5_hourly\arr\\temperature_2m\max\\'
        outdir = data_root+rf'\ERA5\ERA5_hourly\tif\\temperature_2m\max\\'
        T.mk_dir(outdir,force=True)
        for folder in T.listdir(fdir):
            fdir_i = join(fdir,folder)
            # T.open_path_and_file(fdir_i,folder)
            # exit()
            outdir_i = join(outdir,folder)
            T.unzip(fdir_i,outdir_i)
        pass

    def wkt(self):
        wkt = '''
        PROJCS["Sinusoidal",
    GEOGCS["GCS_Undefined",
        DATUM["Undefined",
            SPHEROID["User_Defined_Spheroid",6371007.181,0.0]],
        PRIMEM["Greenwich",0.0],
        UNIT["Degree",0.0174532925199433]],
    PROJECTION["Sinusoidal"],
    PARAMETER["False_Easting",0.0],
    PARAMETER["False_Northing",0.0],
    PARAMETER["Central_Meridian",0.0],
    UNIT["Meter",1.0]]'''
        return wkt

    def reproj(self):
        fdir = join(self.this_class_arr,'unzip')
        outdir = join(self.this_class_arr,'reproj')
        T.mk_dir(outdir)
        for site in T.listdir(fdir):
            fdir_i = join(fdir,site)
            outdir_i = join(outdir,site)
            T.mk_dir(outdir_i)
            for date in T.listdir(fdir_i):
                fdir_i_i = join(fdir_i,date)
                for f in T.listdir(fdir_i_i):
                    fpath = join(fdir_i_i,f)
                    outpath = join(outdir_i,date+'.tif')
                    SRS = DIC_and_TIF().gen_srs_from_wkt(self.wkt())
                    ToRaster().resample_reproj(fpath,outpath,.005,srcSRS=SRS, dstSRS='EPSG:4326')

    def statistic(self):
        fdir = join(self.this_class_arr,'reproj')
        outdir = join(self.this_class_arr,'statistic')
        T.mk_dir(outdir)
        for site in T.listdir(fdir):
            fdir_i = join(fdir,site)
            mean_list = []
            date_list = []
            for f in T.listdir(fdir_i):
                if not f.endswith('.tif'):
                    continue
                date = f.split('.')[0]
                y,m,d = date.split('_')
                y = int(y)
                m = int(m)
                d = int(d)
                date_obj = datetime.datetime(y,m,d)
                fpath = join(fdir_i,f)
                arr = ToRaster().raster2array(fpath)[0]
                arr[arr<=0] = np.nan
                mean = np.nanmean(arr)
                mean_list.append(mean)
                date_list.append(f'{y}-{m:02d}-{d:02d}')
            df = pd.DataFrame({'date':date_list,'NDVI':mean_list})
            outf = join(outdir,site)
            T.df_to_excel(df,outf)

        pass




















def main():
    # ERA5_daily().run()
    # extraction_extreme_event_rainfall().run()
    # extration_extreme_event_temperature().run()
    # extract_rainfall().run()
    # extract_temperature().run()
    build_df().run()
    # ERA5_hourly().run()
    pass

if __name__ == '__main__':
    main()
