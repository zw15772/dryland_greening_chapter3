# coding=utf-8
import matplotlib.pyplot as plt
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
        for year in range(1982,2020):
            self.download_images(year)
        # self.download_images()
        # self.unzip()
        # self.tiff_to_dict()
        # self.reproj()
        # self.statistic()
        # self.transform_ERA()
        # self.define_quantile_extreme()
        # self.extract_extreme_rainfall_event()
        # self.check()
        # self.wet_spell_dry_spell()

        pass

    def download_images(self,year):
        outdir = data_root+rf'\ERA5\ERA5_daily\Tempmin\\{year}\\'
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
        fdir = rf'C:\Users\wenzhang1\Desktop\ERA5\precipitation\\'
        outdir = data_root+rf'\ERA5\ERA5_daily\tif\unzip_precip\\'
        T.mk_dir(outdir)
        for folder in T.listdir(fdir):



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
        fdir_all=data_root+rf'\ERA5\ERA5_daily\tif\unzip_precip\\'
        outdir = data_root+rf'\ERA5\ERA5_daily\dict\\precip\\'
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



                f=fdir_all + fdir + '\\' + fdir_i+f'\\{fdir_i}.total_precipitation.tif'
                # f=fdir_all + fdir + '\\' + fdir_i+f'\\{fdir_i}.maximum_2m_air_temperature.tif'

                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
                    f)
                array = np.array(array, dtype=float)

                array_unify = array[:720][:720,
                              :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

                # array_unify[array_unify < -999] = np.nan

                # array_unify[array_unify > 7] = np.nan
                # array[array ==0] = np.nan

                # array_unify[array_unify < 0] = np.nan
                array_unify = array_unify * 1000  ## precipitation unit is m so we need to multiply 1000 to get mm

                # array_unify = array_unify - 273.15  ## temperature unit is K so we need to minus 273.15 to get Celsius

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

        fdir_all = rf'C:\Users\wenzhang1\Desktop\ERA5\precip\\'
        outdir = data_root + rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
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
                        result_list.append(dic_all_list[i][pix])
                result_dic[pix] = result_list
            ## save
            np.save(outdir + f'per_pix_dic_%03d' % data, result_dic)
            # print(result_dic)
    def define_quantile_extreme(self):
        # 1) extract extreme wet event based on 90th percentile and calculate frequency and total duration
        # 2) extract extreme dry event based on 10th percentile and calculate frequency and total duration
        # 3) extract wet event intensity
        ## 4) extract dry event intensity
        ## extract VPD and calculate the frequency of VPD>2kpa
        fdir=data_root+rf'\ERA5\ERA5_daily\dict\\precip_transform\\'
        outdir = data_root+rf'\ERA5\ERA5_daily\dict\\precip_transform_extreme\\'
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

    def extract_extreme_rainfall_event(self):
        fdir_threshold = data_root+rf'ERA5\ERA5_daily\dict\precip_transform_extreme\\'
        fdir_yearly_all=rf'C:\Users\wenzhang1\Desktop\ERA5\precip\\'
        outdir = data_root + rf'\ERA5\ERA5_daily\dict\\extreme_event_extraction\\'
        T.mk_dir(outdir, force=True)
        spatial_threshold_dic=T.load_npy_dir(fdir_threshold)
        for fdir_yearly in T.listdir(fdir_yearly_all):

            fdir_yearly_data = fdir_yearly_all+fdir_yearly+'\\'
            spatial_dic=T.load_npy_dir(fdir_yearly_data)

            result_dic = {}
            for pix in tqdm(spatial_dic):
                vals = spatial_dic[pix]
                r,c = pix
                growing_season_data = []
                winter_season_data = []
                non_growing_season_data = []
                if r>120 and r<240:

                    ### extract growing season dat and winter season data

                    ## growing season is defined as 1st May to 31st October
                    ## winter season is defined as 1st January to April 30th but not 1st November to 31st December

                    for i in range(len(vals)):
                        if i>=120 and i<=303:
                            growing_season_data.append(vals[i])
                        elif i>=0 and i<=119:
                            winter_season_data.append(vals[i])
                        else:
                            non_growing_season_data.append(vals[i])
                elif r>=240 and r<=480:
                    ## growing season is defined the whole year
                    growing_season_data = vals
                elif r> 480:
                    ## growing season is defined as 1st November to 31st December and next year 1st January to 30th April
                    ## winter season is defined as 1st May to 31st October
                    for i in range(len(vals)):
                        if i>=0 and i<=119:
                            growing_season_data.append(vals[i])
                        elif i>=120 and i<=303:
                            winter_season_data.append(vals[i])
                        else:
                            non_growing_season_data.append(vals[i])



                growing_season_data = np.array(growing_season_data)
                winter_season_data = np.array(winter_season_data)


                if T.is_all_nan(growing_season_data):
                    continue
                if T.is_all_nan(winter_season_data):
                    continue


                threshold_dic=spatial_threshold_dic[pix]
                # val_90th = threshold_dic['90th']
                # print(val_90th)
                # val_10th = threshold_dic['10th']
                # print(val_10th)
                # exit()
                val_95th = threshold_dic['95th']
                val_5th = threshold_dic['5th']
                val_99th = threshold_dic['99th']
                val_1st = threshold_dic['1st']
                extreme_wet_event_winter = []
                extreme_dry_event_winter = []

                extreme_wet_event_growing = []
                extreme_dry_event_growing = []

                for val in growing_season_data:
                    if val>val_95th:
                        extreme_wet_event_growing.append(val)
                    if val<val_5th:
                        extreme_dry_event_growing.append(val)

                for val in winter_season_data:
                    if val>val_95th:
                        extreme_wet_event_winter.append(val)
                    if val<val_5th:
                        extreme_dry_event_winter.append(val)

             ## calculate the frequency and average intensity of extreme wet event and extreme dry event
                extreme_wet_event_winter = np.array(extreme_wet_event_winter)
                extreme_dry_event_winter = np.array(extreme_dry_event_winter)

                extreme_wet_event_growing = np.array(extreme_wet_event_growing)
                extreme_dry_event_growing = np.array(extreme_dry_event_growing)

                average_intensity_extreme_wet_event_winter = np.nanmean(extreme_wet_event_winter)
                average_intensity_extreme_dry_event_winter = np.nanmean(extreme_dry_event_winter)

                average_intensity_extreme_wet_event_growing = np.nanmean(extreme_wet_event_growing)
                average_intensity_extreme_dry_event_growing = np.nanmean(extreme_dry_event_growing)


                ## frequency
                frequency_extreme_wet_event_winter = len(extreme_wet_event_winter)
                frequency_extreme_dry_event_winter = len(extreme_dry_event_winter)

                frequency_extreme_wet_event_growing = len(extreme_wet_event_growing)
                frequency_extreme_dry_event_growing = len(extreme_dry_event_growing)

                dic_i={

                    'average_intensity_extreme_wet_event_winter':average_intensity_extreme_wet_event_winter,
                    'average_intensity_extreme_dry_event_winter':average_intensity_extreme_dry_event_winter,

                    'average_intensity_extreme_wet_event_growing':average_intensity_extreme_wet_event_growing,
                    'average_intensity_extreme_dry_event_growing':average_intensity_extreme_dry_event_growing,

                    'frequency_extreme_wet_event_winter':frequency_extreme_wet_event_winter,
                    'frequency_extreme_dry_event_winter':frequency_extreme_dry_event_winter,

                    'frequency_extreme_wet_event_growing':frequency_extreme_wet_event_growing,
                    'frequency_extreme_dry_event_growing':frequency_extreme_dry_event_growing,

                }




                result_dic[pix]=dic_i
            outf=outdir+fdir_yearly
            np.save(outf,result_dic)

    def check(self):
        f=rf'D:\Project3\Data\ERA5\ERA5_daily\dict\extreme_event_extraction\\1982.npy'
        spatial_dic = np.load(f,allow_pickle=True).item()
        for pix in spatial_dic:
            vals = spatial_dic[pix]
            average_intensity_extreme_wet_event_winter = vals['average_intensity_extreme_wet_event_winter']
            average_intensity_extreme_dry_event_winter = vals['average_intensity_extreme_dry_event_winter']
            average_intensity_wet_event_winter = vals['average_intensity_wet_event_winter']
            average_intensity_dry_event_winter = vals['average_intensity_dry_event_winter']
            average_intensity_extreme_wet_event_growing = vals['average_intensity_extreme_wet_event_growing']


    def wet_spell_dry_spell(self):
        threshold_fdir = data_root+rf'ERA5\ERA5_daily\dict\precip_transform_extreme\\'
        threshold_spatial_dic = T.load_npy_dir(threshold_fdir)
        fdir_yearly_all=rf'C:\Users\wenzhang1\Desktop\ERA5\precip\\'

        outdir = data_root+rf'\ERA5\ERA5_daily\dict\\dry_spell\\'
        T.mk_dir(outdir,force=True)
        for fdir_yearly in T.listdir(fdir_yearly_all):

            fdir_yearly_data = fdir_yearly_all+fdir_yearly+'\\'
            spatial_dic=T.load_npy_dir(fdir_yearly_data)

            result_dic = {}
            for pix in tqdm(spatial_dic):
                if not pix in threshold_spatial_dic:
                    continue
                vals = spatial_dic[pix]
                vals = np.array(vals, dtype=float)

                vals = np.array(vals)
                # print(vals)


                vals_wet = vals.copy()

                vals_wet[vals_wet >= 1] = np.nan

                wet_index = np.where(~np.isnan(vals_wet))
                wet_index = np.array(wet_index)
                wet_index = wet_index.flatten()
                wet_index_groups = T.group_consecutive_vals(wet_index)
                ## calcuate average wet spell
                wet_spell = []
                for group in wet_index_groups:
                    wet_spell.append(len(group))
                wet_spell = np.array(wet_spell)
                average_wet_spell = np.nanmean(wet_spell)
                maxmum_wet_spell = np.nanmax(wet_spell)
                ## calcuate average wet spell


                dic_i = {

                    'average_wet_spell':average_wet_spell,
                    'maxmum_wet_spell':maxmum_wet_spell
                }
                # for group in wet_index_groups:
                #     print(group)
                # plt.bar(range(len(vals)), vals)
                # plt.bar(range(len(vals)), vals_wet)
                # plt.show()
                result_dic[pix] = dic_i
            outf = outdir + fdir_yearly
            np.save(outf, result_dic)





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












        pass


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
    ERA5_daily().run()
    # ERA5_hourly().run()
    pass

if __name__ == '__main__':
    main()
