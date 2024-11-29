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
        # self.resample_ERA5()
        # self.extract_dryland_tiff()
        # self.tiff_to_dict()
        # self.reproj()
        # self.statistic()
        # self.transform_ERA()
        # self.deseasonal()
        # self.detrend_deseasonal()
        self.check_dic()
        # self.spatial_average()



        # self.check()


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
            # Image_product = Image.select('minimum_2m_air_temperature')
            Image_product = Image.select('precipitable')
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
        fdir = rf'C:\Users\wenzhang1\Desktop\max_temp_05\raw\\'
        outdir = rf'E:\Project3\ERA5\\Tmax\\unzip\\'
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

    def resample_ERA5(self):
        fdir_all = rf'E:\Project3\Data\ERA5\Precip\\unzip\\'
        for fdir_i in T.listdir(fdir_all):
            print(fdir_i)
            year=fdir_i

            outdir = rf'E:\Project3\Data\ERA5\Precip\\resample\\{year}\\'

            T.mk_dir(outdir, force=True)
            for fdir in T.listdir(join(fdir_all,fdir_i)):

                for f in T.listdir(join(fdir_all,fdir_i,fdir)):
                    fpath=join(fdir_all,fdir_i,fdir,f)
                    print(fpath)

                    dataset = gdal.Open(fpath)
                    outpath = outdir + '{}.tif'.format(fdir)

                    try:
                        gdal.Warp(outpath, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
                    # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
                    # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
                    except Exception as e:
                        pass

    def extract_dryland_tiff(self):
        NDVI_mask_f = rf'D:\Project3\Data\Base_data\aridity_index05.tif\\dryland_mask05.tif'
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan
        outdir = rf'E:\Project3\Data\ERA5\Precip\\extract_dryland_tiff\\'
        T.mk_dir(outdir,force=True)

        fdir_all = rf'E:\Project3\Data\ERA5\Precip\resample\\'
        for fdir in T.listdir(fdir_all):
            fdir_i = join(fdir_all,fdir)
            outdir_i = join(outdir,fdir)
            T.mk_dir(outdir_i)
            for fi in tqdm(T.listdir(fdir_all+fdir),desc=fdir):

                if not fi.endswith('.tif'):
                    continue

                # fpath = join(fdir_i,fi,fi+'.total_precipitation.tif')
                fpath = join(fdir_i, fi )
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr[np.isnan(array_mask)] = np.nan
                arr=arr*1000  ## precipitation unit is m so we need to multiply 1000 to get mm
                # arr=arr-273.15  ##  temperature unit is c so we need to subtract 273.15 to get K
                # plt.imshow(arr)
                # plt.show()
                outpath = join(outdir_i,fi)

                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)



        pass

    def tiff_to_dict(self):
        fdir_all=rf'E:\Project3\Data\ERA5\Precip\\extract_dryland_tiff\\'
        outdir = rf'E:\Project3\Data\ERA5\Precip\\tiff_to_dict\\'
        T.mk_dir(outdir,force=True)


        year_list = list(range(1982, 2021))

        for fdir in T.listdir(fdir_all):
            all_array = []  #### so important  it should be go with T.mk_dic

            outdir_i =join(outdir,fdir)
            # print(outdir)
            # exit()

            if isdir(outdir_i):
                continue
            T.mk_dir(outdir_i, force=True)

            for f in T.listdir(join(fdir_all,fdir)):
                if not f.endswith('.tif'):
                    continue
                fpath=join(fdir_all,fdir,f)
                print(fpath)

                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(
                    fpath)
                array = np.array(array, dtype=float)

                # array_unify = array[:720][:720,
                #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]

                array_unify = array[:360][:360,
                              :720]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
                # array_unify[array_unify < -999] = np.nan

                # array_unify[array_unify > 7] = np.nan
                # array[array ==0] = np.nan

                # array_unify[array_unify < 0] = np.nan
                array_unify = array_unify


                # plt.imshow(array_unify)
                # plt.show()

                # plt.imshow(array_mask)
                # plt.show()
                array_dryland = array_unify
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
                    np.save(join(outdir_i, 'per_pix_dic_%03d' % (flag / 10000)), temp_dic)

                    temp_dic = {}

            np.save(join(outdir_i, 'per_pix_dic_%03d' % 0), temp_dic)

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

        fdir_all = rf'E:\Project3\Data\ERA5\Precip\tiff_to_dict\\'
        outdir = rf'E:\Project3\Data\ERA5\Precip\transform\\'
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



    def deseasonal(self):  ## temperature does not need to detrend
        fdir_all = rf'C:\Users\wenzhang1\Desktop\ERA5_025_processing\Tmax\transform\\'
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5_025_processing\Tmax\deseasonal\\'
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(106):
            data_list.append(i)

        for data in data_list:
            dic_all = np.load(fdir_all + f'per_pix_dic_%03d.npy' % data, allow_pickle=True).item()
            result_dic = {}
            outf = outdir + f'per_pix_dic_%03d' % data + '.npy'
            print(outf)
            # if isfile(outf):
            #     continue
            for pix in tqdm(dic_all.keys()):
                vals = dic_all[pix]
                vals = np.array(vals)
                vals_flatten = vals.flatten()
                #
                if T.is_all_nan(vals_flatten):
                    continue

                anomaly = self.daily_climatology_anomaly(vals_flatten)
                # anomaly_detrend = T.detrend_vals(anomaly)
                # plt.bar(range(len(anomaly)),anomaly,color='red')
                #
                #
                # #
                # plt.bar(range(len(anomaly_detrend)),anomaly_detrend,color= 'b')
                # plt.show()
                anomaly_detrend_reshape = anomaly.reshape(-1, 365)

                result_dic[pix] = anomaly_detrend_reshape
            np.save(outdir + f'per_pix_dic_%03d' % data, result_dic)

        pass

    def detrend_deseasonal(self):
        fdir_all = data_root + rf'CRU-JRA\Tmax\transform\\'
        outdir = data_root+rf'CRU-JRA\Tmax\\\deseasonal_detrend\\'
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
            # if isfile(outf):
            #     continue
            for pix in tqdm(dic_all.keys()):
                vals = dic_all[pix]
                vals=np.array(vals)
                vals_flatten = vals.flatten()
                #
                if T.is_all_nan(vals_flatten):
                    continue

                anomaly=self.daily_climatology_anomaly(vals_flatten)
                anomaly_detrend=T.detrend_vals(anomaly)
                # plt.bar(range(len(anomaly)),anomaly,color='red')
                #
                #
                # #
                # plt.bar(range(len(anomaly_detrend)),anomaly_detrend,color= 'b')
                # plt.show()
                anomaly_detrend_reshape = anomaly_detrend.reshape(-1, 365)

                result_dic[pix] = anomaly_detrend_reshape
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

    def check_dic(self):
        fdir = rf'E:\Project3\Data\ERA5\Precip\transform\\'
        spatial_dic = T.load_npy_dir(fdir)
        results_dic = {}

        for pix in tqdm(spatial_dic.keys()):
            # print(len(spatial_dic[pix]))
            average_list = []

            for vals in spatial_dic[pix]:
                # print(len(vals))

                if T.is_all_nan(vals):
                    continue

                # if np.nansum(vals)==0:
                #     continue

                average = np.nansum(vals)

                # print(average)
                average_list.append(average)
            annual_average = np.nanmean(average_list)
            results_dic[pix] = annual_average

        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(results_dic)
        plt.imshow(arr, interpolation='nearest', cmap='RdYlGn',vmin=200,vmax=1500)
        plt.colorbar()
        plt.show()

    def spatial_average(self):

        fdir = rf'C:\Users\wenzhang1\Desktop\ERA5_025_processing\Precip\extract_dryland_tiff\1987\\'
        tiff_list=[]
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir, f))
            tiff_list.append(arr)
        tiff_list = np.array(tiff_list)
        tiff_sum = np.sum(tiff_list, axis=0)
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5_025_processing\Precip\average\1987\\'
        T.mk_dir(outdir,force=True)
        # tiff_average = tiff_sum / len(tiff_list)
        ToRaster().array2raster(join(outdir, 'average.tif'), originX, originY, pixelWidth, pixelHeight, tiff_sum)



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
        self.define_quantile_threshold()
        # self.extract_extreme_ENSO_year()
        # self.extract_extreme_heat_event()
        # self.extract_extreme_cold_event()
        # self.heat_spell()
        # self.cold_spell()

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
            # np.save(outf,result_dic)
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


        pass

class Build_df:
    def run (self):
        self.build_df()
        # self.add_attribution_rainfall()
        # self.add_attribution_rainfall_frequency()
        # self.add_attribution_dry_spell()

        # self.add_extreme_cold_frequency()
        # self.add_extreme_heat_frequency()
        # self.add_attribution_heat_spell()
        # self.add_attribution_cold_spell()
        # self.add_GPCP()
        # self.add_fire()
        # self.add_rainfall_CV()
        # self.add_maxmium_LC_change()
        # self.add_greening_moisture_map()
        # self.add_GPCP_lagged()
        # self.add_VPD()
        # self.add_CO2()
        # self.add_LAI4g()
        # self.add_GMST()
        # self.add_tmax()
        # self.show_field()







        pass
    def build_df(self):
        fdir =  rf'E:\Data\ERA5_precip\ERA5_daily\dict\\dry_spell\\'
        outdir = rf'E:\Data\\\\ERA5_precip\ERA5_daily\SHAP\\original_df\\'
        T.mk_dir(outdir, force=True)
        flag = 1981
        result_dic = {}
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.npy'):
                continue
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


        # pix_list = T.get_df_unique_val_list(df, 'pix')
        # spatial_dict = {}
        # for pix in pix_list:
        #     spatial_dict[pix] = 1
        # arr = DIC_and_TIF(pixelsize=.25).pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr)
        # plt.show()
        # T.print_head_n(df)
        # exit()
        outf = outdir + 'RF_df'
        # dff = 'D:\Project3\Data\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        # df = T.load_df(dff)
        # print(outf)
        # exit()
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

    def add_attribution_cold_spell(self):

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
    def add_attribution_heat_spell(self):

        df= T.load_df(data_root+rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        fdir_all = rf'E:\Data\ERA5\max_temp\heat_spell\\'

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







    def add_extreme_cold_frequency(self):
        df = T.load_df(data_root + rf'ERA5\ERA5_daily\SHAP\RF_df\\RF_df')

        fdir_all = rf'E:\Data\ERA5\min_temp\extreme_cold_frequency\\'

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

        fdir_all = rf'E:\Data\ERA5\max_temp\extreme_heat_frequency\\'

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



    def add_GPCP_lagged(self): ##
        df = T.load_df(data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'\extract_GS\OBS_LAI_extend\\GPCC.npy'

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
        df['GPCC_precip_pre'] = NDVI_list

        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass

    def add_GPCP(self): ##
        df = T.load_df(rf'E:\\Data\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'Detrend\detrend_anomaly\\\1982_2020\\CRU.npy'

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
            # if len(vals) < 38:
            #     NDVI_list.append(np.nan)
            #     continue

            v1 = vals[year-0]
            # v1=vals[year-1982]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)

        df['CRU_detrended_anomaly'] = NDVI_list

        outf = 'E:\Data\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass

    def add_fire(self): ##
        df = T.load_df(rf'E:\Data\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'\Detrend\event\\fire_burning_area.npy'

        spatial_dic = T.load_npy(f)

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            year = row.year_range

            pix = row['pix']
            r, c = pix

            if not pix in spatial_dic:
                NDVI_list.append(np.nan)
                continue


            vals = spatial_dic[pix]
            vals= np.array(vals)



            if len(vals)==36:
                for i in range(2):
                    vals=np.append(vals,np.nan)
            # print(len(vals))



            v1 = vals[year-0]
            # v1=vals[year-1982]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)



        df['fire_burned_area'] = NDVI_list

        outf = 'E:\Data\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

        pass

    def add_rainfall_CV(self): ## here add the coefficient of variation of rainfall of seasonality
        df = T.load_df(rf'E:\Data\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'\Detrend\event\\CV_rainfall.npy'


        spatial_dic = T.load_npy(f)

        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            year = row.year_range

            pix = row['pix']
            r, c = pix

            if not pix in spatial_dic:
                NDVI_list.append(np.nan)
                continue

            vals = spatial_dic[pix]
            vals = np.array(vals)


            # print(len(vals))

            v1 = vals[year - 0]
            # v1=vals[year-1982]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)
        df['rainfall_CV_intra'] = NDVI_list
        outf = 'E:\Data\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)



    def add_maxmium_LC_change(self): ##
        df = T.load_df(data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = rf'E:\CCI_landcover\trend_analysis_LC\\LC_max.tif'

        array, origin, pixelWidth, pixelHeight, extent = ToRaster().raster2array(f)
        LC_dic =DIC_and_TIF().spatial_arr_to_dic(array)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix

            val= LC_dic[pix]
            df.loc[i,'LC_max'] = val
        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)



    def add_VPD(self): ##
        df = T.load_df(data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'\extract_GS\OBS_LAI_extend\VPD.npy'

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

            v1 = vals[year-0]
            # v1=vals[year-1982]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)
        df['VPD'] = NDVI_list

        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass

    def add_CO2(self): ##
        df = T.load_df(data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'\extract_GS\OBS_LAI_extend\CO2.npy'

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

            v1 = vals[year-0]
            # v1=vals[year-1982]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)
        df['CO2'] = NDVI_list

        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass

    def add_LAI4g(self): ##
        df = T.load_df(data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'\extract_GS\OBS_LAI_extend\LAI4g.npy'

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

            v1 = vals[year-0]
            # v1=vals[year-1982]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)
        df['LAI4g'] = NDVI_list

        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass

    def add_tmax(self): ##
        df = T.load_df(data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'\extract_GS\OBS_LAI_extend\tmax.npy'

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

            v1 = vals[year-0]
            # v1=vals[year-1982]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)
        df['tmax'] = NDVI_list

        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass


    def add_GMST(self): ##
        df = T.load_df(data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        f = results_root + rf'\extract_GS\OBS_LAI_extend\GMST.npy'

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

            v1 = vals[year-0]
            # v1=vals[year-1982]
            # print(v1,year,len(vals))

            NDVI_list.append(v1)
        df['GMST'] = NDVI_list

        outf = data_root + rf'\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
        T.save_df(df, outf)
        T.df_to_excel(df, outf)
        pass
    def add_greening_moisture_map(self):
        df = T.load_df(rf'E:\Data\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        fdir = rf'D:\Project3\Result\Dataframe\anomaly_LAI\\'

        for f in os.listdir(fdir):
            if not 'greening_moisture_map' in f:
                continue

            if not f.endswith('.tif'):
                continue

            variable = (f.split('.')[0])

            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)

            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

            # val_array = np.load(fdir + f)
            # val_dic=T.load_npy(fdir+f)

            # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -99:
                    val_list.append(np.nan)
                    continue
                if val > 99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f'{f_name}'] = val_list
        outf = rf'E:\Data\ERA5\ERA5_daily\SHAP\RF_df\\RF_df'
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
    def show_field(self):
        df= T.load_df(rf'E:\\Data\\ERA5\ERA5_daily\SHAP\RF_df\\RF_df')
        T.print_head_n(df)
        for key in df.keys():
            print(key)
        pass


class detrend_variables():
    pass ## detrend the data
    def run(self):
        self.detrend()
        pass
    def detrend(self):
        fdir = rf'D:\Project3\Result\extract_GS\fire\\'
        outdir = results_root+rf'detrend\\event\\'
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):

            spatial_dic = np.load(fdir+f,allow_pickle=True).item()
            result_dic = {}
            for pix in spatial_dic:

                vals = spatial_dic[pix]
                vals = np.array(vals)
                if T.is_all_nan(vals):
                    continue
                if len(vals)==0:
                    continue
                vals_detrend = T.detrend_vals(vals)
                # plt.plot(vals)
                # plt.show()
                # plt.plot(vals_detrend)
                # plt.show()

                result_dic[pix] = vals_detrend
            outf = outdir+f
            np.save(outf,result_dic)
        pass

    pass ## deseasonal the data

class fire_extraction():  ## extract_fire_burning_area
    def __init__(self):

        pass
    def run(self):
        self.extract_fire_burning_area()
        pass
    def extract_fire_burning_area(self): ## Northern hemisphere fire is from previous year Octorber to current year october
        ## Southern hemisphere fire is from current year April to next year April

        fdir = data_root+rf'monthly_data\\fire\\'
        outdir = results_root+rf'extract_GS\\'
        T.mk_dir(outdir,force=True)
        spatial_dic= T.load_npy_dir(fdir)
        result_dic = {}
        for pix in spatial_dic:
            r,c = pix
            vals = spatial_dic[pix]
            vals_reshape= np.array(vals).reshape(-1,12)
            # print(len(vals_reshape))
            burning_area_list = []


            for i in tqdm(range(len(vals_reshape))):
                if 120<r<=240:
                    if i - 1 < 0:
                        vals_growing_season_1 = []
                    else:
                        vals_growing_season_1 = vals_reshape[i-1][-3:]
                    vals_growing_season_2 = vals_reshape[i][:10]
                    vals_growing_season_1 = list(vals_growing_season_1)
                    vals_growing_season_2 = list(vals_growing_season_2)
                    vals_growing_season = vals_growing_season_1+vals_growing_season_2

                elif 240<r<480:
                    vals_growing_season = vals_reshape[i]

                else:
                    if i>=len(vals_reshape)-1:
                        break
                    # print(len(vals_reshape))
                    vals_growing_season_1 = vals_reshape[i][4:]
                    vals_growing_season_2 = vals_reshape[i+1][:4]
                    vals_growing_season_1 = list(vals_growing_season_1)
                    vals_growing_season_2 = list(vals_growing_season_2)
                    vals_growing_season = vals_growing_season_1+vals_growing_season_2
                vals_growing_season = np.array(vals_growing_season)
                if T.is_all_nan(vals_growing_season):
                    continue
                average_burned_area = np.nanmean(vals_growing_season)
                burning_area_list.append(average_burned_area)
            result_dic[pix] = burning_area_list


        outf = outdir+'fire_burning_area'
        np.save(outf,result_dic)





        pass

    pass
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

class plot_ERA_df:  ## this figure plot all variables
    def __init__(self):
        pass


    def run(self):

        self.plot_all_variables()
        pass
    def clean_df(self,df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df = df[df['row'] > 120]
        df = df[df['Aridity'] < 0.65]
        df = df[df['LC_max'] < 20]
        # df=df[df['LC_max']>0]
        return df
    def plot_all_variables(self):
        df= T.load_df(rf'E:Data\\ERA5\ERA5_daily\\SHAP\RF_df\\RF_df')
        df= self.clean_df(df)
        # dic_label = {'sig_greening_sig_wetting': 1, 'sig_browning_sig_wetting': 2, 'non_sig_greening_sig_wetting': 3,
        #
        #              'non_sig_browning_sig_wetting': 4, 'sig_greening_sig_drying': 5, 'sig_browning_sig_drying': 6,
        #
        #              'non_sig_greening_sig_drying': 7, 'non_sig_browning_sig_drying': 8, np.nan: 0}
        #

        color_list=['green','brown', 'black', 'r']
        continents = ['Global','Africa','Asia','Australia','North_America','South_America']
        continents = [1,5,]
        variable_list= ['VPD_anomaly','CO2_anomaly','tmax_anomaly','GPCC_anomaly','fire_burned_area','CV_rainfall','frequency_wet','average_dry_spell','frequency_heat_event','average_anomaly_heat_event','average_anomaly_cold_event','frequency_cold_event']
        year_list=list(range(0,38))
        for variable in variable_list:
            continent_dic={}
            for continent in continents:

                # if continent == 'Global':
                #     df_i = df
                # else:
                #     df_i = df[df['continent']==continent]
                df_i=df[df['greening_moisture_map']==continent]

                val_list = []
                # T.print_head_n(df_i)
                # exit()

                for year in year_list:
                    year=int(year)
                    df_year = df_i[df_i['year_range']==year]
                    if len(df_year)==0:
                        continue
                    vals= df_year[variable]
                    vals = np.array(vals)
                    if np.isnan(np.nanmean(vals)):
                        continue
                    val_list.append(np.nanmean(vals))

                continent_dic[continent] = val_list

            fig, ax = plt.subplots()
            for continent in continents:
                val_list = continent_dic[continent]
                ax.plot(range(len(val_list)),val_list,label=continent,color=color_list[continents.index(continent)])
            ax.legend()
            ax.set_title(variable)
            plt.show()
























class CRU_JRA:

    def __init__(self):
        self.datadir = '/home/liyang/Desktop/14T/wen/CRU-JRA'
        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.aggregation()

        # self.check_tiff()
        # self.extract_dryland_tiff()
        # self.tiff_to_dict()
        # self.transform()
        # self.detrend_deseasonal()
        # self.extract_dryland_dic()
        # self.plot_spatial_map()

        pass
    def nc_to_tif(self):
        fdir = join(self.datadir,'Tmax','nc')
        outdir = join(self.datadir,'Tmax','tiff')
        for folder in T.listdir(fdir):
            fdir_i = join(fdir,folder)
            year = folder.split('.')[-4]
            date_list = []
            base_time = datetime.datetime(1901,1,1)
            for d in range(0,365):
                date = base_time + datetime.timedelta(days=d)
                for h in [0,6,12,18]:
                    date_i = date + datetime.timedelta(hours=h)
                    date_list.append(date_i)
            # print(len(date_list))

            outdir_i = join(outdir,year)
            T.mk_dir(outdir_i,force=True)
            for f in T.listdir(fdir_i):
                fpath = join(fdir_i,f)
                outpath = join(outdir_i,f)
                ncin = Dataset(fpath, 'r')
                variables = ncin.variables
                # print(variables.keys())
                arrs = ncin.variables['tmax'][:]

                for i,arr in tqdm(enumerate(arrs),desc=year,total=len(arrs)):
                    arr = arr[::-1]-273.15
                    # print(np.shape(arr));exit()
                    date = date_list[i]
                    mon = date.month
                    day = date.day
                    hour = date.hour
                    date_str = f'{year}{mon:02d}{day:02d}{hour:02d}'
                    newRasterfn = join(outdir_i,f'{date_str}.tif')
                    longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.5, -0.5
                    ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
        pass

    def resample(self):
        fdir_all = join(self.datadir,'Tmax','tiff')
        outdir = join(self.datadir,'resample')
        T.mk_dir(outdir,force=True)

        for fdir in T.listdir(fdir_all):
            for f in T.listdir(join(fdir_all,fdir)):
                if not f.endswith('.tif'):
                    continue

                if f.startswith('._'):
                    continue


                print(f)
                # exit()
                date = f.split('.')[0]

                # print(date)
                # exit()
                dataset = gdal.Open(fdir_all + '\\' + fdir + '\\' + f)
                # print(dataset.GetGeoTransform())
                original_x = dataset.GetGeoTransform()[1]
                original_y = dataset.GetGeoTransform()[5]

                # band = dataset.GetRasterBand(1)
                # newRows = dataset.YSize * 2
                # newCols = dataset.XSize * 2
                try:
                    gdal.Warp(outdir + '{}.tif'.format(date), dataset, xRes=0.25, yRes=0.25, dstSRS='EPSG:4326')
                # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
                # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
                except Exception as e:
                    pass

        pass

    def extract_dryland_tiff(self):
        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan
        outdir = join(self.datadir,'aggregate2_dryland')
        T.mk_dir(outdir,force=True)

        fdir_all = join(self.datadir,'aggregate2')
        for fdir in T.listdir(fdir_all):
            fdir_i = join(fdir_all,fdir)
            outdir_i = join(outdir,fdir)
            T.mk_dir(outdir_i)
            for fi in tqdm(T.listdir(fdir_i),desc=fdir):
                if not fi.endswith('.tif'):
                    continue
                fpath = join(fdir_i,fi)
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr[np.isnan(array_mask)] = np.nan
                # plt.imshow(arr)
                # plt.show()
                outpath = join(outdir_i,fi)

                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)



        pass

    def aggregation(self):
        fdir = join(self.datadir,'resample')
        outdir = join(self.datadir,'aggregate2')
        T.mk_dir(outdir)
        for year in T.listdir(fdir):
            print(year,'\n')
            fdir_i = join(fdir,year)
            outdir_i = join(outdir,year)
            T.mk_dir(outdir_i)
            mon_list = range(1,13)
            day_list = range(1,32)
            date_list = []
            for mon in mon_list:
                for day in day_list:
                    date = f'{year}{mon:02d}{day:02d}'
                    date_list.append(date)
            for date in tqdm(date_list):
                selected_list = []
                for fi in T.listdir(fdir_i):
                    if not fi.endswith('.tif'):
                        continue
                    date_f = fi[:8]
                    if date_f == date:
                        fpath = join(fdir_i,fi)
                        selected_list.append(fpath)
                if len(selected_list) == 0:
                    continue
                outpath = join(outdir_i,f'{date}.tif')
                print(outpath)
                # print(date,'\n')
                if os.path.exists(outpath):
                    continue
                selected_arr = []
                for f in selected_list:
                    arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
                    selected_arr.append(arr)
                selected_arr = np.array(selected_arr)
                selected_arr_sum = np.sum(selected_arr, axis=0)

                longitude_start, latitude_start, pixelWidth, pixelHeight = originX, originY, pixelWidth, pixelHeight
                ToRaster().array2raster(outpath, longitude_start, latitude_start, pixelWidth, pixelHeight, selected_arr_sum)

        pass

    def check_tiff(self):
        fdir = join(self.datadir, 'Tmax','deseasonal_detrend')
        for year in T.listdir(fdir):
            fdir_i = join(fdir, year)
            for fi in tqdm(T.listdir(fdir_i),desc=year):
                fpath = join(fdir_i, fi)
                try:
                    arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                except:
                    print(fpath)
                    os.remove(fpath)

        pass

    def tiff_to_dict(self):

        fdir_all = join(self.datadir,'aggregate2_dryland')
        outdir = join(self.datadir,'dict')
        T.mk_dir(outdir, force=True)

        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan

        year_list = list(range(1982, 2021))

        # 作为筛选条件
        for folder in os.listdir(fdir_all):

            outdir_i = join(outdir, folder)
            T.mk_dir(outdir_i, force=True)
            fdir_i = join(fdir_all, folder)

            Pre_Process().data_transform(fdir_i,outdir_i)

    def transform(self):

        fdir_all = join(self.datadir,'dict')
        outdir = join(self.datadir,'precip_transform')
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(106):
            data_list.append(i)

        for data in data_list:
            dic_all_list = []
            for fdir_i in T.listdir(fdir_all):

                for f in T.listdir(join(fdir_all, fdir_i)):
                    if not f.endswith('.npy'):
                        continue
                    if f.split('.')[0].split('_')[-1] != '%03d' % data:
                        continue

                    spatial_dic = np.load(join(fdir_all, fdir_i, f), allow_pickle=True).item()
                    dic_all_list.append(spatial_dic)

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
            outpath = join(outdir, f'per_pix_dic_%03d' % data)
            ## save
            np.save(outpath, result_dic)

    def detrend_deseasonal(self):
        fdir_all = data_root + rf'\CRU-JRA\Tmax\\transform\\'
        outdir = data_root + rf'\CRU-JRA\Tmax\\deseasonal\\'
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(106):
            data_list.append(i)

        for data in data_list:
            dic_all = np.load(fdir_all + f'per_pix_dic_%03d.npy' % data, allow_pickle=True).item()
            result_dic = {}
            outf = outdir + f'per_pix_dic_%03d' % data + '.npy'
            print(outf)
            if isfile(outf):
                continue
            for pix in tqdm(dic_all.keys()):
                vals = dic_all[pix]
                vals = np.array(vals)
                vals_flatten = vals.flatten()
                #
                if T.is_all_nan(vals_flatten):
                    continue

                anomaly = self.daily_climatology_anomaly(vals_flatten)
                anomaly_detrend = T.detrend_vals(anomaly)
                # plt.bar(range(len(anomaly)),anomaly,color='red')
                #
                #
                # #
                # plt.bar(range(len(anomaly_detrend)),anomaly_detrend,color= 'b')
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

    def check_dic(self):
        fdir = join(self.datadir, 'Tmax','deseasonal_detrend')
        spatial_dic = T.load_npy_dir(fdir)
        results_dic = {}
        for pix in tqdm(spatial_dic.keys()):
            vals = spatial_dic[pix]
            if T.is_all_nan(vals):
                continue
            length=len(vals)
            results_dic[pix] = length
        arr=DIC_and_TIF(pixelsize=0.25).pix_dic_to_spatial_arr(results_dic)
        plt.imshow(arr)
        plt.colorbar()
        plt.show()
    def spatial_average(self):

        fdir = rf'C:\Users\wenzhang1\Desktop\ERA5_025_processing\Precip\extract_dryland_tiff\1987\\'
        tiff_list=[]
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir, f))
            tiff_list.append(arr)
        tiff_list = np.array(tiff_list)
        tiff_sum = np.sum(tiff_list, axis=0)
        outdir = rf'C:\Users\wenzhang1\Desktop\ERA5_025_processing\Precip\average\1987\\'
        T.mk_dir(outdir,force=True)
        # tiff_average = tiff_sum / len(tiff_list)
        ToRaster().array2raster(join(outdir, 'average.tif'), originX, originY, pixelWidth, pixelHeight, tiff_sum)





class CPC():
    def __init__(self):
        self.datadir = '/mnt/14T/wen/CPC'

    def run(self):

        # self.nc_to_tif()
        # self.extract_dryland_tiff()
        # self.tiff_to_dict()
        self.transform()

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tiff')


        date_list = []
        base_time = datetime.datetime(1901,1,1)
        for d in range(0,366):
            date = base_time + datetime.timedelta(days=d)

            date_list.append(date)
        # print(len(date_list))


        for f in T.listdir(fdir):
            year = f.split('.')[1]
            outdir_i = join(outdir, year)
            T.mk_dir(outdir_i, force=True)
            fpath = join(fdir,f)
            outpath = join(outdir_i,f)
            ncin = Dataset(fpath, 'r')
            variables = ncin.variables
            # print(variables.keys())
            arrs = ncin.variables['precip'][:]

            for i,arr in tqdm(enumerate(arrs),desc=f,total=len(arrs)):
                # arr = arr[::-1]-273.15

                ## left and right flip 360
                arr_T=arr.T
                arr_T_1 = arr_T[:360]
                arr_T_2 = arr_T[360:]
                arr_T_new=np.concatenate((arr_T_2,arr_T_1),axis=0)

                arr_new = arr_T_new.T

                ## arr
                # print(np.shape(arr));exit()
                date = date_list[i]
                mon = date.month
                day = date.day
                hour = date.hour
                date_str = f'{year}{mon:02d}{day:02d}'
                newRasterfn = join(outdir_i,f'{date_str}.tif')
                longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.5, -0.5
                ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr_new)

    def extract_dryland_tiff(self):
        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask05.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan
        outdir = join(self.datadir,'tiff_dryland')
        T.mk_dir(outdir,force=True)

        fdir_all = join(self.datadir,'tiff')
        for fdir in T.listdir(fdir_all):
            fdir_i = join(fdir_all,fdir)
            outdir_i = join(outdir,fdir)
            T.mk_dir(outdir_i)
            for fi in tqdm(T.listdir(fdir_i),desc=fdir):
                if not fi.endswith('.tif'):
                    continue
                fpath = join(fdir_i,fi)
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr[np.isnan(array_mask)] = np.nan
                # plt.imshow(arr)
                # plt.show()
                outpath = join(outdir_i,fi)

                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)



        pass

    def tiff_to_dict(self):

        fdir_all = join(self.datadir,'tiff_dryland')
        outdir = join(self.datadir,'dict')
        T.mk_dir(outdir, force=True)



        year_list = list(range(1982, 2021))

        # 作为筛选条件
        for folder in os.listdir(fdir_all):

            outdir_i = join(outdir, folder)
            T.mk_dir(outdir_i, force=True)
            fdir_i = join(fdir_all, folder)

            Pre_Process().data_transform(fdir_i,outdir_i)

    def transform(self):

        fdir_all = join(self.datadir,'dict')
        outdir = join(self.datadir,'transform')
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(106):
            data_list.append(i)

        for data in data_list:
            dic_all_list = []
            for fdir_i in T.listdir(fdir_all):

                for f in T.listdir(join(fdir_all, fdir_i)):
                    if not f.endswith('.npy'):
                        continue
                    if f.split('.')[0].split('_')[-1] != '%03d' % data:
                        continue

                    spatial_dic = np.load(join(fdir_all, fdir_i, f), allow_pickle=True).item()
                    dic_all_list.append(spatial_dic)

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
            outpath = join(outdir, f'per_pix_dic_%03d' % data)
            ## save
            np.save(outpath, result_dic)


class MSWEP():
    def __init__(self):
        self.datadir = '/mnt/14T/wen/MSWEP'

        pass

    def run(self):

        # self.nc_to_tif()
        # self.resample()
        # self.extract_dryland_tiff()
        # self.tiff_to_dict()
        self.transform()

        pass

    def nc_to_tif(self):
        fdir_all = join(self.datadir,'nc')



        for fdir in T.listdir(fdir_all):
            outdir = join(self.datadir, 'tiff', fdir)

            T.mk_dir(outdir, force=True)
            for f in T.listdir(join(fdir_all,fdir)):




                fpath = join(fdir_all,fdir,f)
                outpath = join(outdir,f.replace('.nc','.tif'))
                ncin = Dataset(fpath, 'r')
                variables = ncin.variables
                # print(variables.keys())
                arrs = ncin.variables['precipitation'][:]

                for i,arr in tqdm(enumerate(arrs),desc=f,total=len(arrs)):
                    arr = arr


                    newRasterfn = outpath
                    longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.1, -0.1
                    ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)

    def resample(self):
        fdir_all = join(self.datadir, 'tiff')
        outdir = join(self.datadir, 'resample')
        T.mk_dir(outdir, force=True)

        for fdir in T.listdir(fdir_all):
            outdir_i = join(outdir, fdir)
            T.mk_dir(outdir_i, force=True)
            for f in T.listdir(join(fdir_all, fdir)):
                if not f.endswith('.tif'):
                    continue

                if f.startswith('._'):
                    continue

                print(f)
                # exit()
                date = f.split('.')[0]

                # print(date)
                # exit()
                dataset = gdal.Open(join(fdir_all, fdir, f))
                # print(dataset.GetGeoTransform())
                original_x = dataset.GetGeoTransform()[1]
                original_y = dataset.GetGeoTransform()[5]

                # band = dataset.GetRasterBand(1)
                # newRows = dataset.YSize * 2
                # newCols = dataset.XSize * 2
                try:
                    gdal.Warp(join(outdir_i, f), dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
                # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
                # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
                except Exception as e:
                    pass

        pass

    def extract_dryland_tiff(self):
        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask05.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan
        outdir = join(self.datadir, 'dryland_tiff')
        T.mk_dir(outdir,force=True)

        fdir_all = join(self.datadir,'resample')
        for fdir in T.listdir(fdir_all):
            fdir_i = join(fdir_all,fdir)
            outdir_i = join(outdir,fdir)
            T.mk_dir(outdir_i)
            for fi in tqdm(T.listdir(fdir_i),desc=fdir):
                if not fi.endswith('.tif'):
                    continue
                fpath = join(fdir_i,fi)
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr[np.isnan(array_mask)] = np.nan
                # plt.imshow(arr)
                # plt.show()
                outpath = join(outdir_i,fi)

                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)



        pass

    def tiff_to_dict(self):

        fdir_all = join(self.datadir,'dryland_tiff')
        outdir = join(self.datadir,'dict')
        T.mk_dir(outdir, force=True)

        year_list = list(range(1982, 2021))

        # 作为筛选条件
        for folder in os.listdir(fdir_all):

            outdir_i = join(outdir, folder)
            T.mk_dir(outdir_i, force=True)
            fdir_i = join(fdir_all, folder)

            Pre_Process().data_transform(fdir_i,outdir_i)

    def transform(self):

        fdir_all = join(self.datadir,'dict')
        outdir = join(self.datadir,'transform')
        T.mk_dir(outdir, force=True)
        # create_list from 000 t0 105
        data_list = []
        for i in range(106):
            data_list.append(i)

        for data in data_list:
            dic_all_list = []
            for fdir_i in T.listdir(fdir_all):

                for f in T.listdir(join(fdir_all, fdir_i)):
                    if not f.endswith('.npy'):
                        continue
                    if f.split('.')[0].split('_')[-1] != '%03d' % data:
                        continue

                    spatial_dic = np.load(join(fdir_all, fdir_i, f), allow_pickle=True).item()
                    dic_all_list.append(spatial_dic)

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
            outpath = join(outdir, f'per_pix_dic_%03d' % data)
            ## save
            np.save(outpath, result_dic)


def main():
    ERA5_daily().run()

    # extract_temperature().run()
    # extration_extreme_event_temperature_ENSO().run()
    # fire_extraction().run()
    # detrend_variables().run()
    # Build_df().run()

    # plot_ERA_df().run()
    # CRU_JRA().run()

    # ERA5_hourly().run()
    pass

if __name__ == '__main__':
    main()
