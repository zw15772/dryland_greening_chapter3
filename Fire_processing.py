class Data_processing_2:

    def __init__(self):

        pass

    def run(self):

        # self.dryland_mask()
        # self.test_histogram()
        # self.resampleSOC()
        # self.reclassification_koppen()
        # self.aggregation_soil()
        # self.resample()
        # self.scale()
        # self.extract_dryland_tiff()
        # self.aggregate()

        # self.tif_to_dic()
        # self.interpolation()



        pass
    def dryland_mask(self):
        ## here we extract dryland tif
        fpath=rf'D:\Project3\Data\Base_data\aridity_index05.tif\\aridity_index.tif'
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        NDVI_mask_f = rf'D:\Greening\Data\Base_data\\NDVI_mask.tif'
        array_NDVI_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)

        array[array_NDVI_mask < 0.1] = np.nan


        array[array >= 0.65] = np.nan


        outf=rf'D:\Project3\Data\Base_data\aridity_index05.tif\\dryland_mask.tif'
        DIC_and_TIF().arr_to_tif(array, outf)

    def test_histogram(self):

        fpath = rf'E:\Project3\Data\ERA5_daily\dict\moving_window_average_anaysis\\detrended_annual_LAI4g_CV.npy'
        spatial_dic=np.load(fpath, allow_pickle=True, encoding='latin1').item()
        data_list=[]

        for pix in spatial_dic:
            val=spatial_dic[pix]
            data_list.append(val)

        data_list=np.array(data_list)
        ## histogram

        plt.hist(data_list,bins=50)

        plt.show()




    def resampleSOC(self):
        f=rf'E:\Project3\Data\Base_data\\continent.tif'

        outf = rf'E:\Project3\Data\Base_data\\continent_05.tif'

        dataset = gdal.Open(f)



        try:
            gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
        # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
        # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
        except Exception as e:
            pass

    def nc_to_tif(self):

        f=rf'E:\Project3\Data\GIMMS3g_plus_NDVI\\'
        outdir=rf'D:\Project3\Data\Base_data\HWSD\nc\\tif\\'
        Tools().mk_dir(outdir,force=True)

        nc = Dataset(fdir + f, 'r')

        print(nc)
        print(nc.variables.keys())
        # t = nc['time']
        # print(t)
        # start_year = int(t.units.split(' ')[-1].split('-')[0])

        # basetime = datetime.datetime(start_year, 1, 1)  # 告诉起始时间
        lat_list = nc['lat']
        lon_list = nc['lon']
        # lat_list=lat_list[::-1]  #取反
        print(lat_list[:])
        print(lon_list[:])

        origin_x = lon_list[0]  # 要为负数-180
        origin_y = lat_list[0]  # 要为正数90
        pix_width = lon_list[1] - lon_list[0]  # 经度0.5
        pix_height = lat_list[1] - lat_list[0]  # 纬度-0.5
        print(origin_x)
        print(origin_y)
        print(pix_width)
        print(pix_height)
        # SIF_arr_list = nc['SIF']
        SPEI_arr_list = nc['tmin']
        print(SPEI_arr_list.shape)
        print(SPEI_arr_list[0])
        # plt.imshow(SPEI_arr_list[5])
        # # plt.imshow(SPEI_arr_list[::])
        # plt.show()

        year = int(f.split('.')[-3][0:4])
        month = int(f.split('.')[-3][4:6])

        fname = '{}{:02d}.tif'.format(year, month)
        print(fname)
        newRasterfn = outdir + fname
        print(newRasterfn)
        longitude_start = origin_x
        latitude_start = origin_y
        pixelWidth = pix_width
        pixelHeight = pix_height
        # array = val
        # array=SPEI_arr_list[::-1]
        array = SPEI_arr_list

        array = np.array(array)
        # method 2
        array = array.T
        array = array * 0.001  ### GPP need scale factor
        array[array < 0] = np.nan

        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()
        ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, )

    def reclassification_koppen(self):
        f=rf'E:\Project3\Data\Base_data\Koeppen_tif\\Koeppen.tif_unify.tif'
        dic={0: 'Tropical', 1: 'Tropical', 2: 'Tropical', 3: 'Tropical', 4: 'Dry', 5: 'Dry', 6: 'Dry', 7: 'Dry', 8: 'Temperate', 9: 'Temperate', 10: 'Temperate',
         11: 'Temperate', 12: 'Temperate', 13: 'Temperate', 14: 'Temperate', 15: 'Temperate', 16: 'Temperate', 17: 'Continental', 18: 'Continental', 19: 'Continental', 20: 'Continental',
         21: 'Continental', 22: 'Continental', 23: 'Continental', 24: 'Continental', 25: 'Continental', 26: 'Continental', 27: 'Continental', 28: 'Polar', 29: 'Polar'}

        reclassification_dic = {
            'Tropical': 1,
            'Dry': 2,
            'Temperate': 3,
            'Continental': 4,
            'Polar': 5

        }
        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        spatial_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        for pix in spatial_dic:
            val = spatial_dic[pix]
            if val not in dic:
                spatial_dic[pix] = np.nan
                continue
            if val<-99:
                spatial_dic[pix]=np.nan
                continue

            class_i = dic[val]
            spatial_dic[pix] = reclassification_dic[class_i]


        arr_new = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr_new, interpolation='nearest', cmap='jet')
        plt.colorbar()
        plt.title('Koeppen')
        plt.show()
        outf=rf'E:\Project3\Data\Base_data\Koeppen_tif\\Koeppen_reclassification.tif'
        DIC_and_TIF().arr_to_tif(arr_new, outf)

        pass

    def nc_to_tif_time_series_fast(self):

        fdir=rf'E:\Project3\Data\GIMMS3g_plus_NDVI\\'
        outdir=rf'E:\Project3\Data\GIMMS3g_plus_NDVI\\TIFF\\'
        Tools().mk_dir(outdir,force=True)
        for f in tqdm(os.listdir(fdir)):

            outdir_name = f.split('.')[0]
            # print(outdir_name)

            yearlist = list(range(1982, 2021))
            fpath = join(fdir,f)
            nc_in = xarray.open_dataset(fpath)
            # print(nc_in)
            time_bnds = nc_in['time_bnds']
            for t in range(len(time_bnds)):
                date = time_bnds[t]['time']
                date = pd.to_datetime(date.values)
                date_str = date.strftime('%Y%m%d')
                date_str = date_str.split()[0]
                outf = join(outdir, f'{date_str}.tif')
                array = nc_in['ndvi'][t]/10000.
                array = np.array(array)
                array[array < 0] = np.nan
                longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.0833333333333, -0.0833333333333
                ToRaster().array2raster(outf, longitude_start, latitude_start,
                                        pixelWidth, pixelHeight, array, ndv=-999999)
                # exit()


            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir+f, var_name='ndvi', outdir=outdir, yearlist=yearlist)
            except Exception as e:
                print(e)
                continue


    def aggregation_soil(self):
        ## aggregation soil sand, CEC etc

        fdir = data_root+rf'Base_data\SoilGrid\SOIL_Grid_05_unify\\'

        product_list=['soc', 'sand', 'nitrogen', 'cec']

        for product_i in product_list:
            ##cec_15-30cm_mean_5000_05
            f_layer1 = fdir + rf'{product_i}_0-5cm_mean_5000_05.tif'
            f_layer2 = fdir + rf'{product_i}_5-15cm_mean_5000_05.tif'
            f_layer3 = fdir + rf'{product_i}_15-30cm_mean_5000_05.tif'


            array1, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_layer1)
            array2, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_layer2)
            array3, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_layer3)

            array1 = np.array(array1, dtype=float)*1/6
            array2 = np.array(array2, dtype=float)*1/3
            array3 = np.array(array3, dtype=float)*1/2

            aggregation_array =array1+array2+array3
            aggregation_array[aggregation_array < -99] = np.nan
            ToRaster().array2raster(fdir + rf'{product_i}.tif', originX, originY, pixelWidth, pixelHeight, aggregation_array, )



        pass
    def resample(self):
        fdir=rf'E:\Project3\Data\GIMMS3g_plus_NDVI\TIFF\\'
        outdir=rf'E:\Project3\Data\GIMMS3g_plus_NDVI\resample\\'
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath=fdir+f
            outf=outdir+f
            dataset = gdal.Open(fpath)

            try:
                gdal.Warp(outf, dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326')
            # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                pass
    def scale(self):

        fdir = rf'E:\Project3\Data\NDVI4g\resample\\'
        outdir = rf'E:\Project3\Data\NDVI4g\scale\\'
        Tools().mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
            array = np.array(array, dtype=float)

            array = array * 0.0001
            array[array > 1] = np.nan
            # array=array/10000

            array[array ==65535] = np.nan

            outf = outdir + f
            ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, array)

    def extract_dryland_tiff(self):
        self.datadir=rf'E:\Project3\Data\\'
        NDVI_mask_f = join(self.datadir, 'Base_data', 'dryland_mask05.tif')
        array_mask, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(NDVI_mask_f)
        array_mask[array_mask < 0] = np.nan



        fdir_all = rf'E:\Project3\Data\GIMMS3g_plus_NDVI\\'

        for fdir in T.listdir(fdir_all):
            if not 'resample' in fdir:
                continue


            fdir_i = join(fdir_all, fdir)

            outdir_i = join(fdir_all, 'dryland_tiff')

            T.mk_dir(outdir_i)
            for fi in tqdm(T.listdir(fdir_i), desc=fdir):
                if not fi.endswith('.tif'):
                    continue
                fpath = join(fdir_i, fi)
                arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                arr[np.isnan(array_mask)] = np.nan
                # plt.imshow(arr)
                # plt.show()
                fname=fi.split('_')[-1].split('.')[0]
                # print(fname);exit()
                # outpath = join(outdir_i, fi)
                outpath = join(outdir_i, fname+'.tif')

                ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, arr)

        pass
    def aggregate(self):
        ##every four days to biweekly
        fdir=rf'E:\Project3\Data\MCD15A3H\dryland_tiff\\'
        outdir=rf'E:\Project3\Data\MCD15A3H\aggregate\\'
        month_list=['01','02','03','04','05','06','07','08','09','10','11','12']
        yearlist=list(range(2003,2021))
        T.mk_dir(outdir, force=True)

        for year in yearlist:
            for month in month_list:
                data_aggregate_list=[]

                for f in T.listdir(fdir):
                    if not f.endswith('.tif'):
                        continue
                    if  int(f.split('.')[0][0:4])!=year:
                        continue
                    if  (f.split('.')[0][4:6])!=month:
                        continue


                    array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir + f)
                    array = np.array(array, dtype=float)
                    array[array < -99] = np.nan
                    data_aggregate_list.append(array)
                average_array = np.nanmax(data_aggregate_list, axis=0)
                outf=join(outdir,f'{year}{month}.tif')
                ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, average_array)




        pass

    def tif_to_dic(self):

        fdir_all = rf'E:\Project3\Data\GIMMS3g_plus_NDVI\\'

        year_list = list(range(1982, 2021))

        # 作为筛选条件
        for fdir in os.listdir(fdir_all):
            if not 'dryland' in fdir:
                continue
            outdir=join(fdir_all, 'dic')


            T.mk_dir(outdir, force=True)
            all_array = []  #### so important  it should be go with T.mk_dic

            for f in os.listdir(fdir_all+fdir):
                if not f.endswith('.tif'):
                    continue
                if int(f.split('.')[0][0:4]) not in year_list:
                    continue

                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(join(fdir_all, fdir, f))
                array = np.array(array, dtype=float)


                # array_unify = array[:720][:720,
                #               :1440]  # PAR是361*720   ####specify both a row index and a column index as [row_index, column_index]
                array_unify = array[:360][:360,
                              :720]
                array_unify[array_unify < -999] = np.nan
                array_unify[array_unify > 1] = np.nan
                # array[array ==0] = np.nan

                # array_unify[array_unify < 0] = np.nan

                #
                #
                # plt.imshow(array_unify)
                # plt.show()
                # array_mask = np.array(array_mask, dtype=float)
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
                    np.save(outdir + '\\per_pix_dic_%03d' % (flag / 10000), temp_dic)
                    temp_dic = {}
            np.save(outdir + '\\per_pix_dic_%03d' % 0, temp_dic)

    def interpolation(self):

        fdir = rf'E:\Project3\Data\Landsat\dic\\'
        dic = T.load_npy_dir(fdir)

        dict_clean = {}

        for pix in dic:
            r, c = pix
            vals = dic[pix]
            vals = np.array(vals)
            # vals_clip = vals[36:]
            vals_clip = vals
            vals_no_nan = vals_clip[~np.isnan(vals_clip)]
            ratio = len(vals_no_nan) / len(vals_clip)
            if ratio < 0.5:
                continue
            dict_clean[pix] = vals_clip

        mon_mean_dict = {}
        for pix in tqdm(dict_clean,desc='calculating long term mean'):
            vals = dict_clean[pix]
            vals_reshape = vals.reshape(-1, 12)
            vals_reshape_T = vals_reshape.T
            mon_mean_list = []
            for mon in vals_reshape_T:
                mon_mean = np.nanmean(mon)
                mon_mean_list.append(mon_mean)
            mon_mean_dict[pix] = mon_mean_list

        spatial_dic={}

        for pix in dict_clean:
            vals = dict_clean[pix]
            vals_reshape = vals.reshape(-1, 12)
            vals_reshape_T = vals_reshape.T
            # print(len(vals_reshape));exit()
            mon_vals_interpolated_T = []
            for i in range(len(vals_reshape_T)):
                mon_mean = mon_mean_dict[pix][i]
                mon_vals = vals_reshape_T[i]
                mon_vals = np.array(mon_vals)
                mon_vals[np.isnan(mon_vals)] = mon_mean
                mon_vals_interpolated_T.append(mon_vals)
            mon_vals_interpolated_T = np.array(mon_vals_interpolated_T)
            mon_vals_interpolated = mon_vals_interpolated_T.T
            mon_vals_interpolated_flatten = mon_vals_interpolated.flatten()
            vals_origin = dic[pix]
            spatial_dic[pix] = mon_vals_interpolated_flatten
            # plt.imshow(mon_vals_interpolated)
            # plt.figure(figsize=(15, 6))
            # plt.plot(mon_vals_interpolated_flatten)
            # plt.scatter(np.arange(0, len(vals_origin)), vals_origin, c='r')
            # plt.scatter(np.arange(0, len(mon_vals_interpolated_flatten)), mon_vals_interpolated_flatten, c='g',zorder=-1)
            # plt.show()
        np.save(fdir+'interpolated', spatial_dic)
