import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.interpolate import interp2d
from matplotlib.font_manager import FontProperties
import pandas as pd
from sklearn.cluster import DBSCAN

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

mpl.rcParams['font.family'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False  


def Process(FILE_NAME):
    DATAFIELD_NAME = 'EV_250_Aggr1km_RefSB'
    
    hdf = SD(FILE_NAME, SDC.READ)
    
    attrs = hdf.attributes(full=1)
    nsa = attrs["Number of Scans"]
    number_of_Scans = nsa[0]
    
    # 提取近红外波段
    data2D = hdf.select(DATAFIELD_NAME)
    data = data2D[1,:,:].astype(np.double)
    

    # 获取辐射校正所需的数据（辐射校正消除仪器本身的误差）
    lat = hdf.select('Latitude')
    latitude = lat[:,:]
    lon = hdf.select('Longitude')
    longitude = lon[:,:]
    
    attrs=lat.attributes(full=1)
    fna = attrs["frame_numbers"]
    frame_numbers = fna[0][0]
    lna = attrs["line_numbers"]
    line_numbers = lna[0][0]
    
    attrs = data2D.attributes(full=1)
    lna=attrs["long_name"]
    long_name = lna[0]
    aoa=attrs["radiance_offsets"]
    add_offset = aoa[0][0]
    sfa=attrs["radiance_scales"]
    scale_factor = sfa[0][0]  
    fva=attrs["_FillValue"]
    _FillValue = fva[0]
    vra=attrs["valid_range"]
    valid_min = vra[0][0]        
    valid_max = vra[0][1]
    ua=attrs["units"]
    units = ua[0]
    # 创建插值函数
    lon_interp = interp2d(range(longitude.shape[1]), range(longitude.shape[0]), longitude, kind='linear')
    lat_interp = interp2d(range(latitude.shape[1]), range(latitude.shape[0]), latitude, kind='linear')

    # 生成新的网格
    new_x = np.linspace(0, longitude.shape[1] - 1, data.shape[1])
    new_y = np.linspace(0, latitude.shape[0] - 1, data.shape[0])

    # 插值经纬度数据
    longitude = lon_interp(new_x, new_y)
    latitude = lat_interp(new_x, new_y)

    # 经纬度转换为十进制度
    def dms_to_dd(degrees, minutes, direction):
        dd = float(degrees) + float(minutes)/60
        if direction == 'S' or direction == 'W':
            dd *= -1
        return dd

    # 转换给定的经纬度范围
    lon_min = dms_to_dd(118, 58, 'E')
    lon_max = dms_to_dd(125, 32, 'E')
    lat_min = dms_to_dd(31, 30, 'N')
    lat_max = dms_to_dd(36, 10, 'N')


    # 创建掩码来筛选经纬度范围内的数据点
    lat_mask = (latitude >= lat_min) & (latitude <= lat_max)
    lon_mask = (longitude >= lon_min) & (longitude <= lon_max)
    combined_mask = lat_mask & lon_mask


    invalid = np.logical_or(data > valid_max,
                            data < valid_min)
    invalid = np.logical_or(invalid, data == _FillValue)
    data[invalid] = np.nan
    data = (data - add_offset) * scale_factor 
    data = np.ma.masked_array(data, np.isnan(data))

    # 获取所有为 True 的元素的索引
    true_indices = np.where(combined_mask)

    # 计算行的最小和最大索引
    row_min = np.min(true_indices[0])
    row_max = np.max(true_indices[0])

    # 计算列的最小和最大索引
    col_min = np.min(true_indices[1])
    col_max = np.max(true_indices[1])

    data = data[row_min:row_max + 1, col_min:col_max + 1]
    latitude = latitude[row_min:row_max + 1, col_min:col_max + 1]
    longitude = longitude[row_min:row_max + 1, col_min:col_max + 1]

    # 获取经纬度中心
    lat_m = np.nanmean(latitude)
    lon_m = np.nanmean(longitude)        

    # 读取红光波段的数据
    red_band = hdf.select('EV_250_Aggr1km_RefSB')  # MODIS红光波段1的数据字段
    red_data = red_band[0,:,:].astype(np.double)
    # 读取近红外波段的数据
    nir_band = hdf.select('EV_250_Aggr1km_RefSB')  # MODIS近红外波段2的数据字段
    nir_data = nir_band[1,:,:].astype(np.double)

    RVI = (nir_data) / (red_data)
    RVI = RVI[row_min:row_max + 1, col_min:col_max + 1]
    RVI_min = RVI.min()
    RVI_max = RVI.max()
    RVI_normalized = (RVI - RVI_min) / (RVI_max - RVI_min)

    '''
    Limit = 0.10
    '''
    Limit = 0.10
    # 过滤大部分噪声的清晰图像
    RVI_Filtered = (RVI - RVI_min) / (RVI_max - RVI_min)
    RVI_Filtered[RVI_Filtered > Limit] = 1
    RVI_Filtered[RVI_Filtered <= Limit] = 0
    RVI_Filtered = RVI_Filtered * -1 +1
    RVI_Filtered = RVI_Filtered[::-1]

    RVI_Gray = (RVI - RVI_min) / (RVI_max - RVI_min)
    RVI_Gray[RVI_Gray > Limit] = 0
    RVI_Gray = RVI_Gray * (RVI_max - RVI_min) + RVI_min
    RVI_Gray = RVI_Gray[::-1]

    # 1代表存在点，0代表没有点

    # 提取值为1的点的坐标
    points = np.argwhere(RVI_Filtered == 1)

    # 创建DBSCAN对象
    dbscan = DBSCAN(eps=3, min_samples=2)

    # 拟合数据
    dbscan.fit(points)

    # 获取聚类标签
    labels = dbscan.labels_

    # 提取核心点索引
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    # 提取簇
    unique_labels = set(labels)
    clusters = [points[labels == label] for label in unique_labels if label != -1]

    # 假设最密集区域是包含最多点的簇
    most_dense_cluster = max(clusters, key=len)

    RVI_dense_cluster_matrix = np.zeros_like(RVI_Filtered)

    # 然后遍历most_dense_cluster中的坐标，将这些位置设为1
    for (x, y) in most_dense_cluster:
        RVI_dense_cluster_matrix[x, y] = 1

    # 获取矩阵的维度
    rows, cols = RVI_dense_cluster_matrix.shape

    # 总样本空间的大小
    total_elements = rows * cols

    # 抽样次数
    num_samples = 100000  # 你可以根据需要调整这个数值

    # 随机抽样
    count_ones = 0
    for _ in range(num_samples):
        # 随机选择行和列
        row = np.random.randint(rows)
        col = np.random.randint(cols)
        
        # 检查该位置是否为1
        if RVI_dense_cluster_matrix[row, col] == 1:
            count_ones += 1

    # 计算图形区域占比
    area_ratio = count_ones / num_samples


    Coverage_Area = area_ratio * 31.456

    # 计算质心
    rows, cols = np.where(RVI_dense_cluster_matrix == 1)
    centroid_x = np.mean(cols)
    centroid_y = np.mean(rows)

    # 计算质心经纬度位置
    latitude_start = 31.5
    longitude_start = 118.9667
    latitude_end = 36.1667
    longitude_end = 125.5333

    # 计算每个单元格代表的经纬度增量
    latitude_increment = (latitude_end - latitude_start) / 654
    longitude_increment = (longitude_end - longitude_start) / 609

    # 之前计算的质心的坐标，这里用假设数据
    centroid_row, centroid_col = (int(round(centroid_y)), int(round(centroid_x)))

    # 计算质心的地理坐标
    centroid_latitude = latitude_start + (centroid_row * latitude_increment)
    centroid_longitude = longitude_start + (centroid_col * longitude_increment)

    Volumn = np.sum(RVI_Gray)

    return Coverage_Area, centroid_longitude, centroid_latitude, Volumn


directory = 'Modis/'

result = []
# 遍历文件夹中的所有文件
for filename in os.listdir(directory):
    # 检查文件扩展名是否为 .hdf
    if filename.endswith(".hdf"):
        # 构造完整的文件路径
        try:
            file_path = os.path.join(directory, filename)
            # 调用 process_file 函数处理文件
            result_1, result_2, result_3, result_4 = Process(file_path)
            result.append([result_1, result_2, result_3, result_4])
            # 打印处理结果
            print(result[-1])
        except Exception as e:
            pass
            
