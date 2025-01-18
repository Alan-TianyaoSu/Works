import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.interpolate import interp2d
from matplotlib.font_manager import FontProperties
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor


mpl.rcParams['font.family'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False  
import warnings

# 忽略所有 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_modis_filename(filename):
    """解析 MODIS 数据文件名，返回日期和时间信息。
    
    参数:
        filename (str): MODIS 数据文件名，例如 'MYD021KM.A2023152.0505.061.2023152161300.hdf'
    
    返回:
        dict: 包含 'date' 和 'time' 的字典
    """
    # 从文件名中提取年份和一年中的天数
    parts = filename.split('.')
    year_day = parts[1]  # 如 'A2023152'
    year = int(year_day[1:5])
    day_of_year = int(year_day[5:])

    # 从文件名中提取时间
    time_str = parts[2]  # 如 '0505'
    hour = int(time_str[:2])
    minute = int(time_str[2:])

    # 将一年中的天数转换为日期
    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    
    # 构建并返回结果字典
    result = {
        'date': date.strftime('%Y-%m-%d'),
        'time': f'{hour:02}:{minute:02}'
    }
    return result



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

    # 过滤大部分噪声的清晰图像
    RVI_Filtered = (RVI - RVI_min) / (RVI_max - RVI_min)


    '''
    Limit：可以调整的阈值
    '''

    Limit = 0.08

    RVI_Filtered[RVI_Filtered > Limit] = 1
    RVI_Filtered[RVI_Filtered <= Limit] = 0
    RVI_Filtered = RVI_Filtered * -1 +1
    RVI_Filtered = RVI_Filtered[::-1]

    RVI_Gray = (RVI - RVI_min) / (RVI_max - RVI_min)
    RVI_Gray[RVI_Gray > Limit] = 0
    RVI_min = RVI_Gray.min()
    RVI_max = RVI_Gray.max()
    RVI_Gray = RVI_Gray * (RVI_max - RVI_min) + RVI_min
    RVI_Gray = RVI_Gray[::-1]

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

    # 找到最大簇的点数
    max_cluster_size = max(len(cluster) for cluster in clusters)

    # 计算最小点数阈值（最大簇的20%）
    min_cluster_size = 0.2 * max_cluster_size

    # 筛选出至少有最大簇20%点数的簇
    significant_clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]

    # 合并所有重要的簇成一个大簇
    most_dense_cluster = np.concatenate(significant_clusters)

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

    return Coverage_Area,[centroid_longitude,centroid_latitude],Volumn

def predict_missing(df, feature_columns, target):
    known = df.dropna(subset=[target])
    unknown = df[df[target].isna()]
    if not unknown.empty:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(known[feature_columns], known[target])
        predicted_values = model.predict(unknown[feature_columns])
        df.loc[df[target].isna(), target] = predicted_values
    return df

def predict_missing_HD(df, feature_columns, target, known_index):
    if pd.isnull(df.loc[known_index, target]):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        known = df.dropna(subset=[target])
        unknown = df.loc[[known_index]]
        model.fit(known[feature_columns], known[target])
        predicted_value = model.predict(unknown[feature_columns])
        df.loc[known_index, target] = predicted_value[0]  # Ensure it's a scalar value
    return df

def perform_interpolation(df, feature_columns, targets):
    # Segment-wise interpolation
    for target in targets:
        start = None
        for i in range(len(df)):
            if pd.notna(df.loc[i, target]):
                if start is not None and i > start + 1:
                    df.loc[start:i] = predict_missing(df.loc[start:i], feature_columns, target)
                start = i

    # Edge and single-point interpolation
    for target in targets:
        not_null_indices = df.index[df[target].notna()]
        all_indices = set(df.index)
        processed = set(not_null_indices)

        while processed != all_indices:
            next_to_process = set()
            for idx in processed:
                if idx + 1 in all_indices and idx + 1 not in processed:
                    next_to_process.add(idx + 1)
                if idx - 1 in all_indices and idx - 1 not in processed:
                    next_to_process.add(idx - 1)
            for idx in next_to_process:
                df = predict_missing_HD(df, feature_columns, target, idx)
            processed.update(next_to_process)

    return df


def process_all_files(directory):
    results = []
    t = 10
    i = 1
    for filename in os.listdir(directory):
        # if i > t:
        #     break
        # print(i)
        # i+= 1

        if filename.endswith('.hdf'):
            try:
                file_path = os.path.join(directory, filename)
                parsed_data = parse_modis_filename(file_path)
                process_data = Process(file_path)
                # 结合日期时间和处理结果
                results.append([parsed_data['date'], process_data[0], process_data[1][0], process_data[1][1], process_data[2]])
                print(results[-1])
            except Exception as e:
                # parsed_data = parse_modis_filename(file_path)
                # print(parsed_data)
                pass

    # 创建 DataFrame
    df = pd.DataFrame(results, columns=['Timestamp', 'Coverage_Area', 'Longitude', 'Latitude', 'Volume'])
    # 确保Timestamp列是日期时间类型
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # 删除Timestamp列中的重复行，保留第一次出现的行
    # df = df.drop_duplicates(subset='Timestamp', keep='first')
    # 选择数值类型的列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 对数值类型的列进行平均聚合
    df_mean = df.groupby('Timestamp')[numeric_cols].mean().reset_index()

    # 对非数值类型的列选择众数
    mode_function = lambda x: x.mode()[0] if not x.empty else None
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_cols.remove('Timestamp')  # 去除 Timestamp 列，因为已被用作分组

    df_mode = df.groupby('Timestamp')[non_numeric_cols].agg(mode_function).reset_index()

    # 合并平均值和众数结果
    df = pd.merge(df_mean, df_mode, on='Timestamp', how='left')
    # 创建完整的日期范围
    date_range = pd.date_range(start='2023-06-01', end='2023-08-31', freq='D')
    # date_range = pd.date_range(start='2023-06-01', end='2023-06-05', freq='D')

    # 将原始数据与日期范围合并
    df_full = df.set_index('Timestamp').reindex(date_range).rename_axis('Timestamp').reset_index()
    # 对缺失值进行线性插值
    df_full['Days'] = (df_full['Timestamp'] - df_full['Timestamp'].min()).dt.days
    feature_columns = ['Days']         # Define your feature columns
    targets = ['Coverage_Area', 'Longitude', 'Latitude', 'Volume']  # Define your target columns


    df_interpolated = perform_interpolation(df_full, feature_columns, targets)
    # return df_full
    return df_interpolated



# 使用示例
directory = 'Modis'
df = process_all_files(directory)
print(df)
# 保存 DataFrame 到 CSV 文件
df.to_csv('Modis_DataSet.csv', index=False)