import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# # 创建数据集
# data = [
#     ['2023-06-01', None, None, None],
#     ['2023-06-02', None, None, None],
#     ['2023-06-03', 34.05, -118.25, 100.0],
#     ['2023-06-04', None, None, None],
#     ['2023-06-05', None, None, None],
#     ['2023-06-06', 34.07, -118.22, 105.0],
#     ['2023-06-07', None, None, None]
# ]
# columns = ['Timestamp', 'Longitude', 'Latitude', 'Volume']
# df = pd.DataFrame(data, columns=columns)
# df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# # 将日期转换为连续的天数
# df['Days'] = (df['Timestamp'] - df['Timestamp'].min()).dt.days

# # 预测函数
# def predict_missing(df, feature_columns, target):
#     known = df.dropna(subset=[target])
#     unknown = df[df[target].isna()]
#     if not unknown.empty:
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(known[feature_columns], known[target])
#         predicted_values = model.predict(unknown[feature_columns])
#         df.loc[df[target].isna(), target] = predicted_values
#     return df

# # 预测函数
# def predict_missing_HD(df, feature_columns, target, known_index):
#     if pd.isnull(df.loc[known_index, target]):
#         # 使用随机森林预测单个缺失值
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         known = df.dropna(subset=[target])
#         unknown = df.loc[[known_index]]
#         model.fit(known[feature_columns], known[target])
#         predicted_value = model.predict(unknown[feature_columns])
        
#         df.loc[known_index, target] = predicted_value
#     return df


# # 段内插值
# targets = ['Longitude', 'Latitude', 'Volume']
# for target in targets:
#     start = None
#     for i in range(len(df)):
#         if pd.notna(df.loc[i, target]):
#             if start is not None and i > start + 1:
#                 df.loc[start:i] = predict_missing(df.loc[start:i], ['Days'], target)
#             start = i

# # 段内插值已完成，处理边缘和单点插值
# targets = ['Longitude', 'Latitude', 'Volume']
# for target in targets:
#     # 从每个已知点向外扩展
#     not_null_indices = df.index[df[target].notna()]
#     all_indices = set(df.index)
#     processed = set(not_null_indices)
    
#     while processed != all_indices:
#         next_to_process = set()
#         for idx in processed:
#             # 检查前后是否有未处理的空行
#             if idx + 1 in all_indices and idx + 1 not in processed:
#                 next_to_process.add(idx + 1)
#             if idx - 1 in all_indices and idx - 1 not in processed:
#                 next_to_process.add(idx - 1)
#         for idx in next_to_process:
#             df = predict_missing_HD(df, ['Days'], target, idx)
#         processed.update(next_to_process)

# print(df[['Timestamp', 'Longitude', 'Latitude', 'Volume']])



def predict_missing(df, feature_columns, target):
    known = df.dropna(subset=[target])
    unknown = df[df[target].isna()]
    if not unknown.empty:
        model = LinearRegression()
        model.fit(known[feature_columns], known[target])
        predicted_values = model.predict(unknown[feature_columns])
        df.loc[df[target].isna(), target] = predicted_values
    return df

def predict_missing_HD(df, feature_columns, target, known_index):
    if pd.isnull(df.loc[known_index, target]):
        model = LinearRegression()
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

# 创建数据集
data = [
    ['2023-06-01', None, None, None],
    ['2023-06-02', None, None, None],
    ['2023-06-03', 34.05, -118.25, 100.0],
    ['2023-06-04', None, None, None],
    ['2023-06-05', None, None, None],
    ['2023-06-06', 34.07, -118.22, 105.0],
    ['2023-06-07', None, None, None]
]
columns = ['Timestamp', 'Longitude', 'Latitude', 'Volume']
df = pd.DataFrame(data, columns=columns)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 将日期转换为连续的天数
df['Days'] = (df['Timestamp'] - df['Timestamp'].min()).dt.days
feature_columns = ['Days']  
targets = ['Longitude', 'Latitude', 'Volume'] 
df_interpolated = perform_interpolation(df, feature_columns, targets)

print(df_interpolated)