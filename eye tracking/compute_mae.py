import pandas as pd

# 指定两个文件的绝对路径
path1 = '/Users/chris/Desktop/data/predict_xy_exp.csv'
path2 = '/Users/chris/Desktop/data/true_xy.csv'

# 读取两个CSV文件
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

# 确保两个文件的行数相同
if len(df1) != len(df2):
    raise ValueError("两个文件的行数不相等")

# 计算每行的x值和y值的绝对差异
df1['x_mae'] = (df1['x'] - df2['x']).abs()
df1['y_mae'] = (df1['y'] - df2['y']).abs()

# 可选：如果只需要保存MAE结果，可以选择只保留MAE列
results = df1[['x_mae', 'y_mae']]

# 指定结果文件的保存路径
results_path = '/Users/chris/Desktop/data/results_mae_2.csv'
results.to_csv(results_path, index=False)

print(f"逐行计算完成，结果已保存到 '{results_path}'")
