import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(csv_path):
    # 加载CSV文件
    data = pd.read_csv(csv_path)

    features = ['DATA0_uH', 'DATA1_uH', 'DATA2_uH', 'DATA3_uH']

    # 选择特征和目标数据
    X = data[features].values
    y = data[['x', 'y']].values

    # 创建MinMaxScaler实例，将特征缩放到[0,1]区间
    scaler = MinMaxScaler()
    # 对全部特征数据进行归一化
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
