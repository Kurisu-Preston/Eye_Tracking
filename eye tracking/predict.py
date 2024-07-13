import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def predict_and_save(input_csv_path, model_path, output_csv_path):
    # 从指定路径加载模型
    model = load_model(model_path)

    # 读取CSV文件
    input_data = pd.read_csv(input_csv_path)

    # 创建MinMaxScaler实例，将输入特征归一化到[0, 1]区间
    scaler = MinMaxScaler()
    input_data_normalized = scaler.fit_transform(input_data)

    # 使用归一化后的数据进行推理
    predictions = model.predict(input_data_normalized)

    # 将预测结果保存为新的CSV文件
    predictions_df = pd.DataFrame(predictions, columns=['x', 'y'])
    predictions_df.to_csv(output_csv_path, index=False)

    print(f"Predictions saved to {output_csv_path}")

# 调用函数进行推理并保存结果
# 请根据您的实际文件路径替换下面的路径
input_csv_path = '/Users/chris/Desktop/data/finetune.csv'
model_path = 'model_base.keras'
output_csv_path = '/Users/chris/Desktop/data/predict_xy_exp.csv'

predict_and_save(input_csv_path, model_path, output_csv_path)
