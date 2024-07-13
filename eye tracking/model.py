from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
# 定义一个简单的全连接神经网络
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), activity_regularizer=l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear'))
    # 编译模型，使用adam优化器，均方误差作为损失函数，同时监控平均绝对误差

    # custom_adam = Adam(learning_rate=0.001)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
# 定义回调函数以保存检查点和记录训练损失
checkpoint_callback = ModelCheckpoint('model_exp.keras', save_best_only=True)
csv_logger = CSVLogger('training_log.csv')

# 训练模型
def train_model(model, x_train, y_train, batch_size, epochs):
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[checkpoint_callback, csv_logger])



