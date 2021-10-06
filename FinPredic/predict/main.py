"""
main.py
只用日线指数进行股票预测
@author 1851995刘佳航
"""
import tushare as ts
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras import regularizers
import tensorflow.keras
import time
import matplotlib.pyplot as pyplot


def dataget(start_date, end_date, code):
    """
    数据获取
    :param start_date: string
    :param end_date: string
    :param code: string
    :return: ndarray
    """
    ts.set_token("64acacc4d6dc7fa5cf368ac432f069ba3d041ed0aff877733443585f")
    pro = ts.pro_api()
    df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date,
                         fields='trade_date,close,open,high,low,vol,amount,pct_chg')
    length = df.shape[0]
    data = []
    for i in range(length):
        each_data = [df.loc[length - 1 - i]['open'], df.loc[length - 1 - i]['close'], df.loc[length - 1 - i]['low'],
                     df.loc[length - 1 - i]['high'], df.loc[length - 1 - i]['vol'], df.loc[length - 1 - i]['amount'],
                     df.loc[length - 1 - i]['pct_chg']]
        each_data = np.array(each_data, dtype='float32')
        data.append(each_data)
    data = np.array(data, dtype='float32')
    print('数据类型：', type(data))
    print('数据个数：', data.shape[0])
    print('数据形状：', data.shape)
    print('数据第一行：', data[0])
    return data


def valuestore(data):
    """
    数据集参数存储
    :param data: ndarray
    :return: list
    """
    value = []
    value.append(np.max(data[:, 1]))
    value.append(np.mean(data[:, 1]))
    value.append(np.min(data[:, 1]))
    return value


def slicewindow(data):
    """
    滑窗读取数据，得到形状为（n，30，7）的数组
    :param data: ndarray
    :return: ndarray
    """
    step = 30
    x, y = [], []
    length = data.shape[0]
    for i in range(length - 30):
        end = i + step
        one_x = data[i:end, :]
        one_y = data[end, 1]
        x.append(one_x)
        y.append(one_y)
    return np.array(x, dtype='float32'), np.array(y, dtype='float32')


def datesplit(data):
    """
    数据分割
    :param data: ndarry
    :return: ndarray
    """
    radio = 0.8     # 分割比
    data, label = slicewindow(data)
    train_size = int(data.shape[0] * radio)
    train_data = data[0:train_size, :]
    train_label = label[0:train_size]
    test_data = data[train_size:, :]
    test_label = label[train_size:]
    print("train_data:", train_data.shape)
    print("train_label:", train_label.shape)
    print("test_data:", test_data.shape)
    print("test_label:", test_label.shape)
    return train_data, train_label, test_data, test_label


def normalization(data):
    """
    归一化
    :param data: ndarray
    :return: ndarray
    """
    avg = np.mean(data, axis=0)  # axis=0表示按数组元素的列对numpy取相关操作值
    max_ = np.max(data, axis=0)
    min_ = np.min(data, axis=0)
    print("max", type(max_))
    print(max_.shape)
    result_data = (data - avg) / (max_ - min_)
    return result_data


def datanormalization(train_data, train_label, test_data, test_label):
    """
    归一化
    :param train_data: ndarray
    :param train_label: ndarray
    :param test_data: ndarray
    :param test_label: ndarray
    :return: ndarray
    """
    train_data = normalization(train_data)
    train_label = normalization(train_label)
    test_data = normalization(test_data)
    test_label = normalization(test_label)
    return train_data, train_label, test_data, test_label


def inputnormalization(data, _max, _min, avg):
    """
    输入归一化，如果输入不是原数据集中的数据，须进行相同归一化
    :param data: ndarrar
    :param _max: ndarray
    :param _min: ndarray
    :param avg: ndarray
    :return: ndarray
    """
    return (data - avg) / (_max - _min)


def renormalization(data, _max, _min, avg):
    """
    反归一化，只需对label进行反归一化
    :param data: ndarray
    :param _max: float32
    :param _min: float32
    :param avg: float32
    :return: ndarray
    """
    return data * (_max - _min) + avg


def lstm_model(layers):
    """
    lstm模型建模
    输入参数layers为层数，(layers[0], layers[1])为输入形状，layers[2]为输出形状
    :param layers: list
    :return:
    """
    model = Sequential()
    # LSTM层
    model.add(LSTM(256, activation='relu', return_sequences=False, input_shape=(layers[0], layers[1]),
                   kernel_regularizer=regularizers.l2(0.000003)))
    model.add(Dropout(0.3))
    # BP层
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # 输出层
    model.add(Dense(layers[2]))
    model.add(Activation("linear"))
    # 优化器
    op = tensorflow.keras.optimizers.Adam()
    # 损失函数
    lo = tensorflow.keras.losses.MeanSquaredLogarithmicError()
    model.compile(optimizer=op, loss=lo)
    return model


if __name__ == '__main__':
    # 获取数据，分别输入开始，结束日期与股票代码
    data = dataget("20190309", "20210308", "000001.SH")
    np.savetxt("data.txt", data)
    # 从文本文件中读取数据
    data = np.loadtxt("data.txt")
    # 存储数据集参数，最大值，平均值与最小值，用于归一化
    data_value = valuestore(data)
    # 先对数据归一化
    data = normalization(data)
    # 数据分割
    train_data, train_label, test_data, test_label = datesplit(data)
    # 构建模型
    model = lstm_model([30, train_data.shape[2], 1])
    # 模型显示
    model.summary()
    start = time.time()
    h = model.fit(train_data, train_label, epochs=100, batch_size=100,
                  validation_data=(test_data, test_label), verbose=1, shuffle=True)
    # 画出损失函数
    print("> Compilation Time : ", time.time() - start)
    pyplot.plot(h.history['loss'], label='train')
    pyplot.plot(h.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    # 保存读取模型
    model.save('my_model.h5')
    # 读取模型
    # model = load_model('my_model.h5')
    # 输入归一化（如果需要）
    # input = inputnormalization(input, data_value[0], data_value[2], data_value[1])
    # 画出预测结果
    # 需要对预测结果进行格式转换（矩阵转置）
    predict = model.predict(test_data)
    predict = predict.transpose()[0]
    # 反归一化
    predict = renormalization(predict, data_value[0], data_value[2], data_value[1])
    test_label = renormalization(test_label, data_value[0], data_value[2], data_value[1])
    # print(test_label[0])
    pyplot.plot(predict, label='predict')
    pyplot.plot(test_label, label='real')
    pyplot.legend()
    pyplot.show()
    np.savetxt("predict.txt", predict)
    np.savetxt("test.txt", test_label)
    err = predict - test_label
    for i in range(err.shape[0]):
        err[i] = err[i] ** 2
    cost = np.mean(err)
    print("均方误差为：", np.sqrt(cost))

