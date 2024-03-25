import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

class Data_Preprocess:
    def __init__(self, data_file):
        self.df = data_file

    # 中心对称函数
    def cl(self, x: float):
        if x >= 0:
            output = math.log(x + 1)
        else:
            output = -math.log(1 - x)
        return output

    # 均值填补
    def mean_fillna(self, columns_to_fn):
        df = self.df
        for column in columns_to_fn:
            mean_val = df[column][df[column] != '?'].astype(float).mean()
            df[column] = df[column].replace('?', mean_val).astype(float)
        return self.df
    
    # 中位数填补
    def median_fillna(self, columns_to_fn):
        df = self.df
        for column in columns_to_fn:
            median_val = df[column][df[column] != '?'].astype(float).median()
            df[column] = df[column].replace('?', median_val).astype(float)
        return self.df

    # 零填补
    def zero_fillna(self, columns_to_fn):
        df = self.df
        for column in columns_to_fn:
            df[column] = df[column].replace('?', 0).astype(float)
        return self.df

    # 回归预测填补
    def regression_fillna(self, column):
        df = self.df
        known = df[df[column].notnull()].values
        unknown = df[df[column].isnull()].values
        X_train = known[:, :-1]
        y_train = known[:, -1]
        X_test = unknown[:, :-1]

        # 使用线性回归进行预测
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        # 填充缺失值
        df.loc[df[column].isnull(), column] = y_pred

        return self.df

    # Log transformation: 对数据进行对数变换，将数据映射到更小的范围内。
    def log_transformation(self, column_to_lt):
        self.df[column_to_lt] = np.log1p(self.df[column_to_lt])
        return self.df

    # Z-score normalization: 对数据进行标准化，将数据的均值归0，标准差归1。
    def standardize(self, column_to_standardize):
        mean = self.df[column_to_standardize].mean()
        std = self.df[column_to_standardize].std()
        self.df[column_to_standardize] = (self.df[column_to_standardize] - mean) / std
        return self.df

    # Min-Max normalization: 对数据进行归一化，将数据映射到0~1的区间内。
    def normalize(self, column_to_normalize):
        column = self.df[column_to_normalize]
        self.df[column_to_normalize] = (column - column.mean()) / column.std()
        return self.df

    # Feature scaling: 对数据进行特征缩放，将数据映射到-1~1的区间内。
    def feature_scaling(self, column_to_scaling):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.df[column_to_scaling] = scaler.fit_transform(self.df[column_to_scaling].values.reshape(-1, 1))
        return self.df


if __name__ == "__main__":
    df = pd.read_csv("E:\PycharmProjects\毕设\数据处理\\2015data\dataset.csv")
    # read_csv() 表示从 CSV 文件中读取数据，并创建 DataFrame 对象。

    # 修改原csv格式，添加行索引
    df = df.rename(columns={'Unnamed: 0': 'number'})
    df = df.set_index('number', drop=True)
    # 行索引内容：number, address, function, length, setpoint, gain, reset rate, deadband,
    # cycle time, rate,system mode, control scheme, pump,solenoid, pressure measurement,
    # crc rate, command response, time,binary result, categorized result, specific result

    # 输出数据集
    print(df)

    # 创建 Data_Preprocess 实例
    data_processor = Data_Preprocess(df)

    # 对数据集原始数据空缺值进行填补
    columns_to_fillna = ['setpoint', 'gain', 'reset rate', 'deadband', 'cycle time', 'rate', 'system mode',
                              'control scheme', 'pump', 'solenoid', 'pressure measurement']
    # 均值填补
    #data_processor.mean_fillna(columns_to_fillna)

    # 中位数填补
    data_processor.median_fillna(columns_to_fillna)

    # 零填补
    #data_processor.zero_fillna(columns_to_fillna)

    # 回归预测填补
    #data_processor.regression_fillna(columns_to_fillna)

    # 调用 log_transformation 函数进行对数变换
    columns_to_lt = ['pressure measurement']
    data_processor.log_transformation(columns_to_lt)
    # print(data_processor.df.head(2))

    # 调用 standardization 函数进行标准化
    columns_to_standardization = []
    data_processor.standardize(columns_to_standardization)

    # 调用 normalization 函数进行归一化
    columns_to_normalization = ['setpoint', 'gain', 'reset rate', 'deadband', 'cycle time', 'rate', 'system mode',
                                  'control scheme', 'pump', 'solenoid']
    data_processor.normalize(columns_to_normalization)

    # 调用 feature_scaling 函数进行特征缩放
    # columns_to_scaling = []
    # data_processor.feature_scaling(columns_to_scaling)

    # 保存预处理结果
    csv_data = df.to_csv("E:\PycharmProjects\毕设\数据处理\\2015data\dp_lt.csv")
    print("结果已保存至文件")
