import pandas as pd


csv_file = "E:\PycharmProjects\毕设\数据预处理\\2015data\dp2_2.csv"

# 读取CSV文件
df = pd.read_csv(csv_file)

# 仅保留前100行
df = df.head(100)

# 保存到新的CSV文件
df.to_csv('SMOTE过采样后数据.csv', index=False)
