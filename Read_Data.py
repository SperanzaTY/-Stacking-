import pandas as pd
from scipy.io import arff


class read_file:

    def __init__(self, path):
        self.path = path

    def arff_to_dataframe(self):
        data, _ = arff.loadarff(self.path)
        df = pd.DataFrame(data)
        return df

    def arff_to_csv(self):
        with open(self.path, encoding="utf-8") as f:
            header = []
            for line in f:
                if line.startswith("@attribute"):
                    header.append(line.split('\'')[1])
                elif line.startswith("@data"):
                    break
            df_csv = pd.read_csv(f, header=None)
            df_csv.columns = header
        return df_csv


if __name__ == "__main__":
    file_reader = read_file("2014data/dataset.arff")
    # print(file_reader.read_dataframe())
    # 用csv格式处理数据
    df = file_reader.arff_to_csv()
    print(df)
    # [274628 rows x 20 columns]
    # print(df.dtypes)
    csv_data = df.to_csv("E:/PycharmProjects/毕设/数据预处理/2014data/dataset.csv")
