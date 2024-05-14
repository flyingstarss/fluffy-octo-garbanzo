import pandas as pd

# 读取CSV文件
df = pd.read_csv('../data/segmented_texts.csv')

# 打印列名
print(df.columns)