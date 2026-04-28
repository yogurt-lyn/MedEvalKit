import pandas as pd

# 读取 parquet 文件
df = pd.read_parquet("/media/hdd2/MedEvalKit/utils/VQA_RAD/VQA_RAD/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet")

# 查看前几行
print(df.head())

# 如果想看列名
print(df.columns)
