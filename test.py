import pandas as pd

# 读取数据
df = pd.read_parquet("/mnt/bc_fs/code/med/zengziyang/verl/data/aime-2024.parquet-new")
print(df.columns)
# 提取content字段
df['temp_content'] = df['prompt'].apply(lambda x: x[0]['content'] if x else None)
# 去重
df = df.drop_duplicates(subset=['temp_content'])
df = df.drop(columns=['temp_content'])
print(df.shape)
df.to_parquet("/mnt/bc_fs/code/med/zengziyang/verl/data/aime-2024.parquet-new")