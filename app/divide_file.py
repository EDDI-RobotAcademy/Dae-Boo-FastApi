import pandas as pd

df = pd.read_csv('add_card_data.csv')

chunk_size = 1000000

for i, chunk in enumerate(range(0,len(df),chunk_size),start = 1):
    chunk_df = df.iloc[chunk:chunk+chunk_size]
    chunk_df.to_csv(f'card_file_{i}.csv',index = False)
