# %%
import pandas as pd

df = pd.read_csv("./results/MPF.2021.2.8.csv")
for col in df.columns[1:]:
    print(f"{col}: ", df[col].mean().round(6))
    
# %%