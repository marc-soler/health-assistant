# %%
import pandas as pd

# %%
df = pd.read_csv("../data/medquad.csv")

# %%
df = df.dropna().reset_index(drop=True)
df = df.reset_index().rename(columns={"index": "id"})

# %%
df.to_parquet("../data/medquad_clean.parquet", index=False)
