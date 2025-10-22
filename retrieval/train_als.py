import os, numpy as np, pandas as pd
from pathlib import Path
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix

os.makedirs("data/model", exist_ok=True)

inter = pd.read_parquet("data/interactions.parquet")
uid = {u:i for i,u in enumerate(inter["user_id"].unique())}
iid = {m:i for i,m in enumerate(inter["item_id"].unique())}

rows = inter["user_id"].map(uid).values
cols = inter["item_id"].map(iid).values
vals = (inter.get("like", pd.Series([1]*len(inter))).fillna(1)).astype(float).values

mat = coo_matrix((vals, (rows, cols)), shape=(len(uid), len(iid))).tocsr()

model = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=15)
model.fit(mat)

np.save("data/model/als_user_factors.npy", model.user_factors)
np.save("data/model/als_item_factors.npy", model.item_factors)
print("ALS factors saved in data/model/*.npy")
