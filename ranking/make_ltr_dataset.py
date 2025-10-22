import pandas as pd
from pathlib import Path

Path("data/ranking").mkdir(parents=True, exist_ok=True)

inter = pd.read_parquet("data/interactions.parquet")
u = pd.read_parquet("data/features/user_agg.parquet")
it = pd.read_parquet("data/features/item_agg.parquet")

df = inter.merge(u, on="user_id", how="left").merge(it, on="item_id", how="left")
df["label"] = df.get("like", 1)

df = df.sort_values("ts")
cut = int(len(df) * 0.8)
train, valid = df.iloc[:cut].copy(), df.iloc[cut:].copy()

train.to_parquet("data/ranking/train.parquet", index=False)
valid.to_parquet("data/ranking/valid.parquet", index=False)

print("Training and validation ranking datasets written.")
print(f"train rows: {len(train):,} | valid rows: {len(valid):,}")
print("feature columns:", [c for c in train.columns if c not in ['user_id','item_id','label','ts']])
print("train positives:", int(train['label'].sum()), "| valid positives:", int(valid['label'].sum()))
