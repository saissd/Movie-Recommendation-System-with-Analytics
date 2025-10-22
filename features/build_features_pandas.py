import pandas as pd
from pathlib import Path

Path("data/features").mkdir(parents=True, exist_ok=True)

items = pd.read_csv("data/items.csv")
items.to_parquet("data/items.parquet", index=False)

if Path("data/interactions.csv").exists():
    inter = pd.read_csv("data/interactions.csv")
    inter.to_parquet("data/interactions.parquet", index=False)

    u = inter.groupby("user_id").agg(
        u_cnt=("item_id", "count"),
        u_avg_dwell=("dwell_s", "mean"),
        u_p90_dwell=("dwell_s", lambda x: x.quantile(0.9)),
        u_last_ts=("ts", "max"),
    ).reset_index()
    u.to_parquet("data/features/user_agg.parquet", index=False)

    it = inter.groupby("item_id").agg(
        i_pop=("user_id", "count"),
        i_avg_dwell=("dwell_s", "mean"),
        i_p90_dwell=("dwell_s", lambda x: x.quantile(0.9)),
    ).reset_index()
    it.to_parquet("data/features/item_agg.parquet", index=False)

print("Feature tables written to data/features/*.parquet")
