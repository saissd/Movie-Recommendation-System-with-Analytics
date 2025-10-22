import os
import numpy as np, pandas as pd
import lightgbm as lgb
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "recsys-ranking")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "recsys-ranker")

os.makedirs("data/model", exist_ok=True)

train = pd.read_parquet("data/ranking/train.parquet")
valid = pd.read_parquet("data/ranking/valid.parquet")

FEATURE_EXCLUDE = {"user_id", "item_id", "label", "ts"}
features = [c for c in train.columns if c not in FEATURE_EXCLUDE]

train = train.sort_values(["user_id", "ts"]).reset_index(drop=True)
valid = valid.sort_values(["user_id", "ts"]).reset_index(drop=True)

grp_tr = train.groupby("user_id").size().astype(int).tolist()
grp_va = valid.groupby("user_id").size().astype(int).tolist()

ds_tr = lgb.Dataset(train[features], label=train["label"], group=grp_tr, free_raw_data=False)
ds_va = lgb.Dataset(valid[features], label=valid["label"], group=grp_va, reference=ds_tr, free_raw_data=False)

params = dict(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[10],
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_data_in_leaf=30,
    feature_fraction=0.9,
    verbosity=-1,
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

with mlflow.start_run(run_name="lgbm-ranker"):
    mlflow.log_params(params)
    model = lgb.train(
        params=params,
        train_set=ds_tr,
        valid_sets=[ds_va],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(50)],
    )
    out_path = "data/model/lgbm_ranker.txt"
    model.save_model(out_path)
    mlflow.log_artifact(out_path, artifact_path="model")

    best = model.best_score.get("valid_0", {})
    ndcg10 = best.get("ndcg@10")
    ndcg = best.get("ndcg")
    if ndcg10 is not None:
        mlflow.log_metric("ndcg_valid_at_10", float(ndcg10))
    if ndcg is not None:
        mlflow.log_metric("ndcg_valid", float(ndcg))

    try:
        mlflow.lightgbm.log_model(model, artifact_path="ranker", registered_model_name=MLFLOW_MODEL_NAME)
    except Exception:
        pass

print("LightGBM ranker trained and saved to data/model/lgbm_ranker.txt")
