# JD-Ready Two-Stage Recsys (TikTok-style)

- Implicit feedback with time-decay weights
- Two-stage: ALS retrieval + FAISS → LightGBM LambdaRank
- Session features (recent items)
- Offline metrics: Recall@K, NDCG@K
- FastAPI serving; MLflow hooks; Dockerfile

## Quickstart (toy)
pip install -r requirements.txt
python data/generate_synthetic.py
python features/build_features_pandas.py
python retrieval/train_als.py
python retrieval/build_faiss.py
python ranking/make_ltr_dataset.py
python ranking/train_lgbm_ranker.py
python eval/offline_eval.py
uvicorn serve.api:app --reload

## Use MovieLens 25M
python data/convert_movielens.py --limit-users 100000
python features/build_features_pandas.py
python retrieval/train_als.py && python retrieval/build_faiss.py
python ranking/make_ltr_dataset.py && python ranking/train_lgbm_ranker.py
python eval/offline_eval.py
