import os
from time import perf_counter
from collections import deque
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, multiprocess
from .model import ModelIndex, MODEL_VERSION, DEFAULT_REGION

app = Flask(__name__)

REQUESTS_TOTAL = Counter("http_requests_total", "Total HTTP requests", ["path"])
REQUEST_ERRORS_TOTAL = Counter("http_request_errors_total", "Total HTTP errors", ["path", "code"])
REQUEST_LATENCY = Histogram("http_request_latency_seconds", "HTTP request latency (s)", ["path"])

MODEL_LATENCY = Histogram("model_inference_latency_seconds", "Model inference latency (s)", ["model_version", "region"])
IMPRESSIONS = Counter("impressions_total", "Recommendations served (impressions)", ["model_version", "region"])
CLICKS = Counter("clicks_total", "Clicks recorded", ["model_version", "region"])
CTR_RATIO = Gauge("ctr_ratio", "Rolling CTR ratio", ["model_version", "region"])
FEATURE_INGEST = Counter("feature_ingest_rate_total", "Feature ingest rows", ["region"])
FEATURE_LAG = Gauge("feature_lag_seconds", "Feature lag (s)", ["region"])
DATA_DRIFT = Gauge("data_drift_score", "Lightweight drift score (0..1)", ["region"])

_recent_feedback = deque(maxlen=2000)
_recent_user_vecs = deque(maxlen=200)

@app.before_request
def _start_and_count():
    request._t0 = perf_counter()
    REQUESTS_TOTAL.labels(request.path).inc()

@app.after_request
def _observe(resp):
    try:
        REQUEST_LATENCY.labels(request.path).observe(perf_counter() - request._t0)
        if resp.status_code >= 400:
            REQUEST_ERRORS_TOTAL.labels(request.path, str(resp.status_code)).inc()
    except Exception:
        pass
    return resp

model = ModelIndex()
FEATURE_INGEST.labels(DEFAULT_REGION).inc(0)
FEATURE_LAG.labels(DEFAULT_REGION).set(180)

@app.get("/")
def root():
    return jsonify(ok=True, model_version=MODEL_VERSION, region=DEFAULT_REGION)

@app.get("/healthz")
def healthz():
    return jsonify(status="ok")

@app.post("/recommend")
def recommend():
    payload = request.get_json(silent=True) or {}
    user_text = payload.get("user_text", "")
    k = int(payload.get("k", 10))

    t0 = perf_counter()
    recs = model.recommend(user_text, k=k)
    MODEL_LATENCY.labels(MODEL_VERSION, DEFAULT_REGION).observe(perf_counter() - t0)

    IMPRESSIONS.labels(MODEL_VERSION, DEFAULT_REGION).inc()
    _recent_feedback.append({"clicked": 0})
    try:
        _recent_user_vecs.append(model.embed_user(user_text))
        drift = model.drift_score(list(_recent_user_vecs))
        DATA_DRIFT.labels(DEFAULT_REGION).set(drift)
    except Exception:
        pass

    _update_ctr_gauge()
    return jsonify(recommendations=recs)

@app.post("/feedback")
def feedback():
    payload = request.get_json(silent=True) or {}
    clicked = bool(payload.get("clicked", False))
    if clicked:
        CLICKS.labels(MODEL_VERSION, DEFAULT_REGION).inc()
    _recent_feedback.append({"clicked": int(clicked)})
    _update_ctr_gauge()
    return jsonify(ok=True)

@app.get("/metrics")
def metrics():
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest()
    return Response(data, mimetype=CONTENT_TYPE_LATEST)

def _update_ctr_gauge():
    if not _recent_feedback:
        CTR_RATIO.labels(MODEL_VERSION, DEFAULT_REGION).set(0.0)
        return
    clicks = sum(x["clicked"] for x in _recent_feedback)
    ctr = clicks / max(1, len(_recent_feedback))
    CTR_RATIO.labels(MODEL_VERSION, DEFAULT_REGION).set(ctr)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
