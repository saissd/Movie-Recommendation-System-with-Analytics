# movie-recommendation-engine-with-dashboard

End-to-end **Movie Recommendation Engine** with real-time **monitoring dashboard**, built using **FAISS**, **SentenceTransformers**, **LightGBM**, **Prometheus**, **Grafana**, and **Docker** for production-grade MLOps deployment.

## Quickstart
```powershell
docker compose build --no-cache app
docker compose up -d
docker logs -f recsys-app
```

Test:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
$body = @{ user_text = "romantic thriller 1990s"; k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/recommend -Method POST -Body $body -ContentType "application/json"
```

Prometheus: http://localhost:9090  
Grafana:    http://localhost:3000 (admin/admin)
