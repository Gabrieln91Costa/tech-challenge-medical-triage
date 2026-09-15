import time

from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from app.model import predict


app = FastAPI(
    title="Medical Triage NLP API",
    description="API para classificação automática de textos médicos.",
    version="1.0.0",
)


# ============================================================
# PROMETHEUS METRICS
# ============================================================

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisições HTTP",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latência das requisições HTTP em segundos",
    ["method", "endpoint"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    latency_ms: float


# ============================================================
# MIDDLEWARE - MONITORAMENTO
# ============================================================

@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        status_code = response.status_code

        return response

    except Exception:

        status_code = 500

        raise

    finally:

        latency = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(latency)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def prediction(request: PredictionRequest):

    start_time = time.perf_counter()

    result = predict(request.text)

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "prediction": result["label"],
        "class_name": result["class_name"],
        "latency_ms": round(latency_ms, 4),
    }


# ============================================================
# PROMETHEUS
# ============================================================

metrics_app = make_asgi_app()

app.mount(
    "/metrics",
    metrics_app
)