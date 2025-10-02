# -*- coding: utf-8 -*-
"""
MetricsCollector (optional ClickHouse backend with safe fallback)

환경변수:
  - METRICS_BACKEND: "clickhouse" | "none"
      기본값: clickhouse-driver가 설치되어 있으면 "clickhouse", 아니면 "none"
  - CLICKHOUSE_HOST / CLICKHOUSE_PORT / CLICKHOUSE_USER / CLICKHOUSE_PASSWORD / CLICKHOUSE_DB

Prometheus / OpenTelemetry는 설치되어 있지 않아도 동작하도록 try-import 처리.
ClickHouse 미설치/미사용 시에도 임포트와 런타임이 안전하도록 Null 폴백 제공.
"""
from __future__ import annotations

import os
from typing import Dict, Any, Optional
from datetime import datetime

# ---------- Optional deps (안전한 임포트) ----------
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    )
except Exception:  # pragma: no cover
    # 매우 경량 더미. Prometheus 없으면 메서드 호출만 no-op 되도록.
    class _N:
        def __getattr__(self, _):  # Counter/Histogram/Gauge 대체
            return lambda *a, **k: _N()
        def labels(self, *a, **k): return self
        def inc(self, *a, **k): pass
        def observe(self, *a, **k): pass
        def set(self, *a, **k): pass
    Counter = Histogram = Gauge = _N  # type: ignore
    class _R: pass
    CollectorRegistry = _R  # type: ignore
    def generate_latest(_): return b""

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover
    # OpenTelemetry 미설치 시 완전 무시
    trace = None
    OTLPSpanExporter = TracerProvider = BatchSpanProcessor = None

try:
    from clickhouse_driver import Client as ClickHouseClient  # optional dep
except Exception:  # ImportError 포함
    ClickHouseClient = None

import psutil

# ---------- Logger (structlog optional) ----------
try:
    import structlog
    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    class _L:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def exception(self, *a, **k): pass
    logger = _L()


# ---------- Null Collector ----------
class _NullMetricsCollector:
    """ClickHouse/Prometheus 등이 없을 때 안전하게 작동하는 노옵 수집기."""
    enabled = False

    # API
    async def track_api_request(self, *_, **__): pass
    # LLM
    async def track_llm_inference(self, *_, **__): pass
    # Business/Conversation
    async def track_conversation(self, *_, **__): pass
    # Export
    def get_prometheus_metrics(self) -> bytes: return b""


# ---------- System Metrics ----------
class SystemMetrics:
    """시스템/ GPU 리소스 메트릭 수집 (pynvml 없어도 안전)"""
    def __init__(self):
        self.has_gpu = self._check_gpu_available()

    def _check_gpu_available(self) -> bool:
        try:
            import pynvml  # noqa
            return True
        except Exception:
            return False

    async def get_gpu_stats(self, gpu_id: int = 0) -> Optional[Dict[str, Any]]:
        if not self.has_gpu:
            return None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            return {
                "gpu_id": gpu_id,
                "memory_used_mb": mem_info.used / 1024 / 1024,
                "memory_total_mb": mem_info.total / 1024 / 1024,
                "utilization": utilization.gpu,
                "memory_utilization": utilization.memory,
                "temperature": temp,
                "power_watts": power,
            }
        except Exception as e:
            logger.error(f"Failed to get GPU stats: {e}")
            return None


# ---------- Metrics Collector ----------
class MetricsCollector:
    """
    통합 메트릭 수집 및 관리.
    - METRICS_BACKEND=none 이거나 clickhouse-driver 미설치 시 ClickHouse 경로를 비활성화.
    - Prometheus/OTel은 있으면 사용, 없으면 자동 무시.
    """

    def __new__(cls, *args, **kwargs):
        # 백엔드 결정
        backend_env = os.getenv("METRICS_BACKEND")
        has_clickhouse = ClickHouseClient is not None
        backend = (backend_env or ("clickhouse" if has_clickhouse else "none")).lower()
        if backend == "none":
            logger.info("Metrics backend disabled (METRICS_BACKEND=none). Using Null collector.")
            return _NullMetricsCollector()
        if backend != "clickhouse" or not has_clickhouse:
            logger.warning("Metrics backend set to clickhouse but clickhouse-driver is missing. Using Null collector.")
            return _NullMetricsCollector()
        # clickhouse 사용
        return super().__new__(cls)

    def __init__(self, clickhouse_config: Optional[Dict[str, Any]] = None):
        clickhouse_config = clickhouse_config or {}
        # ----- Prometheus -----
        self.registry = CollectorRegistry()
        self._init_prometheus_metrics()

        # ----- ClickHouse -----
        # 이 지점은 __new__에서 backend 확인을 통과한 경우만 도달.
        host = clickhouse_config.get("host") or os.getenv("CLICKHOUSE_HOST", "localhost")
        port = int(clickhouse_config.get("port") or os.getenv("CLICKHOUSE_PORT", "9000"))
        user = clickhouse_config.get("user") or os.getenv("CLICKHOUSE_USER", "default")
        password = clickhouse_config.get("password") or os.getenv("CLICKHOUSE_PASSWORD", "")
        database = clickhouse_config.get("database") or os.getenv("CLICKHOUSE_DB", "kainexa_metrics")
        self.clickhouse = ClickHouseClient(host=host, port=port, user=user, password=password, database=database)  # type: ignore

        # ----- OpenTelemetry (optional) -----
        self._setup_opentelemetry()

        # ----- System -----
        self.system_metrics = SystemMetrics()
        self.enabled = True
        logger.info("MetricsCollector initialized with ClickHouse backend",
                    host=host, port=port, database=database)

    # -------- Prometheus ----------
    def _init_prometheus_metrics(self):
        def _counter(*a, **k): return Counter(*a, **k, registry=self.registry)
        def _hist(*a, **k): return Histogram(*a, **k, registry=self.registry)
        def _gauge(*a, **k): return Gauge(*a, **k, registry=self.registry)

        self.api_requests = _counter(
            "kainexa_api_requests_total", "Total API requests", ["method", "endpoint", "status"]
        )
        self.api_latency = _hist(
            "kainexa_api_latency_seconds", "API request latency", ["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        self.llm_tokens_processed = _counter(
            "kainexa_llm_tokens_total", "Total tokens processed", ["model", "operation"]
        )
        self.llm_inference_time = _hist(
            "kainexa_llm_inference_seconds", "LLM inference time", ["model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
        )
        self.gpu_memory_used = _gauge(
            "kainexa_gpu_memory_used_mb", "GPU memory usage in MB", ["gpu_id"]
        )
        self.gpu_utilization = _gauge(
            "kainexa_gpu_utilization_percent", "GPU utilization percentage", ["gpu_id"]
        )
        self.conversations_active = _gauge(
            "kainexa_conversations_active", "Active conversations"
        )
        self.messages_processed = _counter(
            "kainexa_messages_processed_total", "Total messages processed", ["intent", "status"]
        )

    # -------- OpenTelemetry ----------
    def _setup_opentelemetry(self):
        if trace is None:
            return
        try:
            trace.set_tracer_provider(TracerProvider())
            tracer_provider = trace.get_tracer_provider()
            exporter = OTLPSpanExporter(endpoint=os.getenv("OTLP_ENDPOINT", "jaeger:14250"),
                                        insecure=True)
            tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            self.tracer = trace.get_tracer(__name__)
        except Exception as e:  # pragma: no cover
            logger.warning(f"OpenTelemetry init skipped: {e}")

    # -------- Tracking APIs ----------
    async def track_api_request(self, method: str, endpoint: str, status: int, duration: float):
        self.api_requests.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.api_latency.labels(method=method, endpoint=endpoint).observe(duration)
        await self._store_api_metric(method, endpoint, status, duration)

    async def track_llm_inference(self, model: str, tokens: int, duration: float, gpu_id: int = 0):
        self.llm_tokens_processed.labels(model=model, operation="inference").inc(tokens)
        self.llm_inference_time.labels(model=model).observe(duration)
        gpu_stats = await self.system_metrics.get_gpu_stats(gpu_id)
        if gpu_stats:
            self.gpu_memory_used.labels(gpu_id=str(gpu_id)).set(gpu_stats["memory_used_mb"])
            self.gpu_utilization.labels(gpu_id=str(gpu_id)).set(gpu_stats["utilization"])
        await self._store_llm_metric(model, tokens, duration, gpu_stats)

    async def track_conversation(self, session_id: str, message_count: int, intent: str, resolved: bool):
        self.messages_processed.labels(
            intent=intent, status=("resolved" if resolved else "ongoing")
        ).inc(message_count)
        await self._store_conversation_event(session_id, message_count, intent, resolved)

    # -------- ClickHouse Writers (guarded) ----------
    async def _store_api_metric(self, method: str, endpoint: str, status: int, duration: float):
        if not getattr(self, "clickhouse", None):
            return
        query = """
        INSERT INTO api_metrics (timestamp, method, endpoint, status, duration_ms)
        VALUES (%(timestamp)s, %(method)s, %(endpoint)s, %(status)s, %(duration_ms)s)
        """
        try:
            self.clickhouse.execute(query, {
                "timestamp": datetime.now(),
                "method": method,
                "endpoint": endpoint,
                "status": status,
                "duration_ms": int(duration * 1000),
            })
        except Exception as e:
            logger.exception(f"ClickHouse insert failed(api_metrics): {e}")

    async def _store_llm_metric(self, model: str, tokens: int, duration: float, gpu_stats: Optional[Dict[str, Any]]):
        if not getattr(self, "clickhouse", None):
            return
        query = """
        INSERT INTO llm_metrics (timestamp, model, tokens, duration_ms, gpu_memory_mb, gpu_utilization)
        VALUES (%(timestamp)s, %(model)s, %(tokens)s, %(duration_ms)s, %(gpu_memory)s, %(gpu_util)s)
        """
        try:
            self.clickhouse.execute(query, {
                "timestamp": datetime.now(),
                "model": model,
                "tokens": int(tokens),
                "duration_ms": int(duration * 1000),
                "gpu_memory": int(gpu_stats.get("memory_used_mb", 0)) if gpu_stats else 0,
                "gpu_util": int(gpu_stats.get("utilization", 0)) if gpu_stats else 0,
            })
        except Exception as e:
            logger.exception(f"ClickHouse insert failed(llm_metrics): {e}")

    async def _store_conversation_event(self, session_id: str, message_count: int, intent: str, resolved: bool):
        if not getattr(self, "clickhouse", None):
            return
        query = """
        INSERT INTO conversation_events (timestamp, session_id, message_count, intent, resolved)
        VALUES (%(timestamp)s, %(session_id)s, %(message_count)s, %(intent)s, %(resolved)s)
        """
        try:
            self.clickhouse.execute(query, {
                "timestamp": datetime.now(),
                "session_id": session_id,
                "message_count": int(message_count),
                "intent": intent,
                "resolved": bool(resolved),
            })
        except Exception as e:
            logger.exception(f"ClickHouse insert failed(conversation_events): {e}")

    # -------- Export ----------
    def get_prometheus_metrics(self) -> bytes:
        try:
            return generate_latest(self.registry)
        except Exception:  # pragma: no cover
            return b""
