# src/monitoring/metrics.py   metrics + collector 통합
import time
import psutil
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import structlog
from clickhouse_driver import Client as ClickHouseClient

logger = structlog.get_logger()

class MetricsCollector:
    """통합 메트릭 수집 및 관리"""
    
    def __init__(self, clickhouse_config: Dict[str, Any]):
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._init_prometheus_metrics()
        
        # ClickHouse client
        self.clickhouse = ClickHouseClient(
            host=clickhouse_config.get('host', 'localhost'),
            port=clickhouse_config.get('port', 9000),
            user=clickhouse_config.get('user', 'kainexa'),
            password=clickhouse_config.get('password', 'kainexa123'),
            database=clickhouse_config.get('database', 'kainexa_metrics')
        )
        
        # OpenTelemetry setup
        self._setup_opentelemetry()
        
        # System metrics
        self.system_metrics = SystemMetrics()
        
    def _init_prometheus_metrics(self):
        """Prometheus 메트릭 정의"""
        
        # API 메트릭
        self.api_requests = Counter(
            'kainexa_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_latency = Histogram(
            'kainexa_api_latency_seconds',
            'API request latency',
            ['method', 'endpoint'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        # LLM 메트릭
        self.llm_tokens_processed = Counter(
            'kainexa_llm_tokens_total',
            'Total tokens processed',
            ['model', 'operation'],
            registry=self.registry
        )
        
        self.llm_inference_time = Histogram(
            'kainexa_llm_inference_seconds',
            'LLM inference time',
            ['model'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self.registry
        )
        
        # GPU 메트릭
        self.gpu_memory_used = Gauge(
            'kainexa_gpu_memory_used_mb',
            'GPU memory usage in MB',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_utilization = Gauge(
            'kainexa_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        # 비즈니스 메트릭
        self.conversations_active = Gauge(
            'kainexa_conversations_active',
            'Active conversations',
            registry=self.registry
        )
        
        self.messages_processed = Counter(
            'kainexa_messages_processed_total',
            'Total messages processed',
            ['intent', 'status'],
            registry=self.registry
        )
        
    def _setup_opentelemetry(self):
        """OpenTelemetry 설정"""
        # Tracer setup
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        # Jaeger exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint="jaeger:14250",
            insecure=True
        )
        
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
        
    async def track_api_request(self, method: str, endpoint: str, 
                                status: int, duration: float):
        """API 요청 메트릭 추적"""
        self.api_requests.labels(method=method, endpoint=endpoint, 
                                status=str(status)).inc()
        self.api_latency.labels(method=method, endpoint=endpoint).observe(duration)
        
        # ClickHouse에 저장
        await self._store_api_metric(method, endpoint, status, duration)
        
    async def track_llm_inference(self, model: str, tokens: int, 
                                  duration: float, gpu_id: int = 0):
        """LLM 추론 메트릭 추적"""
        self.llm_tokens_processed.labels(model=model, 
                                        operation='inference').inc(tokens)
        self.llm_inference_time.labels(model=model).observe(duration)
        
        # GPU 메트릭 업데이트
        gpu_stats = await self.system_metrics.get_gpu_stats(gpu_id)
        if gpu_stats:
            self.gpu_memory_used.labels(gpu_id=str(gpu_id)).set(
                gpu_stats['memory_used_mb']
            )
            self.gpu_utilization.labels(gpu_id=str(gpu_id)).set(
                gpu_stats['utilization']
            )
        
        # ClickHouse에 저장
        await self._store_llm_metric(model, tokens, duration, gpu_stats)
        
    async def track_conversation(self, session_id: str, message_count: int,
                                intent: str, resolved: bool):
        """대화 메트릭 추적"""
        self.messages_processed.labels(
            intent=intent,
            status='resolved' if resolved else 'ongoing'
        ).inc(message_count)
        
        # ClickHouse에 이벤트 저장
        await self._store_conversation_event(session_id, message_count, 
                                            intent, resolved)
        
    async def _store_api_metric(self, method: str, endpoint: str, 
                               status: int, duration: float):
        """API 메트릭을 ClickHouse에 저장"""
        query = """
        INSERT INTO api_metrics (timestamp, method, endpoint, status, duration_ms)
        VALUES (%(timestamp)s, %(method)s, %(endpoint)s, %(status)s, %(duration_ms)s)
        """
        
        self.clickhouse.execute(query, {
            'timestamp': datetime.now(),
            'method': method,
            'endpoint': endpoint,
            'status': status,
            'duration_ms': duration * 1000
        })
        
    async def _store_llm_metric(self, model: str, tokens: int, 
                               duration: float, gpu_stats: Dict):
        """LLM 메트릭을 ClickHouse에 저장"""
        query = """
        INSERT INTO llm_metrics 
        (timestamp, model, tokens, duration_ms, gpu_memory_mb, gpu_utilization)
        VALUES (%(timestamp)s, %(model)s, %(tokens)s, %(duration_ms)s, 
                %(gpu_memory)s, %(gpu_util)s)
        """
        
        self.clickhouse.execute(query, {
            'timestamp': datetime.now(),
            'model': model,
            'tokens': tokens,
            'duration_ms': duration * 1000,
            'gpu_memory': gpu_stats.get('memory_used_mb', 0) if gpu_stats else 0,
            'gpu_util': gpu_stats.get('utilization', 0) if gpu_stats else 0
        })
        
    async def _store_conversation_event(self, session_id: str, 
                                       message_count: int,
                                       intent: str, resolved: bool):
        """대화 이벤트를 ClickHouse에 저장"""
        query = """
        INSERT INTO conversation_events 
        (timestamp, session_id, message_count, intent, resolved)
        VALUES (%(timestamp)s, %(session_id)s, %(message_count)s, 
                %(intent)s, %(resolved)s)
        """
        
        self.clickhouse.execute(query, {
            'timestamp': datetime.now(),
            'session_id': session_id,
            'message_count': message_count,
            'intent': intent,
            'resolved': resolved
        })
        
    def get_prometheus_metrics(self) -> bytes:
        """Prometheus 포맷으로 메트릭 반환"""
        return generate_latest(self.registry)


class SystemMetrics:
    """시스템 리소스 메트릭 수집"""
    
    def __init__(self):
        self.has_gpu = self._check_gpu_available()
        
    def _check_gpu_available(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import pynvml
            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetCount() > 0
        except:
            return False
            
    async def get_gpu_stats(self, gpu_id: int = 0) -> Optional[Dict[str, Any]]:
        """GPU 통계 수집"""
        if not self.has_gpu:
            return None
            
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # 메모리 정보
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 사용률
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # 온도
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            
            # 전력
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
            
            # NVLink 상태 (있는 경우)
            nvlink_stats = self._get_nvlink_stats(handle)
            
            return {
                'gpu_id': gpu_id,
                'memory_used_mb': mem_info.used / 1024 / 1024,
                'memory_total_mb': mem_info.total / 1024 / 1024,
                'utilization': utilization.gpu,
                'memory_utilization': utilization.memory,
                'temperature': temp,
                'power_watts': power,
                'nvlink': nvlink_stats
            }
        except Exception as e:
            logger.error(f"Failed to get GPU stats: {e}")
            return None
            
    def _get_nvlink_stats(self, handle) -> Optional[Dict]:
        """NVLink 통계 수집"""
        try:
            import pynvml
            nvlink_stats = {}
            
            for i in range(pynvml.NVLINK_MAX_LINKS):
                try:
                    # NVLink 상태 확인
                    state = pynvml.nvmlDeviceGetNvLinkState(handle, i)
                    if state == pynvml.NVML_FEATURE_ENABLED:
                        # 처리량 수집
                        tx = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(
                            handle, i, 0, pynvml.NVML_NVLINK_COUNTER_UNIT_BYTES, 
                            pynvml.NVML_NVLINK_COUNTER_TYPE_TX
                        )
                        rx = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(
                            handle, i, 0, pynvml.NVML_NVLINK_COUNTER_UNIT_BYTES,
                            pynvml.NVML_NVLINK_COUNTER_TYPE_RX
                        )
                        nvlink_stats[f'link_{i}'] = {
                            'tx_bytes': tx,
                            'rx_bytes': rx
                        }
                except:
                    continue
                    
            return nvlink_stats if nvlink_stats else None
        except:
            return None
            
    async def get_system_stats(self) -> Dict[str, Any]:
        """시스템 전체 통계"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'used_mb': psutil.virtual_memory().used / 1024 / 1024,
                'total_mb': psutil.virtual_memory().total / 1024 / 1024,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'used_gb': psutil.disk_usage('/').used / 1024 / 1024 / 1024,
                'total_gb': psutil.disk_usage('/').total / 1024 / 1024 / 1024,
                'percent': psutil.disk_usage('/').percent
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            }
        }


class CostTracker:
    """비용 추적 및 관리"""
    
    def __init__(self, pricing_config: Dict[str, float]):
        self.pricing = pricing_config
        self.usage = {
            'tokens': 0,
            'gpu_hours': 0.0,
            'api_calls': 0,
            'storage_gb': 0.0
        }
        
    def track_token_usage(self, model: str, tokens: int):
        """토큰 사용량 추적"""
        self.usage['tokens'] += tokens
        cost = tokens * self.pricing.get(f'{model}_per_token', 0.0001)
        return cost
        
    def track_gpu_usage(self, hours: float, gpu_type: str = 'rtx3090'):
        """GPU 사용 시간 추적"""
        self.usage['gpu_hours'] += hours
        cost = hours * self.pricing.get(f'{gpu_type}_per_hour', 0.5)
        return cost
        
    def get_total_cost(self) -> Dict[str, float]:
        """총 비용 계산"""
        return {
            'tokens': self.usage['tokens'] * self.pricing.get('token_price', 0.0001),
            'gpu': self.usage['gpu_hours'] * self.pricing.get('gpu_hour_price', 0.5),
            'api': self.usage['api_calls'] * self.pricing.get('api_call_price', 0.001),
            'storage': self.usage['storage_gb'] * self.pricing.get('storage_gb_price', 0.1),
            'total': sum([
                self.usage['tokens'] * self.pricing.get('token_price', 0.0001),
                self.usage['gpu_hours'] * self.pricing.get('gpu_hour_price', 0.5),
                self.usage['api_calls'] * self.pricing.get('api_call_price', 0.001),
                self.usage['storage_gb'] * self.pricing.get('storage_gb_price', 0.1)
            ])
        }