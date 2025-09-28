v2 전환 작업 계획
Phase 1: 인프라 업그레이드 (Week 1-2)
yaml# docker-compose.yml 수정 필요
services:
  # 추가 서비스
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
      
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports:
      - "8123:8123"
      - "9000:9000"
Phase 2: DSL 오케스트레이션 구현 (Week 3-4)
python# src/orchestration/dsl_parser.py
class DSLOrchestrator:
    def __init__(self):
        self.graph_executor = GraphExecutor()
        self.policy_engine = PolicyEngine()
    
    def parse_yaml(self, yaml_config):
        """YAML DSL을 실행 그래프로 변환"""
        pass
    
    def execute(self, graph, context):
        """정책 기반 그래프 실행"""
        pass
Phase 3: GPU 병렬 처리 설정 (Week 5-6)
python# src/models/tensor_parallel.py
import torch
import deepspeed
from transformers import AutoModelForCausalLM

class TensorParallelLLM:
    def __init__(self):
        self.model = self._setup_tensor_parallel()
    
    def _setup_tensor_parallel(self):
        """RTX 3090 x2 텐서 병렬 설정"""
        config = {
            "tensor_parallel": {"tp_size": 2},
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 2}
        }
        # DeepSpeed 초기화
        pass
Phase 4: MCP 권한 시스템 (Week 7-8)
python# src/auth/mcp_permissions.py
class MCPPermissionSystem:
    ROLES = {
        "user": ["conversation:read", "conversation:write"],
        "agent": ["retrieve:knowledge", "invoke:action"],
        "admin": ["*:*"]
    }
    
    def check_permission(self, role, action):
        """권한 검증"""
        pass
📁 새로운 디렉토리 구조
kainexa-core/
├── src/
│   ├── orchestration/     # 신규: DSL 오케스트레이션
│   │   ├── dsl_parser.py
│   │   ├── graph_executor.py
│   │   └── policy_engine.py
│   ├── models/            # 수정: GPU 병렬 처리
│   │   ├── tensor_parallel.py
│   │   └── quantization.py
│   ├── monitoring/        # 신규: 관측성
│   │   ├── metrics_collector.py
│   │   └── cost_tracker.py
│   ├── auth/             # 신규: MCP 권한
│   │   └── mcp_permissions.py
│   └── governance/       # 신규: RAG 거버넌스
│       └── document_pipeline.py
├── monitoring/           # 신규: 모니터링 설정
│   ├── prometheus.yml
│   └── grafana/
├── helm/                # 신규: Kubernetes 차트
│   └── kainexa-core/
└── deployment/         # 신규: 배포 스크립트
    ├── ubuntu-setup.sh
    └── gpu-config.sh
🚀 즉시 필요한 작업

requirements.txt 업데이트:

python# 추가 필요
deepspeed==0.12.0
flash-attn==2.3.0
prometheus-client==0.19.0
opentelemetry-api==1.20.0
clickhouse-driver==0.2.6
pyyaml==6.0.1

GPU 설정 스크립트 생성:

bash# deployment/gpu-config.sh
#!/bin/bash
# NVIDIA 드라이버 및 CUDA 설정
nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1

Docker Compose 프로파일 분리:


docker-compose.yml (기본)
docker-compose.gpu.yml (GPU 설정)
docker-compose.monitoring.yml (모니터링)