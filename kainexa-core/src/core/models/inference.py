# src/models/inference.py
"""통합 추론 실행"""
import asyncio
from typing import Dict, Any
import time
import torch

from src.models.model_factory import ModelFactory
from src.monitoring.metrics import MetricsCollector

class InferenceEngine:
    """추론 엔진"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.metrics = MetricsCollector(config.get('clickhouse', {}))
        
    async def initialize(self):
        """초기화"""
        # 모델 로드
        model_type = self.config.get('model_type', 'solar')
        self.model = ModelFactory.create_model(model_type, self.config)
        
        # GPU 설정
        if torch.cuda.is_available():
            print(f"GPU Available: {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    async def run_inference(self, prompt: str, **params) -> Dict[str, Any]:
        """추론 실행"""
        start_time = time.time()
        
        # 추론
        response = await self.model.generate(prompt, **params)
        
        # 메트릭 수집
        duration = time.time() - start_time
        tokens = len(response.split())
        
        await self.metrics.track_llm_inference(
            model=self.config.get('model_type'),
            tokens=tokens,
            duration=duration,
            gpu_id=0
        )
        
        return {
            'response': response,
            'duration': duration,
            'tokens': tokens,
            'tokens_per_sec': tokens / duration if duration > 0 else 0
        }

async def main():
    """메인 실행"""
    config = {
        'model_type': 'solar',
        'device': 'cuda:0',
        'tensor_parallel': True,
        'clickhouse': {
            'host': 'localhost',
            'port': 9000
        }
    }
    
    engine = InferenceEngine(config)
    await engine.initialize()
    
    # 테스트 추론
    result = await engine.run_inference(
        "한국 제조업의 미래는 어떻게 될까요?",
        temperature=0.7,
        max_tokens=256
    )
    
    print(f"\n응답: {result['response']}")
    print(f"소요시간: {result['duration']:.2f}초")
    print(f"토큰/초: {result['tokens_per_sec']:.1f}")

if __name__ == "__main__":
    asyncio.run(main())