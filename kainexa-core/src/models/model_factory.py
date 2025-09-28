# src/models/model_factory.py
"""모델 팩토리 - 모든 모델 로더 통합"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import structlog

logger = structlog.get_logger()

class BaseModel(ABC):
    """모델 베이스 클래스"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

class SolarLLM(BaseModel):
    """Solar-10.7B 모델"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.get('device', 'cuda:0')
        self._load_model()
    
    def _load_model(self):
        """모델 로드"""
        model_name = "upstage/solar-10.7b-instruct"
        
        logger.info(f"Loading Solar model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 텐서 병렬 설정
        if self.config.get('tensor_parallel'):
            # DeepSpeed 설정
            import deepspeed
            self.model = self._load_with_tensor_parallel(model_name)
        else:
            # 단일 GPU 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
    def _load_with_tensor_parallel(self, model_name: str):
        """텐서 병렬 로드"""
        # DeepSpeed 설정으로 로드
        # 실제 구현 필요
        pass
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거
        if prompt in response:
            response = response.replace(prompt, '').strip()
        
        return response

class OpenAIAdapter(BaseModel):
    """OpenAI API 어댑터 (Fallback)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """OpenAI API 호출"""
        import openai
        
        openai.api_key = self.api_key
        
        response = await openai.ChatCompletion.acreate(
            model=self.config.get('model', 'gpt-3.5-turbo'),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 512)
        )
        
        return response.choices[0].message.content

class LightweightLLM(BaseModel):
    """경량 모델 (Fallback)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Polyglot 등 경량 한국어 모델
        self.model_name = "EleutherAI/polyglot-ko-5.8b"
        self._load_model()
    
    def _load_model(self):
        """경량 모델 로드"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda:1"  # 두 번째 GPU 사용
        )
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        # Solar와 동일한 생성 로직
        pass

class ModelFactory:
    """모델 생성 팩토리"""
    
    _models_cache = {}
    
    @classmethod
    def create_model(cls, model_type: str, config: Optional[Dict] = None) -> BaseModel:
        """모델 생성"""
        
        # 캐시 확인
        if model_type in cls._models_cache:
            return cls._models_cache[model_type]
        
        config = config or {}
        
        if model_type == "solar":
            model = SolarLLM(config)
        elif model_type == "openai":
            model = OpenAIAdapter(config)
        elif model_type == "lightweight":
            model = LightweightLLM(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 캐시 저장
        cls._models_cache[model_type] = model
        
        return model
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """사용 가능한 모델 목록"""
        return ["solar", "openai", "lightweight"]