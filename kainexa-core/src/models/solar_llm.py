# src/models/solar_llm.py 생성
"""
 Solar-10.7B LLM 추론 엔진
"""
import torch
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import logging
import importlib.util  # ← accelerate/bitsandbytes 존재 확인용

logger = logging.getLogger(__name__)

class SolarLLM:
    """Solar-10.7B 추론 엔진"""
    
    def __init__(self, 
                 model_path: str = "models/solar-10.7b",
                 device_map: str = "auto",
                 load_in_8bit: bool = True,
                 device: Optional[str] = None):
        
        self.model_path = Path(model_path)
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        
        # 한국어 시스템 프롬프트
        self.system_prompt = """당신은 한국 제조업 전문 AI 어시스턴트입니다. 
        정확하고 실용적인 답변을 제공하며, 필요시 데이터 기반 분석을 수행합니다."""
        
    def load(self):
        """모델 로드"""
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading Solar model from {self.model_path}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드 (INT8 양자화)
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": self.device_map,
            "trust_remote_code": True
        }
        
        if self.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs
        )
        
        # 가용 환경 점검
        has_accelerate = importlib.util.find_spec("accelerate") is not None
        has_bnb = importlib.util.find_spec("bitsandbytes") is not None
        is_gpu = (self.device == "cuda" and torch.cuda.is_available())

        # dtype 결정: GPU면 fp16, CPU면 fp32
        torch_dtype = torch.float16 if is_gpu else torch.float32

        # 기본 로드 인자
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        # 8bit 로딩 가능 여부 판단
        use_8bit = bool(self.load_in_8bit and is_gpu and has_bnb)
        if self.load_in_8bit and not use_8bit:
            if not is_gpu:
                logger.info("8-bit loading disabled: CPU 환경에서는 8bit 로딩을 사용하지 않습니다.")
            elif not has_bnb:
                logger.info("8-bit loading disabled: bitsandbytes 패키지가 없습니다. (pip install bitsandbytes)")

        if use_8bit:
            load_kwargs["load_in_8bit"] = True

        # device_map 처리:
        # - accelerate가 있으면 요청된 device_map을 사용(기본 'auto')
        # - accelerate가 없으면 device_map 인자를 아예 전달하지 않고, 단일 디바이스로 .to(self.device)
        use_device_map = (self.device_map is not None) and has_accelerate
        if use_device_map:
            load_kwargs["device_map"] = self.device_map  # 보통 "auto"
        else:
            if self.device_map is not None and not has_accelerate:
                logger.info("accelerate 미설치로 device_map 인자를 무시하고 단일 디바이스로 로드합니다. (pip install accelerate 권장)")

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

        # accelerate가 없는 경우에는 명시적으로 단일 디바이스에 올림
        if not use_device_map:
            self.model.to(self.device)
         
        # 평가 모드
        self.model.eval()
        
        logger.info("✅ Solar model loaded successfully")
        
    def generate(self, 
                prompt: str,
                max_new_tokens: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.9,
                top_k: int = 50,
                stream: bool = False) -> str:
        """텍스트 생성"""
        
        if self.model is None:
            self.load()
        
        # 프롬프트 구성
        full_prompt = self._build_prompt(prompt)
        
        # 토크나이징
        inputs = self.tokenizer(
            full_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
                
        # 입력 텐서 디바이스 이동
        # - accelerate로 샤딩된 경우(hf_device_map 존재 시)에는 강제 이동하지 않음
        # - 단일 디바이스 로드인 경우에만 self.device로 이동
        if not hasattr(self.model, "hf_device_map"):
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 스트리밍 설정
        streamer = None
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        # 생성
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                streamer=streamer
            )
        
        # 디코딩
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거
        response = response[len(full_prompt):].strip()
        
        # 메트릭
        generation_time = time.time() - start_time
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
        
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} t/s)")
        
        return response
    
    def _build_prompt(self, user_message: str) -> str:
        """프롬프트 템플릿 구성"""
        # Solar 모델용 프롬프트 포맷
        prompt = f"""### System:
{self.system_prompt}

### User:
{user_message}

### Assistant:"""
        return prompt
    
    def batch_generate(self, 
                       prompts: List[str],
                       **kwargs) -> List[str]:
        """배치 생성"""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        return responses
    
    def get_memory_usage(self) -> Dict[str, float]:
        """GPU 메모리 사용량"""
        if not torch.cuda.is_available():
            return {}
        
        memory_stats = {}
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            
            memory_stats[f"gpu_{i}"] = {
                "allocated_gb": round(mem_allocated, 2),
                "reserved_gb": round(mem_reserved, 2),
                "total_gb": round(mem_total, 2),
                "usage_percent": round((mem_allocated / mem_total) * 100, 1)
            }
        
        return memory_stats