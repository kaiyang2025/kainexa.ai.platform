"""
Solar-10.7B LLM 추론 엔진 (한국어 전용 모드)
"""
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import os
import time
import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

logger = logging.getLogger(__name__)


class SolarLLM:
    """Solar-10.7B 추론 엔진 - 한국어 전용"""

    def __init__(
        self,
        model_path: str = "models/solar-10.7b",
        device_map: Optional[str] = "auto",
        load_in_8bit: bool = True,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None

        # 한국어 시스템 프롬프트 (강화)
        self.system_prompt = """당신은 한국 제조업 전문 AI 어시스턴트입니다.
반드시 한국어(한글)로만 답변하세요.
영어 단어나 외국어를 절대 사용하지 마세요.
모든 기술 용어도 한국어로 번역하여 설명하세요."""

    def load(self):
        """모델 로드"""
        if self.model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model from {self.model_path}")

        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드 설정
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        # Quantization 설정
        if torch.cuda.is_available() and BitsAndBytesConfig:
            quant_pref = os.getenv("KXN_QUANT", "4bit").lower()
            if quant_pref == "4bit":
                logger.info("Using 4-bit quantization")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
            elif quant_pref == "8bit" or self.load_in_8bit:
                logger.info("Using 8-bit quantization")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )

        # Device map 설정
        if self.device_map and torch.cuda.is_available():
            load_kwargs["device_map"] = self.device_map
            
            # 메모리 제한 설정
            max_memory = {}
            for i in range(torch.cuda.device_count()):
                max_memory[i] = f"{int(torch.cuda.get_device_properties(i).total_memory * 0.85 / 1e9)}GiB"
            max_memory["cpu"] = "48GiB"
            load_kwargs["max_memory"] = max_memory

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        
        if not self.device_map:
            self.model.to(self.device)
            
        self.model.eval()
        logger.info("✅ Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        stream: bool = False,
        temperature: float = 0.1,  # 낮춰서 일관성 향상
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = False,   # 그리디 모드
        **kwargs
    ) -> str:
        """텍스트 생성 - 한국어 전용"""
        
        if self.model is None:
            self.load()

        # 한국어 강제 프롬프트
        full_prompt = f"""### System:
{self.system_prompt}

중요: 반드시 한국어로만 답변하세요. 영어나 외국어를 사용하지 마세요.
모든 기술 용어를 한국어로 번역하세요.
- OEE → 종합설비효율
- Availability → 가용성
- Performance → 성능
- Quality → 품질
- IoT → 사물인터넷
- Machine Learning → 기계학습
- Predictive Maintenance → 예측 정비

### User:
{prompt}

### Assistant (한국어로만 답변):"""

        # 토크나이징
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        
        # GPU로 이동
        if torch.cuda.is_available():
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # 생성 파라미터
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "top_k": top_k if do_sample else None,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # 스트리밍
        if stream:
            gen_kwargs["streamer"] = TextStreamer(self.tokenizer, skip_prompt=True)

        # 생성
        start_time = time.time()
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM - retrying with smaller settings")
            torch.cuda.empty_cache()
            gen_kwargs["max_new_tokens"] = min(256, max_new_tokens)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

        # 디코딩
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 프롬프트 제거
        response = response.strip()
        if "Assistant" in response:
            response = response.split("Assistant")[-1].strip()
        if "###" in response:
            response = response.split("###")[0].strip()
            
        # 영어를 한국어로 후처리 변환
        response = self._translate_to_korean(response)
        
        # 메트릭
        generation_time = time.time() - start_time
        tokens_generated = len(generated_ids)
        tps = tokens_generated / generation_time if generation_time > 0 else 0
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tps:.1f} t/s)")

        return response

    def _translate_to_korean(self, text: str) -> str:
        """영어 용어를 한국어로 변환"""
        translations = {
            # 제조업 용어
            "OEE": "종합설비효율",
            "Overall Equipment Effectiveness": "종합설비효율",
            "Availability": "가용성",
            "Performance": "성능",
            "Quality": "품질",
            "Predictive Maintenance": "예측 정비",
            "predictive maintenance": "예측 정비",
            "IoT": "사물인터넷",
            "IoT sensors": "사물인터넷 센서",
            "Machine Learning": "기계학습",
            "machine learning": "기계학습",
            "Real-time": "실시간",
            "real-time": "실시간",
            "monitoring": "모니터링",
            "Smart Factory": "스마트 팩토리",
            "smart factory": "스마트 팩토리",
            
            # 일반 용어
            "Improve": "개선",
            "improve": "개선",
            "Enhance": "향상",
            "enhance": "향상",
            "Increase": "증가",
            "increase": "증가",
            "Decrease": "감소",
            "decrease": "감소",
            "Calculate": "계산",
            "calculate": "계산",
            "Implement": "구현",
            "implement": "구현",
            "Optimize": "최적화",
            "optimize": "최적화",
            "Analysis": "분석",
            "analysis": "분석",
            
            # 문장 패턴
            "compared to": "대비",
            "based on": "기반으로",
            "can be": "할 수 있습니다",
            "should be": "해야 합니다",
            "will be": "될 것입니다",
            "for example": "예를 들어",
            "in order to": "위해",
            "as well as": "뿐만 아니라",
            "such as": "같은",
            "and": "그리고",
            "or": "또는",
            "but": "하지만",
            "however": "그러나",
            "therefore": "따라서",
            "because": "왜냐하면",
        }
        
        # 대소문자 구분 없이 치환
        for eng, kor in translations.items():
            # 정규식으로 단어 경계 처리
            pattern = r'\b' + re.escape(eng) + r'\b'
            text = re.sub(pattern, kor, text, flags=re.IGNORECASE)
        
        return text

    def _build_prompt(self, user_message: str) -> str:
        """프롬프트 템플릿 구성"""
        return f"""### System:
{self.system_prompt}

### User:
{user_message}

### Assistant:"""

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """배치 생성"""
        return [self.generate(p, **kwargs) for p in prompts]

    def get_memory_usage(self) -> Dict[str, Any]:
        """GPU 메모리 사용량"""
        if not torch.cuda.is_available():
            return {}
        
        memory_stats = {}
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            memory_stats[f"gpu_{i}"] = {
                "allocated_gb": round(mem_allocated, 2),
                "total_gb": round(mem_total, 2),
                "usage_percent": round((mem_allocated / mem_total) * 100, 1)
            }
        
        return memory_stats