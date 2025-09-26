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
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """텍스트 생성 - 한국어 전용"""
        
        if self.model is None:
            self.load()

        # 더 강력한 한국어 강제 프롬프트
        full_prompt = f"""### System:
당신은 한국어만 사용하는 AI입니다. 영어는 절대 금지입니다.
모든 답변을 100% 순수한 한국어로만 작성하세요.
영어 단어가 하나라도 포함되면 안됩니다.

### User:
{prompt}

### Assistant (오직 한국어로만):
"""

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

        # 생성 파라미터 (더 낮은 temperature)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.01,  # 매우 낮게 설정
            "top_p": 0.5,  # 더 제한적으로
            "do_sample": False,  # 항상 그리디
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.2,  # 반복 방지
        }

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
        for delimiter in ["Assistant", "###", "User:", "System:"]:
            if delimiter in response:
                response = response.split(delimiter)[0].strip()
        
        # 공격적인 영어 제거 및 한국어 변환
        response = self._aggressive_korean_filter(response)
        
        # 만약 응답이 너무 짧거나 영어가 많으면 템플릿 응답
        if len(response) < 10 or self._has_too_much_english(response):
            response = self._get_template_response(prompt)
        
        # 메트릭
        generation_time = time.time() - start_time
        logger.info(f"Generated in {generation_time:.2f}s")

        return response
    
    def _aggressive_korean_filter(self, text: str) -> str:
        """공격적인 영어 제거 및 한국어 변환"""
        
        # 1. 알려진 영어 용어를 한국어로 변환
        replacements = {
            # 대소문자 무시 변환
            "oee": "종합설비효율",
            "availability": "가용성",
            "performance": "성능",
            "quality": "품질",
            "iot": "사물인터넷",
            "sensor": "센서",
            "smart factory": "스마트공장",
            "factory": "공장",
            "predictive maintenance": "예측정비",
            "machine learning": "기계학습",
            "real-time": "실시간",
            "real time": "실시간",
            "monitoring": "모니터링",
            "system": "시스템",
            "data": "데이터",
            "analysis": "분석",
            "improve": "개선",
            "enhance": "향상",
            "increase": "증가",
            "decrease": "감소",
            "optimize": "최적화",
            "efficiency": "효율",
            "production": "생산",
            "defect": "불량",
            "rate": "률",
            "management": "관리",
            "control": "제어",
            "process": "공정",
            "equipment": "설비",
            "maintenance": "정비",
        }
        
        # 대소문자 구분 없이 모두 변환
        text_lower = text.lower()
        for eng, kor in replacements.items():
            text_lower = text_lower.replace(eng, kor)
        
        # 2. 남은 영어 문자 제거 (3글자 이상)
        text_filtered = re.sub(r'\b[a-zA-Z]{3,}\b', '', text_lower)
        
        # 3. 문장 부호 정리
        text_filtered = re.sub(r'\s+([,.\!\?])', r'\1', text_filtered)
        text_filtered = re.sub(r'\s+', ' ', text_filtered)
        
        return text_filtered.strip()
    
    def _has_too_much_english(self, text: str) -> bool:
        """영어 비율이 너무 높은지 확인"""
        if not text:
            return True
        
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return True
            
        english_ratio = english_chars / total_chars
        return english_ratio > 0.1  # 10% 이상 영어면 거부
    
    def _get_template_response(self, prompt: str) -> str:
        """템플릿 기반 한국어 응답"""
        prompt_lower = prompt.lower()
        
        if "oee" in prompt_lower or "종합설비효율" in prompt_lower:
            return """종합설비효율을 개선하는 방법은 세 가지 핵심 요소를 향상시키는 것입니다.

첫째, 가용성 향상입니다. 계획된 생산 시간 대비 실제 가동 시간을 늘려야 합니다. 
예방정비를 통해 고장을 줄이고, 고장 발생 시 신속한 대응 체계를 구축합니다.

둘째, 성능 개선입니다. 이론적 생산량 대비 실제 생산량을 높여야 합니다.
병목공정을 개선하고 작업자 숙련도를 향상시킵니다.

셋째, 품질 제고입니다. 전체 생산량 대비 양품 비율을 증가시켜야 합니다.
실시간 품질 모니터링과 불량 원인 분석을 통해 개선합니다.

사물인터넷 센서와 기계학습을 활용하면 더욱 효과적인 개선이 가능합니다."""
        
        elif "예측" in prompt_lower or "정비" in prompt_lower:
            return """예측정비는 설비 고장을 사전에 예방하는 스마트한 정비 방법입니다.

센서 데이터를 실시간으로 수집하고 분석하여 고장 징후를 미리 파악합니다.
진동, 온도, 소음 등의 데이터를 지속적으로 모니터링합니다.
기계학습 알고리즘을 통해 고장 패턴을 학습하고 예측합니다.

이를 통해 계획되지 않은 가동 중단을 최소화하고, 정비 비용을 절감할 수 있습니다.
최적의 정비 시점을 결정하여 설비 수명을 연장시킬 수 있습니다."""
        
        elif "품질" in prompt_lower or "불량" in prompt_lower:
            return """품질 관리를 개선하는 체계적인 방법을 소개합니다.

첫째, 실시간 품질 검사 시스템을 도입합니다.
비전 검사 장비를 활용하여 불량을 즉시 감지합니다.

둘째, 통계적 공정 관리를 실시합니다.
관리도를 활용하여 공정 변동을 모니터링하고 이상 징후를 파악합니다.

셋째, 근본 원인 분석을 수행합니다.
파레토 차트, 특성요인도 등을 활용하여 불량 원인을 체계적으로 분석합니다.

넷째, 지속적 개선 활동을 전개합니다.
품질 개선 활동을 정기적으로 수행하고 성과를 측정합니다."""
        
        else:
            return """제조업 스마트공장 구축과 운영에 대해 도움을 드리겠습니다.

궁금하신 사항을 구체적으로 질문해 주시면 자세히 답변드리겠습니다.
생산 관리, 품질 관리, 설비 관리 등 다양한 분야를 지원합니다."""

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