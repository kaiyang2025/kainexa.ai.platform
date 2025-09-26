"""
 Solar-10.7B LLM 추론 엔진 (4bit 우선 / 메모리 상한 / OOM 폴백)
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import time
import logging
import importlib.util

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

try:
    # 8bit/4bit는 BitsAndBytesConfig로 지정 (없으면 자동 폴백)
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:
    BitsAndBytesConfig = None  # 런타임 분기용

logger = logging.getLogger(__name__)


class SolarLLM:
    """Solar-10.7B 추론 엔진"""

    def __init__(
        self,
        model_path: str = "models/solar-10.7b",
        device_map: str = "auto",
        load_in_8bit: bool = True,  # 하위호환용 (KXN_QUANT로 대체됨)
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None

        # 한국어 시스템 프롬프트
        self.system_prompt = (
            "당신은 한국 제조업 전문 AI 어시스턴트입니다. "
            "정확하고 실용적인 답변을 제공하며, 필요시 데이터 기반 분석을 수행합니다."
        )

    def load(self):
        """모델 로드(한 번만)"""
        if self.model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading Solar model from {self.model_path}")

        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 환경/가용성
        has_accelerate = importlib.util.find_spec("accelerate") is not None
        has_bnb = importlib.util.find_spec("bitsandbytes") is not None
        is_gpu = (self.device == "cuda" and torch.cuda.is_available())

        # dtype: GPU=fp16, CPU=fp32
        dtype = torch.float16 if is_gpu else torch.float32

        # 기본 로드 인자
        load_kwargs: Dict[str, Any] = {
            "dtype": dtype,                # 최신 권장 키워드
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        # === Quantization 선택: 4bit → 8bit → none ===
        quant_pref = os.getenv("KXN_QUANT", "auto").lower()  # auto|4bit|8bit|none
        def can_bnb() -> bool:
            return bool(is_gpu and has_bnb and BitsAndBytesConfig is not None)

        if (quant_pref in ("4bit", "auto")) and can_bnb():
            logger.info("Using 4-bit NF4 quantization")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        elif (quant_pref in ("8bit", "auto")) and can_bnb():
            logger.info("Using 8-bit quantization")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore
                load_in_8bit=True
            )
        elif self.load_in_8bit and can_bnb() and quant_pref == "auto":
            # 하위호환: load_in_8bit=True가 활성인 환경에서 auto면 8bit 허용
            logger.info("Using 8-bit quantization (legacy flag)")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore
                load_in_8bit=True
            )
        else:
            logger.info("Quantization disabled (fp16/fp32)")

        # device_map / max_memory
        use_device_map = (self.device_map is not None) and has_accelerate
        if use_device_map:
            # 각 GPU 사용 상한(기본 85%) 및 CPU 오프로딩 한도
            max_memory: Dict[str, str] = {}
            if torch.cuda.is_available():
                frac = float(os.getenv("KXN_GPU_MEM_FRACTION", "0.85"))
                for i in range(torch.cuda.device_count()):
                    total = torch.cuda.get_device_properties(i).total_memory
                    limit = max(1, int(total * frac) // (1024 ** 3))  # GiB 단위, 최소 1GiB
                    max_memory[f"cuda:{i}"] = f"{limit}GiB"
            max_memory["cpu"] = os.getenv("KXN_CPU_OFFLOAD_MEM", "48GiB")

            load_kwargs["device_map"] = self.device_map  # 보통 "auto"
            load_kwargs["max_memory"] = max_memory
        else:
            if self.device_map is not None and not has_accelerate:
                logger.info(
                    "accelerate 미설치 → device_map 인자 생략, 단일 디바이스로 로드합니다. "
                    "(pip install accelerate 권장)"
                )

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

        # 단일 디바이스 로드인 경우 명시 이동
        if not use_device_map:
            self.model.to(self.device)

        self.model.eval()
        logger.info("✅ Solar model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False,
    ) -> str:
        """텍스트 생성"""

        if self.model is None:
            self.load()

        # 프롬프트
        full_prompt = self._build_prompt(prompt)

        # 토크나이징
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        prompt_len = inputs["input_ids"].shape[1]

        # 입력 텐서 디바이스 이동
        # - 샤딩(hf_device_map 존재)인 경우: 단일 GPU만 사용 중이면 그쪽으로 이동
        # - 다중 장치면 Accelerate가 처리하므로 이동하지 않음
        # - 단일 디바이스 모델이면 self.device 로 이동
        target_device = None
        if hasattr(self.model, "hf_device_map"):
            devices = {
                str(d)
                for d in set(self.model.hf_device_map.values())
                if str(d) not in {"cpu", "disk", "meta", "offload"}
            }
            if len(devices) == 1:
                only = list(devices)[0]
                if only.startswith("cuda"):
                    target_device = torch.device(only)
        else:
            if self.device == "cuda" and torch.cuda.is_available():
                target_device = torch.device("cuda")

        if target_device is not None:
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

        # 스트리밍
        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if stream else None

        # 생성
        start_time = time.time()
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("⚠️ OOM during generate(); retrying with smaller settings…")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 토큰수 절반(최소 128), 캐시 비활성화로 폴백
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max(128, max_new_tokens // 2),
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    use_cache=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                )

        # 디코딩: 프롬프트 토큰을 제외한 부분만
        gen_ids = outputs[0][prompt_len:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # 메트릭
        generation_time = time.time() - start_time
        tokens_generated = int(gen_ids.shape[0])
        tps = tokens_generated / generation_time if generation_time > 0 else 0.0
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tps:.1f} t/s)")

        return response

    def _build_prompt(self, user_message: str) -> str:
        """프롬프트 템플릿 구성"""
        return f"""### System:
{self.system_prompt}

### User:
{user_message}

### Assistant:"""

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """배치 생성"""
        responses = []
        for p in prompts:
            responses.append(self.generate(p, **kwargs))
        return responses

    def get_memory_usage(self) -> Dict[str, float]:
        """GPU 메모리 사용량 조회(디버그 용)"""
        if not torch.cuda.is_available():
            return {}
        memory_stats: Dict[str, Dict[str, float]] = {}
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
            mem_resv = torch.cuda.memory_reserved(i) / (1024 ** 3)
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            memory_stats[f"gpu_{i}"] = {
                "allocated_gb": round(mem_alloc, 2),
                "reserved_gb": round(mem_resv, 2),
                "total_gb": round(mem_total, 2),
                "usage_percent": round((mem_alloc / mem_total) * 100, 1),
            }
        return memory_stats
