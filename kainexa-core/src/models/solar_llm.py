"""
 Solar-10.7B LLM 추론 엔진 (4bit 우선 / 메모리 상한 / OOM 폴백 / 한국어 일관성)
"""
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import os
import time
import logging
import importlib.util
import re

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
        device_map: Optional[str] = "auto",
        load_in_8bit: bool = True,  # 하위호환용 (KXN_QUANT로 대체됨)
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None

        # bad_words_ids 캐시
        self._cjk_bad_ids: Optional[List[List[int]]] = None   # 한자/중문 토큰 금지
        self._ascii_letter_bad_ids: Optional[List[List[int]]] = None  # 영문자 a-zA-Z 금지

        # 한국어 시스템 프롬프트
        self.system_prompt = (
            "당신은 한국 제조업 전문 AI 어시스턴트입니다. "
            "항상 한국어로만 답변하세요. "
            "외국어 용어는 꼭 필요할 때만 최초 1회 괄호로 병기하고, 그 외에는 한국어를 유지하세요. "
            "본문에서 한국어·영어를 혼용하지 마세요. "
            "모든 수치/지표는 단위를 함께 표기하고, 보고서는 간결한 비즈니스 톤으로 작성하세요."
        )

    # ---------- public ----------

    def load(self):
        """모델 로드(한 번만)"""
        if self.model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading Solar model from {self.model_path}")

        # ✅ Ampere(RTX 3090) 최적화: TF32 사용으로 속도 향상
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

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
            "dtype": dtype,                # 최신 권장 키워드 (torch_dtype deprecated 대체)
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
            logger.info("Using 8-bit quantization (legacy flag)")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore
                load_in_8bit=True
            )
        else:
            logger.info("Quantization disabled (fp16/fp32)")

        # device_map / max_memory
        use_device_map = (self.device_map is not None) and has_accelerate
        if use_device_map:
            max_memory: Dict[Union[int, str], str] = {}
            if torch.cuda.is_available():
                frac = float(os.getenv("KXN_GPU_MEM_FRACTION", "0.85"))
                for i in range(torch.cuda.device_count()):
                    total = torch.cuda.get_device_properties(i).total_memory
                    limit_gib = max(1, int(total * frac) // (1024 ** 3))  # 최소 1GiB
                    max_memory[i] = f"{limit_gib}GiB"   # ← 정수 키만 사용!
            max_memory["cpu"] = os.getenv("KXN_CPU_OFFLOAD_MEM", "48GiB")

            load_kwargs["device_map"] = self.device_map  # 보통 "auto"
            load_kwargs["max_memory"] = max_memory
        else:
            if self.device_map is not None and not has_accelerate:
                logger.info(
                    "accelerate 미설치 → device_map 생략, 단일 디바이스로 로드합니다. "
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
        stream: bool = False,
        do_sample: bool = False,   # 기본은 그리디 → 속도/안정성↑
        ko_only: bool = True,      # 한국어만 출력 강화
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        ko_floor: float = 0.85,    # 한글 비율 하한선
        always_block_ascii: bool = False,  # ✅ 1차부터 영문자 차단
    ) -> str:
        """텍스트 생성(한글 비율 낮을 경우 1회 재생성으로 한국어 강제)"""

        if self.model is None:
            self.load()

        full_prompt = self._build_prompt(prompt)

        # 토크나이징
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        prompt_len = inputs["input_ids"].shape[1]

        # 입력 텐서 디바이스 이동(경고 제거)
        target_device = self._resolve_target_device()
        if target_device is not None and str(target_device).startswith("cuda"):
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

        # 스트리머
        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if stream else None

        # ---- 1차 생성(중문 차단만 적용) ----
        gen_kwargs = self._build_gen_kwargs(
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            streamer=streamer if stream else None,
        )
        
        if ko_only:
            bad_words = self._get_cjk_bad_ids()  # 기본: 중문(한자) 차단
            if always_block_ascii:
                # ✅ 1차부터 영문자 a-zA-Z 전면 차단(숫자/문장부호는 허용)
                bad_words += self._get_ascii_letter_bad_ids()
            gen_kwargs["bad_words_ids"] = bad_words

        start_time = time.time()
        outputs = self._safe_generate(inputs, gen_kwargs, max_new_tokens, stream, do_sample, ko_only)

        # 디코딩: 프롬프트 제외
        gen_ids = outputs[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # 프롬프트 에코 제거
        if full_prompt in text:
            text = text.split(full_prompt, 1)[-1].strip()

        # 한국어 비율 검사
        if ko_only:
            ratio = self._ko_ratio(text)
            if ratio < ko_floor:
                logger.info(f"KO ratio {ratio:.2f} < {ko_floor:.2f} → re-generate with ASCII letters blocked")
                # ---- 2차 생성: 중문 + 영문자 전면 차단 ----
                gen_kwargs2 = self._build_gen_kwargs(
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer if stream else None,
                )
                bad_words = []
                bad_words += self._get_cjk_bad_ids()
                bad_words += self._get_ascii_letter_bad_ids()  # 영문자 a-zA-Z 전면 차단(숫자/문장부호 허용)
                gen_kwargs2["bad_words_ids"] = bad_words

                outputs = self._safe_generate(inputs, gen_kwargs2, max_new_tokens, stream, do_sample, ko_only)
                gen_ids = outputs[0][prompt_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if full_prompt in text:
                    text = text.split(full_prompt, 1)[-1].strip()

        # 메트릭
        generation_time = time.time() - start_time
        tokens_generated = int(gen_ids.shape[0])
        tps = tokens_generated / generation_time if generation_time > 0 else 0.0
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tps:.1f} t/s)")

        return text

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

    # ---------- helpers ----------

    def _build_prompt(self, user_message: str) -> str:
        return f"""### System:
{self.system_prompt}

### User:
{user_message}

### Assistant:"""

    def _resolve_target_device(self) -> Optional[torch.device]:
        """모델 파라미터/샤딩 정보를 바탕으로 입력 텐서를 보낼 디바이스 결정"""
        try:
            param_dev = next(self.model.parameters()).device
            target_device = param_dev
        except StopIteration:
            target_device = None

        if hasattr(self.model, "hf_device_map"):
            unique_devs = {
                str(d)
                for d in set(self.model.hf_device_map.values())
                if str(d) not in {"cpu", "disk", "meta", "offload"}
            }
            cuda_devs = [d for d in unique_devs if d.startswith("cuda")]
            if len(cuda_devs) == 1:
                target_device = torch.device(cuda_devs[0])
        return target_device

    def _build_gen_kwargs(
        self,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        pad_token_id: int,
        max_new_tokens: int,
        streamer: Optional[TextStreamer] = None,
    ) -> Dict[str, Any]:
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
        }
        if streamer is not None:
            gen_kwargs["streamer"] = streamer
        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        else:
            gen_kwargs.update({"do_sample": False})  # temperature 전달 안 함(경고 방지)
        return gen_kwargs

    def _safe_generate(
        self,
        inputs: Dict[str, torch.Tensor],
        gen_kwargs: Dict[str, Any],
        max_new_tokens: int,
        stream: bool,
        do_sample: bool,
        ko_only: bool,
    ):
        try:
            with torch.no_grad():
                return self.model.generate(**inputs, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.warning("⚠️ OOM during generate(); retrying with smaller settings…")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fb_kwargs = {
                "max_new_tokens": max(128, max_new_tokens // 2),
                "use_cache": False,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if stream and "streamer" in gen_kwargs:
                fb_kwargs["streamer"] = gen_kwargs["streamer"]
            if do_sample:
                fb_kwargs.update({
                    "do_sample": True,
                    "temperature": gen_kwargs.get("temperature", 0.7),
                    "top_p": gen_kwargs.get("top_p", 0.9),
                    "top_k": gen_kwargs.get("top_k", 50),
                })
            else:
                fb_kwargs.update({"do_sample": False})
            if ko_only and "bad_words_ids" in gen_kwargs:
                fb_kwargs["bad_words_ids"] = gen_kwargs["bad_words_ids"]
            with torch.no_grad():
                return self.model.generate(**inputs, **fb_kwargs)

    def _ko_ratio(self, text: str) -> float:
        """한글 비율(문자 수 기준)"""
        if not text:
            return 0.0
        ko = re.findall(r"[가-힣]", text)
        return len(ko) / max(1, len(text))

    def _get_cjk_bad_ids(self) -> List[List[int]]:
        """
        중국어/한자(CJK Unified Ideographs 및 확장A, 호환) 토큰을 금지 목록으로 생성.
        최초 1회 스캔 후 캐시.
        """
        if self._cjk_bad_ids is not None:
            return self._cjk_bad_ids
        bad: List[List[int]] = []
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab_size = int(max(self.tokenizer.get_vocab().values())) + 1
        for tid in range(vocab_size):
            s = self.tokenizer.decode([tid], skip_special_tokens=True)
            if not s:
                continue
            # CJK 범위: 기본(4E00-9FFF), 확장A(3400-4DBF), 호환(F900-FAFF)
            if any(
                ("\u4E00" <= ch <= "\u9FFF") or
                ("\u3400" <= ch <= "\u4DBF") or
                ("\uF900" <= ch <= "\uFAFF")
                for ch in s
            ):
                bad.append([tid])
        self._cjk_bad_ids = bad
        return bad

    def _get_ascii_letter_bad_ids(self) -> List[List[int]]:
        """
        영문자 a-zA-Z를 금지 목록으로 생성(숫자와 문장부호는 허용).
        - 2차 재생성 시 한국어 강제용
        """
        if self._ascii_letter_bad_ids is not None:
            return self._ascii_letter_bad_ids
        bad: List[List[int]] = []
        for ch in [*(chr(i) for i in range(65, 91)), *(chr(i) for i in range(97, 123))]:
            tok = self.tokenizer(ch, add_special_tokens=False)
            if tok and tok.input_ids:
                bad.append(tok.input_ids)
        self._ascii_letter_bad_ids = bad
        return bad
