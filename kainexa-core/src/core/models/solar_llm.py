# src/core/models/solar_llm.py
"""
Solar-10.7B LLM 추론 엔진 (한국어 전용 모드, 4bit/8bit, OOM 폴백)
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import time
import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # 양자화 옵션 없이도 동작

logger = logging.getLogger(__name__)


class SolarLLM:
    """Solar-10.7B 추론 엔진 - 한국어 전용"""

    def __init__(
        self,
        model_path: str = "beomi/OPEN-SOLAR-KO-10.7B",
        device_map: Optional[str] = "auto",   # "auto" 권장(2장 NVLink 분산)
        load_in_8bit: bool = True,            # 환경변수 KXN_QUANT가 우선
        device: Optional[str] = None,
    ):
        self.model_path = str(Path(model_path))
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None

        # bad_words 캐시 (토크나이저 단위로 1회 구축)
        self._bad_words_cache: Optional[List[List[int]]] = None

        # 한국어 시스템 프롬프트
        self.system_prompt = (
            "당신은 한국 제조업 전문 AI 어시스턴트입니다.\n"
            "반드시 한국어(한글)로만 답변하세요.\n"
            "영어/한자/가나/키릴 등 한국어 외 문자는 사용하지 마세요.\n"
            "전문 용어는 한국어로 설명하고, 필요 시 약어(OEE/IoT/AI)는 괄호로 병기합니다."
        )

    # ----------------------------
    # 로딩
    # ----------------------------
    def _build_quant_config(self, prefer: str, dtype) -> Optional[BitsAndBytesConfig]:
        if not BitsAndBytesConfig or not torch.cuda.is_available():
            return None
        prefer = prefer.lower()
        if prefer == "4bit":
            logger.info("Using 4-bit NF4 quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        if prefer == "8bit" or (prefer != "4bit" and self.load_in_8bit):
            logger.info("Using 8-bit quantization")
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _build_max_memory(self) -> Optional[Dict[Any, str]]:
        if not torch.cuda.is_available():
            return None
        pct = float(os.getenv("KXN_MAX_MEMORY_PCT", "0.85"))
        max_mem = {}
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            gb = int(total * pct)
            override = os.getenv(f"KXN_MAX_MEMORY_GB_{i}")
            if override:
                max_mem[i] = f"{override}GiB"
            else:
                max_mem[i] = f"{gb}GiB"
        # CPU 버퍼 여유치
        max_mem["cpu"] = os.getenv("KXN_MAX_MEMORY_GB_CPU", "48GiB")
        return max_mem

    def _try_load(self, quant: Optional[str], device_map: Optional[str]) -> bool:
        """주어진 옵션으로 1회 로드 시도"""
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        load_kwargs = {
            "torch_dtype": dtype,  # torch_dtype 경고 회피
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        qcfg = self._build_quant_config(quant or "", dtype)
        if qcfg:
            load_kwargs["quantization_config"] = qcfg

        if device_map and torch.cuda.is_available():
            load_kwargs["device_map"] = device_map
            mm = self._build_max_memory()
            if mm:
                load_kwargs["max_memory"] = mm

        logger.info(
            f"→ load attempt: quant={quant or 'none'}, device_map={device_map}, dtype={dtype}"
        )

        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        if not device_map:
            # 단일 디바이스 (cuda or cpu)
            self.model.to(self.device)
        self.model.eval()
        return True

    def load(self):
        """모델 로드 (4bit → 8bit → FP16 단일 GPU → CPU 순 폴백)"""
        if self.model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model from {self.model_path}")

        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 1차 시도: env 우선
        quant_pref = os.getenv("KXN_QUANT", "4bit").lower()
        devmap = os.getenv("KXN_DEVICE_MAP", self.device_map if self.device_map else "")
        devmap = devmap if devmap in ("auto", "", None) else None if devmap == "single" else "auto"

        tried: List[str] = []
        def _attempt(q, dm):
            tried.append(f"{q or 'none'}/{dm or 'single'}")
            return self._try_load(q, dm)

        try:
            # 4bit/8bit 멀티GPU → 8bit 멀티GPU → FP16 단일GPU → CPU
            if torch.cuda.is_available():
                try:
                    _attempt(quant_pref, devmap or "auto")
                except Exception:
                    if quant_pref != "8bit":
                        logger.warning("Fallback: 8bit on multi-GPU")
                        _attempt("8bit", "auto")
            else:
                raise RuntimeError("CUDA not available")

        except Exception as e1:
            logger.warning(f"Fallback: single-GPU/CPU path due to: {e1}")
            try:
                if torch.cuda.is_available():
                    # FP16 단일 GPU
                    _attempt(None, None)
                else:
                    raise RuntimeError("No CUDA")
            except Exception as e2:
                logger.warning(f"Fallback: CPU float32 due to: {e2}")
                # CPU 최종 폴백
                self.device = "cpu"
                self._try_load(None, None)

        logger.info(f"✅ Model loaded successfully (tried: {', '.join(tried)})")

    # ----------------------------
    # 한국어 전용 제약(토큰 금지)
    # ----------------------------
    def _build_bad_words_ids(self) -> List[List[int]]:
        """한글만 허용: 영문/한자/가나/키릴 스크립트 토큰 금지 목록 생성 (토크나이저별 1회)"""
        if self._bad_words_cache is not None:
            return self._bad_words_cache

        bad: List[List[int]] = []
        
        # 특정 문제 토큰들 직접 차단
        problem_tokens = [
            "△", "▽", "□", "■", "○", "●",  # 도형
            "خ", "ُ", "و", "ش", "َ", "آ",  # 아랍 문자
            "安", "您", "好",  # 중국어
            "こ", "ん", "に", "ち", "は",  # 일본어
        ]        
        for token in problem_tokens:
            try:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                for id in ids:
                    bad.append([id])
            except:
                pass
        
        pat_list = [
            r"[A-Za-z]",                            # 영문
            r"[\u3400-\u4DBF\u4E00-\u9FFF]",        # 한자(CJK Unified + Ext-A)
            r"[\u3040-\u30FF]",                     # 일본어 가나(히라/가타)
            r"[\u0400-\u04FF]",                     # 키릴
        ]
        compiled = [re.compile(p) for p in pat_list]

        # vocab 전수 스캔 (1회 캐시)
        for tok_id in range(self.tokenizer.vocab_size):
            txt = self.tokenizer.decode([tok_id], skip_special_tokens=True)
            if not txt:
                continue
            if any(p.search(txt) for p in compiled):
                bad.append([tok_id])

        self._bad_words_cache = bad
        logger.info(f"bad_words_ids prepared: {len(bad)} tokens banned")
        return bad

    # ----------------------------
    # 후처리(최종 안전장치)
    # ----------------------------
    def _force_korean(self, text: str) -> str:
        """한국어만 남기고 정리 - 더 엄격한 처리"""
        if not text:
            return text
        
        # 1. 약어 화이트리스트 보호
        WL = {
            "OEE": "__WL_OEE__", "IoT": "__WL_IOT__", "AI": "__WL_AI__",
            "API": "__WL_API__", "LLM": "__WL_LLM__", "GPU": "__WL_GPU__"
        }
        for k, v in WL.items():
            text = text.replace(k, v)
        
        # 2. 깨진 한글 패턴 수정
        text = text.replace("안녫", "안녕")
        text = text.replace("업 엔", "업무")
        text = text.replace("하요", "하세요")
        
        # 3. 연속된 특수문자 제거
        text = re.sub(r'[△▽□■○●]+', '', text)
        text = re.sub(r'[\u0600-\u06FF]+', '', text)  # 아랍 문자
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)  # 중국어
        text = re.sub(r'[\u3040-\u30ff]+', '', text)  # 일본어
        
        # 4. 괄호 안의 이상한 문자 제거
        text = re.sub(r'\([^가-힣a-zA-Z0-9\s]*\)', '', text)
        
        # 5. 약어 복원
        for k, v in WL.items():
            text = text.replace(v, k)
        
        # 6. 공백 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    # ----------------------------
    # 프롬프트 & 생성
    # ----------------------------
    def _build_prompt(self, user_message: str) -> str:
        return f"""### System:
{self.system_prompt}

### User:
{user_message}

### Assistant:"""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        stream: bool = False,
        temperature: float = 0.7,  # do_sample=False일 땐 무시됨
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        ko_only: bool = True,
        **kwargs
    ) -> str:
        """텍스트 생성 - 한국어 전용"""
        if self.model is None:
            self.load()

        # 기본 토큰 길이 (속도/안정 타협점)
        if max_new_tokens is None:
            max_new_tokens = int(os.getenv("KXN_MAX_NEW_TOKENS", "320"))

        full_prompt = self._build_prompt(prompt)

        # 토크나이징
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(os.getenv("KXN_MAX_PROMPT_TOKENS", "2048")),
        )

        # 모델 디바이스로 이동
        if torch.cuda.is_available():
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,          # 기본: 그리디            
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": float(os.getenv("KXN_REP_PENALTY", "1.2")),
        }
        if do_sample:
            gen_kwargs.update({
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        })

        # 한국어 외 스크립트 토큰 금지
        if ko_only:
            gen_kwargs["bad_words_ids"] = self._build_bad_words_ids()

        # 생성
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM during generate() — retry with shorter length")
            torch.cuda.empty_cache()
            gen_kwargs["max_new_tokens"] = min(256, max_new_tokens)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

        # 디코딩 (프롬프트 부분 제외)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 방어적 프롬프트 잔여 태그 제거
        for delimiter in ("###", "User:", "System:", "Assistant:"):
            if delimiter in response:
                response = response.split(delimiter)[0].strip()

        # 최종 한국어 강제 후처리
        response = self._force_korean(response)

        # 너무 짧거나 비한글 과다일 경우 템플릿 응답
        if len(response) < 10 or self._non_korean_ratio(response) > 0.05:
            response = self._template_response(prompt)

        logger.info(f"Generated in {time.time() - start_time:.2f}s")
        return response

    # ----------------------------
    # 품질 보정
    # ----------------------------
    def _non_korean_ratio(self, text: str) -> float:
        if not text:
            return 1.0
        # 한글(가-힣), 숫자, 기호 제외 문자의 비율
        total = len(re.sub(r"\s", "", text))
        non_ko = len(re.findall(r"[A-Za-z\u0400-\u04FF\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]", text))
        return (non_ko / total) if total else 1.0

    def _template_response(self, prompt: str) -> str:
        p = prompt.lower()
        if "oee" in p or "종합설비효율" in p:
            return (
                "종합설비효율(OEE)은 가용성×성능×품질로 계산합니다. "
                "① 가용성: 고장·대기 시간을 줄여 계획 대비 실제 가동시간을 늘립니다. "
                "② 성능: 병목 해소·표준사이클 준수로 이상 생산속도에 근접합니다. "
                "③ 품질: 공정·최종 검사 고도화와 원인분석으로 양품률을 높입니다. "
                "예지보전, 실시간 모니터링, 데이터 기반 개선으로 세 요소를 체계적으로 개선하세요."
            )
        if "정비" in p or "예지" in p:
            return (
                "예지보전은 센서 데이터(진동·온도 등)를 상시 수집·학습하여 고장을 사전 감지하고, "
                "가동 중단을 최소화하는 정비 방식입니다. 고장 확률과 잔여수명을 추정하여 최적 정비시점을 "
                "결정하고, 필요 자재·인력을 사전에 준비해 비용과 다운타임을 줄입니다."
            )
        if "품질" in p or "불량" in p:
            return (
                "품질 개선은 실시간 검사, 통계적 공정관리(관리도), 근본원인분석(파레토·특성요인도), "
                "표준작업·교육 강화가 핵심입니다. 공정변동을 줄이고 재발방지 대책을 문서화해 지속 개선하세요."
            )
        return (
            "스마트팩토리·생산·품질·설비 관리 전반을 한국어로 지원합니다. "
            "구체적인 현황이나 목표를 알려주시면 맞춤형 실행조치를 제안드리겠습니다."
        )

    # ----------------------------
    # 모니터링
    # ----------------------------
    def get_memory_usage(self) -> Dict[str, Any]:
        """GPU 메모리 사용량 조회"""
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
                "usage_percent": round((mem_allocated / mem_total) * 100, 1),
            }
        return memory_stats
