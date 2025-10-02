# -*- coding: utf-8 -*-
"""
ModelFactory (test-safe, pluggable)

- 테스트/수집 단계에서 무거운 의존성(torch/transformers)이 없어도 안전하도록 지연 임포트.
- 기본값은 EchoChatModel 로 안전하게 동작 (GPU/네트워크 불필요).
- 운영 시 "solar", "lightweight", "openai" 등을 환경변수/설정으로 활성화.

사용:
    from src.core.models.model_factory import ModelFactory
    model = ModelFactory.get_model("echo")  # 또는 "solar" / "openai" / "lightweight"
    text = await model.generate("Hello")
    text = await model.chat([{"role":"user","content":"Hi"}])
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import os

# ---------------------------------------------------------
# Base Interface
# ---------------------------------------------------------
class BaseModel:
    async def generate(self, prompt: str, **kwargs) -> str:  # abstract-like
        raise NotImplementedError

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        messages: [{"role":"system|user|assistant", "content":"..."}]
        """
        # 기본 구현: 가장 최근 user 메시지 추출 후 generate
        last_user = next((m["content"] for m in reversed(messages)
                          if m.get("role") == "user"), "")
        return await self.generate(last_user, **kwargs)


# ---------------------------------------------------------
# Safe default: Echo model (no external deps)
# ---------------------------------------------------------
class EchoChatModel(BaseModel):
    def __init__(self, name: str = "echo", **_):
        self.name = name

    async def generate(self, prompt: str, **kwargs) -> str:
        await asyncio.sleep(0)  # cooperative
        return f"[{self.name}] {prompt}".strip()


# ---------------------------------------------------------
# Optional backends (lazy imports)
# ---------------------------------------------------------
class SolarLLM(BaseModel):
    """
    Upstage SOLAR 10.7B Instruct wrapper (lazy import).
    실제 사용 시 transformers/torch 설치 및 모델 로컬/원격 서빙 필요.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = {**{"device": "cuda:0", "tensor_parallel": False}, **(config or {})}
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        # 지연 임포트
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        model_name = self.config.get("model_name", "upstage/solar-10.7b-instruct")
        device = self.config.get("device", "cuda:0")

        # 텐서 병렬 분기는 실제 구현에 맞게 확장
        if self.config.get("tensor_parallel"):
            # TODO: DeepSpeed / vLLM 등과 연계 구현
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, "float16", None),
            device_map="auto",
        )
        self.device = device
        self._loaded = True

    async def generate(self, prompt: str, **kwargs) -> str:
        self._lazy_load()
        import torch  # type: ignore

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', 256),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=True
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 프롬프트 제거(간단 처리)
        return response.replace(prompt, "", 1).strip()


class LightweightLLM(BaseModel):
    """
    경량 한국어 모델 래퍼(예: Polyglot 5.8B). 지연 임포트.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.device = self.config.get("device", "cuda:0")
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        model_name = self.config.get("model_name", "EleutherAI/polyglot-ko-5.8b")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, "float16", None),
            device_map="auto",
        )
        self._loaded = True

    async def generate(self, prompt: str, **kwargs) -> str:
        self._lazy_load()
        import torch  # type: ignore
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', 256),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=True
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "", 1).strip()


class OpenAIAdapter(BaseModel):
    """
    OpenAI 어댑터(테스트 친화 보호장치 포함).
    - API 키가 없거나 네트워크 불가 환경이면 Echo로 폴백.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = self.config.get("model", "gpt-4o-mini")

        # 키가 없으면 안전 폴백
        if not self.api_key:
            self._fallback = EchoChatModel("openai-fallback")
        else:
            self._fallback = None

    async def generate(self, prompt: str, **kwargs) -> str:
        if self._fallback:
            return await self._fallback.generate(prompt, **kwargs)

        # 최신 openai 패키지 기준 예시 (실 서비스에서 조정)
        try:
            import openai  # type: ignore
            client = openai.OpenAI(api_key=self.api_key)  # type: ignore
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 256),
            )
            return resp.choices[0].message.content or ""
        except Exception:
            # 어떤 예외든 조용히 에코 폴백
            return await EchoChatModel("openai-error-fallback").generate(prompt, **kwargs)


# ---------------------------------------------------------
# Factory
# ---------------------------------------------------------
class ModelFactory:
    _cache: Dict[str, BaseModel] = {}

    @classmethod
    def get_model(cls, model_type: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        공통 진입점. 기본값은 'echo'.
        """
        name = (model_type or os.getenv("DEFAULT_CHAT_MODEL") or "echo").lower()
        key = f"{name}:{hash(str(sorted((config or {}).items())))}"

        if key in cls._cache:
            return cls._cache[key]

        cfg = config or {}
        if name in ("echo", "test", "debug"):
            inst = EchoChatModel(name)
        elif name in ("solar", "upstage-solar"):
            inst = SolarLLM(cfg)
        elif name in ("lightweight", "polyglot"):
            inst = LightweightLLM(cfg)
        elif name in ("openai", "gpt"):
            inst = OpenAIAdapter(cfg)
        else:
            # 알 수 없는 이름은 안전 폴백
            inst = EchoChatModel(name)

        cls._cache[key] = inst
        return inst

    # 호환성: 기존 코드가 create_model()을 호출하더라도 동작하도록 별칭 제공
    @classmethod
    def create_model(cls, model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        return cls.get_model(model_type, config)

    @classmethod
    def get_available_models(cls) -> List[str]:
        return ["echo", "solar", "lightweight", "openai"]
