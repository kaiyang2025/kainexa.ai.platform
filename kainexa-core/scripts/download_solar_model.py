# scripts/download_solar_model.py 생성
#!/usr/bin/env python3
"""
Solar-10.7B 모델 다운로드 및 설정
"""
import os
import sys
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_solar_model():
    """Solar-10.7B 모델 다운로드"""
    
    model_id = "upstage/SOLAR-10.7B-Instruct-v1.0"  #upstage/solar-10.7b-instruct
    cache_dir = Path("models/solar-10.7b")
    
    print("="*60)
    print("Solar-10.7B 모델 다운로드")
    print("="*60)
    print(f"모델: {model_id}")
    print(f"저장 경로: {cache_dir}")
    print("")
    
    # 디렉토리 생성
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 모델 다운로드 (약 20GB)
        print("⬇️  모델 다운로드 중... (약 20GB, 시간이 걸립니다)")
        
        # HuggingFace Hub에서 다운로드
        snapshot_download(
            repo_id=model_id,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            resume_download=True,  # 중단된 다운로드 재개
            ignore_patterns=["*.onnx", "*.msgpack"]  # 불필요한 파일 제외
        )
        
        print("✅ 모델 다운로드 완료!")
        
        # 모델 파일 확인
        model_files = list(cache_dir.glob("*.safetensors"))
        if model_files:
            total_size = sum(f.stat().st_size for f in model_files) / (1024**3)
            print(f"📦 모델 크기: {total_size:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return False

def test_model_loading():
    """모델 로딩 테스트"""
    
    print("\n" + "="*60)
    print("모델 로딩 테스트")
    print("="*60)
    
    model_path = Path("models/solar-10.7b")
    
    if not model_path.exists():
        print("❌ 모델이 다운로드되지 않았습니다. 먼저 다운로드를 실행하세요.")
        return False
    
    try:
        print("🔄 토크나이저 로딩...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✅ 토크나이저 로드 성공")
        
        print("\n🔄 모델 로딩 (INT8 양자화)...")
        
        # GPU 메모리 절약을 위해 INT8 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,  # INT8 양자화
            trust_remote_code=True
        )
        
        print("✅ 모델 로드 성공")
        
        # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_used = torch.cuda.memory_allocated(i) / (1024**3)
                mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"🎮 GPU {i}: {mem_used:.1f}/{mem_total:.1f} GB 사용")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return False

if __name__ == "__main__":
    # 1. 모델 다운로드
    if "--download" in sys.argv or not Path("models/solar-10.7b").exists():
        success = download_solar_model()
        if not success:
            sys.exit(1)
    
    # 2. 로딩 테스트
    if "--test" in sys.argv:
        test_model_loading()