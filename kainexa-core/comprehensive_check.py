#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')
import sys

print("="*60)
print("Kainexa AI Platform - Complete Environment Check")
print("="*60)
print(f"Python: {sys.version}\n")

# 체크할 패키지 목록
packages = {
    'Core ML': {
        'numpy': '1.24.3',
        'scipy': '1.10.1',
        'pandas': None,
        'scikit-learn': '1.3.0'
    },
    'PyTorch': {
        'torch': '2.1.0',
        'torchvision': None,
        'torchaudio': None
    },
    'Hugging Face': {
        'transformers': '4.35.2',
        'tokenizers': '0.15.0',
        'datasets': '2.14.5',
        'huggingface_hub': '0.19.4',
        'accelerate': '0.24.1',
        'peft': '0.6.0',
        'safetensors': None
    },
    'Deep Learning': {
        'deepspeed': '0.12.0',
        'bitsandbytes': None,
    },
    'Korean NLP': {
        'konlpy': None,
        'kiwipiepy': None,
        'sentencepiece': None
    },
    'Database': {
        'pyarrow': '11.0.0',
        'qdrant_client': None,
        'redis': None,
        'sqlalchemy': None
    }
}

# 패키지 체크
for category, pkg_dict in packages.items():
    print(f"\n{category}:")
    print("-" * 40)
    for package, expected_version in pkg_dict.items():
        try:
            mod = __import__(package.replace('_', '-').replace('-', '_'))
            actual_version = getattr(mod, '__version__', 'unknown')
            
            if expected_version and actual_version != expected_version:
                status = f"⚠️  {actual_version} (expected {expected_version})"
            else:
                status = f"✅ {actual_version}"
            
            print(f"  {package:20s}: {status}")
        except ImportError as e:
            print(f"  {package:20s}: ❌ Not installed")

# GPU/CUDA 상태
print("\n" + "="*40)
print("GPU/CUDA Status:")
print("-" * 40)

try:
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name}")
            print(f"         Memory: {memory_gb:.1f} GB")
            print(f"         Compute Capability: {props.major}.{props.minor}")
except Exception as e:
    print(f"Error checking CUDA: {e}")

# 시스템 정보
print("\n" + "="*40)
print("System Information:")
print("-" * 40)

import platform
import os

print(f"OS: {platform.system()} {platform.release()}")
print(f"Architecture: {platform.machine()}")
print(f"CPU Cores: {os.cpu_count()}")

try:
    import psutil
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
except:
    pass

print("\n" + "="*60)
print("✨ Environment check completed!")
print("="*60)
