#!/usr/bin/env python3

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 경로 추가
sys.path.insert(0, os.getcwd())

print("Testing Kainexa module imports...")
print("-" * 40)

modules = [
    ('src.core.config', 'Settings'),
    ('src.core.database', 'get_db'),
    ('src.api.main', 'app'),
    ('src.models.model_factory', 'ModelFactory'),
    ('src.orchestration.graph_executor', 'GraphExecutor'),
]

for module_path, attr in modules:
    try:
        mod = __import__(module_path, fromlist=[attr])
        if hasattr(mod, attr):
            print(f"✅ {module_path}.{attr}")
        else:
            print(f"⚠️  {module_path} (missing {attr})")
    except ImportError as e:
        print(f"❌ {module_path}: {e}")
    except Exception as e:
        print(f"❌ {module_path}: {e}")

print("-" * 40)
print("Import test completed!")
