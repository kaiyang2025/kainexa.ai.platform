# conftest.py (프로젝트 루트)
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # 'src' 패키지 루트가 경로에 올라갑니다.