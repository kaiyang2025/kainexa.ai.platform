# ============================================
# src/utils/logger.py - 통합 로거
# ============================================
"""src/utils/logger.py"""
import logging
import sys
from pathlib import Path
import structlog
from datetime import datetime

# 로그 디렉토리 생성
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

def setup_logging():
    """로깅 설정"""
    
    # Structlog 설정
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 기본 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"app_{datetime.now():%Y%m%d}.log")
        ]
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """로거 인스턴스 생성"""
    return structlog.get_logger(name)

# 초기화
setup_logging()