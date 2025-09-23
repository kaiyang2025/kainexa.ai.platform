# scripts/init_database.py 생성

#!/usr/bin/env python3
"""
데이터베이스 초기화
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from src.core.config import settings
from src.core.models import Base

async def init_database():
    """데이터베이스 초기화"""
    
    print("="*60)
    print("데이터베이스 초기화")
    print("="*60)
    
    # 1. SQL 파일 실행
    print("\n1. SQL 스키마 생성...")
    
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="kainexa",
        password="password",
        database="kainexa_db"
    )
    
    # SQL 파일 읽기 및 실행
    sql_file = Path("migrations/001_create_tables.sql")
    if sql_file.exists():
        sql_content = sql_file.read_text()
        await conn.execute(sql_content)
        print("   ✅ SQL 스키마 생성 완료")
    else:
        print("   ⚠️ SQL 파일이 없습니다. SQLAlchemy로 생성합니다.")
        
        # SQLAlchemy로 테이블 생성
        engine = create_async_engine(settings.DATABASE_URL, echo=True)
        
        async with engine.begin() as conn_alchemy:
            await conn_alchemy.run_sync(Base.metadata.create_all)
        
        await engine.dispose()
        print("   ✅ SQLAlchemy 스키마 생성 완료")
    
    # 2. 초기 데이터 삽입
    print("\n2. 초기 데이터 삽입...")
    
    # 테스트 사용자 생성
    await conn.execute("""
        INSERT INTO users (username, email, full_name, role, department)
        VALUES 
            ('admin', 'admin@kainexa.ai', '시스템 관리자', 'admin', 'IT'),
            ('kim_manager', 'kim@company.com', '김부장', 'manager', '생산관리팀'),
            ('lee_staff', 'lee@company.com', '이대리', 'user', '품질관리팀')
        ON CONFLICT (username) DO NOTHING;
    """)
    
    # 기본 워크플로우 저장
    await conn.execute("""
        INSERT INTO workflows (name, description, dsl_content, version)
        VALUES 
            ('production_monitoring', '생산 모니터링 워크플로우', 
             '{"name": "production_monitoring", "steps": []}', '1.0'),
            ('quality_control', '품질 관리 워크플로우',
             '{"name": "quality_control", "steps": []}', '1.0')
        ON CONFLICT DO NOTHING;
    """)
    
    print("   ✅ 초기 데이터 삽입 완료")
    
    # 3. 통계 확인
    print("\n3. 데이터베이스 통계:")
    
    tables = [
        'users', 'sessions', 'conversations', 'messages',
        'knowledge_documents', 'audit_logs', 'workflows'
    ]
    
    for table in tables:
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
        print(f"   - {table}: {count}개")
    
    await conn.close()
    
    print("\n✅ 데이터베이스 초기화 완료!")

if __name__ == "__main__":
    asyncio.run(init_database())