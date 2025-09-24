#!/usr/bin/env python3
"""
데이터베이스 정리
"""
import asyncio
import asyncpg

async def clean_database():
    print("데이터베이스 정리 중...")
    
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="kainexa",
        password="password",
        database="kainexa_db"
    )
    
    # 모든 테이블 삭제 (CASCADE)
    tables_to_drop = [
        'analytics_events',
        'audit_logs',
        'knowledge_chunks',
        'knowledge_documents',
        'messages',
        'conversations',
        'sessions',
        'workflows',
        'model_configurations',
        'users'
    ]
    
    for table in tables_to_drop:
        try:
            await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            print(f"   ✅ Dropped table: {table}")
        except Exception as e:
            print(f"   ⚠️ Error dropping {table}: {e}")
    
    await conn.close()
    print("✅ 정리 완료!")

if __name__ == "__main__":
    response = input("정말로 모든 테이블을 삭제하시겠습니까? (yes/no): ")
    if response.lower() == 'yes':
        asyncio.run(clean_database())
    else:
        print("취소되었습니다.")
