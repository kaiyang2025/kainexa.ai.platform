#!/usr/bin/env python3
"""
데이터베이스 초기화
"""
import asyncio
import sys
from pathlib import Path
import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent))

async def create_tables_directly(conn):
    """테이블을 직접 생성"""
    
    # 1. Users 테이블
    print("   Creating users table...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            full_name VARCHAR(255),
            role VARCHAR(50) DEFAULT 'user',
            department VARCHAR(100),
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ users table created")
    
    # 2. Sessions 테이블
    print("   Creating sessions table...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            session_token VARCHAR(500) UNIQUE NOT NULL,
            ip_address INET,
            user_agent TEXT,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ sessions table created")
    
    # 3. Conversations 테이블
    print("   Creating conversations table...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID,
            user_id UUID,
            title VARCHAR(255),
            context JSONB DEFAULT '{}',
            status VARCHAR(50) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ conversations table created")
    
    # 4. Messages 테이블
    print("   Creating messages table...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            tokens INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ messages table created")
    
    # 5. Knowledge Documents 테이블
    print("   Creating knowledge_documents table...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title VARCHAR(255) NOT NULL,
            source VARCHAR(500),
            content TEXT,
            file_path VARCHAR(500),
            file_type VARCHAR(50),
            file_size INTEGER,
            checksum VARCHAR(64),
            access_level VARCHAR(50) DEFAULT 'internal',
            tags TEXT[],
            metadata JSONB DEFAULT '{}',
            quality_score FLOAT DEFAULT 0.0,
            usage_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
    """)
    print("   ✅ knowledge_documents table created")
    
    # 6. Knowledge Chunks 테이블
    print("   Creating knowledge_chunks table...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding_id VARCHAR(100),
            start_char INTEGER,
            end_char INTEGER,
            tokens INTEGER,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ knowledge_chunks table created")
    
    # 7. Audit Logs 테이블
    print("   Creating audit_logs table...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID,
            session_id UUID,
            action VARCHAR(100) NOT NULL,
            resource_type VARCHAR(50),
            resource_id VARCHAR(100),
            details JSONB DEFAULT '{}',
            ip_address INET,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ audit_logs table created")
    
    # 8. Analytics Events 테이블
    print("   Creating analytics_events table...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS analytics_events (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            event_type VARCHAR(100) NOT NULL,
            user_id UUID,
            session_id UUID,
            conversation_id UUID,
            properties JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ analytics_events table created")
    
    # 9. Model Configurations 테이블
    print("   Creating model_configurations table...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS model_configurations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50),
            version VARCHAR(20),
            configuration JSONB NOT NULL,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ model_configurations table created")
    
    # 10. Workflows 테이블
    print("   Creating workflows table...")
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS workflows (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            dsl_content TEXT NOT NULL,
            version VARCHAR(20),
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("   ✅ workflows table created")

async def create_indexes(conn):
    """인덱스 생성"""
    print("\n3. 인덱스 생성...")
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)",
        "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_documents_access_level ON knowledge_documents(access_level)",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_id ON knowledge_chunks(document_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at)"
    ]
    
    for idx_sql in indexes:
        try:
            await conn.execute(idx_sql)
            print(f"   ✅ Index created: {idx_sql.split('idx_')[1].split(' ')[0]}")
        except Exception as e:
            print(f"   ⚠️ Index skipped: {e}")

async def insert_initial_data(conn):
    """초기 데이터 삽입"""
    print("\n4. 초기 데이터 삽입...")
    
    # 테스트 사용자
    try:
        await conn.execute("""
            INSERT INTO users (username, email, full_name, role, department)
            VALUES 
                ('admin', 'admin@kainexa.ai', '시스템 관리자', 'admin', 'IT'),
                ('kim_manager', 'kim@company.com', '김부장', 'manager', '생산관리팀'),
                ('lee_staff', 'lee@company.com', '이대리', 'user', '품질관리팀'),
                ('park_engineer', 'park@company.com', '박과장', 'engineer', '기술팀')
            ON CONFLICT (username) DO NOTHING
        """)
        print("   ✅ Users data inserted")
    except Exception as e:
        print(f"   ⚠️ Users data: {e}")
    
    # 기본 워크플로우
    try:
        await conn.execute("""
            INSERT INTO workflows (name, description, dsl_content, version, is_active)
            VALUES 
                ('production_monitoring', '생산 모니터링 워크플로우', 
                 '{"name": "production_monitoring", "steps": [{"step": "collect_data"}, {"step": "analyze"}, {"step": "report"}]}', 
                 '1.0', true),
                ('quality_control', '품질 관리 워크플로우',
                 '{"name": "quality_control", "steps": [{"step": "inspect"}, {"step": "classify"}, {"step": "action"}]}', 
                 '1.0', true),
                ('predictive_maintenance', '예측적 유지보수 워크플로우',
                 '{"name": "predictive_maintenance", "steps": [{"step": "monitor"}, {"step": "predict"}, {"step": "schedule"}]}',
                 '1.0', true)
            ON CONFLICT DO NOTHING
        """)
        print("   ✅ Workflows data inserted")
    except Exception as e:
        print(f"   ⚠️ Workflows data: {e}")
    
    # 모델 설정
    try:
        await conn.execute("""
            INSERT INTO model_configurations (model_name, model_type, version, configuration, is_active)
            VALUES 
                ('solar-10.7b', 'llm', '1.0', 
                 '{"model_path": "models/solar-10.7b", "device": "cuda", "load_in_8bit": true}', 
                 true),
                ('sentence-transformers', 'embedding', '1.0',
                 '{"model_name": "sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens", "device": "cuda"}',
                 true)
            ON CONFLICT DO NOTHING
        """)
        print("   ✅ Model configurations inserted")
    except Exception as e:
        print(f"   ⚠️ Model configurations: {e}")

async def init_database():
    """데이터베이스 초기화 메인 함수"""
    
    print("="*60)
    print("Kainexa 데이터베이스 초기화")
    print("="*60)
    
    try:
        # 데이터베이스 연결
        print("\n1. 데이터베이스 연결...")
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="kainexa",
            password="password",
            database="kainexa_db"
        )
        print("   ✅ 데이터베이스 연결 성공")
        
        # 테이블 생성
        print("\n2. 테이블 생성...")
        await create_tables_directly(conn)
        
        # 인덱스 생성
        await create_indexes(conn)
        
        # 초기 데이터 삽입
        await insert_initial_data(conn)
        
        # 통계 확인
        print("\n5. 데이터베이스 통계:")
        tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """
        
        tables = await conn.fetch(tables_query)
        
        for table_row in tables:
            table_name = table_row['table_name']
            try:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                print(f"   📊 {table_name}: {count} rows")
            except Exception:
                print(f"   ⚠️ {table_name}: 조회 실패")
        
        await conn.close()
        
        print("\n" + "="*60)
        print("✅ 데이터베이스 초기화 완료!")
        print("="*60)
        
    except asyncpg.exceptions.InvalidPasswordError:
        print("❌ 데이터베이스 연결 실패: 비밀번호가 틀렸습니다.")
        print("   docker-compose.yml의 POSTGRES_PASSWORD를 확인하세요.")
    except asyncpg.exceptions.InvalidCatalogNameError:
        print("❌ 데이터베이스가 존재하지 않습니다.")
        print("   먼저 데이터베이스를 생성하세요:")
        print("   docker exec -it kainexa-postgres psql -U kainexa -c 'CREATE DATABASE kainexa_db;'")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(init_database())