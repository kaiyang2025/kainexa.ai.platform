"""
데이터베이스 초기화
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

async def init_database():
    """데이터베이스 초기화"""
    
    print("="*60)
    print("데이터베이스 초기화")
    print("="*60)
    
    # asyncpg로 직접 연결
    import asyncpg
    
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
        
        # 2. 테이블 생성
        print("\n2. 테이블 생성...")
        
        # SQL 파일 확인
        sql_file = Path("migrations/001_create_tables.sql")
        if sql_file.exists():
            print("   SQL 파일 실행...")
            sql_content = sql_file.read_text()
            
            # SQL을 개별 명령으로 분리 (세미콜론 기준)
            sql_commands = [cmd.strip() for cmd in sql_content.split(';') if cmd.strip()]
            
            for i, cmd in enumerate(sql_commands, 1):
                if cmd and not cmd.startswith('--'):
                    try:
                        await conn.execute(cmd)
                        print(f"   ✅ 명령 {i}/{len(sql_commands)} 실행 완료")
                    except Exception as e:
                        if "already exists" in str(e):
                            print(f"   ⚠️ 이미 존재함 (스킵)")
                        else:
                            print(f"   ❌ 오류: {e}")
        else:
            # SQL 파일이 없으면 기본 테이블만 생성
            print("   SQL 파일이 없습니다. 기본 테이블 생성...")
            
            # Users 테이블
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
            
            # Sessions 테이블
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
            
            # Conversations 테이블
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
            
            # Messages 테이블
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
            
            print("   ✅ 기본 테이블 생성 완료")
        
        # 3. 초기 데이터 삽입
        print("\n3. 초기 데이터 삽입...")
        
        # 테스트 사용자 생성
        try:
            await conn.execute("""
                INSERT INTO users (username, email, full_name, role, department)
                VALUES 
                    ('admin', 'admin@kainexa.ai', '시스템 관리자', 'admin', 'IT'),
                    ('kim_manager', 'kim@company.com', '김부장', 'manager', '생산관리팀'),
                    ('lee_staff', 'lee@company.com', '이대리', 'user', '품질관리팀')
                ON CONFLICT (username) DO NOTHING;
            """)
            print("   ✅ 사용자 데이터 삽입 완료")
        except Exception as e:
            print(f"   ⚠️ 사용자 데이터 삽입 스킵: {e}")
        
        # 4. 통계 확인
        print("\n4. 데이터베이스 통계:")
        
        # 테이블 목록 확인
        tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """
        
        tables = await conn.fetch(tables_query)
        
        for table_row in tables:
            table_name = table_row['table_name']
            try:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                print(f"   - {table_name}: {count}개")
            except Exception as e:
                print(f"   - {table_name}: 조회 실패")
        
        await conn.close()
        
        print("\n✅ 데이터베이스 초기화 완료!")
        
    except asyncpg.exceptions.InvalidPasswordError:
        print("❌ 데이터베이스 연결 실패: 비밀번호가 틀렸습니다.")
        print("   docker-compose.yml의 POSTGRES_PASSWORD를 확인하세요.")
    except asyncpg.exceptions.InvalidCatalogNameError:
        print("❌ 데이터베이스가 존재하지 않습니다.")
        print("   다음 명령어로 데이터베이스를 생성하세요:")
        print("   docker exec -it kainexa-postgres psql -U kainexa -c 'CREATE DATABASE kainexa_db;'")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(init_database())