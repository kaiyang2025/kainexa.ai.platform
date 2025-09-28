# scripts/upload_documents.py 생성
#!/usr/bin/env python3
"""
문서 업로드 및 인덱싱
"""
import sys
import asyncio
from pathlib import Path
from typing import List
import PyPDF2
import docx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.governance.rag_pipeline import RAGGovernance, DocumentMetadata, AccessLevel
from datetime import datetime

class DocumentUploader:
    """문서 업로드 관리자"""
    
    def __init__(self):
        self.rag = RAGGovernance(
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="kainexa_knowledge"
        )
        
    async def upload_pdf(self, file_path: Path) -> bool:
        """PDF 문서 업로드"""
        try:
            # PDF 텍스트 추출
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # 메타데이터 생성
            metadata = DocumentMetadata(
                doc_id=f"pdf_{file_path.stem}",
                title=file_path.stem,
                source=str(file_path),
                created_at=datetime.now(),
                access_level=AccessLevel.INTERNAL,
                tags=["pdf", "manufacturing"],
                language="ko"
            )
            
            # RAG에 추가
            success = await self.rag.add_document(text, metadata)
            
            if success:
                print(f"✅ PDF 업로드 성공: {file_path.name}")
            else:
                print(f"❌ PDF 업로드 실패: {file_path.name}")
                
            return success
            
        except Exception as e:
            print(f"❌ PDF 처리 오류: {e}")
            return False
    
    async def upload_docx(self, file_path: Path) -> bool:
        """DOCX 문서 업로드"""
        try:
            # DOCX 텍스트 추출
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # 메타데이터 생성
            metadata = DocumentMetadata(
                doc_id=f"docx_{file_path.stem}",
                title=file_path.stem,
                source=str(file_path),
                created_at=datetime.now(),
                access_level=AccessLevel.INTERNAL,
                tags=["docx", "manufacturing"],
                language="ko"
            )
            
            # RAG에 추가
            success = await self.rag.add_document(text, metadata)
            
            if success:
                print(f"✅ DOCX 업로드 성공: {file_path.name}")
            else:
                print(f"❌ DOCX 업로드 실패: {file_path.name}")
                
            return success
            
        except Exception as e:
            print(f"❌ DOCX 처리 오류: {e}")
            return False
    
    async def upload_text(self, file_path: Path) -> bool:
        """텍스트 문서 업로드"""
        try:
            # 텍스트 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 메타데이터 생성
            metadata = DocumentMetadata(
                doc_id=f"txt_{file_path.stem}",
                title=file_path.stem,
                source=str(file_path),
                created_at=datetime.now(),
                access_level=AccessLevel.PUBLIC,
                tags=["text", "manual"],
                language="ko"
            )
            
            # RAG에 추가
            success = await self.rag.add_document(text, metadata)
            
            if success:
                print(f"✅ TXT 업로드 성공: {file_path.name}")
            else:
                print(f"❌ TXT 업로드 실패: {file_path.name}")
                
            return success
            
        except Exception as e:
            print(f"❌ TXT 처리 오류: {e}")
            return False
    
    async def upload_directory(self, dir_path: Path):
        """디렉토리 내 모든 문서 업로드"""
        
        print(f"📁 디렉토리 스캔: {dir_path}")
        
        # 지원 파일 확장자
        supported_extensions = {
            '.pdf': self.upload_pdf,
            '.docx': self.upload_docx,
            '.txt': self.upload_text,
            '.md': self.upload_text
        }
        
        uploaded = 0
        failed = 0
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                if ext in supported_extensions:
                    success = await supported_extensions[ext](file_path)
                    if success:
                        uploaded += 1
                    else:
                        failed += 1
        
        print(f"\n📊 업로드 완료: 성공 {uploaded}개, 실패 {failed}개")

async def main():
    """메인 실행"""
    uploader = DocumentUploader()
    
    # 샘플 문서 디렉토리
    docs_dir = Path("data/sample_docs")
    
    if not docs_dir.exists():
        print(f"❌ 디렉토리가 없습니다: {docs_dir}")
        print("샘플 문서를 생성합니다...")
        
        # 샘플 문서 생성
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # 샘플 텍스트 파일들
        sample_docs = {
            "smart_factory_guide.txt": """스마트 팩토리 구축 가이드
            
1. OEE (Overall Equipment Effectiveness) 개선
   - 가용성(Availability): 계획된 생산 시간 대비 실제 가동 시간
   - 성능(Performance): 이론적 생산량 대비 실제 생산량
   - 품질(Quality): 전체 생산량 대비 양품 비율
   
2. 예측적 유지보수
   - IoT 센서를 통한 실시간 모니터링
   - 머신러닝 기반 고장 예측
   - 최적 정비 시점 결정
   
3. 품질 관리 자동화
   - 비전 검사 시스템 도입
   - 실시간 불량 감지
   - 자동 분류 및 처리
""",
            "production_manual.txt": """생산 관리 매뉴얼

1. 생산 계획
   - 수요 예측 기반 생산 계획 수립
   - 자재 소요량 계산 (MRP)
   - 생산 일정 최적화
   
2. 공정 관리
   - 작업 지시서 발행
   - 실시간 생산 모니터링
   - 병목 공정 관리
   
3. 재고 관리
   - JIT (Just-In-Time) 방식 적용
   - 안전 재고 수준 유지
   - ABC 분석을 통한 중요도 관리
""",
            "quality_control.txt": """품질 관리 시스템

1. 품질 검사
   - 수입 검사: 원자재 품질 확인
   - 공정 검사: 생산 과정 중 품질 확인
   - 출하 검사: 최종 제품 품질 확인
   
2. 통계적 품질 관리
   - 관리도(Control Chart) 활용
   - Cpk 지수 관리 (목표: 1.33 이상)
   - 6시그마 기법 적용
   
3. 불량 분석
   - 파레토 차트를 통한 주요 불량 파악
   - 특성요인도(Fishbone Diagram) 작성
   - 근본 원인 분석(RCA)
"""
        }
        
        for filename, content in sample_docs.items():
            file_path = docs_dir / filename
            file_path.write_text(content, encoding='utf-8')
            print(f"✅ 샘플 문서 생성: {filename}")
    
    # 문서 업로드
    await uploader.upload_directory(docs_dir)

if __name__ == "__main__":
    asyncio.run(main())