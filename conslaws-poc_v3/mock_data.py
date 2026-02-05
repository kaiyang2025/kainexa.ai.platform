# mock_data.py
import pandas as pd

def get_project_list():
    return ["강원대 도서관 공기지연 건", "OO 아파트 추가공사비 분쟁", "세종시 복합단지 클레임"]

def get_risk_data(project_name):
    return {
        "score": 85 if "강원대" in project_name else 45,
        "status": "High Risk" if "강원대" in project_name else "Stable",
        "missing_docs": ["7월 16일 작업일보", "도로폐쇄 통보문"] if "강원대" in project_name else []
    }

def get_timeline_data(project_name):
    return [
        {"date": "2024-07-10", "event": "호우주의보 발령", "type": "Fact", "status": "Verified"},
        {"date": "2024-07-12", "event": "현장 침수 피해", "type": "Fact", "status": "Verified"},
        {"date": "2024-07-15", "event": "자재 반입 지연", "type": "Gap", "status": "Missing"},
        {"date": "2024-07-21", "event": "작업 재개 승인", "type": "Notice", "status": "Verified"},
    ]

def get_templates():
    return ["발주처 제출용 (EOT 청구)", "대한상사중재원 제출용", "내부 보고용 리포트"]