# mock_data.py
import pandas as pd

# [NEW] 리스크 대시보드 데이터
def get_risk_metrics():
    return {
        "overall_risk": "High",
        "win_probability": 85,
        "missing_evidences": 2,
        "total_claim_amount": "₩2.4B"
    }
    
# [NEW] 코파일럿 대화 데이터
def get_chat_history():
    return [
        {"role": "assistant", "content": "안녕하세요, Kainexa Copilot입니다. 강원대 도서관 프로젝트의 '7월 공기지연' 건에 대해 분석을 완료했습니다. 어떤 도움이 필요하신가요?"},
        {"role": "user", "content": "발주처가 공기연장(EOT)을 거부할 경우, 가장 강력한 반박 논리는 무엇인가요?"},
        {"role": "assistant", "content": "도급계약서 제25조와 기상 데이터를 결합했을 때, '도로 침수로 인한 자재 반입 불가'가 가장 강력한 면책 사유입니다. 이를 바탕으로 반박 서면 초안을 작성해 드릴까요?"}
    ]

# [NEW] 시각적 타임라인 데이터
def get_visual_timeline():
    return [
        {"date": "07-10", "title": "호우주의보 발령", "status": "Done", "desc": "기상청 데이터 연동 완료", "color": "#28a745"},
        {"date": "07-12", "title": "현장 침수 피해", "status": "Done", "desc": "현장 사진 4장 매핑", "color": "#28a745"},
        {"date": "07-15", "title": "자재 반입 지연", "status": "Risk", "desc": "증거 문서 누락됨 (입증 필요)", "color": "#dc3545"},
        {"date": "07-21", "title": "작업 재개", "status": "Done", "desc": "공문 Out-101 확인", "color": "#28a745"},
    ]    
    
# 1. 초기 타임라인 (증거 공백 존재)
def get_initial_timeline():
    return [
        {"date": "2024-07-10", "event": "기상청 호우주의보 발령", "source": "기상청 데이터", "type": "Fact"},
        {"date": "2024-07-12", "event": "현장 침수 및 장비 중단", "source": "현장사진첩_0712", "type": "Fact"},
        {"date": "2024-07-14", "event": "공기연장(EOT) 의향 통지", "source": "공문(IS-24-055)", "type": "Notice"},
        {"date": "2024-07-21", "event": "작업 재개 승인 요청", "source": "공문(Out-24-101)", "type": "Submission"},
    ]

# 2. 보완된 타임라인 (사용자가 문서 업로드 후)
def get_filled_timeline():
    data = get_initial_timeline()
    new_data = [
        {"date": "2024-07-16", "event": "자재 반입 지연 (도로 침수)", "source": "운반일지(7/16)", "type": "Evidence"},
        {"date": "2024-07-18", "event": "현장 복구 및 배수 작업", "source": "작업일보(7/18)", "type": "Evidence"}
    ]
    return sorted(data, key=lambda x: x['date'])

# 3. 요건-증거 매핑 매트릭스 (Element-Evidence Matrix)
def get_element_matrix():
    # 투자자에게 보여줄 "요건-증거 매핑"의 정석 [cite: 196, 239]
    return [
        {"요건(Legal Element)": "E1. 불가항력적 사유 발생", "입증 증거": "기상청 데이터 (80mm/h)", "상태": "✅ 충족"},
        {"요건(Legal Element)": "E2. 시공사의 통제 불능성", "입증 증거": "현장사진, 도로 폐쇄 통보", "상태": "✅ 충족"},
        {"요건(Legal Element)": "E3. 적기 통지 의무 준수", "입증 증거": "공문(IS-24-055) - 24h 내", "상태": "✅ 충족"},
        {"요건(Legal Element)": "E4. 인과관계 증명(Gap)", "입증 증거": "작업일보(7/16-18) 보완 완료", "상태": "✅ 충족"},
    ]

# 4. 에이전틱 추론이 반영된 최종 서면
def get_advanced_draft():
    return """
    **[지체상금 면책 청구서 초안]**
    본 시공사는 2024년 7월 발생한 기록적 폭우[[Doc:기상데이터]] 및 이에 따른 현장 침수로 공사가 12일간 중단되었습니다.
    도급계약서 제25조(불가항력)[[Clause:Art.25]]에 의거하여 해당 기간의 지체상금 면책을 청구합니다.
    """