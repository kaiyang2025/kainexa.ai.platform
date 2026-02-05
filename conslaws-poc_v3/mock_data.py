# mock_data.py
import pandas as pd

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
    
    1. 2024년 7월 집중호우[[Doc:기상데이터]] 및 이에 따른 현장 침수로 공사가 12일간 중단되었습니다.
    2. 도급계약서 제25조(불가항력)[[Clause:Art.25]]에 의거하여 해당 기간의 지체상금 면책을 청구합니다.
    3. AI 분석 결과: 추가 확보된 운반일지[[Doc:운반일지]]를 통해 도로 침수로 인한 자재 반입 불가능 사실이 입증되어 승소 확률이 **88%**로 상향되었습니다.
    """