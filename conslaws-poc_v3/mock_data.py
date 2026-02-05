# mock_data.py
import pandas as pd

# 1. 초기 타임라인 (증거 공백 존재)
def get_initial_timeline():
    return [
        {"date": "2024-07-10", "event": "기상청 호우주의보 발령 (집중호우)", "source": "기상청 데이터", "type": "Fact"},
        {"date": "2024-07-12", "event": "현장 침수 및 장비 가동 중단", "source": "현장사진첩_0712", "type": "Fact"},
        {"date": "2024-07-14", "event": "발주처 앞 공기연장(EOT) 의향 통지", "source": "공문(IS-24-055)", "type": "Notice"},
        # 7/15 ~ 7/20 공백
        {"date": "2024-07-21", "event": "작업 재개 승인 요청", "source": "공문(Out-24-101)", "type": "Submission"},
    ]

# 2. 보완된 타임라인 (사용자가 문서 업로드 후)
def get_filled_timeline():
    data = get_initial_timeline()
    new_data = [
        {"date": "2024-07-16", "event": "자재 반입 지연 (도로 침수)", "source": "운반일지(7/16)", "type": "Evidence", "is_new": True},
        {"date": "2024-07-18", "event": "추가 배수 및 현장 복구 작업", "source": "작업일보(7/18)", "type": "Evidence", "is_new": True}
    ]
    data.extend(new_data)
    return sorted(data, key=lambda x: x['date'])

# 3. 요건-증거 매핑 매트릭스 (Element-Evidence Matrix)
def get_element_matrix():
    return [
        {"요건(Legal Element)": "E1. 불가항력적 사유 발생", "입증 증거": "기상청 강수량 데이터 (80mm/h)", "상태": "✅ 충족"},
        {"요건(Legal Element)": "E2. 시공사의 통제 불능성", "입증 증거": "현장사진첩, 도로 폐쇄 통보", "상태": "✅ 충족"},
        {"요건(Legal Element)": "E3. 적기 통지 의무 준수", "입증 증거": "공문(IS-24-055) - 24시간 내 통지", "상태": "✅ 충족"},
        {"요건(Legal Element)": "E4. 인과관계 증명 (Critical Path)", "입증 증거": "공정표(As-built) 대비 지연 분석", "상태": "⚠️ 보완 필요"},
    ]

# 4. 에이전틱 추론이 반영된 최종 서면
def get_advanced_draft():
    return """
    ### [지체상금 면책 및 공기연장 청구서]
    
    **1. 청구 사유 및 법리적 근거**
    본 시공사는 도급계약서 제25조(불가항력)[[Clause:Art.25]]에 의거하여, 2024년 7월 발생한 기록적 폭우[[Doc:기상데이터]]에 따른 공기 지연 12일에 대한 면책을 청구합니다.
    
    **2. 에이전트 분석 결과 (Strategic Insight)**
    - **Evidence Agent:** 단순 기상 데이터 외에 도로 침수로 인한 '자재 반입 불가'[[Doc:운반일지]]를 추가 포착하여 인과관계를 강화함.
    - **Strategy Agent:** 발주처의 지시 지연과 천재지변을 결합한 '복합 귀책' 전략을 수립함.
    
    **3. 결론**
    상기 증거 패키지(Evidence Pack)를 바탕으로 분석한 결과, 본 건의 **면책 승인 확률은 88.5%**로 추산됩니다.
    """