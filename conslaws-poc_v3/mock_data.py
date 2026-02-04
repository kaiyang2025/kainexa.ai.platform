# mock_data.py
import pandas as pd

# 시나리오: 7월 장마로 인한 공기 지연 클레임
# 상황: 7월 10일~14일은 문서가 있는데, 15일~20일은 문서가 누락된 상태

def get_initial_timeline():
    return [
        {"date": "2024-07-10", "event": "호우주의보 발령 및 작업 중지 지시", "source": "공문(IS-24-055)", "type": "instruction"},
        {"date": "2024-07-12", "event": "현장 침수 피해 확인", "source": "현장사진첩_0712", "type": "fact"},
        {"date": "2024-07-14", "event": "펌프 배수 작업 실시", "source": "작업일보(7/14)", "type": "activity"},
        # 여기에 7월 15일~20일 데이터가 비어있음 (Gap)
        {"date": "2024-07-21", "event": "작업 재개 승인 요청", "source": "공문(Out-24-101)", "type": "submission"},
    ]

def get_filled_timeline():
    # 사용자가 문서를 추가했을 때 나타날 데이터
    data = get_initial_timeline()
    new_data = [
        {"date": "2024-07-16", "event": "장비(백호) 대기 일지 기록", "source": "장비가동일보(7/16)", "type": "evidence", "is_new": True},
        {"date": "2024-07-18", "event": "추가 배수 작업 및 현장 정리", "source": "작업일보(7/18)", "type": "evidence", "is_new": True}
    ]
    data.extend(new_data)
    # 날짜순 정렬
    return sorted(data, key=lambda x: x['date'])

def get_draft_text():
    # Evidence Pack 생성 결과물 (Citation 포함)
    return """
    **1. 공기 지연의 사유**
    2024년 7월 10일 발주처의 작업 중지 지시[[Doc-IS-055]]가 있었으며, 
    이에 따라 7월 14일까지 배수 작업[[Doc-Daily-0714]]을 진행하였습니다.
    
    **2. 영향 분석**
    해당 기간 동안 Critical Path 상의 기초 타설 공정이 중단되었으며, 
    이는 전체 공정률에 0.5%의 지연을 초래하였습니다[[Sch-Update-v3]].
    """