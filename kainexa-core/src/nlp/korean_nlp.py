# src/nlp/korean_nlp.py
"""한국어 NLP 통합 모듈"""
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re
from kiwipiepy import Kiwi
from konlpy.tag import Okt
import numpy as np
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

class HonorificLevel(Enum):
    """한국어 존댓말 7단계"""
    HASIPSIO = 1      # 하십시오체 (최고 존대)
    HAMNIDA = 2       # 합니다체 (격식체)
    HAO = 3           # 하오체 (중간 격식)
    HAEYO = 4         # 해요체 (일반 존대)
    HAE = 5           # 해체 (보통)
    HAERA = 6         # 해라체 (낮춤)
    BANMAL = 7        # 반말 (최하위)

@dataclass
class KoreanNLPResult:
    """한국어 처리 결과"""
    text: str
    tokens: List[Tuple[str, str]]  # (형태소, 품사)
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = None
    honorific_level: HonorificLevel = HonorificLevel.HAEYO
    sentiment: float = 0.0
    dialect: Optional[str] = None

class KoreanNLPPipeline:
    """한국어 NLP 파이프라인"""
    
    def __init__(self):
        self.kiwi = Kiwi()
        self.okt = Okt()
        
        # 업계별 전문용어 사전
        self.industry_terms = {
            'manufacturing': ['금형', '프레스', 'CNC', '불량률', 'OEE'],
            'finance': ['대출', '이자율', '신용등급', '담보', '상환'],
            'healthcare': ['처방', '진료', '입원', '검사', '수술']
        }
        
        # 지역 방언 패턴
        self.dialect_patterns = {
            'gyeongsang': ['가가', '마카', '와이라노'],
            'jeolla': ['거시기', '겁나', '잉'],
            'jeju': ['하영', '게메', '혼저']
        }
        
        # 존댓말 변환 규칙
        self.honorific_rules = self._load_honorific_rules()
    
    async def process(self, text: str, 
                     target_honorific: Optional[HonorificLevel] = None) -> KoreanNLPResult:
        """텍스트 처리"""
        
        # 1. 형태소 분석
        tokens = self._tokenize(text)
        
        # 2. 의도 파악
        intent = await self._classify_intent(text, tokens)
        
        # 3. 개체 추출
        entities = self._extract_entities(text, tokens)
        
        # 4. 존댓말 레벨 분석
        current_honorific = self._detect_honorific_level(text)
        
        # 5. 감정 분석
        sentiment = self._analyze_sentiment(tokens)
        
        # 6. 방언 감지
        dialect = self._detect_dialect(text)
        
        # 7. 존댓말 변환 (필요시)
        if target_honorific and target_honorific != current_honorific:
            text = self._convert_honorific(text, current_honorific, target_honorific)
        
        return KoreanNLPResult(
            text=text,
            tokens=tokens,
            intent=intent,
            entities=entities,
            honorific_level=current_honorific,
            sentiment=sentiment,
            dialect=dialect
        )
    
    def _tokenize(self, text: str) -> List[Tuple[str, str]]:
        """형태소 분석"""
        result = self.kiwi.tokenize(text)
        return [(token.form, token.tag) for token in result]
    
    async def _classify_intent(self, text: str, tokens: List) -> str:
        """의도 분류"""
        # 키워드 기반 간단한 분류 (실제는 ML 모델 사용)
        intent_keywords = {
            'question': ['뭐', '어떻게', '왜', '언제', '누가'],
            'request': ['해줘', '부탁', '요청', '원해'],
            'complaint': ['불만', '문제', '안돼', '이상해'],
            'greeting': ['안녕', '반가워', '처음'],
            'farewell': ['잘가', '안녕히', '다음에']
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in text for keyword in keywords):
                return intent
        
        return 'general'
    
    def _extract_entities(self, text: str, tokens: List) -> List[Dict]:
        """개체명 추출"""
        entities = []
        
        for i, (form, tag) in enumerate(tokens):
            # 고유명사 (NNP)
            if tag == 'NNP':
                entities.append({
                    'type': 'proper_noun',
                    'value': form,
                    'position': i
                })
            
            # 숫자 + 단위
            if tag == 'NR':  # 수사
                if i + 1 < len(tokens) and tokens[i+1][1] == 'NNB':  # 의존명사
                    entities.append({
                        'type': 'quantity',
                        'value': f"{form}{tokens[i+1][0]}",
                        'position': i
                    })
            
            # 시간 표현
            if form in ['오늘', '내일', '어제', '지금']:
                entities.append({
                    'type': 'time',
                    'value': form,
                    'position': i
                })
        
        return entities
    
    def _detect_honorific_level(self, text: str) -> HonorificLevel:
        """존댓말 레벨 감지"""
        
        # 종결어미 패턴으로 판단
        patterns = {
            HonorificLevel.HASIPSIO: r'십니다|십니까|시겠습니까',
            HonorificLevel.HAMNIDA: r'습니다|습니까|ㅂ니다|ㅂ니까',
            HonorificLevel.HAO: r'하오|하구려|하시오',
            HonorificLevel.HAEYO: r'[해|돼|줘|와]요|어요|아요|예요|이에요',
            HonorificLevel.HAE: r'[해|돼|줘|와](?!요)',
            HonorificLevel.HAERA: r'[하|되|오|가]라|어라|거라',
            HonorificLevel.BANMAL: r'[야|아|어](?!요)$'
        }
        
        for level, pattern in patterns.items():
            if re.search(pattern, text):
                return level
        
        return HonorificLevel.HAEYO  # 기본값
    
    def _convert_honorific(self, text: str, 
                          current: HonorificLevel, 
                          target: HonorificLevel) -> str:
        """존댓말 변환"""
        
        # 간단한 종결어미 변환 (실제는 더 복잡)
        conversions = {
            ('해', HonorificLevel.HASIPSIO): '하십시오',
            ('해', HonorificLevel.HAMNIDA): '합니다',
            ('해', HonorificLevel.HAEYO): '해요',
            ('해', HonorificLevel.BANMAL): '해',
            
            ('했어', HonorificLevel.HASIPSIO): '하셨습니다',
            ('했어', HonorificLevel.HAMNIDA): '했습니다',
            ('했어', HonorificLevel.HAEYO): '했어요',
            
            ('갈게', HonorificLevel.HASIPSIO): '가겠습니다',
            ('갈게', HonorificLevel.HAMNIDA): '가겠습니다',
            ('갈게', HonorificLevel.HAEYO): '갈게요',
        }
        
        result = text
        for (pattern, level), replacement in conversions.items():
            if level == target and pattern in text:
                result = text.replace(pattern, replacement)
        
        return result
    
    def _analyze_sentiment(self, tokens: List) -> float:
        """감정 분석"""
        positive_words = ['좋', '행복', '기쁘', '감사', '최고']
        negative_words = ['싫', '나쁘', '화나', '짜증', '최악']
        
        score = 0.0
        for form, tag in tokens:
            if any(pos in form for pos in positive_words):
                score += 1.0
            if any(neg in form for neg in negative_words):
                score -= 1.0
        
        # -1 ~ 1 사이로 정규화
        return max(-1.0, min(1.0, score / len(tokens) if tokens else 0))
    
    def _detect_dialect(self, text: str) -> Optional[str]:
        """방언 감지"""
        for dialect, patterns in self.dialect_patterns.items():
            if any(pattern in text for pattern in patterns):
                return dialect
        return None
    
    def _load_honorific_rules(self) -> Dict:
        """존댓말 변환 규칙 로드"""
        return {
            'verb_endings': {
                'hasipsio': ['십니다', '십니까', '시겠습니까'],
                'hamnida': ['습니다', '습니까', 'ㅂ니다'],
                'haeyo': ['어요', '아요', '해요', '예요'],
                'banmal': ['어', '아', '해', '야']
            },
            'particles': {
                'subject_honorific': ['께서', '님'],
                'object_honorific': ['께', '님께']
            }
        }

class IntentClassifier:
    """의도 분류기"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        # 실제로는 fine-tuned BERT 모델 로드
        
    async def classify(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """의도 분류"""
        # 간단한 규칙 기반 (실제는 ML 모델)
        intents = {
            'sales_inquiry': 0.0,
            'technical_support': 0.0,
            'complaint': 0.0,
            'general_question': 0.0
        }
        
        # 키워드 매칭으로 점수 계산
        if '매출' in text or '판매' in text:
            intents['sales_inquiry'] = 0.9
        elif '고장' in text or '오류' in text:
            intents['technical_support'] = 0.85
        elif '불만' in text or '환불' in text:
            intents['complaint'] = 0.8
        else:
            intents['general_question'] = 0.7
        
        # 최고 점수 의도 반환
        best_intent = max(intents.items(), key=lambda x: x[1])
        
        return {
            'intent': best_intent[0] if best_intent[1] > threshold else 'unknown',
            'confidence': best_intent[1],
            'all_scores': intents
        }