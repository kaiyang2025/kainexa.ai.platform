"""
// ========================================
// Kainexa Studio - 한국어 특화 기능
// packages/korean-nlp/src/
// ========================================

// ============================
// 1. 한국어 처리 서비스 (Python 백엔드)
// apps/studio-api/src/modules/korean/korean.service.py
// ============================

"""
# Python 백엔드 서비스 (FastAPI)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from kiwipiepy import Kiwi
import re

app = FastAPI()
kiwi = Kiwi()

class HonorificRequest(BaseModel):
    text: str
    target_level: int  # 1-7
    current_level: Optional[int] = None

class TextAnalysisRequest(BaseModel):
    text: str

class HonorificResponse(BaseModel):
    original: str
    converted: str
    detected_level: int
    target_level: int

@app.post("/analyze-honorific")
async def analyze_honorific(request: TextAnalysisRequest) -> Dict:
    '''존댓말 레벨 분석'''
    tokens = kiwi.tokenize(request.text)
    
    # 종결어미 분석을 통한 존댓말 레벨 판단
    level = detect_honorific_level(tokens)
    
    return {
        "text": request.text,
        "level": level,
        "description": get_level_description(level),
        "tokens": [{"form": t.form, "tag": t.tag} for t in tokens]
    }

@app.post("/convert-honorific")
async def convert_honorific(request: HonorificRequest) -> HonorificResponse:
    '''존댓말 레벨 변환'''
    current_level = request.current_level or detect_honorific_level(
        kiwi.tokenize(request.text)
    )
    
    if current_level == request.target_level:
        return HonorificResponse(
            original=request.text,
            converted=request.text,
            detected_level=current_level,
            target_level=request.target_level
        )
    
    converted = convert_text_honorific(
        request.text, 
        current_level, 
        request.target_level
    )
    
    return HonorificResponse(
        original=request.text,
        converted=converted,
        detected_level=current_level,
        target_level=request.target_level
    )

def detect_honorific_level(tokens) -> int:
    '''
    한국어 존댓말 7단계 감지
    1: 해라체 (명령)
    2: 해체 (반말)  
    3: 해요체 (보통 존댓말)
    4: 하오체 (예사 높임)
    5: 하게체 (예사 낮춤)
    6: 합쇼체 (정중한 존댓말)
    7: 하십시오체 (매우 정중)
    '''
    
    # 종결어미 패턴
    patterns = {
        1: [r'.*라$', r'.*어$', r'.*아$'],
        2: [r'.*야$', r'.*지$', r'.*어$'],
        3: [r'.*요$', r'.*어요$', r'.*아요$'],
        4: [r'.*오$', r'.*소$'],
        5: [r'.*게$', r'.*네$'],
        6: [r'.*습니다$', r'.*ㅂ니다$'],
        7: [r'.*십시오$', r'.*시옵니다$']
    }
    
    # 마지막 어절 분석
    for level, pattern_list in patterns.items():
        for pattern in pattern_list:
            if any(re.match(pattern, t.form) for t in tokens):
                return level
    
    return 3  # 기본값: 해요체

def get_level_description(level: int) -> str:
    descriptions = {
        1: "해라체 - 아주 낮춤",
        2: "해체 - 반말",
        3: "해요체 - 보통 존댓말", 
        4: "하오체 - 예사 높임",
        5: "하게체 - 예사 낮춤",
        6: "합쇼체 - 정중한 존댓말",
        7: "하십시오체 - 매우 정중"
    }
    return descriptions.get(level, "알 수 없음")
"""

// ============================
// 2. 한국어 처리 클라이언트 (TypeScript)
// packages/korean-nlp/src/korean-processor.ts
// ============================

export interface HonorificLevel {
  level: number;
  name: string;
  description: string;
  examples: string[];
}

export const HONORIFIC_LEVELS: HonorificLevel[] = [
  {
    level: 1,
    name: '해라체',
    description: '아주 낮춤 (명령, 일방적)',
    examples: ['해라', '가라', '먹어라']
  },
  {
    level: 2,
    name: '해체',
    description: '반말 (친근, 편안)',
    examples: ['해', '가', '먹어']
  },
  {
    level: 3,
    name: '해요체',
    description: '보통 존댓말 (일반적)',
    examples: ['해요', '가요', '먹어요']
  },
  {
    level: 4,
    name: '하오체',
    description: '예사 높임 (격식)',
    examples: ['하오', '가오', '먹소']
  },
  {
    level: 5,
    name: '하게체',
    description: '예사 낮춤 (중년층)',
    examples: ['하게', '가게', '먹게']
  },
  {
    level: 6,
    name: '합쇼체',
    description: '정중한 존댓말 (공식적)',
    examples: ['합니다', '갑니다', '먹습니다']
  },
  {
    level: 7,
    name: '하십시오체',
    description: '매우 정중 (최고 존경)',
    examples: ['하십시오', '가십시오', '드십시오']
  }
];

export class KoreanProcessor {
  private apiUrl: string;

  constructor(apiUrl: string = 'http://localhost:8001') {
    this.apiUrl = apiUrl;
  }

  /**
   * 텍스트의 존댓말 레벨 감지
   */
  async detectHonorificLevel(text: string): Promise<{
    level: number;
    description: string;
    confidence: number;
  }> {
    try {
      const response = await fetch(`${this.apiUrl}/analyze-honorific`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await response.json();
      return {
        level: data.level,
        description: data.description,
        confidence: 0.9 // 실제로는 모델에서 계산
      };
    } catch (error) {
      console.error('Failed to detect honorific level:', error);
      // 폴백: 간단한 규칙 기반 감지
      return this.detectHonorificLevelFallback(text);
    }
  }

  /**
   * 존댓말 레벨 변환
   */
  async convertHonorificLevel(
    text: string,
    targetLevel: number
  ): Promise<string> {
    try {
      const response = await fetch(`${this.apiUrl}/convert-honorific`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          target_level: targetLevel
        })
      });

      const data = await response.json();
      return data.converted;
    } catch (error) {
      console.error('Failed to convert honorific level:', error);
      // 폴백: 템플릿 기반 변환
      return this.convertHonorificLevelFallback(text, targetLevel);
    }
  }

  /**
   * 폴백: 간단한 규칙 기반 존댓말 감지
   */
  private detectHonorificLevelFallback(text: string): {
    level: number;
    description: string;
    confidence: number;
  } {
    const patterns = {
      6: /습니다|ㅂ니다/,
      3: /어요|아요|예요|에요|요$/,
      2: /야|어|아|지$/,
      1: /어라|아라|라$/
    };

    for (const [level, pattern] of Object.entries(patterns)) {
      if (pattern.test(text)) {
        const levelNum = parseInt(level);
        return {
          level: levelNum,
          description: HONORIFIC_LEVELS[levelNum - 1].description,
          confidence: 0.7
        };
      }
    }

    return {
      level: 3,
      description: '보통 존댓말 (추정)',
      confidence: 0.5
    };
  }

  /**
   * 폴백: 템플릿 기반 존댓말 변환
   */
  private convertHonorificLevelFallback(
    text: string,
    targetLevel: number
  ): string {
    // 간단한 변환 규칙
    const conversions: Record<number, Record<string, string>> = {
      2: { // 반말
        '습니다': '어',
        '합니다': '해',
        '입니다': '야',
        '어요': '어',
        '아요': '아',
        '요': ''
      },
      3: { // 해요체
        '습니다': '어요',
        '합니다': '해요',
        '입니다': '예요',
        '어': '어요',
        '아': '아요',
        '야': '예요'
      },
      6: { // 합쇼체
        '어요': '습니다',
        '해요': '합니다',
        '예요': '입니다',
        '어': '습니다',
        '해': '합니다',
        '야': '입니다'
      }
    };

    let converted = text;
    const rules = conversions[targetLevel] || {};

    for (const [pattern, replacement] of Object.entries(rules)) {
      const regex = new RegExp(pattern + '$', 'g');
      converted = converted.replace(regex, replacement);
    }

    return converted;
  }

  /**
   * 문장 분리 (한국어 특화)
   */
  splitSentences(text: string): string[] {
    // 한국어 문장 종결 패턴
    const sentenceEndings = /[.!?。！？]+[\s]*/g;
    const sentences = text.split(sentenceEndings)
      .filter(s => s.trim().length > 0);
    return sentences;
  }

  /**
   * 형태소 분석 (간단 버전)
   */
  async tokenize(text: string): Promise<Array<{
    surface: string;
    pos: string;
    semantic: string;
  }>> {
    // 실제로는 KoNLPy 또는 Kiwi API 호출
    // 여기서는 간단한 시뮬레이션
    const tokens = text.split(/\s+/).map(word => ({
      surface: word,
      pos: 'NNG', // 일반명사
      semantic: ''
    }));

    return tokens;
  }
}

// ============================
// 3. 카카오톡 연동 서비스
// packages/integrations/src/kakao/kakao-channel.ts
// ============================

export interface KakaoMessage {
  user_key: string;
  type: 'text' | 'photo' | 'button';
  content: string;
}

export interface KakaoResponse {
  message: {
    text?: string;
    photo?: {
      url: string;
      width: number;
      height: number;
    };
    message_button?: {
      label: string;
      url: string;
    };
  };
  keyboard?: {
    type: 'buttons' | 'text';
    buttons?: string[];
  };
}

export class KakaoChannelService {
  private apiKey: string;
  private channelId: string;
  private webhookUrl: string;

  constructor(config: {
    apiKey: string;
    channelId: string;
    webhookUrl: string;
  }) {
    this.apiKey = config.apiKey;
    this.channelId = config.channelId;
    this.webhookUrl = config.webhookUrl;
  }

  /**
   * 카카오톡 메시지 수신 처리
   */
  async handleIncomingMessage(message: KakaoMessage): Promise<KakaoResponse> {
    // 메시지 타입별 처리
    switch (message.type) {
      case 'text':
        return this.handleTextMessage(message);
      case 'photo':
        return this.handlePhotoMessage(message);
      case 'button':
        return this.handleButtonMessage(message);
      default:
        return this.createTextResponse('지원하지 않는 메시지 타입입니다.');
    }
  }

  /**
   * 텍스트 메시지 처리
   */
  private async handleTextMessage(message: KakaoMessage): Promise<KakaoResponse> {
    const userInput = message.content;
    
    // 워크플로우 실행을 위한 컨텍스트 생성
    const context = {
      channel: 'kakao',
      userId: message.user_key,
      message: userInput
    };

    // TODO: 워크플로우 엔진과 연동
    const response = await this.processWithWorkflow(context);

    return this.createTextResponse(response.text, response.buttons);
  }

  /**
   * 사진 메시지 처리
   */
  private async handlePhotoMessage(message: KakaoMessage): Promise<KakaoResponse> {
    return this.createTextResponse('사진을 받았습니다. 처리 중입니다...');
  }

  /**
   * 버튼 메시지 처리
   */
  private async handleButtonMessage(message: KakaoMessage): Promise<KakaoResponse> {
    const selectedButton = message.content;
    
    // 버튼 선택에 따른 처리
    return this.createTextResponse(`'${selectedButton}'을(를) 선택하셨습니다.`);
  }

  /**
   * 텍스트 응답 생성
   */
  private createTextResponse(
    text: string,
    buttons?: string[]
  ): KakaoResponse {
    const response: KakaoResponse = {
      message: { text }
    };

    if (buttons && buttons.length > 0) {
      response.keyboard = {
        type: 'buttons',
        buttons
      };
    }

    return response;
  }

  /**
   * 사진 응답 생성
   */
  createPhotoResponse(
    url: string,
    width: number = 640,
    height: number = 480,
    text?: string
  ): KakaoResponse {
    return {
      message: {
        text,
        photo: { url, width, height }
      }
    };
  }

  /**
   * 버튼 응답 생성
   */
  createButtonResponse(
    text: string,
    buttons: Array<{ label: string; url: string }>
  ): KakaoResponse {
    return {
      message: {
        text,
        message_button: buttons[0] // 카카오톡은 하나의 링크 버튼만 지원
      },
      keyboard: {
        type: 'buttons',
        buttons: buttons.map(b => b.label)
      }
    };
  }

  /**
   * 워크플로우와 연동하여 응답 생성
   */
  private async processWithWorkflow(context: any): Promise<{
    text: string;
    buttons?: string[];
  }> {
    // TODO: 실제 워크플로우 엔진 호출
    // 여기서는 시뮬레이션
    return {
      text: '안녕하세요! 무엇을 도와드릴까요?',
      buttons: ['주문 조회', '상품 문의', '환불/교환', '기타 문의']
    };
  }

  /**
   * 프로액티브 메시지 전송
   */
  async sendProactiveMessage(
    userKey: string,
    message: string
  ): Promise<boolean> {
    try {
      const response = await fetch('https://kapi.kakao.com/v1/api/talk/memo/send', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
          template_object: JSON.stringify({
            object_type: 'text',
            text: message
          })
        })
      });

      return response.ok;
    } catch (error) {
      console.error('Failed to send proactive message:', error);
      return false;
    }
  }
}

// ============================
// 4. 한국어 NLP 노드
// packages/workflow-engine/src/nodes/korean-nlp-node.ts
// ============================

import { AbstractNode, NodeType, ExecutionContext, NodeResult } from './base';
import { KoreanProcessor, HONORIFIC_LEVELS } from '@kainexa/korean-nlp';
import { z } from 'zod';

const KoreanNLPConfigSchema = z.object({
  operation: z.enum(['detect-honorific', 'convert-honorific', 'tokenize', 'sentiment']),
  targetHonorificLevel: z.number().min(1).max(7).optional(),
  autoAdjust: z.boolean().default(true),
  preserveEmoticons: z.boolean().default(true)
});

export class KoreanNLPNode extends AbstractNode {
  private config: z.infer<typeof KoreanNLPConfigSchema>;
  private processor: KoreanProcessor;

  constructor(node: any) {
    super(node);
    this.config = KoreanNLPConfigSchema.parse(node.data.config);
    this.processor = new KoreanProcessor();
  }

  async execute(context: ExecutionContext): Promise<NodeResult> {
    try {
      const text = context.variables.get('text') || 
                   context.history[context.history.length - 1]?.content;

      if (!text) {
        return {
          success: false,
          error: 'No text input found'
        };
      }

      let result: any;

      switch (this.config.operation) {
        case 'detect-honorific':
          result = await this.detectHonorific(text);
          break;
        case 'convert-honorific':
          result = await this.convertHonorific(text, context);
          break;
        case 'tokenize':
          result = await this.tokenizeText(text);
          break;
        case 'sentiment':
          result = await this.analyzeSentiment(text);
          break;
      }

      // 컨텍스트 업데이트
      context.variables.set('koreanNLPResult', result);
      
      return {
        success: true,
        output: result,
        context: {
          variables: context.variables,
          metadata: {
            ...context.metadata,
            koreanProcessing: {
              operation: this.config.operation,
              timestamp: new Date().toISOString()
            }
          }
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  private async detectHonorific(text: string) {
    const detection = await this.processor.detectHonorificLevel(text);
    
    return {
      text,
      detectedLevel: detection.level,
      levelName: HONORIFIC_LEVELS[detection.level - 1].name,
      description: detection.description,
      confidence: detection.confidence,
      recommendation: this.getHonorificRecommendation(detection.level)
    };
  }

  private async convertHonorific(text: string, context: ExecutionContext) {
    const targetLevel = this.config.targetHonorificLevel || 
                       context.variables.get('targetHonorificLevel') || 3;
    
    const converted = await this.processor.convertHonorificLevel(text, targetLevel);
    const detection = await this.processor.detectHonorificLevel(text);
    
    return {
      original: text,
      converted,
      originalLevel: detection.level,
      targetLevel,
      changed: text !== converted
    };
  }

  private async tokenizeText(text: string) {
    const tokens = await this.processor.tokenize(text);
    
    return {
      text,
      tokens,
      wordCount: tokens.length,
      uniqueWords: new Set(tokens.map(t => t.surface)).size
    };
  }

  private async analyzeSentiment(text: string) {
    // 간단한 감정 분석 (실제로는 ML 모델 사용)
    const positiveWords = ['좋아', '감사', '훌륭', '최고', '만족'];
    const negativeWords = ['나쁜', '실망', '화가', '짜증', '불만'];
    
    let positiveScore = 0;
    let negativeScore = 0;
    
    for (const word of positiveWords) {
      if (text.includes(word)) positiveScore++;
    }
    
    for (const word of negativeWords) {
      if (text.includes(word)) negativeScore++;
    }
    
    const sentiment = positiveScore > negativeScore ? 'positive' : 
                     negativeScore > positiveScore ? 'negative' : 'neutral';
    
    return {
      text,
      sentiment,
      positiveScore,
      negativeScore,
      confidence: 0.7
    };
  }

  private getHonorificRecommendation(currentLevel: number): string {
    if (currentLevel <= 2) {
      return '고객 응대시에는 더 높은 존댓말 레벨(3-6)을 권장합니다.';
    } else if (currentLevel >= 6) {
      return '적절한 존댓말 레벨입니다. 상황에 따라 조금 낮춰도 괜찮습니다.';
    } else {
      return '일반적인 고객 응대에 적합한 레벨입니다.';
    }
  }

  validate(): boolean {
    try {
      KoreanNLPConfigSchema.parse(this.data.config);
      return true;
    } catch {
      return false;
    }
  }
}

// ============================
// 5. 통합 테스트 및 사용 예제
// packages/workflow-engine/src/test/korean-integration.test.ts
// ============================

import { describe, it, expect, beforeAll } from '@jest/globals';
import { KoreanProcessor } from '@kainexa/korean-nlp';
import { KakaoChannelService } from '@kainexa/integrations';
import { AdvancedWorkflowExecutor } from '../executor/advanced-executor';
import { Node, Edge } from 'reactflow';

describe('한국어 특화 기능 통합 테스트', () => {
  let koreanProcessor: KoreanProcessor;
  let kakaoService: KakaoChannelService;
  
  beforeAll(() => {
    koreanProcessor = new KoreanProcessor();
    kakaoService = new KakaoChannelService({
      apiKey: process.env.KAKAO_API_KEY || 'test-key',
      channelId: process.env.KAKAO_CHANNEL_ID || 'test-channel',
      webhookUrl: 'https://api.kainexa.com/webhook/kakao'
    });
  });

  describe('존댓말 처리', () => {
    it('존댓말 레벨을 정확히 감지해야 함', async () => {
      const testCases = [
        { text: '안녕하세요', expectedLevel: 3 },
        { text: '안녕하십니까', expectedLevel: 6 },
        { text: '안녕', expectedLevel: 2 },
        { text: '가십시오', expectedLevel: 7 }
      ];

      for (const testCase of testCases) {
        const result = await koreanProcessor.detectHonorificLevel(testCase.text);
        expect(result.level).toBe(testCase.expectedLevel);
      }
    });

    it('존댓말 레벨을 변환할 수 있어야 함', async () => {
      const original = '안녕하세요';
      const converted = await koreanProcessor.convertHonorificLevel(original, 6);
      
      expect(converted).toContain('습니다');
    });
  });

  describe('카카오톡 통합', () => {
    it('카카오톡 메시지를 처리할 수 있어야 함', async () => {
      const message = {
        user_key: 'test-user',
        type: 'text' as const,
        content: '주문 조회하고 싶어요'
      };

      const response = await kakaoService.handleIncomingMessage(message);
      
      expect(response.message.text).toBeDefined();
      expect(response.keyboard?.buttons).toBeDefined();
    });

    it('버튼 응답을 생성할 수 있어야 함', () => {
      const response = kakaoService.createButtonResponse(
        '무엇을 도와드릴까요?',
        [
          { label: '주문 조회', url: 'https://example.com/orders' },
          { label: '고객 센터', url: 'https://example.com/support' }
        ]
      );

      expect(response.message.text).toBe('무엇을 도와드릴까요?');
      expect(response.keyboard?.buttons).toHaveLength(2);
    });
  });

  describe('워크플로우 통합', () => {
    it('한국어 처리 노드가 포함된 워크플로우를 실행할 수 있어야 함', async () => {
      const nodes: Node[] = [
        {
          id: 'start',
          type: 'intent',
          position: { x: 0, y: 0 },
          data: {
            label: '시작',
            config: {
              intents: [
                {
                  name: 'greeting',
                  examples: ['안녕', '안녕하세요'],
                  entities: []
                }
              ]
            }
          }
        },
        {
          id: 'korean-nlp',
          type: 'koreanNLP',
          position: { x: 200, y: 0 },
          data: {
            label: '존댓말 처리',
            config: {
              operation: 'convert-honorific',
              targetHonorificLevel: 6
            }
          }
        },
        {
          id: 'response',
          type: 'llm',
          position: { x: 400, y: 0 },
          data: {
            label: '응답 생성',
            config: {
              model: 'gpt-3.5-turbo',
              systemPrompt: '친절한 한국어 상담원입니다.',
              userPromptTemplate: '{{converted}}'
            }
          }
        }
      ];

      const edges: Edge[] = [
        { id: 'e1', source: 'start', target: 'korean-nlp' },
        { id: 'e2', source: 'korean-nlp', target: 'response' }
      ];

      const executor = new AdvancedWorkflowExecutor(nodes, edges, {
        mode: 'sequential',
        maxDepth: 10,
        timeout: 30000,
        debug: true
      });

      const result = await executor.execute({
        variables: new Map([
          ['userInput', '안녕'],
          ['text', '안녕']
        ])
      });

      expect(result.success).toBe(true);
      expect(result.results).toHaveLength(3);
    });
  });
});

// ============================
// 6. 한국어 챗봇 템플릿
// packages/templates/src/korean-chatbot.ts
// ============================

export const KoreanChatbotTemplate = {
  name: '한국어 고객 상담 챗봇',
  description: '존댓말 자동 조절과 카카오톡 연동이 포함된 한국어 챗봇 템플릿',
  nodes: [
    {
      id: 'welcome',
      type: 'intent',
      position: { x: 100, y: 100 },
      data: {
        label: '인사',
        config: {
          intents: [
            {
              name: 'greeting',
              examples: ['안녕', '안녕하세요', '반가워', '반갑습니다'],
              entities: []
            }
          ],
          threshold: 0.7
        }
      }
    },
    {
      id: 'detect-honorific',
      type: 'koreanNLP',
      position: { x: 300, y: 100 },
      data: {
        label: '존댓말 감지',
        config: {
          operation: 'detect-honorific',
          autoAdjust: true
        }
      }
    },
    {
      id: 'adjust-response',
      type: 'condition',
      position: { x: 500, y: 100 },
      data: {
        label: '응답 레벨 조정',
        config: {
          conditions: [
            {
              field: 'koreanNLPResult.detectedLevel',
              operator: 'less_than',
              value: 3,
              nextNode: 'casual-response'
            },
            {
              field: 'koreanNLPResult.detectedLevel',
              operator: 'greater_than',
              value: 5,
              nextNode: 'formal-response'
            }
          ],
          defaultBranch: 'normal-response'
        }
      }
    },
    {
      id: 'casual-response',
      type: 'llm',
      position: { x: 700, y: 50 },
      data: {
        label: '캐주얼 응답',
        config: {
          model: 'solar',
          temperature: 0.8,
          systemPrompt: '친근하고 캐주얼한 말투로 대화하세요. 반말을 사용하되 무례하지 않게 합니다.',
          userPromptTemplate: '고객이 "{{userInput}}"라고 했습니다. 친근하게 응답하세요.'
        }
      }
    },
    {
      id: 'normal-response',
      type: 'llm',
      position: { x: 700, y: 150 },
      data: {
        label: '일반 응답',
        config: {
          model: 'solar',
          temperature: 0.7,
          systemPrompt: '정중하고 친절한 한국어로 응답하세요. 해요체를 사용합니다.',
          userPromptTemplate: '고객이 "{{userInput}}"라고 했습니다. 정중하게 응답하세요.'
        }
      }
    },
    {
      id: 'formal-response',
      type: 'llm',
      position: { x: 700, y: 250 },
      data: {
        label: '격식 응답',
        config: {
          model: 'solar',
          temperature: 0.6,
          systemPrompt: '매우 정중하고 격식 있는 한국어로 응답하세요. 합쇼체를 사용합니다.',
          userPromptTemplate: '고객님께서 "{{userInput}}"라고 말씀하셨습니다. 매우 정중하게 응답하세요.'
        }
      }
    },
    {
      id: 'send-kakao',
      type: 'api',
      position: { x: 900, y: 150 },
      data: {
        label: '카카오톡 전송',
        config: {
          url: 'https://kapi.kakao.com/v1/api/talk/memo/send',
          method: 'POST',
          headers: {
            'Authorization': 'Bearer {{kakaoToken}}'
          },
          bodyTemplate: '{"template_object": {"object_type": "text", "text": "{{response}}"}}'
        }
      }
    }
  ],
  edges: [
    { id: 'e1', source: 'welcome', target: 'detect-honorific' },
    { id: 'e2', source: 'detect-honorific', target: 'adjust-response' },
    { id: 'e3', source: 'adjust-response', target: 'casual-response' },
    { id: 'e4', source: 'adjust-response', target: 'normal-response' },
    { id: 'e5', source: 'adjust-response', target: 'formal-response' },
    { id: 'e6', source: 'casual-response', target: 'send-kakao' },
    { id: 'e7', source: 'normal-response', target: 'send-kakao' },
    { id: 'e8', source: 'formal-response', target: 'send-kakao' }
  ],
  variables: {
    defaultHonorificLevel: 3,
    companyName: '카인엑사',
    supportEmail: 'support@kainexa.com'
  }
};