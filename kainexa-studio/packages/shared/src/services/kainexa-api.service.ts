// kainexa-studio/packages/shared/src/services/kainexa-api.service.ts
// Kainexa Core API와 통신하는 통합 서비스

import axios, { AxiosInstance, AxiosError } from 'axios';

// API 응답 타입 정의
export interface ChatRequest {
  message: string;
  session_id?: string;
  user_email?: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  timestamp: string;
  intent?: string;
  confidence?: number;
}

export interface WorkflowExecuteRequest {
  nodes: any[];
  edges: any[];
  context?: Record<string, any>;
}

export interface WorkflowExecuteResponse {
  execution_id: string;
  status: 'running' | 'completed' | 'failed';
  results?: any[];
  error?: string;
}

export interface HealthResponse {
  status: string;
  services?: {
    llm?: { status: string; model?: string };
    rag?: { status: string };
    database?: { status: string };
  };
}

// Kainexa API 서비스 클래스
export class KainexaAPIService {
  private coreAPI: AxiosInstance;
  private studioAPI: AxiosInstance;
  private wsConnection: WebSocket | null = null;

  constructor(
    coreBaseURL: string = process.env.NEXT_PUBLIC_CORE_API_URL || 'http://localhost:8000',
    studioBaseURL: string = process.env.NEXT_PUBLIC_STUDIO_API_URL || 'http://localhost:4000'
  ) {
    // Core API 클라이언트 (kainexa-core)
    this.coreAPI = axios.create({
      baseURL: coreBaseURL,
      timeout: 60000, // LLM 응답 시간 고려
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Studio API 클라이언트 (kainexa-studio 백엔드)
    this.studioAPI = axios.create({
      baseURL: studioBaseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // 요청 인터셉터
    this.setupInterceptors();
  }

  private setupInterceptors() {
    // 요청 로깅
    this.coreAPI.interceptors.request.use(
      (config) => {
        console.log(`[Core API] ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('[Core API] Request error:', error);
        return Promise.reject(error);
      }
    );

    // 응답 에러 처리
    this.coreAPI.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.code === 'ECONNREFUSED') {
          console.error('Core API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.');
        }
        return Promise.reject(this.handleAPIError(error));
      }
    );
  }

  private handleAPIError(error: AxiosError): Error {
    if (error.response) {
      // 서버 응답 에러
      const message = (error.response.data as any)?.detail || error.message;
      return new Error(`API Error (${error.response.status}): ${message}`);
    } else if (error.request) {
      // 응답 없음
      return new Error('서버로부터 응답이 없습니다. 네트워크를 확인하세요.');
    } else {
      return new Error(`요청 설정 오류: ${error.message}`);
    }
  }

  // ========== Core API 메서드 ==========

  /**
   * 헬스 체크
   */
  async checkHealth(): Promise<HealthResponse> {
    try {
      const { data } = await this.coreAPI.get<HealthResponse>('/api/v1/health/full');
      return data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  /**
   * 채팅 메시지 전송
   */
  async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const { data } = await this.coreAPI.post<ChatResponse>('/api/v1/chat', request);
      return data;
    } catch (error) {
      console.error('Chat request failed:', error);
      throw error;
    }
  }

  /**
   * 생산 모니터링 시나리오
   */
  async runProductionScenario(query: string = '어제 밤사 생산 현황') {
    try {
      const { data } = await this.coreAPI.post('/api/v1/scenarios/production', null, {
        params: { query }
      });
      return data;
    } catch (error) {
      console.error('Production scenario failed:', error);
      throw error;
    }
  }

  /**
   * 예지보전 시나리오
   */
  async runMaintenanceScenario(equipmentId: string) {
    try {
      const { data } = await this.coreAPI.post('/api/v1/scenarios/maintenance', null, {
        params: { equipment_id: equipmentId }
      });
      return data;
    } catch (error) {
      console.error('Maintenance scenario failed:', error);
      throw error;
    }
  }

  /**
   * 품질 분석 시나리오
   */
  async runQualityScenario() {
    try {
      const { data } = await this.coreAPI.post('/api/v1/scenarios/quality');
      return data;
    } catch (error) {
      console.error('Quality scenario failed:', error);
      throw error;
    }
  }

  // ========== Studio API 메서드 ==========

  /**
   * 워크플로우 실행
   */
  async executeWorkflow(request: WorkflowExecuteRequest): Promise<WorkflowExecuteResponse> {
    try {
      // 먼저 Studio API로 워크플로우 검증
      const validationResponse = await this.studioAPI.post('/api/workflow/validate', request);
      
      if (!validationResponse.data.valid) {
        throw new Error(`Workflow validation failed: ${validationResponse.data.errors}`);
      }

      // Core API로 실제 실행 요청
      const { data } = await this.coreAPI.post<WorkflowExecuteResponse>(
        '/api/v1/workflow/execute',
        request
      );
      
      return data;
    } catch (error) {
      console.error('Workflow execution failed:', error);
      throw error;
    }
  }

  /**
   * 워크플로우 저장
   */
  async saveWorkflow(workflow: any) {
    try {
      const { data } = await this.studioAPI.post('/api/workflow/save', workflow);
      return data;
    } catch (error) {
      console.error('Workflow save failed:', error);
      throw error;
    }
  }

  /**
   * 워크플로우 불러오기
   */
  async loadWorkflow(id: string) {
    try {
      const { data } = await this.studioAPI.get(`/api/workflow/${id}`);
      return data;
    } catch (error) {
      console.error('Workflow load failed:', error);
      throw error;
    }
  }

  // ========== WebSocket 연결 (실시간 기능) ==========

  /**
   * WebSocket 연결 초기화
   */
  connectWebSocket(sessionId: string, onMessage: (data: any) => void) {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
    this.wsConnection = new WebSocket(`${wsUrl}/api/v1/chat/ws/${sessionId}`);

    this.wsConnection.onopen = () => {
      console.log('WebSocket connected');
    };

    this.wsConnection.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    this.wsConnection.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.wsConnection.onclose = () => {
      console.log('WebSocket disconnected');
    };
  }

  /**
   * WebSocket으로 메시지 전송
   */
  sendWebSocketMessage(message: string) {
    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      this.wsConnection.send(JSON.stringify({ text: message }));
    }
  }

  /**
   * WebSocket 연결 종료
   */
  disconnectWebSocket() {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }

  // ========== 유틸리티 메서드 ==========

  /**
   * 한국어 처리 - 존댓말 변환
   */
  async convertHonorific(text: string, level: number) {
    try {
      const { data } = await this.studioAPI.post('/api/korean/honorific', {
        text,
        level
      });
      return data;
    } catch (error) {
      console.error('Honorific conversion failed:', error);
      throw error;
    }
  }

  /**
   * 템플릿 목록 조회
   */
  async getTemplates() {
    try {
      const { data } = await this.studioAPI.get('/api/templates');
      return data;
    } catch (error) {
      console.error('Template fetch failed:', error);
      throw error;
    }
  }
}

// 싱글톤 인스턴스 export
export const kainexaAPI = new KainexaAPIService();