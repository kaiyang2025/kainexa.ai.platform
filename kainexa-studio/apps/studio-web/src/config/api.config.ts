// kainexa-studio/apps/studio-web/src/config/api.config.ts
// API URL을 환경에 따라 자동으로 설정하는 파일

interface APIConfig {
  CORE_API_URL: string;
  STUDIO_API_URL: string;
  WS_URL: string;
}

// 환경별 설정
const configs: Record<string, APIConfig> = {
  // 로컬 개발 환경 (같은 PC에서 실행)
  local: {
    CORE_API_URL: 'http://localhost:8000',
    STUDIO_API_URL: 'http://localhost:4000',
    WS_URL: 'ws://localhost:8000',
  },
  
  // 네트워크 환경 (다른 PC에서 접속)
  network: {
    CORE_API_URL: 'http://192.168.1.215:8000',
    STUDIO_API_URL: 'http://192.168.1.215:4000',
    WS_URL: 'ws://192.168.1.215:8000',
  },
  
  // 프로덕션 환경
  production: {
    CORE_API_URL: process.env.NEXT_PUBLIC_CORE_API_URL || 'http://api.kainexa.com',
    STUDIO_API_URL: process.env.NEXT_PUBLIC_STUDIO_API_URL || 'http://studio-api.kainexa.com',
    WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'wss://api.kainexa.com',
  }
};

// 현재 환경 감지 함수
function detectEnvironment(): string {
  // 브라우저에서 실행 중인지 확인
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    
    console.log('🌐 Current hostname:', hostname);
    
    // localhost로 접속한 경우
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return 'local';
    }
    
    // 192.168.x.x 네트워크 IP로 접속한 경우
    if (hostname.startsWith('192.168.')) {
      return 'network';
    }
    
    // 그 외 (프로덕션)
    return 'production';
  }
  
  // 서버 사이드 렌더링 시 환경 변수 사용
  return process.env.NODE_ENV === 'production' ? 'production' : 'local';
}

// 현재 환경 설정 가져오기
export function getAPIConfig(): APIConfig {
  const env = detectEnvironment();
  console.log(`🔧 API Environment: ${env}`);
  
  const config = configs[env] || configs.local;
  console.log('📡 API URLs:', config);
  
  return config;
}

// 편의 함수들 - API 엔드포인트 URL을 쉽게 가져올 수 있음
export const API = {
  // Core API endpoints
  health: () => `${getAPIConfig().CORE_API_URL}/api/v1/health`,
  healthFull: () => `${getAPIConfig().CORE_API_URL}/api/v1/health/full`,
  chat: () => `${getAPIConfig().CORE_API_URL}/api/v1/chat`,
  workflowExecute: () => `${getAPIConfig().CORE_API_URL}/api/v1/workflow/execute`,
  scenarioProduction: () => `${getAPIConfig().CORE_API_URL}/api/v1/scenarios/production`,
  scenarioMaintenance: () => `${getAPIConfig().CORE_API_URL}/api/v1/scenarios/maintenance`,
  scenarioQuality: () => `${getAPIConfig().CORE_API_URL}/api/v1/scenarios/quality`,
  
  // Studio API endpoints
  workflowSave: () => `${getAPIConfig().STUDIO_API_URL}/api/workflow/save`,
  workflowLoad: (id: string) => `${getAPIConfig().STUDIO_API_URL}/api/workflow/${id}`,
  workflowValidate: () => `${getAPIConfig().STUDIO_API_URL}/api/workflow/validate`,
  templates: () => `${getAPIConfig().STUDIO_API_URL}/api/templates`,
  koreanHonorific: () => `${getAPIConfig().STUDIO_API_URL}/api/korean/honorific`,
  
  // WebSocket endpoints
  chatWS: (sessionId: string) => `${getAPIConfig().WS_URL}/api/v1/chat/ws/${sessionId}`,
  
  // Base URLs (필요시 직접 사용)
  coreBase: () => getAPIConfig().CORE_API_URL,
  studioBase: () => getAPIConfig().STUDIO_API_URL,
  wsBase: () => getAPIConfig().WS_URL,
};

// Export default config
export default getAPIConfig();