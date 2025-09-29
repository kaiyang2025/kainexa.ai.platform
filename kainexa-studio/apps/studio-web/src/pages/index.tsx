import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'

// 클라이언트 전용 컴포넌트
const ClientOnlyInfo = dynamic(
  () => Promise.resolve(({ url }: { url: string }) => (
    <p className="text-xs opacity-60">현재 페이지: {url}</p>
  )),
  { ssr: false }
)

export default function Home() {
  const [mounted, setMounted] = useState(false)
  const [status, setStatus] = useState({
    api: false,
    loading: true,
    error: null as string | null
  })

  const API_URL = 'http://192.168.1.215:4000'

  useEffect(() => {
    // 컴포넌트가 마운트되었음을 표시
    setMounted(true)
    
    // API 체크
    checkAPI()
  }, [])

  const checkAPI = async () => {
    try {
      console.log(`Checking API at: ${API_URL}/health`)
      
      const res = await fetch(`${API_URL}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors'
      })
      
      if (res.ok) {
        const data = await res.json()
        console.log('API Response:', data)
        setStatus({
          api: data.status === 'healthy',
          loading: false,
          error: null
        })
      } else {
        throw new Error(`HTTP ${res.status}`)
      }
    } catch (error: any) {
      console.error('API Connection Error:', error)
      setStatus({
        api: false,
        loading: false,
        error: error.message || 'Connection failed'
      })
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-600 text-white p-8">
      <div className="max-w-6xl mx-auto">
        {/* 헤더 */}
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4">🚀 Kainexa Studio</h1>
          <p className="text-xl opacity-90">한국형 AI 워크플로우 빌더</p>
        </header>

        {/* 시스템 상태 카드 */}
        <div className="bg-white/10 backdrop-blur rounded-xl p-6 mb-8">
          <h2 className="text-2xl font-bold mb-4">시스템 상태</h2>
          
          <div className="space-y-3">
            {/* Web UI 상태 */}
            <div className="flex items-center justify-between">
              <span className="font-medium">Web UI:</span>
              <span className="text-green-300">✅ 실행 중</span>
            </div>
            
            {/* API 상태 */}
            <div className="flex items-center justify-between">
              <span className="font-medium">API 서버:</span>
              <span className={
                status.loading ? 'text-yellow-300' : 
                status.api ? 'text-green-300' : 'text-red-300'
              }>
                {status.loading ? '⏳ 확인 중...' : 
                 status.api ? '✅ 연결됨' : '❌ 연결 안됨'}
              </span>
            </div>
            
            {/* API URL 정보 */}
            <div className="text-xs opacity-70 pt-2 border-t border-white/10">
              <p>API Endpoint: {API_URL}</p>
            </div>
          </div>

          {/* 에러 메시지 */}
          {status.error && (
            <div className="mt-4 p-3 bg-red-500/20 rounded">
              <p className="text-sm font-semibold mb-1">연결 오류</p>
              <p className="text-xs opacity-80">{status.error}</p>
              <p className="text-xs opacity-60 mt-1">
                API 서버가 http://192.168.1.215:4000 에서 실행 중인지 확인하세요.
              </p>
            </div>
          )}

          {/* 재시도 버튼 */}
          <button 
            onClick={checkAPI}
            disabled={status.loading}
            className="mt-4 px-4 py-2 bg-white/20 rounded hover:bg-white/30 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status.loading ? '확인 중...' : '연결 상태 다시 확인'}
          </button>
        </div>

        {/* 기능 카드 그리드 */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* 워크플로우 에디터 */}
          <Card
            href="/editor"
            icon="📊"
            title="워크플로우 에디터"
            description="드래그 & 드롭으로 AI 워크플로우를 만들어보세요"
          />

          {/* 템플릿 */}
          <Card
            href="/templates"
            icon="📚"
            title="템플릿"
            description="미리 만들어진 워크플로우 템플릿을 사용하세요"
          />

          {/* 한국어 처리 */}
          <Card
            href="/korean"
            icon="🇰🇷"
            title="한국어 처리"
            description="존댓말 감지, 변환 등 한국어 특화 기능"
          />

          {/* API 테스트 */}
          <Card
            href={`${API_URL}/health`}
            icon="🔧"
            title="API 테스트"
            description="API 엔드포인트를 직접 확인해보세요"
            external
          />

          {/* 문서 */}
          <Card
            href="/docs"
            icon="📖"
            title="문서"
            description="Kainexa Studio 사용법과 API 문서"
          />

          {/* 접속 정보 */}
          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <div className="text-3xl mb-3">🌐</div>
            <h3 className="text-xl font-bold mb-2">접속 정보</h3>
            <div className="text-sm space-y-1 opacity-80">
              <p>Web UI: http://192.168.1.215:3000</p>
              <p>API: http://192.168.1.215:4000</p>
            </div>
          </div>
        </div>

        {/* 푸터 - 클라이언트 전용 정보 */}
        <footer className="mt-12 text-center">
          {/* mounted 상태일 때만 클라이언트 정보 표시 */}
          {mounted && (
            <div className="text-xs opacity-60 space-y-1">
              <ClientOnlyInfo url={window.location.href} />
              <p>User Agent: {navigator.userAgent.substring(0, 50)}...</p>
            </div>
          )}
          
          {/* 항상 표시되는 정적 정보 */}
          <p className="text-sm opacity-75 mt-4">
            Windows에서 접속: http://192.168.1.215:3000
          </p>
        </footer>
      </div>
    </div>
  )
}

// 재사용 가능한 카드 컴포넌트
interface CardProps {
  href: string
  icon: string
  title: string
  description: string
  external?: boolean
}

function Card({ href, icon, title, description, external }: CardProps) {
  const className = "bg-white/10 backdrop-blur rounded-xl p-6 hover:bg-white/20 transition cursor-pointer"
  
  if (external) {
    return (
      <div 
        className={className}
        onClick={() => window.open(href, '_blank')}
      >
        <div className="text-3xl mb-3">{icon}</div>
        <h3 className="text-xl font-bold mb-2">{title}</h3>
        <p className="opacity-80 text-sm">{description}</p>
      </div>
    )
  }
  
  return (
    <a href={href} className={className}>
      <div className="text-3xl mb-3">{icon}</div>
      <h3 className="text-xl font-bold mb-2">{title}</h3>
      <p className="opacity-80 text-sm">{description}</p>
    </a>
  )
}