import { useState, useEffect } from 'react'

export default function Home() {
  const [status, setStatus] = useState({
    api: false,
    loading: true,
    error: null
  })

  // API URL을 환경 변수에서 가져오기 (없으면 기본값 사용)
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://192.168.1.215:4000'

  useEffect(() => {
    checkAPI()
  }, [])

  const checkAPI = async () => {
    try {
      console.log(`Checking API at: ${API_URL}/health`)
      
      const res = await fetch(`${API_URL}/health`, {
        method: 'GET',
        headers: { 
          'Content-Type': 'application/json'
        },
        // CORS 문제 방지
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
    } catch (error) {
      console.error('API Connection Error:', error)
      setStatus({
        api: false,
        loading: false,
        error: error.message
      })
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-600 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-5xl font-bold mb-4 text-center">🚀 Kainexa Studio</h1>
        <p className="text-xl text-center mb-12">한국형 AI 워크플로우 빌더</p>

        {/* 상태 표시 */}
        <div className="bg-white/10 backdrop-blur rounded-xl p-6 mb-8">
          <h2 className="text-2xl font-bold mb-4">시스템 상태</h2>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span>Web UI:</span>
              <span className="text-green-300">✅ 실행 중</span>
            </div>
            <div className="flex items-center justify-between">
              <span>API 서버:</span>
              {status.loading ? (
                <span className="text-yellow-300">⏳ 확인 중...</span>
              ) : status.api ? (
                <span className="text-green-300">✅ 연결됨</span>
              ) : (
                <span className="text-red-300">❌ 연결 안됨</span>
              )}
            </div>
            <div className="text-xs opacity-70 mt-2">
              API URL: {API_URL}
            </div>
          </div>
          {status.error && (
            <div className="mt-4 p-3 bg-red-500/20 rounded text-sm">
              <p className="font-semibold">오류 정보:</p>
              <p>{status.error}</p>
              <p className="text-xs mt-1">API 서버가 실행 중인지 확인하세요.</p>
            </div>
          )}
          <button 
            onClick={checkAPI}
            className="mt-4 px-4 py-2 bg-white/20 rounded hover:bg-white/30 transition"
            disabled={status.loading}
          >
            {status.loading ? '확인 중...' : '다시 확인'}
          </button>
        </div>

        {/* 기능 카드들 */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <a 
            href="/editor" 
            className="bg-white/10 backdrop-blur rounded-xl p-6 hover:bg-white/20 transition cursor-pointer"
          >
            <div className="text-3xl mb-3">📊</div>
            <h2 className="text-xl font-bold mb-2">워크플로우 에디터</h2>
            <p className="opacity-80">드래그 & 드롭으로 워크플로우 생성</p>
          </a>

          <a 
            href="/templates" 
            className="bg-white/10 backdrop-blur rounded-xl p-6 hover:bg-white/20 transition cursor-pointer"
          >
            <div className="text-3xl mb-3">📚</div>
            <h2 className="text-xl font-bold mb-2">템플릿</h2>
            <p className="opacity-80">미리 만들어진 워크플로우 템플릿</p>
          </a>

          <a 
            href="/korean" 
            className="bg-white/10 backdrop-blur rounded-xl p-6 hover:bg-white/20 transition cursor-pointer"
          >
            <div className="text-3xl mb-3">🇰🇷</div>
            <h2 className="text-xl font-bold mb-2">한국어 처리</h2>
            <p className="opacity-80">존댓말 감지, 변환 등 한국어 특화 기능</p>
          </a>

          <div 
            className="bg-white/10 backdrop-blur rounded-xl p-6 hover:bg-white/20 transition cursor-pointer"
            onClick={() => window.open(`${API_URL}/health`, '_blank')}
          >
            <div className="text-3xl mb-3">🔧</div>
            <h2 className="text-xl font-bold mb-2">API 테스트</h2>
            <p className="opacity-80">API 엔드포인트 직접 확인</p>
          </div>

          <a 
            href="/docs" 
            className="bg-white/10 backdrop-blur rounded-xl p-6 hover:bg-white/20 transition cursor-pointer"
          >
            <div className="text-3xl mb-3">📖</div>
            <h2 className="text-xl font-bold mb-2">문서</h2>
            <p className="opacity-80">사용법과 API 문서</p>
          </a>

          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <div className="text-3xl mb-3">🌐</div>
            <h2 className="text-xl font-bold mb-2">접속 정보</h2>
            <div className="text-sm space-y-1 opacity-80">
              <p>Web: http://192.168.1.215:3000</p>
              <p>API: http://192.168.1.215:4000</p>
            </div>
          </div>
        </div>

        {/* 디버그 정보 */}
        <div className="mt-8 text-center text-xs opacity-60">
          <p>현재 페이지: {typeof window !== 'undefined' ? window.location.href : 'Loading...'}</p>
          <p>API 엔드포인트: {API_URL}/health</p>
        </div>
      </div>
    </div>
  )
}
