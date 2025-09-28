import { useState } from 'react'

export default function Home() {
  const [isConnected, setIsConnected] = useState(false)

  const checkConnection = async () => {
    try {
      const res = await fetch('http://localhost:4000/health')
      const data = await res.json()
      setIsConnected(data.status === 'healthy')
    } catch (error) {
      setIsConnected(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto p-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-4">
          🚀 Kainexa Studio
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          AI Workflow Builder for Korea
        </p>
        
        <div className="bg-white rounded-lg shadow-md p-6 max-w-md">
          <h2 className="text-2xl font-semibold mb-4">시스템 상태</h2>
          
          <div className="flex items-center justify-between mb-4">
            <span>API 서버</span>
            <span className={`px-3 py-1 rounded-full text-white ${
              isConnected ? 'bg-green-500' : 'bg-red-500'
            }`}>
              {isConnected ? '연결됨' : '연결 안됨'}
            </span>
          </div>
          
          <button
            onClick={checkConnection}
            className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 transition"
          >
            연결 테스트
          </button>
        </div>
      </div>
    </div>
  )
}
