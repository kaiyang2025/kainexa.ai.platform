import { useState } from 'react'
import Link from 'next/link'

export default function Editor() {
  const [nodes, setNodes] = useState([])

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/" className="text-2xl font-bold text-primary">
                Kainexa Studio
              </Link>
              <span className="text-gray-500">/ 워크플로우 에디터</span>
            </div>
            <div className="flex space-x-2">
              <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                💾 저장
              </button>
              <button className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">
                ▶️ 실행
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="flex h-[calc(100vh-73px)]">
        {/* 좌측 패널 */}
        <div className="w-64 bg-white border-r p-4">
          <h3 className="font-semibold mb-4">노드 팔레트</h3>
          <div className="space-y-2">
            <div className="p-3 bg-purple-100 rounded cursor-move">
              🧠 의도 분류
            </div>
            <div className="p-3 bg-blue-100 rounded cursor-move">
              💬 AI 응답
            </div>
            <div className="p-3 bg-green-100 rounded cursor-move">
              🌐 API 호출
            </div>
          </div>
        </div>

        {/* 중앙 캔버스 */}
        <div className="flex-1 p-8">
          <div className="w-full h-full bg-white rounded-lg shadow-lg flex items-center justify-center text-gray-400">
            <div className="text-center">
              <p className="text-2xl mb-2">📊</p>
              <p>노드를 드래그하여 시작하세요</p>
            </div>
          </div>
        </div>

        {/* 우측 패널 */}
        <div className="w-80 bg-white border-l p-4">
          <h3 className="font-semibold mb-4">속성</h3>
          <p className="text-gray-500">노드를 선택하세요</p>
        </div>
      </main>
    </div>
  )
}