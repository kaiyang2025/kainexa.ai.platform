import { useState, useEffect } from 'react'
import Link from 'next/link'
import styles from '@/styles/Home.module.css'

export default function Home() {
  const [isConnected, setIsConnected] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    checkConnection()
  }, [])

  const checkConnection = async () => {
    setLoading(true)
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000'
      const res = await fetch(`${apiUrl}/health`)
      const data = await res.json()
      setIsConnected(data.status === 'healthy')
    } catch (error) {
      console.error('Connection check failed:', error)
      setIsConnected(false)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.main}>
      <h1 className={styles.title}>
        🚀 Kainexa Studio
      </h1>
      <p className={styles.description}>
        한국형 AI 워크플로우 빌더
      </p>

      <div className={styles.grid}>
        <Link href="/editor" className={styles.card}>
          <h2 className={styles.cardTitle}>📊 워크플로우 에디터</h2>
          <p className={styles.cardDescription}>
            드래그 & 드롭으로 AI 워크플로우를 만들어보세요
          </p>
        </Link>

        <div className={styles.card} onClick={checkConnection}>
          <h2 className={styles.cardTitle}>🔌 시스템 상태</h2>
          <p className={styles.cardDescription}>
            API 서버: {loading ? '확인 중...' : (
              <span className={isConnected ? 'text-green-600' : 'text-red-600'}>
                {isConnected ? '✅ 연결됨' : '❌ 연결 안됨'}
              </span>
            )}
          </p>
        </div>

        <Link href="/templates" className={styles.card}>
          <h2 className={styles.cardTitle}>📚 템플릿</h2>
          <p className={styles.cardDescription}>
            미리 만들어진 워크플로우 템플릿을 사용해보세요
          </p>
        </Link>

        <Link href="/docs" className={styles.card}>
          <h2 className={styles.cardTitle}>📖 문서</h2>
          <p className={styles.cardDescription}>
            Kainexa Studio 사용법과 API 문서를 확인하세요
          </p>
        </Link>
      </div>

      <div className="mt-8 text-white text-center">
        <p>접속 주소: {typeof window !== 'undefined' && window.location.origin}</p>
        <p className="text-sm opacity-75 mt-2">
          외부 접속: http://192.168.1.215:3000
        </p>
      </div>
    </div>
  )
}