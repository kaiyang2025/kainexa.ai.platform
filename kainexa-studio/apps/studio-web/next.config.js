/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  
  // 환경 변수 기본값 설정
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://192.168.1.215:4000'
  },
  
  // API 프록시 설정 (선택적)
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://192.168.1.215:4000/:path*'
      }
    ];
  },
  
  // CORS 헤더 추가
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'Access-Control-Allow-Origin',
            value: '*'
          }
        ]
      }
    ];
  }
}

module.exports = nextConfig