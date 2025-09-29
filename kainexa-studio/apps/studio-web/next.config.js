/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  
  // 외부 접속 허용
  images: {
    domains: ['localhost', '192.168.1.215'],
  },
  
  // API 프록시 설정
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://192.168.1.215:4000/api/:path*'
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