// apps/studio-web/next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  // ⚠️ StrictMode를 false로 변경 - 드래그 이벤트가 두 번 실행되는 문제 해결
  reactStrictMode: false,
  
  // 환경 변수 기본값 설정
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://192.168.1.215:4000'
  },
  
  // Webpack 설정 추가 (개발 모드 최적화)
  webpack: (config, { dev, isServer }) => {
    // 개발 모드에서 HMR 관련 설정
    if (dev && !isServer) {
      config.watchOptions = {
        poll: 1000,
        aggregateTimeout: 300,
        ignored: /node_modules/,
      };
      
      // 소스맵 설정 (디버깅 용이)
      config.devtool = 'cheap-module-source-map';
    }
    
    return config;
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
  },
  
  // 컴파일러 옵션 (선택사항)
  compiler: {
    // React DevTools에서 컴포넌트 이름 표시
    // displayName: true,
  },
  
  // 실험적 기능 (필요시 활성화)
  experimental: {
    // optimizeCss: true,
  }
}

module.exports = nextConfig