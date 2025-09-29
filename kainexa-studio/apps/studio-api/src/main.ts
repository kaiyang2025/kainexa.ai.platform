import express from 'express';
import cors from 'cors';

const app = express();
const PORT = process.env.PORT || 4000;

// CORS 설정 - 모든 출처 허용 (개발용)
app.use(cors({
  origin: ['http://localhost:3000', 'http://192.168.1.215:3000', 'http://192.168.1.*:3000'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'Kainexa Studio API',
    host: req.hostname
  });
});

app.post('/api/workflow/execute', async (req, res) => {
  const { nodes, edges } = req.body;
  console.log(`Workflow execution request from ${req.ip}`);
  
  res.json({
    success: true,
    result: 'Workflow executed successfully',
    executionId: `exec_${Date.now()}`
  });
});

// 0.0.0.0으로 바인딩하여 모든 네트워크 인터페이스에서 접속 가능
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 API Server running on:`);
  console.log(`   - Local: http://localhost:${PORT}`);
  console.log(`   - Network: http://192.168.1.215:${PORT}`);
});