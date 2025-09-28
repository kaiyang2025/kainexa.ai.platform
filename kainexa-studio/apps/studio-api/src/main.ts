import express from 'express';
import cors from 'cors';
import { WorkflowExecutor } from '@kainexa/workflow-engine';
import { generateId } from '@kainexa/shared';

const app = express();
const PORT = process.env.PORT || 4000;

app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'Kainexa Studio API',
    sessionId: generateId()
  });
});

app.post('/api/workflow/execute', async (req, res) => {
  const { nodes, edges } = req.body;
  
  const executor = new WorkflowExecutor(nodes || [], edges || []);
  const result = await executor.execute({
    sessionId: generateId(),
    userId: 'user_001',
    variables: new Map(),
    history: [],
    currentNode: nodes?.[0]?.id || 'start',
    metadata: {}
  });
  
  res.json(result);
});

app.listen(PORT, () => {
  console.log(`🚀 API Server running on http://localhost:${PORT}`);
});
