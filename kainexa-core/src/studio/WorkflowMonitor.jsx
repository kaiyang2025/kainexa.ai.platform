// src/studio/WorkflowMonitor.jsx
import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';

const WorkflowMonitor = ({ workflowId }) => {
  const [metrics, setMetrics] = useState({});
  const [executions, setExecutions] = useState([]);
  
  useEffect(() => {
    // WebSocket 연결로 실시간 모니터링
    const ws = new WebSocket(`ws://localhost:8000/ws/workflows/${workflowId}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'execution') {
        setExecutions(prev => [...prev, data.execution].slice(-50));
      }
      
      if (data.type === 'metrics') {
        setMetrics(data.metrics);
      }
    };
    
    return () => ws.close();
  }, [workflowId]);
  
  return (
    <div className="workflow-monitor">
      {/* 실행 상태 */}
      <div className="status-panel">
        <h3>Current Executions</h3>
        <div className="execution-list">
          {executions.map(exec => (
            <div key={exec.id} className="execution-item">
              <span className={`status ${exec.status}`}>{exec.status}</span>
              <span>{exec.sessionId}</span>
              <span>{exec.currentStep}</span>
              <span>{exec.duration}ms</span>
            </div>
          ))}
        </div>
      </div>
      
      {/* 성능 메트릭 */}
      <div className="metrics-panel">
        <h3>Performance Metrics</h3>
        <div className="metric-cards">
          <div className="metric-card">
            <h4>Success Rate</h4>
            <div className="metric-value">{metrics.successRate}%</div>
          </div>
          <div className="metric-card">
            <h4>Avg Duration</h4>
            <div className="metric-value">{metrics.avgDuration}ms</div>
          </div>
          <div className="metric-card">
            <h4>Total Executions</h4>
            <div className="metric-value">{metrics.totalExecutions}</div>
          </div>
        </div>
      </div>
      
      {/* 단계별 성능 */}
      <div className="step-performance">
        <h3>Step Performance</h3>
        <Bar 
          data={{
            labels: Object.keys(metrics.stepDurations || {}),
            datasets: [{
              label: 'Average Duration (ms)',
              data: Object.values(metrics.stepDurations || {}),
              backgroundColor: 'rgba(99, 102, 241, 0.5)'
            }]
          }}
        />
      </div>
    </div>
  );
};