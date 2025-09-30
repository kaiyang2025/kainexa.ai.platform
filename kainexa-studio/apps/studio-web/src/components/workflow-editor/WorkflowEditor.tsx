// kainexa-studio/apps/studio-web/src/components/workflow-editor/WorkflowEditor.tsx
// API Config를 사용하여 환경에 맞게 자동으로 URL 선택

import React, { useCallback, useRef, useState, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  ReactFlowProvider,
  BackgroundVariant,
  useReactFlow,
} from 'reactflow';
import 'reactflow/dist/style.css';

// API Configuration
import { API } from '@/config/api.config';

// Custom Nodes (기존과 동일)
import IntentNode from '../nodes/IntentNode';
import LLMNode from '../nodes/LLMNode';
import APINode from '../nodes/APINode';
import ConditionNode from '../nodes/ConditionNode';
import LoopNode from '../nodes/LoopNode';

import NodePalette from '../panels/NodePalette';
import PropertiesPanel from '../panels/PropertiesPanel';
import DebugPanel from '../panels/DebugPanel';

const nodeTypes = {
  intent: IntentNode,
  llm: LLMNode,
  api: APINode,
  condition: ConditionNode,
  loop: LoopNode,
};

const STORAGE_KEY = 'kainexa.workflow.v1';

let nodeCounter = 0;
function generateNodeId(prefix = 'node') {
  nodeCounter += 1;
  return `${prefix}_${nodeCounter}`;
}

function getDefaultConfig(type: string) {
  switch (type) {
    case 'intent':     return { threshold: 0.7, intents: [] };
    case 'llm':        return { model: 'solar', temperature: 0.7, systemPrompt: '' };
    case 'api':        return { url: 'https://api.example.com/endpoint', method: 'GET', timeout: 30000 };
    case 'condition':  return { expression: '' };
    case 'loop':       return { iterations: 3, breakCondition: '' };
    default:           return {};
  }
}

// 실행 툴바 컴포넌트
function ExecutionToolbar({ 
  nodes, 
  edges, 
  onExecute, 
  onSave, 
  onLoad, 
  onNew 
}: {
  nodes: Node[];
  edges: Edge[];
  onExecute: () => void;
  onSave: () => void;
  onLoad: () => void;
  onNew: () => void;
}) {
  const [isExecuting, setIsExecuting] = useState(false);
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [apiUrl, setApiUrl] = useState<string>('');

  // API 상태 체크
  useEffect(() => {
    checkAPIStatus();
    const interval = setInterval(checkAPIStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkAPIStatus = async () => {
    try {
      // API Config를 사용하여 URL 자동 선택
      const healthUrl = API.health();
      setApiUrl(healthUrl.replace('/api/v1/health', '')); // 베이스 URL 표시용
      
      console.log(`Checking API at: ${healthUrl}`);
      
      const response = await fetch(healthUrl);
      if (response.ok) {
        const data = await response.json();
        console.log('API connected:', data);
        setApiStatus('connected');
      } else {
        console.warn('API returned non-OK status:', response.status);
        setApiStatus('disconnected');
      }
    } catch (error) {
      console.error('API connection error:', error);
      setApiStatus('disconnected');
    }
  };

  const handleExecute = async () => {
    setIsExecuting(true);
    try {
      console.log('Executing workflow with nodes:', nodes);
      console.log('Edges:', edges);
      
      // API Config를 사용하여 워크플로우 실행
      const response = await fetch(API.workflowExecute(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ nodes, edges }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Execution result:', result);
        alert(`✅ 워크플로우 실행 성공!\n${result.message || '실행이 완료되었습니다.'}`);
      } else {
        const error = await response.text();
        throw new Error(error || '실행 실패');
      }
    } catch (error) {
      console.error('Execution error:', error);
      alert(`❌ 워크플로우 실행 실패\n${error.message || '알 수 없는 오류'}`);
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div style={{
      position: 'absolute',
      top: '20px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 10,
      display: 'flex',
      gap: '8px',
      padding: '8px',
      background: 'white',
      borderRadius: '8px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    }}>
      {/* API 상태 표시 */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        padding: '8px 12px',
        borderRadius: '6px',
        backgroundColor: apiStatus === 'connected' ? '#e6ffed' : 
                        apiStatus === 'disconnected' ? '#ffebe9' : '#fff3cd',
        border: `1px solid ${apiStatus === 'connected' ? '#52c41a' : 
                             apiStatus === 'disconnected' ? '#ff4d4f' : '#faad14'}`,
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: apiStatus === 'connected' ? '#52c41a' : 
                          apiStatus === 'disconnected' ? '#ff4d4f' : '#faad14',
          animation: apiStatus === 'checking' ? 'pulse 1.5s infinite' : 'none',
        }} />
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
          <span style={{ fontSize: '14px', fontWeight: 500 }}>
            {apiStatus === 'connected' ? '✅ API 연결됨' : 
             apiStatus === 'disconnected' ? '❌ API 연결 안됨' : '⏳ API 확인 중...'}
          </span>
          {apiUrl && (
            <span style={{ fontSize: '11px', opacity: 0.7 }}>
              {apiUrl}
            </span>
          )}
        </div>
      </div>

      <div style={{ width: '1px', background: '#e0e0e0', margin: '0 4px' }} />

      {/* 실행 버튼 */}
      <button
        onClick={() => {
          onExecute();
          handleExecute();
        }}
        disabled={isExecuting || nodes.length === 0 || apiStatus !== 'connected'}
        style={{
          padding: '8px 16px',
          borderRadius: '6px',
          border: 'none',
          backgroundColor: isExecuting ? '#faad14' : '#52c41a',
          color: 'white',
          fontWeight: 'bold',
          cursor: (isExecuting || nodes.length === 0 || apiStatus !== 'connected') ? 'not-allowed' : 'pointer',
          opacity: (isExecuting || nodes.length === 0 || apiStatus !== 'connected') ? 0.6 : 1,
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}
      >
        {isExecuting ? (
          <>
            <span style={{ display: 'inline-block', animation: 'spin 1s linear infinite' }}>⚙️</span>
            실행 중...
          </>
        ) : (
          <>
            ▶️ 실행
          </>
        )}
      </button>

      {/* 저장 버튼 */}
      <button
        onClick={onSave}
        style={{
          padding: '8px 16px',
          borderRadius: '6px',
          border: 'none',
          backgroundColor: '#1890ff',
          color: 'white',
          fontWeight: 'bold',
          cursor: 'pointer',
        }}
      >
        💾 저장
      </button>

      {/* 불러오기 버튼 */}
      <button
        onClick={onLoad}
        style={{
          padding: '8px 16px',
          borderRadius: '6px',
          border: '1px solid #d9d9d9',
          backgroundColor: 'white',
          color: '#000',
          fontWeight: 'bold',
          cursor: 'pointer',
        }}
      >
        📂 불러오기
      </button>

      {/* 새로 만들기 버튼 */}
      <button
        onClick={onNew}
        style={{
          padding: '8px 16px',
          borderRadius: '6px',
          border: '1px solid #d9d9d9',
          backgroundColor: 'white',
          color: '#000',
          fontWeight: 'bold',
          cursor: 'pointer',
        }}
      >
        🗑 새로 만들기
      </button>
    </div>
  );
}

// 메인 워크플로우 에디터 컴포넌트
function WorkflowEditorContent() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node[]>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge[]>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  const rfWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, fitView } = useReactFlow();

  const onConnect = useCallback((params: Connection) => {
    setEdges((eds) =>
      addEdge({ ...params, animated: true, style: { stroke: '#6366f1', strokeWidth: 2 } }, eds),
    );
  }, [setEdges]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();

    const raw = event.dataTransfer.getData('application/reactflow') || 
                 event.dataTransfer.getData('text/plain');
    if (!raw) return;

    let parsed: { type: string; label?: string };
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = { type: raw };
    }

    const position = screenToFlowPosition({
      x: event.clientX,
      y: event.clientY,
    });

    const newNode: Node = {
      id: generateNodeId(parsed.type),
      type: parsed.type,
      position,
      data: {
        label: parsed.label || `${parsed.type} Node`,
        config: getDefaultConfig(parsed.type),
      },
    };

    setNodes((nds) => nds.concat(newNode));
  }, [screenToFlowPosition, setNodes]);

  const updateNodeData = useCallback((nodeId: string, newData: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...newData } }
          : node
      )
    );
  }, [setNodes]);

  const handleExecute = useCallback(() => {
    console.log('Executing workflow...');
    console.log('Nodes:', nodes);
    console.log('Edges:', edges);
  }, [nodes, edges]);

  const handleSave = useCallback(() => {
    const workflow = { nodes, edges, version: '1.0' };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(workflow));
    alert('💾 워크플로우가 저장되었습니다.');
  }, [nodes, edges]);

  const handleLoad = useCallback(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) {
      alert('저장된 워크플로우가 없습니다.');
      return;
    }
    
    try {
      const workflow = JSON.parse(saved);
      setNodes(workflow.nodes || []);
      setEdges(workflow.edges || []);
      setTimeout(() => fitView(), 50);
      alert('📂 워크플로우를 불러왔습니다.');
    } catch (error) {
      alert('워크플로우를 불러오는 중 오류가 발생했습니다.');
      console.error(error);
    }
  }, [setNodes, setEdges, fitView]);

  const handleNew = useCallback(() => {
    if (confirm('현재 워크플로우를 초기화하시겠습니까?')) {
      setNodes([]);
      setEdges([]);
      setSelectedNode(null);
    }
  }, [setNodes, setEdges]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr 320px', height: '100vh', overflow: 'hidden' }}>
      <div style={{ borderRight: '1px solid #e5e7eb', overflow: 'auto', backgroundColor: '#f9fafb' }}>
        <NodePalette />
      </div>

      <div style={{ position: 'relative', backgroundColor: '#fafafa' }}>
        <ExecutionToolbar
          nodes={nodes}
          edges={edges}
          onExecute={handleExecute}
          onSave={handleSave}
          onLoad={handleLoad}
          onNew={handleNew}
        />
        
        <div ref={rfWrapper} style={{ position: 'absolute', inset: 0 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onDragOver={onDragOver}
            onDrop={onDrop}
            fitView
            snapToGrid
            snapGrid={[16, 16]}
            style={{ background: '#fff', width: '100%', height: '100%' }}
            onNodeClick={(_, n) => setSelectedNode(n)}
          >
            <MiniMap style={{ height: 120 }} zoomable pannable />
            <Controls />
            <Background variant={BackgroundVariant.Lines} gap={16} lineWidth={1} />
          </ReactFlow>
        </div>

        {nodes.length === 0 && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
            color: '#666',
            pointerEvents: 'none',
          }}>
            <h3 style={{ fontSize: '18px', marginBottom: '8px' }}>워크플로우를 시작하세요</h3>
            <p style={{ fontSize: '14px' }}>왼쪽 팔레트에서 노드를 드래그하여 캔버스에 놓으세요</p>
          </div>
        )}
      </div>

      <div style={{ borderLeft: '1px solid #e5e7eb', overflow: 'auto', backgroundColor: '#f9fafb' }}>
        <PropertiesPanel selectedNode={selectedNode} updateNodeData={updateNodeData} />
      </div>
    </div>
  );
}

// CSS 애니메이션 추가
if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes pulse {
      0% { opacity: 1; }
      50% { opacity: 0.5; }
      100% { opacity: 1; }
    }
    
    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(style);
}

export default function WorkflowEditor() {
  return (
    <ReactFlowProvider>
      <WorkflowEditorContent />
    </ReactFlowProvider>
  );
}