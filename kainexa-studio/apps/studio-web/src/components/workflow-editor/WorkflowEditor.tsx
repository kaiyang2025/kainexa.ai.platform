// kainexa-studio/apps/studio-web/src/components/workflow-editor/WorkflowEditor.tsx
// 실행 버튼이 포함된 수정된 워크플로우 에디터

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
  Panel,
  useReactFlow,
} from 'reactflow';
import 'reactflow/dist/style.css';

// Custom Nodes
import IntentNode from './nodes/IntentNode';
import LLMNode from './nodes/LLMNode';
import APINode from './nodes/APINode';
import ConditionNode from './nodes/ConditionNode';
import LoopNode from './nodes/LoopNode';

// Panels
import NodePalette from './panels/NodePalette';
import PropertiesPanel from './panels/PropertiesPanel';
import DebugPanel from './panels/DebugPanel';

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

// 실행 버튼을 별도 컴포넌트로 분리
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

  // API 상태 체크
  useEffect(() => {
    checkAPIStatus();
    const interval = setInterval(checkAPIStatus, 30000); // 30초마다 체크
    return () => clearInterval(interval);
  }, []);

  const checkAPIStatus = async () => {
    try {
      // Core API 헬스 체크
      const response = await fetch('http://localhost:8000/api/v1/health');
      if (response.ok) {
        setApiStatus('connected');
      } else {
        setApiStatus('disconnected');
      }
    } catch (error) {
      setApiStatus('disconnected');
    }
  };

  const handleExecute = async () => {
    setIsExecuting(true);
    try {
      // 실제 실행 로직
      console.log('Executing workflow with nodes:', nodes);
      console.log('Edges:', edges);
      
      // Core API 호출 시뮬레이션
      const response = await fetch('http://localhost:8000/api/v1/workflow/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ nodes, edges }),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Execution result:', result);
        alert('워크플로우가 성공적으로 실행되었습니다!');
      } else {
        throw new Error('실행 실패');
      }
    } catch (error) {
      console.error('Execution error:', error);
      alert('워크플로우 실행 중 오류가 발생했습니다.');
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
        <span style={{ fontSize: '14px', fontWeight: 500 }}>
          {apiStatus === 'connected' ? 'API 연결됨' : 
           apiStatus === 'disconnected' ? 'API 연결 안됨' : 'API 확인 중...'}
        </span>
      </div>

      <div style={{ width: '1px', background: '#e0e0e0', margin: '0 4px' }} />

      {/* 실행 버튼 */}
      <button
        onClick={() => {
          onExecute();
          handleExecute();
        }}
        disabled={isExecuting || nodes.length === 0}
        style={{
          padding: '8px 16px',
          borderRadius: '6px',
          border: 'none',
          backgroundColor: isExecuting ? '#faad14' : '#52c41a',
          color: 'white',
          fontWeight: 'bold',
          cursor: isExecuting || nodes.length === 0 ? 'not-allowed' : 'pointer',
          opacity: isExecuting || nodes.length === 0 ? 0.6 : 1,
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

function WorkflowEditorContent() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node[]>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge[]>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  const rfWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, fitView } = useReactFlow();

  // 연결 핸들러
  const onConnect = useCallback((params: Connection) => {
    setEdges((eds) =>
      addEdge({ ...params, animated: true, style: { stroke: '#6366f1', strokeWidth: 2 } }, eds),
    );
  }, [setEdges]);

  // 드래그 오버
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // 드롭 핸들러
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

  // 노드 데이터 업데이트
  const updateNodeData = useCallback((nodeId: string, newData: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...newData } }
          : node
      )
    );
  }, [setNodes]);

  // 워크플로우 실행
  const handleExecute = useCallback(() => {
    console.log('Executing workflow...');
    console.log('Nodes:', nodes);
    console.log('Edges:', edges);
  }, [nodes, edges]);

  // 저장
  const handleSave = useCallback(() => {
    const workflow = { nodes, edges, version: '1.0' };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(workflow));
    alert('워크플로우가 저장되었습니다.');
  }, [nodes, edges]);

  // 불러오기
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
      alert('워크플로우를 불러왔습니다.');
    } catch (error) {
      alert('워크플로우를 불러오는 중 오류가 발생했습니다.');
      console.error(error);
    }
  }, [setNodes, setEdges, fitView]);

  // 새로 만들기
  const handleNew = useCallback(() => {
    if (confirm('현재 워크플로우를 초기화하시겠습니까?')) {
      setNodes([]);
      setEdges([]);
      setSelectedNode(null);
    }
  }, [setNodes, setEdges]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr 320px', height: '100vh', overflow: 'hidden' }}>
      {/* 좌측 노드 팔레트 */}
      <div style={{ borderRight: '1px solid #e5e7eb', overflow: 'auto', backgroundColor: '#f9fafb' }}>
        <NodePalette />
      </div>

      {/* 중앙 캔버스 */}
      <div style={{ position: 'relative', backgroundColor: '#fafafa' }}>
        {/* 실행 툴바를 캔버스 내부에 배치 */}
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
            <MiniMap 
              style={{
                height: 120,
              }}
              zoomable
              pannable
            />
            <Controls />
            <Background variant={BackgroundVariant.Lines} gap={16} lineWidth={1} />
          </ReactFlow>
        </div>

        {/* 노드가 없을 때 안내 메시지 */}
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

      {/* 우측 속성 패널 */}
      <div style={{ borderLeft: '1px solid #e5e7eb', overflow: 'auto', backgroundColor: '#f9fafb' }}>
        <PropertiesPanel selectedNode={selectedNode} updateNodeData={updateNodeData} />
      </div>
    </div>
  );
}

// CSS 애니메이션 추가
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

export default function WorkflowEditor() {
  return (
    <ReactFlowProvider>
      <WorkflowEditorContent />
    </ReactFlowProvider>
  );
}