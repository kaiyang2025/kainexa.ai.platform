// apps/studio-web/src/components/workflow-editor/WorkflowEditor.tsx
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

// === 커스텀 노드 === (프로젝트 폴더 구조에 맞춰 경로 확인)
import IntentNode from './nodes/IntentNode';
import LLMNode from './nodes/LLMNode';
import APINode from './nodes/APINode';
import ConditionNode from './nodes/ConditionNode';
import LoopNode from './nodes/LoopNode';

// === 패널 ===
import NodePalette from './panels/NodePalette';
import PropertiesPanel from './panels/PropertiesPanel';
import DebugPanel from './panels/DebugPanel';

// === 유틸 ===
const nodeTypes = {
  intent: IntentNode,
  llm: LLMNode,
  api: APINode,
  condition: ConditionNode,
  loop: LoopNode,
};

const STORAGE_KEY = 'kainexa.workflow.v1';

// 간단 ID 생성기
let nodeCounter = 0;
function generateNodeId(prefix = 'node') {
  nodeCounter += 1;
  return `${prefix}_${nodeCounter}`;
}

// 타입별 기본 설정
function getDefaultConfig(type: string) {
  switch (type) {
    case 'intent':     return { threshold: 0.7, intents: [] };
    case 'llm':        return { model: 'gpt-3.5-turbo', temperature: 0.7, systemPrompt: '' };
    case 'api':        return { url: '', method: 'GET', timeout: 30000 };
    case 'condition':  return { expression: '' };
    case 'loop':       return { iterations: 3, breakCondition: '' };
    default:           return {};
  }
}

// === 메인 에디터(Provider 내부에서만 렌더) ===
function WorkflowEditorContent() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node[]>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge[]>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);

  const rfWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, fitView } = useReactFlow();

  // 연결 핸들러
  const onConnect = useCallback((params: Connection) => {
    setEdges((eds) =>
      addEdge({ ...params, animated: true, style: { stroke: '#6366f1', strokeWidth: 2 } }, eds),
    );
  }, [setEdges]);

  // 드래그 오버 허용
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // 드롭 핸들러 (JSON + text/plain 폴백)
  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();

    const raw =
      event.dataTransfer.getData('application/reactflow') ||
      event.dataTransfer.getData('text/plain');
    if (!raw) return;

    let parsed: { type: string; label?: string } | null = null;
    try { parsed = JSON.parse(raw); } catch { parsed = { type: raw, label: raw }; }

    const nodeType = parsed.type;
    const nodeLabel = parsed.label || parsed.type;
    if (!nodeType) return;

    const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });

    const newNode: Node = {
      id: generateNodeId(),
      type: nodeType,
      position,
      data: { label: nodeLabel, config: getDefaultConfig(nodeType) },
      dragHandle: '.custom-drag-handle',
    };

    setNodes((nds) => nds.concat(newNode));
    setSelectedNode(newNode);
  }, [screenToFlowPosition, setNodes]);

  // 우측 패널에서 노드 데이터 업데이트
  const updateNodeData = useCallback((nodeId: string, data: any) => {
    setNodes((nds) =>
      nds.map((node) => (node.id === nodeId ? { ...node, data: { ...node.data, ...data } } : node)),
    );
  }, [setNodes]);

  // === 로컬 저장/불러오기 ===
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const { nodes: savedNodes = [], edges: savedEdges = [] } = JSON.parse(raw);
      setNodes(savedNodes);
      setEdges(savedEdges);
      setTimeout(() => fitView(), 0);
    } catch (e) {
      console.error('Failed to load workflow:', e);
    }
  }, [setNodes, setEdges, fitView]);

  const handleSave = useCallback(() => {
    const payload = JSON.stringify({ nodes, edges });
    localStorage.setItem(STORAGE_KEY, payload);
  }, [nodes, edges]);

  const handleLoad = useCallback(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const { nodes: savedNodes = [], edges: savedEdges = [] } = JSON.parse(raw);
      setNodes(savedNodes);
      setEdges(savedEdges);
      setTimeout(() => fitView(), 0);
    } catch (e) {
      console.error('Failed to load workflow:', e);
    }
  }, [setNodes, setEdges, fitView]);

  const handleNew = useCallback(() => {
    setNodes([]);
    setEdges([]);
    localStorage.removeItem(STORAGE_KEY);
  }, [setNodes, setEdges]);

  // 자동 저장 (디바운스 500ms)
  useEffect(() => {
    const t = setTimeout(() => {
      const payload = JSON.stringify({ nodes, edges });
      localStorage.setItem(STORAGE_KEY, payload);
    }, 500);
    return () => clearTimeout(t);
  }, [nodes, edges]);

  // 단축키: 저장(S) / 불러오기(O)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === 's') { e.preventDefault(); handleSave(); }
      if ((e.ctrlKey || e.metaKey) && k === 'o') { e.preventDefault(); handleLoad(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleSave, handleLoad]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr 320px', height: '100vh', overflow: 'hidden' }}>
      {/* 좌측 팔레트 */}
      <div style={{ borderRight: '1px solid #e5e7eb', overflow: 'auto' }}>
        <NodePalette />
      </div>

      {/* 중앙 캔버스 */}
      <div style={{ position: 'relative' }}>
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
            <MiniMap />
            <Controls />
            <Background variant={BackgroundVariant.Lines} gap={16} lineWidth={1} />

            {/* 상단 툴바 (캔버스 위, 패널로 배치) */}
            <Panel position="top-center">
              <div className="flex gap-2 bg-white p-2 rounded-lg shadow-lg">
                <button
                  onClick={handleSave}
                  className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 transition-colors"
                >
                  💾 저장
                </button>
                <button
                  onClick={handleLoad}
                  className="px-4 py-2 rounded border border-gray-300 hover:bg-gray-50 transition-colors"
                >
                  📂 불러오기
                </button>
                <button
                  onClick={handleNew}
                  className="px-4 py-2 rounded border border-gray-300 hover:bg-gray-50 transition-colors"
                >
                  🗑 새로 만들기
                </button>
              </div>
            </Panel>
          </ReactFlow>
        </div>
      </div>

      {/* 우측 속성 패널 */}
      <div style={{ borderLeft: '1px solid #e5e7eb', overflow: 'auto' }}>
        <PropertiesPanel selectedNode={selectedNode} updateNodeData={updateNodeData} />
      </div>

      {/* 디버그 패널 (필요 시 표시) */}
      {isExecuting && <DebugPanel />}
    </div>
  );
}

// === 외부 래퍼(Provider) ===
export default function WorkflowEditor() {
  return (
    <ReactFlowProvider>
      <WorkflowEditorContent />
    </ReactFlowProvider>
  );
}
