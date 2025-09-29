// src/components/workflow-editor/WorkflowEditor.tsx
import React, { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  ReactFlowProvider,
  useReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
} from 'reactflow';

import 'reactflow/dist/style.css';

import NodePalette from './panels/NodePalette';
import PropertiesPanel from './panels/PropertiesPanel';

// 커스텀 노드들
import IntentNode from './nodes/IntentNode';
import LLMNode from './nodes/LLMNode';
import APINode from './nodes/APINode';
import ConditionNode from './nodes/ConditionNode';
import LoopNode from './nodes/LoopNode';

// 노드 타입 매핑
const nodeTypes = {
  intent: IntentNode,
  llm: LLMNode,
  api: APINode,
  condition: ConditionNode,
  loop: LoopNode,
};

// 기본 설정 생성기
function getDefaultConfig(type: string) {
  switch (type) {
    case 'intent':
      return { threshold: 0.7, intents: [] };
    case 'llm':
      return { model: 'gpt-3.5-turbo', temperature: 0.7, systemPrompt: '' };
    case 'api':
      return { url: '', method: 'GET', timeout: 30000 };
    case 'condition':
      return { expression: '' };
    case 'loop':
      return { iterations: 3, breakCondition: '' };
    default:
      return {};
  }
}

// 간단 ID 생성기
let nodeCounter = 0;
function generateNodeId(prefix = 'node') {
  nodeCounter += 1;
  return `${prefix}_${nodeCounter}`;
}

// 로컬 저장 키
const STORAGE_KEY = 'kainexa.workflow.v1';

// ReactFlow 내부에서만 쓰는 캔버스 컴포넌트
function FlowCanvas({
  nodes, setNodes, edges, setEdges, setSelectedNode,
}: {
  nodes: Node[]; setNodes: React.Dispatch<React.SetStateAction<Node[]>>;
  edges: Edge[]; setEdges: React.Dispatch<React.SetStateAction<Edge[]>>;
  setSelectedNode: (node: Node | null) => void;
}) {
  const rfWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  // 연결
  const onConnect = useCallback((params: Connection) => {
    setEdges((eds) => addEdge({ ...params, animated: true, style: { stroke: '#6366f1', strokeWidth: 2 } }, eds));
  }, [setEdges]);

  // 드래그 오버
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  // 드롭
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
  }, [screenToFlowPosition, setNodes, setSelectedNode]);

  // 노드/엣지 체인지
  const onNodesChange = useCallback(
    (changes: any) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [setNodes]
  );
  const onEdgesChange = useCallback(
    (changes: any) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [setEdges]
  );

  // 위는 타입 단순화를 위한 방어. reactflow v11에서는 import { applyNodeChanges, applyEdgeChanges } 사용 권장.

  return (
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
        {/* 격자(선) 배경: Lines 또는 Cross로 취향 선택 */}
        <Background variant={BackgroundVariant.Lines} gap={16} lineWidth={1} />
      </ReactFlow>
    </div>
  );
}

function WorkflowEditorInner() {
  const [nodes, setNodes] = useNodesState([]);
  const [edges, setEdges] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  
  //자동 저장 디바운스 500ms
  useEffect(() => {
  const t = setTimeout(() => {
    const payload = JSON.stringify({ nodes, edges });
    localStorage.setItem(STORAGE_KEY, payload);
  }, 500);
  return () => clearTimeout(t);
  }, [nodes, edges]);

  // ── 로컬 저장/불러오기 ──────────────────────────────
  // 초기 로드 (한 번만)
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const { nodes: savedNodes = [], edges: savedEdges = [] } = JSON.parse(raw);
      setNodes(savedNodes);
      setEdges(savedEdges);
    } catch (e) {
      console.error('Failed to load workflow:', e);
    }
  }, [setNodes, setEdges]);

  const saveWorkflow = useCallback(() => {
    const payload = JSON.stringify({ nodes, edges });
    localStorage.setItem(STORAGE_KEY, payload);
  }, [nodes, edges]);

  const loadWorkflow = useCallback(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const { nodes: savedNodes = [], edges: savedEdges = [] } = JSON.parse(raw);
      setNodes(savedNodes);
      setEdges(savedEdges);
    } catch (e) {
      console.error('Failed to load workflow:', e);
    }
  }, [setNodes, setEdges]);

  const clearWorkflow = useCallback(() => {
    setNodes([]);
    setEdges([]);
    localStorage.removeItem(STORAGE_KEY);
  }, [setNodes, setEdges]);

  // 단축키: Ctrl/Cmd+S 저장, Ctrl/Cmd+O 불러오기
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        saveWorkflow();
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') {
        e.preventDefault();
        loadWorkflow();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [saveWorkflow, loadWorkflow]);
  // ───────────────────────────────────────────────────

  const updateNodeData = useCallback((nodeId: string, data: any) => {
    setNodes((nds) =>
      nds.map((n) => (n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n)),
    );
  }, [setNodes]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr 320px', height: '100vh', overflow: 'hidden' }}>
      {/* 좌측 팔레트 */}
      <div style={{ borderRight: '1px solid #e5e7eb', overflow: 'auto' }}>
        <NodePalette />
    </div>

    {/* 중앙 캔버스 + 툴바 */}
    <div style={{ position: 'relative' }}>
      <FlowCanvas ... />
      {/* 우상단 툴바 (컨테이너 안으로 이동) */}
    <div style={{ position:'absolute', top:8, right:8, display:'flex', gap:8, zIndex:10,
                    background:'rgba(255,255,255,0.8)', padding:6, borderRadius:8,
                    boxShadow:'0 2px 8px rgba(0,0,0,0.08)' }}>
        <button onClick={saveWorkflow} style={{ padding:'6px 10px', borderRadius:6, border:'1px solid #e5e7eb' }}>저장</button>
        <button onClick={loadWorkflow} style={{ padding:'6px 10px', borderRadius:6, border:'1px solid #e5e7eb' }}>불러오기</button>
        <button onClick={clearWorkflow} style={{ padding:'6px 10px', borderRadius:6, border:'1px solid #e5e7eb' }}>새로 만들기</button>
      </div>
    </div>

      {/* 우측 속성 패널 */}
      <div style={{ borderLeft: '1px solid #e5e7eb', overflow: 'auto' }}>
        <PropertiesPanel selectedNode={selectedNode} updateNodeData={updateNodeData} />
       </div>
    </div>
  );
}

export default function WorkflowEditor() {
  return (
    <ReactFlowProvider>
      <WorkflowEditorInner />
    </ReactFlowProvider>
  );
}
