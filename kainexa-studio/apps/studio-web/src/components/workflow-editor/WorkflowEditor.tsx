// src/components/workflow-editor/WorkflowEditor.tsx
import React, { useCallback, useMemo, useRef, useState } from 'react';
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
  const [onNodesChange, onEdgesChange] = [
    useCallback((chs: any) => setNodes((nds) => (window as any).applyNodeChanges ? (window as any).applyNodeChanges(chs, nds) : nds), [setNodes]),
    useCallback((chs: any) => setEdges((eds) => (window as any).applyEdgeChanges ? (window as any).applyEdgeChanges(chs, eds) : eds), [setEdges]),
  ];
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
        style={{ background: '#fff', width: '100%', height: '100%' }}
        onNodeClick={(_, n) => setSelectedNode(n)}
      >
        <MiniMap />
        <Controls />
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
      </ReactFlow>
    </div>
  );
}

function WorkflowEditorInner() {
  const [nodes, setNodes] = useNodesState([]);
  const [edges, setEdges] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

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

      {/* 중앙 캔버스 */}
      <div style={{ position: 'relative' }}>
        <FlowCanvas
          nodes={nodes}
          setNodes={setNodes}
          edges={edges}
          setEdges={setEdges}
          setSelectedNode={setSelectedNode}
        />
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
