// ========================================
// Kainexa Studio - React Flow 비주얼 에디터
// apps/studio-web/src/components/workflow-editor/
// ========================================

// ============================
// 1. 메인 워크플로우 에디터 컴포넌트
// apps/studio-web/src/components/workflow-editor/WorkflowEditor.tsx
// ============================

import React, { useCallback, useRef, useState } from 'react';
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
  ReactFlowInstance,
  BackgroundVariant,
  NodeChange,
  EdgeChange,
  Panel,
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

// Hooks & Utils
import { useWorkflowStore } from '../../stores/workflowStore';
import { generateNodeId } from '../../utils/nodeUtils';

const nodeTypes = {
  intent: IntentNode,
  llm: LLMNode,
  api: APINode,
  condition: ConditionNode,
  loop: LoopNode,
};

const initialNodes: Node[] = [
  {
    id: '1',
    type: 'intent',
    position: { x: 250, y: 100 },
    data: { 
      label: '시작',
      config: {
        intents: [],
        threshold: 0.7
      }
    },
  },
];

const initialEdges: Edge[] = [];

export default function WorkflowEditor() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  
  const { 
    currentWorkflow, 
    saveWorkflow, 
    executeWorkflow 
  } = useWorkflowStore();

  // 노드 연결
  const onConnect = useCallback(
    (params: Connection) => {
      setEdges((eds) => addEdge({ 
        ...params, 
        animated: true,
        style: { stroke: '#6366f1', strokeWidth: 2 }
      }, eds));
    },
    [setEdges]
  );

  // 드래그 앤 드롭
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (!reactFlowWrapper.current || !reactFlowInstance) return;

      const type = event.dataTransfer.getData('nodeType');
      const label = event.dataTransfer.getData('nodeLabel');

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode: Node = {
        id: generateNodeId(),
        type,
        position,
        data: { 
          label,
          config: getDefaultConfig(type)
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  // 노드 선택
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  // 노드 설정 업데이트
  const updateNodeData = useCallback((nodeId: string, data: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...data } }
          : node
      )
    );
  }, [setNodes]);

  // 워크플로우 저장
  const handleSave = useCallback(async () => {
    const workflow = {
      id: currentWorkflow?.id || generateNodeId(),
      name: currentWorkflow?.name || '새 워크플로우',
      nodes,
      edges,
      updatedAt: new Date().toISOString(),
    };
    
    await saveWorkflow(workflow);
    alert('워크플로우가 저장되었습니다!');
  }, [nodes, edges, currentWorkflow, saveWorkflow]);

  // 워크플로우 실행
  const handleExecute = useCallback(async () => {
    setIsExecuting(true);
    try {
      const result = await executeWorkflow({
        nodes,
        edges,
        testInput: '안녕하세요'
      });
      console.log('Execution result:', result);
    } catch (error) {
      console.error('Execution error:', error);
    } finally {
      setIsExecuting(false);
    }
  }, [nodes, edges, executeWorkflow]);

  return (
    <div className="h-screen flex">
      {/* 왼쪽 패널 - 노드 팔레트 */}
      <NodePalette />

      {/* 중앙 - 에디터 */}
      <div className="flex-1" ref={reactFlowWrapper}>
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
          >
            <Background 
              variant={BackgroundVariant.Dots} 
              gap={12} 
              size={1} 
              color="#e5e7eb"
            />
            <Controls />
            <MiniMap 
              style={{
                height: 120,
                backgroundColor: '#f3f4f6'
              }}
              maskColor="rgb(243, 244, 246, 0.7)"
            />

            {/* 상단 툴바 */}
            <Panel position="top-center" className="flex gap-2 bg-white p-2 rounded-lg shadow-lg">
              <button
                onClick={handleSave}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
              >
                💾 저장
              </button>
              <button
                onClick={handleExecute}
                disabled={isExecuting}
                className={`px-4 py-2 rounded transition-colors ${
                  isExecuting 
                    ? 'bg-gray-300 cursor-not-allowed' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isExecuting ? '⏳ 실행 중...' : '▶️ 실행'}
              </button>
            </Panel>
          </ReactFlow>
        </ReactFlowProvider>
      </div>

      {/* 오른쪽 패널 - 속성 편집기 */}
      <PropertiesPanel 
        selectedNode={selectedNode} 
        updateNodeData={updateNodeData}
      />

      {/* 디버그 패널 (하단) */}
      {isExecuting && <DebugPanel />}
    </div>
  );
}

// 노드 타입별 기본 설정
function getDefaultConfig(type: string) {
  switch (type) {
    case 'intent':
      return {
        intents: [],
        threshold: 0.7,
        fallback: 'unknown'
      };
    case 'llm':
      return {
        model: 'gpt-3.5-turbo',
        temperature: 0.7,
        maxTokens: 500,
        systemPrompt: '',
        userPromptTemplate: ''
      };
    case 'api':
      return {
        url: '',
        method: 'GET',
        headers: {},
        timeout: 30000
      };
    case 'condition':
      return {
        conditions: [],
        defaultBranch: null
      };
    case 'loop':
      return {
        maxIterations: 10,
        breakCondition: ''
      };
    default:
      return {};
  }
}
