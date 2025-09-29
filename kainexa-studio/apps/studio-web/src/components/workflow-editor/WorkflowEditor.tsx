// apps/studio-web/src/components/workflow-editor/WorkflowEditor.tsx
import React, { useCallback, useRef, useState, DragEvent } from 'react';
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

// 메인 에디터 컴포넌트 (ReactFlowProvider 내부)
function WorkflowEditorContent() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  
  const { screenToFlowPosition } = useReactFlow();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  
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

  // 드래그 오버 이벤트
  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // 교체본: onDrop 전체
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      // 1) dataTransfer에서 우선 JSON(payload) 시도 → 없으면 text/plain 폴백
      const raw =
        event.dataTransfer.getData('application/reactflow') ||
        event.dataTransfer.getData('text/plain');

      if (!raw) return;

      // 2) JSON 파싱 시도 (팔레트가 JSON으로 넣은 경우)
      let parsed: { type: string; label?: string } | null = null;
      try {
        parsed = JSON.parse(raw);
      } catch {
        // 문자열만 온 구버전 폴백
        parsed = { type: raw };
      }

      // 3) 타입/라벨 확정 (구버전 'nodeLabel' 키도 함께 폴백)
      const nodeType = parsed?.type;
      const nodeLabel =
        parsed?.label ||
        event.dataTransfer.getData('nodeLabel') || // 과거 방식 호환
        nodeType;

      if (!nodeType) return;

      // 4) 드롭 위치를 Flow 좌표로 변환
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      // 5) 새 노드 생성
      const newNode: Node = {
        id: generateNodeId(),
        type: nodeType,
        position,
        data: {
          label: nodeLabel,
          config: getDefaultConfig(nodeType),
        },
        dragHandle: '.custom-drag-handle', // (선택)
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [screenToFlowPosition, setNodes]
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
      alert('워크플로우 실행이 완료되었습니다!');
    } catch (error) {
      console.error('Execution error:', error);
      alert('워크플로우 실행 중 오류가 발생했습니다.');
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
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          defaultEdgeOptions={{
            animated: true,
            style: { stroke: '#6366f1', strokeWidth: 2 }
          }}
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
            nodeColor={(node) => {
              switch (node.type) {
                case 'intent': return '#9333ea';
                case 'llm': return '#3b82f6';
                case 'api': return '#10b981';
                case 'condition': return '#f97316';
                case 'loop': return '#ec4899';
                default: return '#6b7280';
              }
            }}
          />

          {/* 상단 툴바 */}
          <Panel position="top-center">
            <div className="flex gap-2 bg-white p-2 rounded-lg shadow-lg">
              <button
                onClick={handleSave}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors flex items-center gap-2"
              >
                💾 저장
              </button>
              <button
                onClick={handleExecute}
                disabled={isExecuting}
                className={`px-4 py-2 rounded transition-colors flex items-center gap-2 ${
                  isExecuting 
                    ? 'bg-gray-300 cursor-not-allowed' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isExecuting ? '⏳ 실행 중...' : '▶️ 실행'}
              </button>
            </div>
          </Panel>
        </ReactFlow>
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
        model: 'solar',
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

// 메인 컴포넌트 - ReactFlowProvider로 감싸기
export default function WorkflowEditor() {
  return (
    <ReactFlowProvider>
      <WorkflowEditorContent />
    </ReactFlowProvider>
  );
}