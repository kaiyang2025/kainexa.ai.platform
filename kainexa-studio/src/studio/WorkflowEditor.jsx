// src/studio/WorkflowEditor.jsx
import React, { useState, useCallback, useRef } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';

// 커스텀 노드 타입
import IntentNode from './nodes/IntentNode';
import LLMNode from './nodes/LLMNode';
import APINode from './nodes/APINode';
import ConditionNode from './nodes/ConditionNode';
import ParallelNode from './nodes/ParallelNode';

const nodeTypes = {
  intent: IntentNode,
  llm: LLMNode,
  api: APINode,
  condition: ConditionNode,
  parallel: ParallelNode,
};

const WorkflowEditor = () => {
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [workflowName, setWorkflowName] = useState('Untitled Workflow');
  
  // 노드 연결
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // 노드 추가
  const addNode = (type) => {
    const newNode = {
      id: `${type}_${Date.now()}`,
      type,
      position: { x: 250, y: 250 },
      data: { 
        label: `New ${type} Node`,
        params: getDefaultParams(type)
      },
    };
    setNodes((nds) => nds.concat(newNode));
  };

  // YAML로 변환
  const exportToYAML = () => {
    const workflow = {
      name: workflowName,
      version: '1.0',
      entry_point: nodes[0]?.id || 'start',
      graph: nodes.map(node => ({
        step: node.id,
        type: node.type,
        params: node.data.params,
        next: edges
          .filter(e => e.source === node.id)
          .map(e => e.target),
        policy: node.data.policy
      }))
    };
    
    return yaml.dump(workflow);
  };

  // YAML에서 가져오기
  const importFromYAML = (yamlContent) => {
    const workflow = yaml.load(yamlContent);
    
    // 노드 생성
    const newNodes = workflow.graph.map((step, index) => ({
      id: step.step,
      type: step.type,
      position: { 
        x: 100 + (index % 3) * 200, 
        y: 100 + Math.floor(index / 3) * 150 
      },
      data: {
        label: step.step,
        params: step.params,
        policy: step.policy
      }
    }));
    
    // 엣지 생성
    const newEdges = [];
    workflow.graph.forEach(step => {
      step.next?.forEach(target => {
        newEdges.push({
          id: `${step.step}-${target}`,
          source: step.step,
          target: target
        });
      });
    });
    
    setNodes(newNodes);
    setEdges(newEdges);
    setWorkflowName(workflow.name);
  };

  return (
    <div className="workflow-editor" style={{ height: '100vh' }}>
      {/* 툴바 */}
      <div className="toolbar" style={{
        position: 'absolute',
        top: 10,
        left: 10,
        zIndex: 10,
        background: 'white',
        padding: '10px',
        borderRadius: '8px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
      }}>
        <input 
          type="text" 
          value={workflowName}
          onChange={(e) => setWorkflowName(e.target.value)}
          style={{ marginBottom: '10px', width: '100%' }}
        />
        
        <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
          <button onClick={() => addNode('intent')}>
            + Intent Classify
          </button>
          <button onClick={() => addNode('llm')}>
            + LLM Generate
          </button>
          <button onClick={() => addNode('api')}>
            + API Call
          </button>
          <button onClick={() => addNode('condition')}>
            + Condition
          </button>
          <button onClick={() => addNode('parallel')}>
            + Parallel
          </button>
        </div>
        
        <div style={{ marginTop: '10px', display: 'flex', gap: '5px' }}>
          <button onClick={() => console.log(exportToYAML())}>
            Export YAML
          </button>
          <button onClick={() => document.getElementById('import').click()}>
            Import YAML
          </button>
          <input 
            id="import"
            type="file" 
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files[0];
              const reader = new FileReader();
              reader.onload = (e) => importFromYAML(e.target.result);
              reader.readAsText(file);
            }}
          />
          <button onClick={saveWorkflow}>Save</button>
          <button onClick={deployWorkflow}>Deploy</button>
        </div>
      </div>

      {/* 노드 속성 패널 */}
      {selectedNode && (
        <NodeProperties 
          node={selectedNode}
          onUpdate={(updates) => {
            setNodes((nds) =>
              nds.map((node) =>
                node.id === selectedNode.id
                  ? { ...node, data: { ...node.data, ...updates } }
                  : node
              )
            );
          }}
          onClose={() => setSelectedNode(null)}
        />
      )}

      {/* React Flow 캔버스 */}
      <ReactFlowProvider>
        <div ref={reactFlowWrapper} style={{ height: '100%' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={(event, node) => setSelectedNode(node)}
            nodeTypes={nodeTypes}
            fitView
          >
            <Controls />
            <MiniMap />
            <Background variant="dots" gap={12} size={1} />
          </ReactFlow>
        </div>
      </ReactFlowProvider>
    </div>
  );
};