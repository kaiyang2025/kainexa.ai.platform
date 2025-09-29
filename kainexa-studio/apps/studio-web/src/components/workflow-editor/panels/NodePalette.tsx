// apps/studio-web/src/components/workflow-editor/panels/NodePalette.tsx
// React 환경에서 작동하는 드래그 구현

import React, { useState, useRef, useEffect } from 'react';

interface NodeItem {
  type: string;
  label: string;
  color: string;
  icon: string;
}

const nodes: NodeItem[] = [
  { type: 'intent', label: '의도 분류', color: '#9333ea', icon: '🧠' },
  { type: 'llm', label: 'AI 응답', color: '#3b82f6', icon: '💬' },
  { type: 'api', label: 'API 호출', color: '#10b981', icon: '🌐' },
  { type: 'condition', label: '조건 분기', color: '#f97316', icon: '🔀' },
  { type: 'loop', label: '반복', color: '#ec4899', icon: '🔄' },
];

export default function NodePalette() {
  const [isDragging, setIsDragging] = useState(false);
  const [draggedNode, setDraggedNode] = useState<NodeItem | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const dragRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      setMousePos({ x: e.clientX, y: e.clientY });
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (draggedNode) {
        // React Flow 캔버스 찾기
        const reactFlowElement = document.querySelector('.react-flow');
        const targetElement = document.elementFromPoint(e.clientX, e.clientY);
        
        // React Flow 영역 안에 드롭했는지 확인
        if (reactFlowElement && reactFlowElement.contains(targetElement)) {
          console.log('✅ Dropping node:', draggedNode);
          
          // CustomEvent 대신 직접 React Flow API 호출
          // 이벤트 버블링을 통해 상위 컴포넌트로 전달
          const dropEvent = new CustomEvent('nodepalette:drop', {
            bubbles: true,
            detail: {
              type: draggedNode.type,
              label: draggedNode.label,
              clientX: e.clientX,
              clientY: e.clientY
            }
          });
          
          // React Flow 컨테이너에 이벤트 발생
          reactFlowElement.dispatchEvent(dropEvent);
        }
      }
      
      setIsDragging(false);
      setDraggedNode(null);
      document.body.style.cursor = 'default';
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, draggedNode]);

  const handleNodeMouseDown = (e: React.MouseEvent, node: NodeItem) => {
    e.preventDefault();
    e.stopPropagation();
    
    console.log('🎯 Starting drag:', node.label);
    setDraggedNode(node);
    setIsDragging(true);
    setMousePos({ x: e.clientX, y: e.clientY });
    document.body.style.cursor = 'grabbing';
  };

  return (
    <>
      <div 
        className="node-palette"
        style={{
          width: '250px',
          backgroundColor: '#f9fafb',
          borderRight: '1px solid #e5e7eb',
          padding: '16px',
          height: '100%',
          overflowY: 'auto',
          userSelect: 'none',
        }}
      >
        <h2 style={{ 
          fontSize: '18px', 
          fontWeight: 'bold', 
          marginBottom: '16px',
          color: '#111827' 
        }}>
          노드 팔레트
        </h2>
        
        <div style={{ marginBottom: '16px' }}>
          {nodes.map((node) => (
            <div
              key={node.type}
              className="node-item"
              onMouseDown={(e) => handleNodeMouseDown(e, node)}
              style={{
                padding: '12px',
                marginBottom: '8px',
                backgroundColor: 'white',
                border: `2px solid ${node.color}`,
                borderLeft: `5px solid ${node.color}`,
                borderRadius: '6px',
                cursor: 'grab',
                transition: 'transform 0.2s, box-shadow 0.2s',
                opacity: isDragging && draggedNode?.type !== node.type ? 0.5 : 1,
              }}
              onMouseEnter={(e) => {
                if (!isDragging) {
                  e.currentTarget.style.transform = 'translateX(4px)';
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
                }
              }}
              onMouseLeave={(e) => {
                if (!isDragging) {
                  e.currentTarget.style.transform = 'translateX(0)';
                  e.currentTarget.style.boxShadow = 'none';
                }
              }}
            >
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px',
                pointerEvents: 'none' 
              }}>
                <span style={{ fontSize: '20px' }}>{node.icon}</span>
                <span style={{ 
                  fontSize: '14px',
                  fontWeight: '500',
                  color: '#374151'
                }}>
                  {node.label}
                </span>
              </div>
            </div>
          ))}
        </div>

        <div style={{
          padding: '12px',
          backgroundColor: '#dbeafe',
          borderRadius: '6px',
          fontSize: '12px',
          color: '#1e40af'
        }}>
          {isDragging ? 
            `🎯 "${draggedNode?.label}" 드래그 중...` : 
            '💡 노드를 클릭하고 캔버스로 드래그하세요'
          }
        </div>
      </div>

      {/* 드래그 중인 고스트 이미지 */}
      {isDragging && draggedNode && (
        <div
          ref={dragRef}
          style={{
            position: 'fixed',
            left: mousePos.x - 60,
            top: mousePos.y - 20,
            padding: '8px 16px',
            backgroundColor: 'white',
            border: `2px solid ${draggedNode.color}`,
            borderRadius: '6px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            pointerEvents: 'none',
            zIndex: 10000,
            opacity: 0.9,
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <span>{draggedNode.icon}</span>
          <span style={{ fontSize: '14px', fontWeight: '500' }}>
            {draggedNode.label}
          </span>
        </div>
      )}
    </>
  );
}