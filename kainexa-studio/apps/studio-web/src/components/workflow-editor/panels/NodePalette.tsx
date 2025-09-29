// apps/studio-web/src/components/workflow-editor/panels/NodePalette.tsx
// 완전히 새로 작성한 간단한 버전

import React, { useState } from 'react';

interface NodeItem {
  type: string;
  label: string;
  color: string;
}

const nodes: NodeItem[] = [
  { type: 'intent', label: '의도 분류', color: '#9333ea' },
  { type: 'llm', label: 'AI 응답', color: '#3b82f6' },
  { type: 'api', label: 'API 호출', color: '#10b981' },
  { type: 'condition', label: '조건 분기', color: '#f97316' },
  { type: 'loop', label: '반복', color: '#ec4899' },
];

export default function NodePalette() {
  const [isDragging, setIsDragging] = useState<string | null>(null);

  const handleDragStart = (e: React.DragEvent<HTMLDivElement>, node: NodeItem) => {
    console.log('✅ Drag started:', node);
    
    // 드래그 데이터 설정
    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('application/reactflow', node.type);
    e.dataTransfer.setData('nodeLabel', node.label);
    
    setIsDragging(node.type);
    
    // 드래그 중 커서 스타일
    if (e.dataTransfer.setDragImage) {
      const ghost = e.currentTarget.cloneNode(true) as HTMLElement;
      ghost.style.opacity = '0.5';
      ghost.style.position = 'absolute';
      ghost.style.top = '-1000px';
      document.body.appendChild(ghost);
      e.dataTransfer.setDragImage(ghost, 50, 20);
      setTimeout(() => document.body.removeChild(ghost), 0);
    }
  };

  const handleDragEnd = () => {
    console.log('✅ Drag ended');
    setIsDragging(null);
  };

  return (
    <div className="w-64 bg-white border-r border-gray-200 p-4 h-full overflow-y-auto">
      <h2 className="text-lg font-bold mb-4">노드 팔레트</h2>
      
      <div className="space-y-2">
        {nodes.map((node) => (
          <div
            key={node.type}
            draggable
            onDragStart={(e) => handleDragStart(e, node)}
            onDragEnd={handleDragEnd}
            style={{
              padding: '12px',
              borderRadius: '8px',
              backgroundColor: isDragging === node.type ? '#f3f4f6' : 'white',
              border: `2px solid ${node.color}`,
              borderLeft: `6px solid ${node.color}`,
              cursor: 'grab',
              opacity: isDragging && isDragging !== node.type ? 0.5 : 1,
              transition: 'all 0.2s',
              userSelect: 'none',
            }}
            onMouseDown={(e) => {
              e.currentTarget.style.cursor = 'grabbing';
            }}
            onMouseUp={(e) => {
              e.currentTarget.style.cursor = 'grab';
            }}
          >
            <div className="flex items-center gap-2">
              <span style={{ fontSize: '18px' }}>
                {node.type === 'intent' && '🧠'}
                {node.type === 'llm' && '💬'}
                {node.type === 'api' && '🌐'}
                {node.type === 'condition' && '🔀'}
                {node.type === 'loop' && '🔄'}
              </span>
              <span className="font-medium text-gray-800">
                {node.label}
              </span>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-6 p-3 bg-blue-50 rounded-lg">
        <p className="text-xs text-blue-700">
          {isDragging ? 
            `🎯 "${nodes.find(n => n.type === isDragging)?.label}" 드래그 중...` : 
            '💡 노드를 드래그하여 캔버스에 놓으세요'
          }
        </p>
      </div>
      
      {/* 디버그 정보 */}
      <div className="mt-4 p-2 bg-gray-100 rounded text-xs">
        <p>드래그 상태: {isDragging || 'none'}</p>
      </div>
    </div>
  );
}