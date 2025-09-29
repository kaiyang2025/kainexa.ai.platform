// apps/studio-web/src/components/workflow-editor/panels/NodePalette.tsx
import React from 'react';

interface NodeItem {
  type: string;
  label: string;
  color: string;
  icon: string;
}

const nodes: NodeItem[] = [
  { type: 'intent',    label: '의도 분류', icon: '🧠', color: '#9333ea' },
  { type: 'llm',       label: 'AI 응답',   icon: '💬', color: '#3b82f6' },
  { type: 'api',       label: 'API 호출',  icon: '🌐', color: '#10b981' },
  { type: 'condition', label: '조건 분기', icon: '🔀', color: '#f97316' },
  { type: 'loop',      label: '반복',     icon: '🔄', color: '#ec4899' },
];

export default function NodePalette() {

  const onDragStart = (e: React.DragEvent, node: NodeItem) => {
  // 1) 드래그 페이로드를 JSON으로 포장
  const payload = JSON.stringify({ type: node.type, label: node.label });

  // 2) 표준 키 + 호환 키 모두 설정
  e.dataTransfer.setData('application/reactflow', payload);
  e.dataTransfer.setData('text/plain', payload);

  // 3) 드래그 이펙트
  e.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div
      className="node-palette"
      style={{
        width: 250, backgroundColor: '#f9fafb', borderRight: '1px solid #e5e7eb',
        padding: 16, height: '100%', overflowY: 'auto', userSelect: 'none',
      }}
    >
      <h2 style={{ fontSize: 18, fontWeight: 'bold', marginBottom: 16, color: '#111827' }}>
        노드 팔레트
      </h2>

      <div style={{ marginBottom: 16 }}>
        {nodes.map((node) => (
          <div
            key={node.type}
            draggable
            onDragStart={(e) => onDragStart(e, node)}
            style={{
              padding: 12, marginBottom: 8, background: '#fff',
              border: `2px solid ${node.color}`, borderLeft: `5px solid ${node.color}`,
              borderRadius: 6, cursor: 'grab', transition: 'transform .2s, box-shadow .2s',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, pointerEvents: 'none' }}>
              <span style={{ fontSize: 20 }}>{node.icon}</span>
              <span style={{ fontSize: 14, fontWeight: 500, color: '#374151' }}>{node.label}</span>
            </div>
          </div>
        ))}
      </div>

      <div style={{ padding: 12, background: '#dbeafe', borderRadius: 6, fontSize: 12, color: '#1e40af' }}>
        💡 카드를 캔버스로 드래그하여 놓으세요
      </div>
    </div>
  );
}
