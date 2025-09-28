// ============================
// 3. 노드 팔레트 컴포넌트
// apps/studio-web/src/components/workflow-editor/panels/NodePalette.tsx
// ============================

import React from 'react';
import { Brain, MessageSquare, Globe, GitBranch, Repeat } from 'lucide-react';

const nodeCategories = [
  {
    title: '기본 노드',
    nodes: [
      { type: 'intent', label: '의도 분류', icon: Brain, color: 'purple' },
      { type: 'llm', label: 'AI 응답', icon: MessageSquare, color: 'blue' },
      { type: 'api', label: 'API 호출', icon: Globe, color: 'green' },
    ]
  },
  {
    title: '흐름 제어',
    nodes: [
      { type: 'condition', label: '조건 분기', icon: GitBranch, color: 'orange' },
      { type: 'loop', label: '반복', icon: Repeat, color: 'pink' },
    ]
  }
];

export default function NodePalette() {
  const onDragStart = (event: React.DragEvent, nodeType: string, label: string) => {
    event.dataTransfer.setData('nodeType', nodeType);
    event.dataTransfer.setData('nodeLabel', label);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="w-64 bg-gray-50 p-4 border-r border-gray-200 overflow-y-auto">
      <h2 className="text-lg font-bold mb-4">노드 팔레트</h2>
      
      {nodeCategories.map((category) => (
        <div key={category.title} className="mb-6">
          <h3 className="text-sm font-semibold text-gray-600 mb-2">
            {category.title}
          </h3>
          <div className="space-y-2">
            {category.nodes.map((node) => {
              const Icon = node.icon;
              return (
                <div
                  key={node.type}
                  className={`bg-white p-3 rounded-lg shadow cursor-move border-l-4 border-${node.color}-500 hover:shadow-md transition-shadow`}
                  draggable
                  onDragStart={(e) => onDragStart(e, node.type, node.label)}
                >
                  <div className="flex items-center gap-2">
                    <Icon size={18} className={`text-${node.color}-500`} />
                    <span className="text-sm font-medium">{node.label}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}