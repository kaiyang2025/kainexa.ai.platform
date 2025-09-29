// apps/studio-web/src/components/workflow-editor/panels/NodePalette.tsx
// 디버깅이 추가된 수정 버전

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
    console.log('🎯 Drag Start:', { nodeType, label }); // 디버깅용
    
    // 여러 데이터 형식으로 설정 (호환성 향상)
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.setData('text/plain', nodeType); // 폴백용
    event.dataTransfer.setData('nodeLabel', label);
    event.dataTransfer.effectAllowed = 'move';
    
    // 드래그 중 시각적 피드백
    const dragImage = event.currentTarget.cloneNode(true) as HTMLElement;
    dragImage.style.opacity = '0.8';
    document.body.appendChild(dragImage);
    event.dataTransfer.setDragImage(dragImage, 0, 0);
    setTimeout(() => document.body.removeChild(dragImage), 0);
  };

  const onDragEnd = (event: React.DragEvent) => {
    console.log('🏁 Drag End'); // 디버깅용
    const target = event.target as HTMLElement;
    target.style.opacity = '1';
  };

  // 색상 클래스 매핑 (Tailwind 동적 클래스 문제 해결)
  const getColorClasses = (color: string) => {
    switch(color) {
      case 'purple':
        return 'border-purple-500 hover:bg-purple-50';
      case 'blue':
        return 'border-blue-500 hover:bg-blue-50';
      case 'green':
        return 'border-green-500 hover:bg-green-50';
      case 'orange':
        return 'border-orange-500 hover:bg-orange-50';
      case 'pink':
        return 'border-pink-500 hover:bg-pink-50';
      default:
        return 'border-gray-500 hover:bg-gray-50';
    }
  };

  const getIconColorClass = (color: string) => {
    switch(color) {
      case 'purple': return 'text-purple-500';
      case 'blue': return 'text-blue-500';
      case 'green': return 'text-green-500';
      case 'orange': return 'text-orange-500';
      case 'pink': return 'text-pink-500';
      default: return 'text-gray-500';
    }
  };

  return (
    <div className="w-64 bg-gray-50 p-4 border-r border-gray-200 overflow-y-auto">
      <h2 className="text-lg font-bold mb-4 text-gray-800">노드 팔레트</h2>
      
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
                  className={`
                    bg-white p-3 rounded-lg shadow cursor-move 
                    border-l-4 transition-all duration-200
                    hover:shadow-md active:shadow-lg active:scale-95
                    ${getColorClasses(node.color)}
                  `}
                  draggable={true}
                  onDragStart={(e) => onDragStart(e, node.type, node.label)}
                  onDragEnd={onDragEnd}
                  role="button"
                  tabIndex={0}
                  aria-label={`드래그 가능한 ${node.label} 노드`}
                >
                  <div className="flex items-center gap-2 pointer-events-none">
                    <Icon size={18} className={getIconColorClass(node.color)} />
                    <span className="text-sm font-medium text-gray-700">
                      {node.label}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
      
      {/* 사용 안내 */}
      <div className="mt-6 p-3 bg-blue-50 rounded-lg text-xs text-blue-700">
        <p className="font-semibold mb-1">💡 사용 방법</p>
        <p>노드를 드래그하여 캔버스에 놓으세요</p>
      </div>
    </div>
  );
}