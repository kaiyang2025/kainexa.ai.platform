// ============================
// 2. 커스텀 노드 컴포넌트 - IntentNode
// apps/studio-web/src/components/workflow-editor/nodes/IntentNode.tsx
// ============================

import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Brain } from 'lucide-react';

const IntentNode = memo(({ data, isConnectable }: NodeProps) => {
  return (
    <div className="bg-white rounded-lg shadow-lg border-2 border-purple-500 min-w-[200px]">
      <div className="bg-purple-500 text-white px-3 py-2 rounded-t-md flex items-center gap-2">
        <Brain size={18} />
        <span className="font-semibold">의도 분류</span>
      </div>
      
      <div className="p-3">
        <div className="text-sm text-gray-700">{data.label}</div>
        {data.config?.intents && (
          <div className="mt-2 text-xs text-gray-500">
            {data.config.intents.length}개 의도 정의됨
          </div>
        )}
      </div>

      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#6366f1' }}
        isConnectable={isConnectable}
      />
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#6366f1' }}
        isConnectable={isConnectable}
      />
    </div>
  );
});

IntentNode.displayName = 'IntentNode';
export default IntentNode;