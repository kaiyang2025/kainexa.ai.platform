import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

function IntentNodeComp({ data, isConnectable }: NodeProps) {
  return (
    <div className="bg-white rounded-lg shadow border-2 border-purple-500 min-w-[200px]">
      <div className="bg-purple-500 text-white px-3 py-2 rounded-t-md">
        <span className="font-semibold">의도 분류</span>
      </div>
      <div className="p-3">
        <div className="text-sm text-gray-700">{data?.label || '의도 분류'}</div>
      </div>
      <Handle type="source" position={Position.Right} style={{ background: '#9333ea' }} isConnectable={isConnectable} />
    </div>
  );
}
const IntentNode = memo(IntentNodeComp);
IntentNode.displayName = 'IntentNode';
export default IntentNode;
