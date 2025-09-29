import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

function LLMNodeComp({ data, isConnectable }: NodeProps) {
  return (
    <div className="bg-white rounded-lg shadow border-2 border-blue-500 min-w-[200px]">
      <div className="bg-blue-500 text-white px-3 py-2 rounded-t-md">
        <span className="font-semibold">AI 응답</span>
      </div>
      <div className="p-3">
        <div className="text-sm text-gray-700">{data?.label || 'AI 응답'}</div>
      </div>
      <Handle type="target" position={Position.Left} style={{ background: '#3b82f6' }} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Right} style={{ background: '#3b82f6' }} isConnectable={isConnectable} />
    </div>
  );
}
const LLMNode = memo(LLMNodeComp);
LLMNode.displayName = 'LLMNode';
export default LLMNode;
