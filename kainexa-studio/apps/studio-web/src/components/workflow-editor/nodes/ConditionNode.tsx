import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

function ConditionNodeComp({ data, isConnectable }: NodeProps) {
  return (
    <div className="bg-white rounded-lg shadow border-2 border-orange-500 min-w-[200px]">
      <div className="bg-orange-500 text-white px-3 py-2 rounded-t-md">
        <span className="font-semibold">조건 분기</span>
      </div>
      <div className="p-3">
        <div className="text-sm text-gray-700">{data?.label || '조건 분기'}</div>
      </div>
      <Handle type="target" position={Position.Left} style={{ background: '#f97316' }} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Right} style={{ background: '#f97316' }} isConnectable={isConnectable} />
    </div>
  );
}
const ConditionNode = memo(ConditionNodeComp);
ConditionNode.displayName = 'ConditionNode';
export default ConditionNode;
