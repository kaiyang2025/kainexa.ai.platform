import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

function APINodeComp({ data, isConnectable }: NodeProps) {
  return (
    <div className="bg-white rounded-lg shadow border-2 border-green-500 min-w-[200px]">
      <div className="bg-green-500 text-white px-3 py-2 rounded-t-md">
        <span className="font-semibold">API 호출</span>
      </div>
      <div className="p-3">
        <div className="text-sm text-gray-700">{data?.label || 'API 호출'}</div>
      </div>
      <Handle type="target" position={Position.Left} style={{ background: '#10b981' }} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Right} style={{ background: '#10b981' }} isConnectable={isConnectable} />
    </div>
  );
}
const APINode = memo(APINodeComp);
APINode.displayName = 'APINode';
export default APINode;
