import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

function LoopNodeComp({ data, isConnectable }: NodeProps) {
  return (
    <div className="bg-white rounded-lg shadow border-2 border-pink-500 min-w-[200px]">
      <div className="bg-pink-500 text-white px-3 py-2 rounded-t-md">
        <span className="font-semibold">반복</span>
      </div>
      <div className="p-3">
        <div className="text-sm text-gray-700">{data?.label || '반복'}</div>
      </div>
      <Handle type="target" position={Position.Left} style={{ background: '#ec4899' }} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Right} style={{ background: '#ec4899' }} isConnectable={isConnectable} />
    </div>
  );
}
const LoopNode = memo(LoopNodeComp);
LoopNode.displayName = 'LoopNode';
export default LoopNode;
