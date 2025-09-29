// LoopNode.tsx
export const LoopNode = memo(({ data, isConnectable }: NodeProps) => {
  return (
    <div className="bg-white rounded-lg shadow-lg border-2 border-pink-500 min-w-[200px]">
      <div className="bg-pink-500 text-white px-3 py-2 rounded-t-md">
        <span className="font-semibold">반복</span>
      </div>
      <div className="p-3">
        <div className="text-sm text-gray-700">{data.label || '반복'}</div>
      </div>
      <Handle type="target" position={Position.Left} style={{ background: '#6366f1' }} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Right} style={{ background: '#6366f1' }} isConnectable={isConnectable} />
    </div>
  );
});

LoopNode.displayName = 'LoopNode';
export default LoopNode;