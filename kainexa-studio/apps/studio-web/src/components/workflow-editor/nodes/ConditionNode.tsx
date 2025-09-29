// ConditionNode.tsx
export const ConditionNode = memo(({ data, isConnectable }: NodeProps) => {
  return (
    <div className="bg-white rounded-lg shadow-lg border-2 border-orange-500 min-w-[200px]">
      <div className="bg-orange-500 text-white px-3 py-2 rounded-t-md">
        <span className="font-semibold">조건 분기</span>
      </div>
      <div className="p-3">
        <div className="text-sm text-gray-700">{data.label || '조건 분기'}</div>
      </div>
      <Handle type="target" position={Position.Left} style={{ background: '#6366f1' }} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Right} id="a" style={{ background: '#6366f1', top: '30%' }} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Right} id="b" style={{ background: '#6366f1', top: '70%' }} isConnectable={isConnectable} />
    </div>
  );
});

ConditionNode.displayName = 'ConditionNode';
export default ConditionNode;