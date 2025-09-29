export const LLMNode = memo(({ data, isConnectable }: NodeProps) => {
  return (
    <div className="bg-white rounded-lg shadow-lg border-2 border-blue-500 min-w-[200px]">
      <div className="bg-blue-500 text-white px-3 py-2 rounded-t-md">
        <span className="font-semibold">AI 응답</span>
      </div>
      <div className="p-3">
        <div className="text-sm text-gray-700">{data.label || 'AI 응답'}</div>
      </div>
      <Handle type="target" position={Position.Left} style={{ background: '#6366f1' }} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Right} style={{ background: '#6366f1' }} isConnectable={isConnectable} />
    </div>
  );
});

IntentNode.displayName = 'LLMNode';
export default LLMNode;