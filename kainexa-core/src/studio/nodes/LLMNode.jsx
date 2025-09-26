// src/studio/nodes/LLMNode.jsx
import React from 'react';
import { Handle, Position } from 'reactflow';

const LLMNode = ({ data, selected }) => {
  return (
    <div style={{
      background: selected ? '#6366F1' : '#8B5CF6',
      color: 'white',
      padding: '10px 20px',
      borderRadius: '8px',
      minWidth: '150px',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    }}>
      <Handle type="target" position={Position.Top} />
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span>🤖</span>
        <div>
          <div style={{ fontWeight: 'bold' }}>{data.label}</div>
          <div style={{ fontSize: '10px', opacity: 0.7 }}>
            LLM Generate
          </div>
        </div>
      </div>
      
      {data.params?.template && (
        <div style={{ 
          marginTop: '8px', 
          fontSize: '11px', 
          opacity: 0.8,
          maxWidth: '200px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        }}>
          Template: {data.params.template.substring(0, 30)}...
        </div>
      )}
      
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

// src/studio/nodes/ConditionNode.jsx
const ConditionNode = ({ data, selected }) => {
  return (
    <div style={{
      background: selected ? '#F59E0B' : '#FCD34D',
      color: '#7C2D12',
      padding: '10px 20px',
      borderRadius: '8px',
      minWidth: '150px',
      position: 'relative'
    }}>
      <Handle type="target" position={Position.Top} />
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span>🔀</span>
        <div>
          <div style={{ fontWeight: 'bold' }}>{data.label}</div>
          <div style={{ fontSize: '10px' }}>Condition</div>
        </div>
      </div>
      
      {data.params?.condition && (
        <div style={{ marginTop: '8px', fontSize: '11px' }}>
          {data.params.condition}
        </div>
      )}
      
      {/* True/False 출력 */}
      <Handle 
        type="source" 
        position={Position.Bottom} 
        id="true"
        style={{ left: '30%', background: '#10B981' }}
      />
      <Handle 
        type="source" 
        position={Position.Bottom} 
        id="false"
        style={{ left: '70%', background: '#EF4444' }}
      />
      
      <div style={{
        position: 'absolute',
        bottom: '-20px',
        left: '20%',
        fontSize: '10px',
        color: '#10B981'
      }}>
        True
      </div>
      <div style={{
        position: 'absolute',
        bottom: '-20px',
        left: '60%',
        fontSize: '10px',
        color: '#EF4444'
      }}>
        False
      </div>
    </div>
  );
};