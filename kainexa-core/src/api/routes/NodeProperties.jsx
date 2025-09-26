// src/studio/NodeProperties.jsx
import React from 'react';
import MonacoEditor from '@monaco-editor/react';

const NodeProperties = ({ node, onUpdate, onClose }) => {
  const [params, setParams] = useState(node.data.params || {});
  const [policy, setPolicy] = useState(node.data.policy || {});
  
  const handleParamChange = (key, value) => {
    const newParams = { ...params, [key]: value };
    setParams(newParams);
    onUpdate({ params: newParams });
  };
  
  const handlePolicyChange = (key, value) => {
    const newPolicy = { ...policy, [key]: value };
    setPolicy(newPolicy);
    onUpdate({ policy: newPolicy });
  };
  
  return (
    <div style={{
      position: 'absolute',
      right: 10,
      top: 10,
      width: '400px',
      maxHeight: '80vh',
      background: 'white',
      borderRadius: '8px',
      boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
      zIndex: 100,
      overflow: 'auto'
    }}>
      <div style={{
        padding: '20px',
        borderBottom: '1px solid #e5e7eb'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0 }}>Node Properties</h3>
          <button onClick={onClose} style={{ background: 'none', border: 'none', fontSize: '20px' }}>
            ✕
          </button>
        </div>
      </div>
      
      <div style={{ padding: '20px' }}>
        {/* 기본 정보 */}
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            Node ID
          </label>
          <input 
            type="text" 
            value={node.id} 
            disabled
            style={{ width: '100%', padding: '8px', border: '1px solid #e5e7eb', borderRadius: '4px' }}
          />
        </div>
        
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
            Label
          </label>
          <input 
            type="text" 
            value={node.data.label}
            onChange={(e) => onUpdate({ label: e.target.value })}
            style={{ width: '100%', padding: '8px', border: '1px solid #e5e7eb', borderRadius: '4px' }}
          />
        </div>
        
        {/* 노드 타입별 파라미터 */}
        {node.type === 'llm' && (
          <>
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Template
              </label>
              <MonacoEditor
                height="200px"
                language="yaml"
                value={params.template || ''}
                onChange={(value) => handleParamChange('template', value)}
                options={{
                  minimap: { enabled: false },
                  fontSize: 12,
                }}
              />
            </div>
            
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Max Tokens
              </label>
              <input 
                type="number" 
                value={params.max_tokens || 256}
                onChange={(e) => handleParamChange('max_tokens', parseInt(e.target.value))}
                style={{ width: '100%', padding: '8px', border: '1px solid #e5e7eb', borderRadius: '4px' }}
              />
            </div>
            
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Temperature
              </label>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.1"
                value={params.temperature || 0.7}
                onChange={(e) => handleParamChange('temperature', parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
              <span>{params.temperature || 0.7}</span>
            </div>
          </>
        )}
        
        {node.type === 'api' && (
          <>
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                URL
              </label>
              <input 
                type="text" 
                value={params.url || ''}
                onChange={(e) => handleParamChange('url', e.target.value)}
                style={{ width: '100%', padding: '8px', border: '1px solid #e5e7eb', borderRadius: '4px' }}
              />
            </div>
            
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Method
              </label>
              <select 
                value={params.method || 'GET'}
                onChange={(e) => handleParamChange('method', e.target.value)}
                style={{ width: '100%', padding: '8px', border: '1px solid #e5e7eb', borderRadius: '4px' }}
              >
                <option value="GET">GET</option>
                <option value="POST">POST</option>
                <option value="PUT">PUT</option>
                <option value="DELETE">DELETE</option>
              </select>
            </div>
          </>
        )}
        
        {node.type === 'condition' && (
          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
              Condition
            </label>
            <input 
              type="text" 
              value={params.condition || ''}
              placeholder="e.g., intent == 'refund'"
              onChange={(e) => handleParamChange('condition', e.target.value)}
              style={{ width: '100%', padding: '8px', border: '1px solid #e5e7eb', borderRadius: '4px' }}
            />
          </div>
        )}
        
        {/* 정책 설정 */}
        <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: '20px', marginTop: '20px' }}>
          <h4>Policy Settings</h4>
          
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <input 
                type="checkbox" 
                checked={policy.escalate || false}
                onChange={(e) => handlePolicyChange('escalate', e.target.checked)}
              />
              Enable Escalation
            </label>
          </div>
          
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>
              Confidence Threshold
            </label>
            <input 
              type="number" 
              min="0" 
              max="1" 
              step="0.1"
              value={policy.confidence_threshold || 0.7}
              onChange={(e) => handlePolicyChange('confidence_threshold', parseFloat(e.target.value))}
              style={{ width: '100%', padding: '8px', border: '1px solid #e5e7eb', borderRadius: '4px' }}
            />
          </div>
          
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>
              Retry Count
            </label>
            <input 
              type="number" 
              min="0" 
              max="5"
              value={node.data.retry || 0}
              onChange={(e) => onUpdate({ retry: parseInt(e.target.value) })}
              style={{ width: '100%', padding: '8px', border: '1px solid #e5e7eb', borderRadius: '4px' }}
            />
          </div>
          
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <input 
                type="checkbox" 
                checked={node.data.cache || false}
                onChange={(e) => onUpdate({ cache: e.target.checked })}
              />
              Enable Caching
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};