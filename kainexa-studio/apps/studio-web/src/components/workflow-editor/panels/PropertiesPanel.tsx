// ============================
// 4. 속성 패널 컴포넌트
// apps/studio-web/src/components/workflow-editor/panels/PropertiesPanel.tsx
// ============================

import React, { useEffect, useState } from 'react';
import { Node } from 'reactflow';
import { X } from 'lucide-react';

interface PropertiesPanelProps {
  selectedNode: Node | null;
  updateNodeData: (nodeId: string, data: any) => void;
}

export default function PropertiesPanel({ selectedNode, updateNodeData }: PropertiesPanelProps) {
  const [formData, setFormData] = useState<any>({});

  useEffect(() => {
    if (selectedNode) {
      setFormData(selectedNode.data);
    }
  }, [selectedNode]);

  if (!selectedNode) {
    return (
      <div className="w-80 bg-gray-50 p-4 border-l border-gray-200">
        <p className="text-gray-500 text-sm">노드를 선택하세요</p>
      </div>
    );
  }

  const handleChange = (field: string, value: any) => {
    const newData = {
      ...formData,
      config: {
        ...formData.config,
        [field]: value
      }
    };
    setFormData(newData);
    updateNodeData(selectedNode.id, newData);
  };

  const renderNodeProperties = () => {
    switch (selectedNode.type) {
      case 'intent':
        return <IntentProperties data={formData} onChange={handleChange} />;
      case 'llm':
        return <LLMProperties data={formData} onChange={handleChange} />;
      case 'api':
        return <APIProperties data={formData} onChange={handleChange} />;
      default:
        return <div>속성 편집기 준비 중...</div>;
    }
  };

  return (
    <div className="w-80 bg-white border-l border-gray-200 overflow-y-auto">
      <div className="p-4 border-b border-gray-200 flex justify-between items-center">
        <h3 className="font-semibold">노드 속성</h3>
        <button className="text-gray-400 hover:text-gray-600">
          <X size={20} />
        </button>
      </div>
      
      <div className="p-4">
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            레이블
          </label>
          <input
            type="text"
            value={formData.label || ''}
            onChange={(e) => {
              const newData = { ...formData, label: e.target.value };
              setFormData(newData);
              updateNodeData(selectedNode.id, newData);
            }}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {renderNodeProperties()}
      </div>
    </div>
  );
}

// Intent 노드 속성 편집기
function IntentProperties({ data, onChange }: any) {
  const [intents, setIntents] = useState(data.config?.intents || []);

  const addIntent = () => {
    const newIntent = {
      name: `intent_${intents.length + 1}`,
      examples: [],
      entities: []
    };
    const updated = [...intents, newIntent];
    setIntents(updated);
    onChange('intents', updated);
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          임계값
        </label>
        <input
          type="number"
          min="0"
          max="1"
          step="0.1"
          value={data.config?.threshold || 0.7}
          onChange={(e) => onChange('threshold', parseFloat(e.target.value))}
          className="w-full px-3 py-2 border border-gray-300 rounded-md"
        />
      </div>

      <div>
        <div className="flex justify-between items-center mb-2">
          <label className="text-sm font-medium text-gray-700">의도 목록</label>
          <button
            onClick={addIntent}
            className="text-sm bg-blue-500 text-white px-2 py-1 rounded hover:bg-blue-600"
          >
            + 추가
          </button>
        </div>
        
        <div className="space-y-2">
          {intents.map((intent: any, index: number) => (
            <div key={index} className="p-2 border border-gray-200 rounded">
              <input
                type="text"
                value={intent.name}
                onChange={(e) => {
                  const updated = [...intents];
                  updated[index].name = e.target.value;
                  setIntents(updated);
                  onChange('intents', updated);
                }}
                className="w-full px-2 py-1 text-sm border border-gray-300 rounded"
                placeholder="의도 이름"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// LLM 노드 속성 편집기
function LLMProperties({ data, onChange }: any) {
  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          모델
        </label>
        <select
          value={data.config?.model || 'gpt-3.5-turbo'}
          onChange={(e) => onChange('model', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md"
        >
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          <option value="gpt-4">GPT-4</option>
          <option value="claude-3">Claude 3</option>
          <option value="solar">Solar (한국어)</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Temperature
        </label>
        <input
          type="number"
          min="0"
          max="2"
          step="0.1"
          value={data.config?.temperature || 0.7}
          onChange={(e) => onChange('temperature', parseFloat(e.target.value))}
          className="w-full px-3 py-2 border border-gray-300 rounded-md"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          시스템 프롬프트
        </label>
        <textarea
          value={data.config?.systemPrompt || ''}
          onChange={(e) => onChange('systemPrompt', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md"
          rows={4}
          placeholder="당신은 친절한 AI 어시스턴트입니다..."
        />
      </div>
    </div>
  );
}

// API 노드 속성 편집기
function APIProperties({ data, onChange }: any) {
  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          URL
        </label>
        <input
          type="text"
          value={data.config?.url || ''}
          onChange={(e) => onChange('url', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md"
          placeholder="https://api.example.com/endpoint"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          메서드
        </label>
        <select
          value={data.config?.method || 'GET'}
          onChange={(e) => onChange('method', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md"
        >
          <option value="GET">GET</option>
          <option value="POST">POST</option>
          <option value="PUT">PUT</option>
          <option value="DELETE">DELETE</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          타임아웃 (ms)
        </label>
        <input
          type="number"
          value={data.config?.timeout || 30000}
          onChange={(e) => onChange('timeout', parseInt(e.target.value))}
          className="w-full px-3 py-2 border border-gray-300 rounded-md"
        />
      </div>
    </div>
  );
}