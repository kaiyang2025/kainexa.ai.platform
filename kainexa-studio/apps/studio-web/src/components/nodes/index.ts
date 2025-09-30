// ============================
// 5. Node Factory
// packages/workflow-engine/src/nodes/index.ts
// ============================

import { BaseNode, NodeType, AbstractNode } from './base';
import { IntentNode } from './IntentNode';
import { LLMNode } from './LLMNode';
import { APINode } from './APINode';

export class NodeFactory {
  static createNode(nodeData: BaseNode): AbstractNode {
    switch (nodeData.type) {
      case NodeType.INTENT:
        return new IntentNode(nodeData);
      
      case NodeType.LLM:
        return new LLMNode(nodeData);
      
      case NodeType.API:
        return new APINode(nodeData);
      
      // 추가 노드 타입들
      case NodeType.CONDITION:
      case NodeType.LOOP:
      case NodeType.HUMAN_HANDOFF:
      case NodeType.KNOWLEDGE:
        // TODO: 구현 예정
        throw new Error(`Node type ${nodeData.type} not yet implemented`);
      
      default:
        throw new Error(`Unknown node type: ${nodeData.type}`);
    }
  }
}

export * from './base';
export { IntentNode } from './IntentNode';
export { LLMNode } from './LLMNode';
export { APINode } from './APINode';