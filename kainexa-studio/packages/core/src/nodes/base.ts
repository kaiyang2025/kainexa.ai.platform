// ========================================
// Kainexa Studio - 핵심 노드 타입 구현
// packages/workflow-engine/src/nodes/
// ========================================

// ============================
// 1. Base Node Interface
// packages/workflow-engine/src/nodes/base.ts
// ============================

import { z } from 'zod';

export enum NodeType {
  INTENT = 'intent',
  LLM = 'llm',
  API = 'api',
  CONDITION = 'condition',
  LOOP = 'loop',
  HUMAN_HANDOFF = 'humanHandoff',
  KNOWLEDGE = 'knowledge'
}

export interface NodePosition {
  x: number;
  y: number;
}

export interface NodeData {
  label: string;
  description?: string;
  config: Record<string, any>;
}

export interface BaseNode {
  id: string;
  type: NodeType;
  position: NodePosition;
  data: NodeData;
}

export interface ExecutionContext {
  sessionId: string;
  userId: string;
  variables: Map<string, any>;
  history: Message[];
  currentNode: string;
  metadata: Record<string, any>;
}

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface NodeResult {
  success: boolean;
  output?: any;
  error?: string;
  nextNode?: string;
  context?: Partial<ExecutionContext>;
}

export abstract class AbstractNode implements BaseNode {
  id: string;
  type: NodeType;
  position: NodePosition;
  data: NodeData;

  constructor(node: BaseNode) {
    this.id = node.id;
    this.type = node.type;
    this.position = node.position;
    this.data = node.data;
  }

  abstract execute(context: ExecutionContext): Promise<NodeResult>;
  abstract validate(): boolean;
}