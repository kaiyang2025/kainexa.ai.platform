// Shared types and utilities
export interface BaseNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: any;
}

export interface ExecutionContext {
  sessionId: string;
  userId: string;
  variables: Map<string, any>;
  history: any[];
  currentNode: string;
  metadata: Record<string, any>;
}

export const VERSION = '1.0.0';

export function generateId(): string {
  return `id_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
