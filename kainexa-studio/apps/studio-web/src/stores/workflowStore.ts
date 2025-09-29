// src/stores/workflowStore.ts
import create from 'zustand';

interface WorkflowState {
  nodes: any[];
  edges: any[];
  setNodes: (nodes: any[]) => void;
  setEdges: (edges: any[]) => void;
}

// 간단한 zustand 스토어 예제
export const useWorkflowStore = create<WorkflowState>((set) => ({
  nodes: [],
  edges: [],
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
}));
