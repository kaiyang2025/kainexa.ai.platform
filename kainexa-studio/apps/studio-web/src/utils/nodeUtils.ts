// src/utils/nodeUtils.ts

let counter = 0;

/**
 * 고유한 노드 ID를 생성합니다.
 * ReactFlow 노드 id는 string이어야 합니다.
 */
export function generateNodeId(prefix: string = "node"): string {
  counter += 1;
  return `${prefix}_${counter}`;
}
