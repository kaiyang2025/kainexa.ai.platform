// ============================
// 1. 병렬 실행 지원
// packages/workflow-engine/src/executor/parallel-executor.ts
// ============================

import { Node, Edge } from 'reactflow';
import { ExecutionContext, NodeResult } from '../nodes/base';
import { NodeFactory } from '../nodes';

export interface ParallelExecutionConfig {
  maxConcurrency: number;
  timeout: number;
  failureStrategy: 'fail-fast' | 'best-effort';
}

export class ParallelExecutor {
  private config: ParallelExecutionConfig;
  
  constructor(config: Partial<ParallelExecutionConfig> = {}) {
    this.config = {
      maxConcurrency: config.maxConcurrency || 5,
      timeout: config.timeout || 30000,
      failureStrategy: config.failureStrategy || 'best-effort'
    };
  }

  async executeParallel(
    nodes: Node[],
    context: ExecutionContext
  ): Promise<Map<string, NodeResult>> {
    const results = new Map<string, NodeResult>();
    const batches = this.createBatches(nodes);
    
    for (const batch of batches) {
      const batchPromises = batch.map(async (node) => {
        try {
          const nodeInstance = NodeFactory.createNode(node);
          const result = await this.executeWithTimeout(
            nodeInstance.execute(context),
            this.config.timeout
          );
          return { nodeId: node.id, result };
        } catch (error) {
          if (this.config.failureStrategy === 'fail-fast') {
            throw error;
          }
          return {
            nodeId: node.id,
            result: {
              success: false,
              error: error.message
            }
          };
        }
      });
      
      const batchResults = await Promise.all(batchPromises);
      batchResults.forEach(({ nodeId, result }) => {
        results.set(nodeId, result);
      });
    }
    
    return results;
  }

  private createBatches(nodes: Node[]): Node[][] {
    const batches: Node[][] = [];
    for (let i = 0; i < nodes.length; i += this.config.maxConcurrency) {
      batches.push(nodes.slice(i, i + this.config.maxConcurrency));
    }
    return batches;
  }

  private executeWithTimeout<T>(
    promise: Promise<T>,
    timeout: number
  ): Promise<T> {
    return Promise.race([
      promise,
      new Promise<T>((_, reject) =>
        setTimeout(() => reject(new Error('Execution timeout')), timeout)
      )
    ]);
  }
}