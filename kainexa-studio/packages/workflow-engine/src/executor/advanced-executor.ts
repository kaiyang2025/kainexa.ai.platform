// ============================
// 4. 고도화된 워크플로우 실행기
// packages/workflow-engine/src/executor/advanced-executor.ts
// ============================


import { Node, Edge } from 'reactflow';
import { ExecutionContext, NodeResult } from '../nodes/base';
import { NodeFactory } from '../nodes';
import { ParallelExecutor } from './parallel-executor';

export interface ExecutionOptions {
  mode: 'sequential' | 'parallel' | 'hybrid';
  maxDepth: number;
  timeout: number;
  debug: boolean;
}

export class AdvancedWorkflowExecutor {
  private nodes: Map<string, Node>;
  private edges: Edge[];
  private parallelExecutor: ParallelExecutor;
  private executionStack: string[] = [];
  private visitedNodes: Set<string> = new Set();
  
  constructor(
    nodes: Node[],
    edges: Edge[],
    private options: ExecutionOptions = {
      mode: 'hybrid',
      maxDepth: 100,
      timeout: 300000,
      debug: false
    }
  ) {
    this.nodes = new Map(nodes.map(n => [n.id, n]));
    this.edges = edges;
    this.parallelExecutor = new ParallelExecutor({
      maxConcurrency: 5,
      timeout: options.timeout
    });
  }

  async execute(initialContext: Partial<ExecutionContext>): Promise<any> {
    const context: ExecutionContext = {
      sessionId: initialContext.sessionId || `session-${Date.now()}`,
      userId: initialContext.userId || 'system',
      variables: new Map(initialContext.variables || []),
      history: initialContext.history || [],
      currentNode: this.findStartNode(),
      metadata: initialContext.metadata || {}
    };

    const results = [];
    
    try {
      await this.executeNode(context.currentNode, context, results);
    } catch (error) {
      if (this.options.debug) {
        console.error('Execution error:', error);
        console.log('Execution stack:', this.executionStack);
      }
      throw error;
    }

    return {
      success: true,
      results,
      context: {
        variables: Array.from(context.variables.entries()),
        history: context.history,
        metadata: context.metadata
      }
    };
  }

  private async executeNode(
    nodeId: string,
    context: ExecutionContext,
    results: any[],
    depth: number = 0
  ): Promise<void> {
    // 최대 깊이 체크
    if (depth > this.options.maxDepth) {
      throw new Error(`Maximum execution depth (${this.options.maxDepth}) exceeded`);
    }

    // 순환 참조 체크
    if (this.visitedNodes.has(nodeId)) {
      if (this.options.debug) {
        console.warn(`Circular reference detected at node ${nodeId}`);
      }
      return;
    }

    const node = this.nodes.get(nodeId);
    if (!node) {
      throw new Error(`Node ${nodeId} not found`);
    }

    this.visitedNodes.add(nodeId);
    this.executionStack.push(nodeId);

    if (this.options.debug) {
      console.log(`Executing node: ${nodeId} (depth: ${depth})`);
    }

    // 노드 실행
    const nodeInstance = NodeFactory.createNode(node);
    const result = await nodeInstance.execute(context);
    
    results.push({
      nodeId,
      type: node.type,
      result,
      timestamp: new Date().toISOString()
    });

    // 결과에 따른 다음 노드 결정
    if (result.success) {
      // 병렬 분기 처리
      if (context.metadata.parallelBranches) {
        await this.executeParallelBranches(
          context.metadata.parallelBranches,
          context,
          results,
          depth + 1
        );
        delete context.metadata.parallelBranches;
      } 
      // 단일 다음 노드
      else if (result.nextNode) {
        await this.executeNode(result.nextNode, context, results, depth + 1);
      }
      // 엣지 기반 다음 노드 찾기
      else {
        const nextNodes = this.findNextNodes(nodeId);
        for (const nextNode of nextNodes) {
          await this.executeNode(nextNode, context, results, depth + 1);
        }
      }
    }

    this.executionStack.pop();
  }

  private async executeParallelBranches(
    branches: string[],
    context: ExecutionContext,
    results: any[],
    depth: number
  ): Promise<void> {
    const parallelNodes = branches.map(id => this.nodes.get(id)).filter(Boolean);
    
    if (parallelNodes.length === 0) return;

    const parallelResults = await this.parallelExecutor.executeParallel(
      parallelNodes as Node[],
      context
    );

    parallelResults.forEach((result, nodeId) => {
      results.push({
        nodeId,
        result,
        parallel: true,
        timestamp: new Date().toISOString()
      });
    });
  }

  private findStartNode(): string {
    // 들어오는 엣지가 없는 노드를 시작 노드로 간주
    const nodesWithIncoming = new Set(this.edges.map(e => e.target));
    const startNodes = Array.from(this.nodes.keys()).filter(
      id => !nodesWithIncoming.has(id)
    );
    
    return startNodes[0] || this.nodes.keys().next().value;
  }

  private findNextNodes(nodeId: string): string[] {
    return this.edges
      .filter(e => e.source === nodeId)
      .map(e => e.target);
  }
}