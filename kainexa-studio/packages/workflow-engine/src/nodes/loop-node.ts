// ============================
// 3. 루프 처리 노드
// packages/workflow-engine/src/nodes/loop-node.ts
// ============================

import { AbstractNode, NodeType, ExecutionContext, NodeResult } from './base';
import { z } from 'zod';

const LoopConfigSchema = z.object({
  loopType: z.enum(['for', 'while', 'foreach']),
  maxIterations: z.number().default(100),
  breakCondition: z.string().optional(),
  iteratorVariable: z.string().default('index'),
  collection: z.string().optional(), // foreach용
  startValue: z.number().default(0), // for용
  endValue: z.number().optional(), // for용
  step: z.number().default(1), // for용
  loopBody: z.string() // 루프 내부에서 실행할 노드 ID
});

export class LoopNode extends AbstractNode {
  private config: z.infer<typeof LoopConfigSchema>;
  private currentIteration: number = 0;

  constructor(node: any) {
    super(node);
    this.config = LoopConfigSchema.parse(node.data.config);
  }

  async execute(context: ExecutionContext): Promise<NodeResult> {
    try {
      const results = [];
      this.currentIteration = 0;

      switch (this.config.loopType) {
        case 'for':
          results.push(...await this.executeForLoop(context));
          break;
        case 'while':
          results.push(...await this.executeWhileLoop(context));
          break;
        case 'foreach':
          results.push(...await this.executeForEachLoop(context));
          break;
      }

      return {
        success: true,
        output: {
          totalIterations: this.currentIteration,
          results
        },
        context: {
          ...context,
          variables: new Map([
            ...context.variables,
            ['loopResults', results]
          ])
        }
      };
    } catch (error) {
      return {
        success: false,
        error: `Loop execution failed: ${error.message}`
      };
    }
  }

  private async executeForLoop(context: ExecutionContext): Promise<any[]> {
    const results = [];
    const start = this.config.startValue;
    const end = this.config.endValue || 10;
    const step = this.config.step;

    for (let i = start; i < end && this.currentIteration < this.config.maxIterations; i += step) {
      // 반복 변수 설정
      context.variables.set(this.config.iteratorVariable, i);
      this.currentIteration++;

      // break 조건 체크
      if (this.config.breakCondition && this.evaluateBreakCondition(context)) {
        break;
      }

      // 루프 바디 실행 (실제로는 loopBody 노드를 실행)
      const iterationResult = await this.executeLoopBody(context);
      results.push({
        iteration: this.currentIteration,
        value: i,
        result: iterationResult
      });
    }

    return results;
  }

  private async executeWhileLoop(context: ExecutionContext): Promise<any[]> {
    const results = [];

    while (this.currentIteration < this.config.maxIterations) {
      // break 조건 체크
      if (this.config.breakCondition && !this.evaluateBreakCondition(context)) {
        break;
      }

      this.currentIteration++;
      context.variables.set(this.config.iteratorVariable, this.currentIteration);

      // 루프 바디 실행
      const iterationResult = await this.executeLoopBody(context);
      results.push({
        iteration: this.currentIteration,
        result: iterationResult
      });
    }

    return results;
  }

  private async executeForEachLoop(context: ExecutionContext): Promise<any[]> {
    const results = [];
    const collection = context.variables.get(this.config.collection || 'items') || [];

    for (const [index, item] of collection.entries()) {
      if (this.currentIteration >= this.config.maxIterations) break;

      this.currentIteration++;
      context.variables.set(this.config.iteratorVariable, item);
      context.variables.set('index', index);

      // break 조건 체크
      if (this.config.breakCondition && this.evaluateBreakCondition(context)) {
        break;
      }

      // 루프 바디 실행
      const iterationResult = await this.executeLoopBody(context);
      results.push({
        iteration: this.currentIteration,
        item,
        result: iterationResult
      });
    }

    return results;
  }

  private evaluateBreakCondition(context: ExecutionContext): boolean {
    if (!this.config.breakCondition) return false;

    try {
      // 간단한 표현식 평가 (실제로는 더 복잡한 평가 엔진 필요)
      const expression = this.config.breakCondition;
      const variables = Object.fromEntries(context.variables);
      
      // 예: "index > 5" 같은 간단한 조건 평가
      const func = new Function(...Object.keys(variables), `return ${expression}`);
      return func(...Object.values(variables));
    } catch {
      return false;
    }
  }

  private async executeLoopBody(context: ExecutionContext): Promise<any> {
    // 실제로는 loopBody 노드를 실행해야 함
    // 여기서는 시뮬레이션
    return {
      message: `Loop iteration ${this.currentIteration} executed`,
      variables: Object.fromEntries(context.variables)
    };
  }

  validate(): boolean {
    try {
      LoopConfigSchema.parse(this.data.config);
      return true;
    } catch {
      return false;
    }
  }
}
