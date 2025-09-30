// ============================
// 2. 조건부 분기 노드
// packages/workflow-engine/src/nodes/condition-node.ts
// ============================

import { AbstractNode, NodeType, ExecutionContext, NodeResult } from './base';
import { z } from 'zod';

const ConditionSchema = z.object({
  field: z.string(),
  operator: z.enum(['equals', 'not_equals', 'contains', 'greater_than', 'less_than', 'regex']),
  value: z.any(),
  nextNode: z.string()
});

const ConditionConfigSchema = z.object({
  conditions: z.array(ConditionSchema),
  defaultBranch: z.string().nullable(),
  evaluationMode: z.enum(['first-match', 'all-matches']).default('first-match')
});

export class ConditionNode extends AbstractNode {
  private config: z.infer<typeof ConditionConfigSchema>;

  constructor(node: any) {
    super(node);
    this.config = ConditionConfigSchema.parse(node.data.config);
  }

  async execute(context: ExecutionContext): Promise<NodeResult> {
    try {
      const matchedConditions = this.evaluateConditions(context);
      
      if (matchedConditions.length === 0) {
        // 기본 분기로 이동
        return {
          success: true,
          output: { matched: false, branch: 'default' },
          nextNode: this.config.defaultBranch
        };
      }

      // 평가 모드에 따른 처리
      if (this.config.evaluationMode === 'first-match') {
        return {
          success: true,
          output: {
            matched: true,
            condition: matchedConditions[0],
            branch: matchedConditions[0].nextNode
          },
          nextNode: matchedConditions[0].nextNode
        };
      } else {
        // all-matches 모드: 여러 분기를 병렬로 실행할 수 있도록 결과 반환
        return {
          success: true,
          output: {
            matched: true,
            conditions: matchedConditions,
            branches: matchedConditions.map(c => c.nextNode)
          },
          nextNode: null, // 병렬 실행을 위해 null 반환
          context: {
            ...context,
            metadata: {
              ...context.metadata,
              parallelBranches: matchedConditions.map(c => c.nextNode)
            }
          }
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  private evaluateConditions(context: ExecutionContext): any[] {
    const matched = [];
    
    for (const condition of this.config.conditions) {
      const fieldValue = this.getFieldValue(condition.field, context);
      
      if (this.evaluateCondition(fieldValue, condition.operator, condition.value)) {
        matched.push(condition);
      }
    }
    
    return matched;
  }

  private getFieldValue(field: string, context: ExecutionContext): any {
    // 점 표기법 지원 (e.g., "user.name")
    const parts = field.split('.');
    let value: any = context.variables.get(parts[0]);
    
    for (let i = 1; i < parts.length && value !== undefined; i++) {
      value = value[parts[i]];
    }
    
    return value;
  }

  private evaluateCondition(fieldValue: any, operator: string, compareValue: any): boolean {
    switch (operator) {
      case 'equals':
        return fieldValue === compareValue;
      case 'not_equals':
        return fieldValue !== compareValue;
      case 'contains':
        return String(fieldValue).includes(String(compareValue));
      case 'greater_than':
        return Number(fieldValue) > Number(compareValue);
      case 'less_than':
        return Number(fieldValue) < Number(compareValue);
      case 'regex':
        return new RegExp(compareValue).test(String(fieldValue));
      default:
        return false;
    }
  }

  validate(): boolean {
    try {
      ConditionConfigSchema.parse(this.data.config);
      return true;
    } catch {
      return false;
    }
  }
}