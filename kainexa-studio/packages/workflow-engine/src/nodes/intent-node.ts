// ============================
// 2. Intent Node
// packages/workflow-engine/src/nodes/intent-node.ts
// ============================

import { AbstractNode, NodeType, ExecutionContext, NodeResult } from './base';

// Intent 설정 스키마
const IntentConfigSchema = z.object({
  intents: z.array(z.object({
    name: z.string(),
    examples: z.array(z.string()),
    entities: z.array(z.string()).optional()
  })),
  threshold: z.number().min(0).max(1).default(0.7),
  fallback: z.string().optional()
});

export class IntentNode extends AbstractNode {
  private intentConfig: z.infer<typeof IntentConfigSchema>;

  constructor(node: any) {
    super(node);
    this.intentConfig = IntentConfigSchema.parse(node.data.config);
  }

  async execute(context: ExecutionContext): Promise<NodeResult> {
    try {
      // 사용자 입력 가져오기
      const userInput = context.variables.get('userInput') || 
                       context.history[context.history.length - 1]?.content;

      if (!userInput) {
        return {
          success: false,
          error: 'No user input found'
        };
      }

      // Intent 분류 (실제 구현시 ML 모델 사용)
      const intent = await this.classifyIntent(userInput);
      
      // Entity 추출
      const entities = await this.extractEntities(userInput, intent);

      // Context 업데이트
      context.variables.set('intent', intent);
      context.variables.set('entities', entities);
      context.variables.set('confidence', intent.confidence);

      // 다음 노드 결정
      const nextNode = this.determineNextNode(intent);

      return {
        success: true,
        output: {
          intent: intent.name,
          confidence: intent.confidence,
          entities
        },
        nextNode,
        context: {
          variables: context.variables,
          metadata: {
            ...context.metadata,
            lastIntent: intent.name
          }
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  private async classifyIntent(text: string): Promise<any> {
    // 간단한 키워드 매칭 (실제로는 ML 모델 사용)
    const normalizedText = text.toLowerCase();
    
    for (const intent of this.intentConfig.intents) {
      for (const example of intent.examples) {
        if (normalizedText.includes(example.toLowerCase())) {
          return {
            name: intent.name,
            confidence: 0.9
          };
        }
      }
    }

    return {
      name: this.intentConfig.fallback || 'unknown',
      confidence: 0.3
    };
  }

  private async extractEntities(text: string, intent: any): Promise<Record<string, any>> {
    const entities: Record<string, any> = {};
    
    // 간단한 entity 추출 로직
    // 실제로는 NER 모델 사용
    const intentConfig = this.intentConfig.intents.find(i => i.name === intent.name);
    
    if (intentConfig?.entities) {
      for (const entity of intentConfig.entities) {
        // 날짜 추출 예시
        if (entity === 'date') {
          const dateMatch = text.match(/\d{4}-\d{2}-\d{2}/);
          if (dateMatch) entities.date = dateMatch[0];
        }
        // 숫자 추출 예시
        if (entity === 'number') {
          const numberMatch = text.match(/\d+/);
          if (numberMatch) entities.number = parseInt(numberMatch[0]);
        }
      }
    }

    return entities;
  }

  private determineNextNode(intent: any): string {
    // Intent에 따른 다음 노드 결정
    const routes = this.data.config.routes || {};
    return routes[intent.name] || routes.default || null;
  }

  validate(): boolean {
    try {
      IntentConfigSchema.parse(this.data.config);
      return true;
    } catch {
      return false;
    }
  }
}