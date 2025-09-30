// ============================
// 3. LLM Node
// packages/workflow-engine/src/nodes/llm-node.ts
// ============================

import { AbstractNode, NodeType, ExecutionContext, NodeResult } from './base';

const LLMConfigSchema = z.object({
  model: z.enum(['gpt-4', 'gpt-3.5-turbo', 'claude-3', 'solar']),
  temperature: z.number().min(0).max(2).default(0.7),
  maxTokens: z.number().default(500),
  systemPrompt: z.string(),
  userPromptTemplate: z.string(),
  outputFormat: z.enum(['text', 'json', 'structured']).default('text'),
  fallbackModel: z.string().optional()
});

export class LLMNode extends AbstractNode {
  private llmConfig: z.infer<typeof LLMConfigSchema>;

  constructor(node: any) {
    super(node);
    this.llmConfig = LLMConfigSchema.parse(node.data.config);
  }

  async execute(context: ExecutionContext): Promise<NodeResult> {
    try {
      // 프롬프트 생성
      const prompt = this.buildPrompt(context);
      
      // LLM 호출
      const response = await this.callLLM(prompt);
      
      // 응답 파싱
      const parsedResponse = this.parseResponse(response);
      
      // 대화 기록 업데이트
      context.history.push({
        role: 'assistant',
        content: parsedResponse,
        timestamp: new Date(),
        metadata: {
          model: this.llmConfig.model,
          nodeId: this.id
        }
      });

      return {
        success: true,
        output: parsedResponse,
        context: {
          history: context.history,
          variables: new Map([
            ...context.variables,
            ['lastResponse', parsedResponse]
          ])
        }
      };
    } catch (error) {
      // Fallback 모델 시도
      if (this.llmConfig.fallbackModel) {
        return this.executeFallback(context, error);
      }
      
      return {
        success: false,
        error: error.message
      };
    }
  }

  private buildPrompt(context: ExecutionContext): string {
    // 템플릿에 변수 치환
    let userPrompt = this.llmConfig.userPromptTemplate;
    
    // {{variable}} 형식의 변수를 실제 값으로 치환
    context.variables.forEach((value, key) => {
      const regex = new RegExp(`{{${key}}}`, 'g');
      userPrompt = userPrompt.replace(regex, String(value));
    });

    // 대화 기록 추가 (최근 5개)
    const recentHistory = context.history.slice(-5)
      .map(m => `${m.role}: ${m.content}`)
      .join('\n');

    return `${this.llmConfig.systemPrompt}\n\n${recentHistory}\n\nuser: ${userPrompt}`;
  }

  private async callLLM(prompt: string): Promise<string> {
    // 실제 LLM API 호출 시뮬레이션
    // 실제 구현시에는 OpenAI, Anthropic 등의 SDK 사용
    
    console.log(`Calling ${this.llmConfig.model} with prompt:`, prompt);
    
    // Mock response
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(`안녕하세요! ${this.llmConfig.model} 모델의 응답입니다. 무엇을 도와드릴까요?`);
      }, 1000);
    });
  }

  private parseResponse(response: string): any {
    if (this.llmConfig.outputFormat === 'json') {
      try {
        return JSON.parse(response);
      } catch {
        return { text: response };
      }
    }
    return response;
  }

  private async executeFallback(context: ExecutionContext, originalError: Error): Promise<NodeResult> {
    console.log('Executing fallback model:', this.llmConfig.fallbackModel);
    // Fallback 로직 구현
    return {
      success: true,
      output: 'Fallback response',
      context: {
        metadata: {
          ...context.metadata,
          usedFallback: true
        }
      }
    };
  }

  validate(): boolean {
    try {
      LLMConfigSchema.parse(this.data.config);
      return true;
    } catch {
      return false;
    }
  }
}