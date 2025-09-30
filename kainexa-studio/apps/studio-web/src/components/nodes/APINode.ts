// ============================
// 4. API Node
// ============================

import { AbstractNode, NodeType, ExecutionContext, NodeResult } from './base';
import { z } from 'zod';

const APIConfigSchema = z.object({
  url: z.string().url(),
  method: z.enum(['GET', 'POST', 'PUT', 'DELETE', 'PATCH']),
  headers: z.record(z.string()).optional(),
  bodyTemplate: z.string().optional(),
  authentication: z.object({
    type: z.enum(['none', 'bearer', 'apiKey', 'oauth2']),
    config: z.record(z.any()).optional()
  }).optional(),
  timeout: z.number().default(30000),
  retries: z.number().default(3),
  responseMapping: z.record(z.string()).optional()
});

export class APINode extends AbstractNode {
  private apiConfig: z.infer<typeof APIConfigSchema>;

  constructor(node: any) {
    super(node);
    this.apiConfig = APIConfigSchema.parse(node.data.config);
  }

  async execute(context: ExecutionContext): Promise<NodeResult> {
    try {
      // Request 준비
      const request = this.prepareRequest(context);
      
      // API 호출 (재시도 로직 포함)
      const response = await this.callAPIWithRetry(request);
      
      // Response 매핑
      const mappedResponse = this.mapResponse(response);
      
      // 변수 저장
      context.variables.set('apiResponse', mappedResponse);
      
      return {
        success: true,
        output: mappedResponse,
        context: {
          variables: context.variables,
          metadata: {
            ...context.metadata,
            lastApiCall: {
              url: this.apiConfig.url,
              status: response.status,
              timestamp: new Date()
            }
          }
        }
      };
    } catch (error) {
      return {
        success: false,
        error: `API call failed: ${error.message}`
      };
    }
  }

  private prepareRequest(context: ExecutionContext): any {
    // URL 변수 치환
    let url = this.apiConfig.url;
    context.variables.forEach((value, key) => {
      url = url.replace(`{${key}}`, String(value));
    });

    // Body 준비
    let body = null;
    if (this.apiConfig.bodyTemplate) {
      body = this.apiConfig.bodyTemplate;
      context.variables.forEach((value, key) => {
        body = body.replace(`{{${key}}}`, JSON.stringify(value));
      });
      body = JSON.parse(body);
    }

    // Headers 준비
    const headers = {
      'Content-Type': 'application/json',
      ...this.apiConfig.headers
    };

    // Authentication 추가
    if (this.apiConfig.authentication?.type === 'bearer') {
      const token = context.variables.get('authToken');
      headers['Authorization'] = `Bearer ${token}`;
    }

    return {
      url,
      method: this.apiConfig.method,
      headers,
      body
    };
  }

  private async callAPIWithRetry(request: any): Promise<any> {
    let lastError: Error;
    
    for (let i = 0; i < this.apiConfig.retries; i++) {
      try {
        // 실제 API 호출 시뮬레이션
        console.log(`API Call (attempt ${i + 1}):`, request);
        
        // Mock response
        return {
          status: 200,
          data: {
            success: true,
            result: 'Mock API response'
          }
        };
      } catch (error) {
        lastError = error;
        // Exponential backoff
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 1000));
      }
    }
    
    throw lastError;
  }

  private mapResponse(response: any): any {
    if (!this.apiConfig.responseMapping) {
      return response.data;
    }

    const mapped: Record<string, any> = {};
    
    for (const [key, path] of Object.entries(this.apiConfig.responseMapping)) {
      // JSONPath 형식으로 데이터 추출
      const value = this.getValueByPath(response.data, path as string);
      mapped[key] = value;
    }
    
    return mapped;
  }

  private getValueByPath(obj: any, path: string): any {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  }

  validate(): boolean {
    try {
      APIConfigSchema.parse(this.data.config);
      return true;
    } catch {
      return false;
    }
  }
}