import { ExecutionContext } from '@kainexa/shared';

export interface NodeResult {
  success: boolean;
  output?: any;
  error?: string;
  nextNode?: string;
}

export abstract class AbstractNode {
  id: string;
  type: string;
  data: any;

  constructor(config: any) {
    this.id = config.id;
    this.type = config.type;
    this.data = config.data;
  }

  abstract execute(context: ExecutionContext): Promise<NodeResult>;
}
