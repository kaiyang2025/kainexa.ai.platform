// studio/src/utils/dsl/index.ts
/**
 * Kainexa Studio DSL Export/Import Utilities
 */
import yaml from 'js-yaml';
import { Node, Edge } from 'reactflow';

// ========== Types ==========
export interface WorkflowDSL {
  version: string;
  workflow: WorkflowMetadata;
  nodes: DSLNode[];
  edges: DSLEdge[];
  policies?: PolicyConfig;
  environments?: EnvironmentConfig;
}

export interface WorkflowMetadata {
  namespace: string;
  name: string;
  version: string;
  metadata: {
    author: string;
    description?: string;
    tags?: string[];
    created_at?: string;
    updated_at?: string;
  };
}

export interface DSLNode {
  id: string;
  type: NodeType;
  position?: { x: number; y: number };
  config: Record<string, any>;
}

export interface DSLEdge {
  id?: string;
  source: string;
  target: string;
  condition?: string;
  label?: string;
}

export interface PolicyConfig {
  sla?: {
    max_latency_ms?: number;
    timeout_ms?: number;
  };
  fallback?: {
    on_llm_error?: {
      action: string;
      model?: string;
    };
    on_api_error?: {
      action: string;
      ttl?: number;
    };
  };
  escalation?: {
    on_sentiment?: {
      trigger: string;
      action: string;
      queue?: string;
    };
    on_intent?: {
      trigger: string;
      action: string;
    };
  };
  cost?: {
    max_tokens_per_session?: number;
    max_cost_per_day?: number;
  };
  security?: {
    pii_masking?: boolean;
    audit_logging?: boolean;
    retention_days?: number;
  };
}

export interface EnvironmentConfig {
  dev?: EnvironmentSettings;
  stage?: EnvironmentSettings;
  prod?: EnvironmentSettings;
}

export interface EnvironmentSettings {
  models?: {
    llm?: string;
    intent?: string;
  };
  api_base?: string;
}

export enum NodeType {
  INTENT = 'intent',
  LLM = 'llm',
  API = 'api',
  CONDITION = 'condition',
  LOOP = 'loop'
}

// ========== DSL Exporter ==========
export class DSLExporter {
  static toYAML(nodes: Node[], edges: Edge[], metadata?: Partial<WorkflowMetadata>): string {
    const dsl = this.toDSL(nodes, edges, metadata);
    return yaml.dump(dsl, {
      indent: 2,
      lineWidth: -1,
      noRefs: true,
      sortKeys: false
    });
  }

  static toJSON(nodes: Node[], edges: Edge[], metadata?: Partial<WorkflowMetadata>): string {
    const dsl = this.toDSL(nodes, edges, metadata);
    return JSON.stringify(dsl, null, 2);
  }

  private static toDSL(nodes: Node[], edges: Edge[], metadata?: Partial<WorkflowMetadata>): WorkflowDSL {
    const currentUser = this.getCurrentUser();
    const now = new Date().toISOString();

    // Build DSL structure
    const dsl: WorkflowDSL = {
      version: '1.0',
      workflow: {
        namespace: metadata?.namespace || 'default',
        name: metadata?.name || 'unnamed-flow',
        version: metadata?.version || '1.0.0',
        metadata: {
          author: metadata?.metadata?.author || currentUser.email,
          description: metadata?.metadata?.description,
          tags: metadata?.metadata?.tags || [],
          created_at: metadata?.metadata?.created_at || now,
          updated_at: now
        }
      },
      nodes: this.convertNodes(nodes),
      edges: this.convertEdges(edges),
      policies: this.getDefaultPolicies(),
      environments: this.getDefaultEnvironments()
    };

    return dsl;
  }

  private static convertNodes(nodes: Node[]): DSLNode[] {
    return nodes.map(node => ({
      id: node.id,
      type: node.type as NodeType,
      position: node.position,
      config: this.extractNodeConfig(node)
    }));
  }

  private static extractNodeConfig(node: Node): Record<string, any> {
    const config = node.data?.config || {};
    
    // Type-specific config extraction
    switch (node.type) {
      case NodeType.INTENT:
        return {
          model: config.model || 'solar-ko-intent',
          categories: config.categories || [],
          confidence_threshold: config.confidence_threshold || 0.7,
          fallback_intent: config.fallback_intent || 'general'
        };
      
      case NodeType.LLM:
        return {
          model: config.model || 'solar-10.7b',
          temperature: config.temperature || 0.7,
          max_tokens: config.max_tokens || 150,
          top_p: config.top_p || 0.9,
          system_prompt: config.system_prompt || '',
          prompt_template: config.prompt_template || ''
        };
      
      case NodeType.API:
        return {
          method: config.method || 'GET',
          url: config.url || '',
          headers: config.headers || {},
          timeout: config.timeout || 5000,
          retry: config.retry || { max_attempts: 3, backoff: 'exponential' },
          response_mapping: config.response_mapping || {}
        };
      
      case NodeType.CONDITION:
        return {
          expression: config.expression || '',
          variables: config.variables || []
        };
      
      case NodeType.LOOP:
        return {
          iterator: config.iterator || '',
          max_iterations: config.max_iterations || 10,
          parallel: config.parallel || false,
          body: config.body || []
        };
      
      default:
        return config;
    }
  }

  private static convertEdges(edges: Edge[]): DSLEdge[] {
    return edges.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      condition: edge.data?.condition,
      label: edge.data?.label || edge.label
    }));
  }

  private static getDefaultPolicies(): PolicyConfig {
    return {
      sla: {
        max_latency_ms: 3000,
        timeout_ms: 10000
      },
      fallback: {
        on_llm_error: {
          action: 'use_model',
          model: 'gpt2-small'
        },
        on_api_error: {
          action: 'use_cache',
          ttl: 3600
        }
      },
      escalation: {
        on_sentiment: {
          trigger: 'angry',
          action: 'transfer_agent',
          queue: 'priority'
        }
      },
      cost: {
        max_tokens_per_session: 1000,
        max_cost_per_day: 100.0
      },
      security: {
        pii_masking: true,
        audit_logging: true,
        retention_days: 90
      }
    };
  }

  private static getDefaultEnvironments(): EnvironmentConfig {
    return {
      dev: {
        models: {
          llm: 'solar-10.7b-dev',
          intent: 'solar-intent-dev'
        },
        api_base: 'https://dev-api.kainexa.local'
      },
      stage: {
        models: {
          llm: 'solar-10.7b-stage',
          intent: 'solar-intent-stage'
        },
        api_base: 'https://stage-api.kainexa.local'
      },
      prod: {
        models: {
          llm: 'solar-10.7b',
          intent: 'solar-intent'
        },
        api_base: 'https://api.kainexa.local'
      }
    };
  }

  private static getCurrentUser(): { email: string } {
    // Get from auth context or localStorage
    return {
      email: localStorage.getItem('userEmail') || 'unknown@kainexa.ai'
    };
  }
}

// ========== DSL Importer ==========
export class DSLImporter {
  static fromYAML(yamlContent: string): { nodes: Node[], edges: Edge[], metadata: WorkflowMetadata } {
    const dsl = yaml.load(yamlContent) as WorkflowDSL;
    return this.fromDSL(dsl);
  }

  static fromJSON(jsonContent: string): { nodes: Node[], edges: Edge[], metadata: WorkflowMetadata } {
    const dsl = JSON.parse(jsonContent) as WorkflowDSL;
    return this.fromDSL(dsl);
  }

  private static fromDSL(dsl: WorkflowDSL): { nodes: Node[], edges: Edge[], metadata: WorkflowMetadata } {
    // Validate DSL structure
    this.validateDSL(dsl);

    // Convert nodes
    const nodes = this.convertToNodes(dsl.nodes);
    
    // Convert edges
    const edges = this.convertToEdges(dsl.edges);

    // Auto-layout if positions are missing
    const layoutedNodes = this.autoLayout(nodes, edges);

    return {
      nodes: layoutedNodes,
      edges,
      metadata: dsl.workflow
    };
  }

  private static validateDSL(dsl: WorkflowDSL): void {
    if (!dsl.version) {
      throw new Error('DSL version is required');
    }

    if (!dsl.workflow) {
      throw new Error('Workflow metadata is required');
    }

    if (!dsl.nodes || dsl.nodes.length === 0) {
      throw new Error('At least one node is required');
    }

    // Validate node types
    for (const node of dsl.nodes) {
      if (!Object.values(NodeType).includes(node.type)) {
        throw new Error(`Invalid node type: ${node.type}`);
      }
    }

    // Validate edges reference existing nodes
    const nodeIds = new Set(dsl.nodes.map(n => n.id));
    for (const edge of dsl.edges) {
      if (!nodeIds.has(edge.source)) {
        throw new Error(`Edge source '${edge.source}' not found in nodes`);
      }
      if (!nodeIds.has(edge.target)) {
        throw new Error(`Edge target '${edge.target}' not found in nodes`);
      }
    }
  }

  private static convertToNodes(dslNodes: DSLNode[]): Node[] {
    return dslNodes.map(node => ({
      id: node.id,
      type: node.type,
      position: node.position || { x: 0, y: 0 },
      data: {
        label: this.getNodeLabel(node),
        config: node.config
      }
    }));
  }

  private static getNodeLabel(node: DSLNode): string {
    switch (node.type) {
      case NodeType.INTENT:
        return `Intent: ${node.id}`;
      case NodeType.LLM:
        return `LLM: ${node.config.model || 'default'}`;
      case NodeType.API:
        return `API: ${node.config.method || 'GET'}`;
      case NodeType.CONDITION:
        return `Condition: ${node.id}`;
      case NodeType.LOOP:
        return `Loop: ${node.config.iterator || 'items'}`;
      default:
        return node.id;
    }
  }

  private static convertToEdges(dslEdges: DSLEdge[]): Edge[] {
    return dslEdges.map((edge, index) => ({
      id: edge.id || `edge-${index}`,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      type: 'smoothstep',
      animated: true,
      data: {
        condition: edge.condition,
        label: edge.label
      }
    }));
  }

  private static autoLayout(nodes: Node[], edges: Edge[]): Node[] {
    if (nodes.every(n => n.position.x !== 0 || n.position.y !== 0)) {
      // Positions already set
      return nodes;
    }

    // Simple hierarchical layout
    const nodeMap = new Map(nodes.map(n => [n.id, n]));
    const levels = this.calculateNodeLevels(nodes, edges);
    
    const horizontalSpacing = 200;
    const verticalSpacing = 150;
    
    // Group nodes by level
    const nodesByLevel = new Map<number, Node[]>();
    levels.forEach((level, nodeId) => {
      const node = nodeMap.get(nodeId);
      if (node) {
        if (!nodesByLevel.has(level)) {
          nodesByLevel.set(level, []);
        }
        nodesByLevel.get(level)!.push(node);
      }
    });

    // Position nodes
    nodesByLevel.forEach((levelNodes, level) => {
      levelNodes.forEach((node, index) => {
        node.position = {
          x: index * horizontalSpacing,
          y: level * verticalSpacing
        };
      });
    });

    return nodes;
  }

  private static calculateNodeLevels(nodes: Node[], edges: Edge[]): Map<string, number> {
    const levels = new Map<string, number>();
    const incoming = new Map<string, Set<string>>();
    
    // Build incoming edges map
    nodes.forEach(n => incoming.set(n.id, new Set()));
    edges.forEach(e => {
      incoming.get(e.target)?.add(e.source);
    });

    // Find root nodes (no incoming edges)
    const queue: string[] = [];
    nodes.forEach(n => {
      if (incoming.get(n.id)?.size === 0) {
        queue.push(n.id);
        levels.set(n.id, 0);
      }
    });

    // BFS to assign levels
    while (queue.length > 0) {
      const nodeId = queue.shift()!;
      const currentLevel = levels.get(nodeId)!;

      // Find outgoing edges
      edges.filter(e => e.source === nodeId).forEach(e => {
        if (!levels.has(e.target) || levels.get(e.target)! < currentLevel + 1) {
          levels.set(e.target, currentLevel + 1);
          if (!queue.includes(e.target)) {
            queue.push(e.target);
          }
        }
      });
    }

    return levels;
  }
}

// ========== DSL Validator ==========
export class DSLValidator {
  static validate(dsl: WorkflowDSL): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Version validation
    if (!this.isValidVersion(dsl.workflow.version)) {
      errors.push(`Invalid version format: ${dsl.workflow.version}. Use semantic versioning (e.g., 1.0.0)`);
    }

    // Namespace validation
    if (!this.isValidNamespace(dsl.workflow.namespace)) {
      errors.push(`Invalid namespace: ${dsl.workflow.namespace}. Use lowercase letters and hyphens only`);
    }

    // Name validation
    if (!this.isValidName(dsl.workflow.name)) {
      errors.push(`Invalid workflow name: ${dsl.workflow.name}. Use lowercase letters, numbers, and hyphens only`);
    }

    // Node validation
    dsl.nodes.forEach(node => {
      const nodeErrors = this.validateNode(node);
      errors.push(...nodeErrors);
    });

    // Edge validation
    const nodeIds = new Set(dsl.nodes.map(n => n.id));
    dsl.edges.forEach(edge => {
      if (!nodeIds.has(edge.source)) {
        errors.push(`Edge source '${edge.source}' references non-existent node`);
      }
      if (!nodeIds.has(edge.target)) {
        errors.push(`Edge target '${edge.target}' references non-existent node`);
      }
    });

    // Graph validation
    const graphErrors = this.validateGraph(dsl.nodes, dsl.edges);
    errors.push(...graphErrors);

    return {
      valid: errors.length === 0,
      errors
    };
  }

  private static isValidVersion(version: string): boolean {
    return /^\d+\.\d+\.\d+$/.test(version);
  }

  private static isValidNamespace(namespace: string): boolean {
    return /^[a-z][a-z0-9-]*$/.test(namespace);
  }

  private static isValidName(name: string): boolean {
    return /^[a-z][a-z0-9-]*$/.test(name);
  }

  private static validateNode(node: DSLNode): string[] {
    const errors: string[] = [];

    if (!node.id) {
      errors.push('Node must have an id');
    }

    if (!Object.values(NodeType).includes(node.type)) {
      errors.push(`Invalid node type: ${node.type}`);
    }

    // Type-specific validation
    switch (node.type) {
      case NodeType.INTENT:
        if (!node.config.categories || node.config.categories.length === 0) {
          errors.push(`Intent node '${node.id}' must have at least one category`);
        }
        break;
      
      case NodeType.LLM:
        if (!node.config.model) {
          errors.push(`LLM node '${node.id}' must specify a model`);
        }
        if (node.config.temperature < 0 || node.config.temperature > 1) {
          errors.push(`LLM node '${node.id}' temperature must be between 0 and 1`);
        }
        break;
      
      case NodeType.API:
        if (!node.config.url) {
          errors.push(`API node '${node.id}' must have a URL`);
        }
        if (!['GET', 'POST', 'PUT', 'DELETE', 'PATCH'].includes(node.config.method)) {
          errors.push(`API node '${node.id}' has invalid HTTP method: ${node.config.method}`);
        }
        break;
      
      case NodeType.CONDITION:
        if (!node.config.expression) {
          errors.push(`Condition node '${node.id}' must have an expression`);
        }
        break;
      
      case NodeType.LOOP:
        if (!node.config.iterator) {
          errors.push(`Loop node '${node.id}' must have an iterator`);
        }
        if (node.config.max_iterations < 1) {
          errors.push(`Loop node '${node.id}' max_iterations must be at least 1`);
        }
        break;
    }

    return errors;
  }

  private static validateGraph(nodes: DSLNode[], edges: DSLEdge[]): string[] {
    const errors: string[] = [];

    // Check for entry points
    const hasIncoming = new Set(edges.map(e => e.target));
    const entryPoints = nodes.filter(n => !hasIncoming.has(n.id));
    
    if (entryPoints.length === 0) {
      errors.push('Workflow has no entry point (node with no incoming edges)');
    } else if (entryPoints.length > 1) {
      errors.push(`Workflow has multiple entry points: ${entryPoints.map(n => n.id).join(', ')}`);
    }

    // Check for unreachable nodes
    const reachable = new Set<string>();
    const queue = entryPoints.map(n => n.id);
    
    while (queue.length > 0) {
      const nodeId = queue.shift()!;
      reachable.add(nodeId);
      
      edges.filter(e => e.source === nodeId && !reachable.has(e.target))
        .forEach(e => queue.push(e.target));
    }

    const unreachable = nodes.filter(n => !reachable.has(n.id));
    if (unreachable.length > 0) {
      errors.push(`Unreachable nodes: ${unreachable.map(n => n.id).join(', ')}`);
    }

    // Check for cycles
    if (this.hasCycle(nodes, edges)) {
      errors.push('Workflow contains a cycle');
    }

    return errors;
  }

  private static hasCycle(nodes: DSLNode[], edges: DSLEdge[]): boolean {
    const adjacency = new Map<string, string[]>();
    nodes.forEach(n => adjacency.set(n.id, []));
    edges.forEach(e => adjacency.get(e.source)?.push(e.target));

    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const hasCycleDFS = (nodeId: string): boolean => {
      visited.add(nodeId);
      recursionStack.add(nodeId);

      for (const neighbor of adjacency.get(nodeId) || []) {
        if (!visited.has(neighbor)) {
          if (hasCycleDFS(neighbor)) {
            return true;
          }
        } else if (recursionStack.has(neighbor)) {
          return true;
        }
      }

      recursionStack.delete(nodeId);
      return false;
    };

    for (const node of nodes) {
      if (!visited.has(node.id)) {
        if (hasCycleDFS(node.id)) {
          return true;
        }
      }
    }

    return false;
  }
}