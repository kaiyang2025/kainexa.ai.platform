-- Kainexa Core Database Schema
-- PostgreSQL 15+

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create database
-- CREATE DATABASE kainexa_workflow;

-- ========== Core Tables ==========

-- Workflow registry table
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace VARCHAR(100) NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ NULL,
    metadata JSONB DEFAULT '{}',
    UNIQUE(namespace, name) WHERE deleted_at IS NULL
);

-- Workflow versions table
CREATE TABLE IF NOT EXISTS workflow_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    version VARCHAR(20) NOT NULL,
    dsl_raw TEXT NOT NULL,
    dsl_format VARCHAR(10) DEFAULT 'yaml', -- yaml or json
    compiled_graph JSONB,
    status VARCHAR(20) DEFAULT 'uploaded',
    checksums JSONB DEFAULT '{}',
    validation_errors JSONB DEFAULT '[]',
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    compiled_at TIMESTAMPTZ,
    UNIQUE(workflow_id, version),
    CHECK (status IN ('uploaded', 'compiling', 'compiled', 'failed', 'deprecated'))
);

-- Environment routing table
CREATE TABLE IF NOT EXISTS env_routes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    environment VARCHAR(20) NOT NULL,
    active_version VARCHAR(20),
    version_id UUID REFERENCES workflow_versions(id),
    activated_at TIMESTAMPTZ,
    activated_by VARCHAR(100),
    previous_version VARCHAR(20),
    rollback_count INTEGER DEFAULT 0,
    UNIQUE(workflow_id, environment),
    CHECK (environment IN ('dev', 'stage', 'prod'))
);

-- Execution logs table
CREATE TABLE IF NOT EXISTS executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflows(id),
    version VARCHAR(20) NOT NULL,
    version_id UUID REFERENCES workflow_versions(id),
    environment VARCHAR(20) NOT NULL,
    tenant_id VARCHAR(100),
    session_id VARCHAR(200),
    user_id VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    latency_ms INTEGER,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    total_cost DECIMAL(10, 4) DEFAULT 0,
    model_used VARCHAR(100),
    error_code VARCHAR(50),
    error_message TEXT,
    trace_id VARCHAR(100),
    request_payload JSONB,
    response_payload JSONB,
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'timeout', 'cancelled'))
);

-- Node execution details table
CREATE TABLE IF NOT EXISTS node_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES executions(execution_id) ON DELETE CASCADE,
    node_id VARCHAR(100) NOT NULL,
    node_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    inputs JSONB,
    outputs JSONB,
    error TEXT,
    metrics JSONB DEFAULT '{}',
    CHECK (node_type IN ('intent', 'llm', 'api', 'condition', 'loop'))
);

-- Deployment history table
CREATE TABLE IF NOT EXISTS deployment_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflows(id),
    version_id UUID NOT NULL REFERENCES workflow_versions(id),
    version VARCHAR(20) NOT NULL,
    environment VARCHAR(20) NOT NULL,
    action VARCHAR(20) NOT NULL,
    deployed_by VARCHAR(100) NOT NULL,
    deployed_at TIMESTAMPTZ DEFAULT NOW(),
    deployment_metadata JSONB DEFAULT '{}',
    CHECK (action IN ('deploy', 'rollback', 'deactivate'))
);

-- A/B test configurations table
CREATE TABLE IF NOT EXISTS ab_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_name VARCHAR(200) NOT NULL,
    workflow_a_id UUID NOT NULL REFERENCES workflows(id),
    version_a VARCHAR(20) NOT NULL,
    workflow_b_id UUID NOT NULL REFERENCES workflows(id),
    version_b VARCHAR(20) NOT NULL,
    traffic_split DECIMAL(3, 2) DEFAULT 0.50,
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    results JSONB DEFAULT '{}',
    CHECK (traffic_split >= 0 AND traffic_split <= 1),
    CHECK (status IN ('pending', 'running', 'completed', 'cancelled'))
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    user_email VARCHAR(200),
    user_role VARCHAR(50),
    ip_address INET,
    user_agent TEXT,
    request_method VARCHAR(10),
    request_path TEXT,
    request_body JSONB,
    response_status INTEGER,
    changes JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ========== Indexes ==========

-- Workflow indexes
CREATE INDEX idx_workflows_namespace ON workflows(namespace) WHERE deleted_at IS NULL;
CREATE INDEX idx_workflows_created_at ON workflows(created_at DESC);
CREATE INDEX idx_workflows_namespace_name ON workflows(namespace, name) WHERE deleted_at IS NULL;

-- Version indexes
CREATE INDEX idx_workflow_versions_workflow_id ON workflow_versions(workflow_id);
CREATE INDEX idx_workflow_versions_status ON workflow_versions(status);
CREATE INDEX idx_workflow_versions_created_at ON workflow_versions(created_at DESC);

-- Environment routing indexes
CREATE INDEX idx_env_routes_workflow_id ON env_routes(workflow_id);
CREATE INDEX idx_env_routes_environment ON env_routes(environment);

-- Execution indexes
CREATE INDEX idx_executions_workflow_id ON executions(workflow_id, created_at DESC);
CREATE INDEX idx_executions_tenant_id ON executions(tenant_id, created_at DESC);
CREATE INDEX idx_executions_session_id ON executions(session_id);
CREATE INDEX idx_executions_status ON executions(status);
CREATE INDEX idx_executions_created_at ON executions(created_at DESC);
CREATE INDEX idx_executions_trace_id ON executions(trace_id);

-- Node execution indexes
CREATE INDEX idx_node_executions_execution_id ON node_executions(execution_id);
CREATE INDEX idx_node_executions_node_id ON node_executions(node_id);

-- Deployment history indexes
CREATE INDEX idx_deployment_history_workflow_id ON deployment_history(workflow_id);
CREATE INDEX idx_deployment_history_environment ON deployment_history(environment);
CREATE INDEX idx_deployment_history_deployed_at ON deployment_history(deployed_at DESC);

-- Audit log indexes
CREATE INDEX idx_audit_logs_entity ON audit_logs(entity_type, entity_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);

-- ========== Functions & Triggers ==========

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers for updated_at
CREATE TRIGGER update_workflows_updated_at
    BEFORE UPDATE ON workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Audit log function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_logs (
        entity_type,
        entity_id,
        action,
        user_id,
        changes,
        created_at
    ) VALUES (
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        TG_OP,
        COALESCE(NEW.created_by, OLD.created_by, 'system'),
        jsonb_build_object(
            'old', to_jsonb(OLD),
            'new', to_jsonb(NEW)
        ),
        NOW()
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add audit triggers for critical tables
CREATE TRIGGER audit_workflows
    AFTER INSERT OR UPDATE OR DELETE ON workflows
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_workflow_versions
    AFTER INSERT OR UPDATE OR DELETE ON workflow_versions
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_env_routes
    AFTER INSERT OR UPDATE OR DELETE ON env_routes
    FOR EACH ROW
    EXECUTE FUNCTION audit_trigger_function();

-- ========== Views ==========

-- Active workflows view
CREATE OR REPLACE VIEW active_workflows AS
SELECT 
    w.id,
    w.namespace,
    w.name,
    w.description,
    w.created_by,
    w.created_at,
    w.updated_at,
    jsonb_object_agg(
        er.environment,
        jsonb_build_object(
            'version', er.active_version,
            'activated_at', er.activated_at,
            'activated_by', er.activated_by
        )
    ) FILTER (WHERE er.active_version IS NOT NULL) as environments
FROM workflows w
LEFT JOIN env_routes er ON w.id = er.workflow_id
WHERE w.deleted_at IS NULL
GROUP BY w.id;

-- Execution statistics view
CREATE OR REPLACE VIEW execution_stats AS
SELECT 
    w.namespace,
    w.name,
    e.environment,
    e.version,
    COUNT(*) as execution_count,
    AVG(e.latency_ms) as avg_latency_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY e.latency_ms) as p50_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY e.latency_ms) as p95_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY e.latency_ms) as p99_latency_ms,
    SUM(e.tokens_in + e.tokens_out) as total_tokens,
    SUM(e.total_cost) as total_cost,
    COUNT(*) FILTER (WHERE e.status = 'completed') as success_count,
    COUNT(*) FILTER (WHERE e.status = 'failed') as failure_count,
    DATE_TRUNC('hour', e.created_at) as hour
FROM executions e
JOIN workflows w ON e.workflow_id = w.id
WHERE e.created_at >= NOW() - INTERVAL '7 days'
GROUP BY w.namespace, w.name, e.environment, e.version, DATE_TRUNC('hour', e.created_at);

-- ========== Row Level Security (RLS) ==========

-- Enable RLS on critical tables
ALTER TABLE workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE executions ENABLE ROW LEVEL SECURITY;

-- Create policies (example for multi-tenant setup)
-- Note: Adjust based on your authentication method

-- Policy for workflows (users can only see their tenant's workflows)
CREATE POLICY workflow_tenant_policy ON workflows
    FOR ALL
    USING (metadata->>'tenant_id' = current_setting('app.current_tenant', true));

-- Policy for executions (users can only see their own executions)
CREATE POLICY execution_tenant_policy ON executions
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant', true));

-- ========== Initial Data ==========

-- Insert default workflow for testing
INSERT INTO workflows (namespace, name, description, created_by, metadata)
VALUES 
    ('system', 'hello-world', 'Sample hello world workflow', 'system', '{"tags": ["sample", "test"]}'),
    ('cs', 'refund-flow', 'Customer refund processing workflow', 'admin@kainexa.ai', '{"tags": ["refund", "customer-service"]}');

-- Grant permissions (adjust based on your user setup)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO kainexa_api;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO kainexa_api;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO kainexa_api;