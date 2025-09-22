-- docker/postgres/init.sql
CREATE SCHEMA IF NOT EXISTS kainexa;

-- Conversations table
CREATE TABLE kainexa.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    channel VARCHAR(50),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Messages table
CREATE TABLE kainexa.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES kainexa.conversations(id),
    role VARCHAR(50) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    intent VARCHAR(100),
    entities JSONB,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Knowledge base table
CREATE TABLE kainexa.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255),
    content TEXT,
    embedding_id VARCHAR(255),
    category VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Analytics table
CREATE TABLE kainexa.analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES kainexa.conversations(id),
    metric_type VARCHAR(100),
    metric_value JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_conversations_session ON kainexa.conversations(session_id);
CREATE INDEX idx_messages_conversation ON kainexa.messages(conversation_id);
CREATE INDEX idx_analytics_conversation ON kainexa.analytics(conversation_id);
CREATE INDEX idx_analytics_timestamp ON kainexa.analytics(timestamp);