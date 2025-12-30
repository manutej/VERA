# VERA Production Specification v1.0

**Version**: 1.0.0
**Status**: Draft
**Date**: 2025-12-29
**Classification**: PRODUCTION SPECIFICATION DOCUMENT
**Prerequisites**: MVP-SPEC.md, synthesis.md
**Extends**: MVP-SPEC.md v1.0.0

---

## 1. Executive Summary

This Production Specification extends VERA MVP to enterprise-grade infrastructure. While MVP demonstrates categorical verification with a single provider and in-memory storage, Production VERA delivers multi-tenant, scalable, secure deployment with comprehensive observability.

### 1.1 Production Goals

| Goal | Metric | Target |
|------|--------|--------|
| Availability | Uptime SLA | 99.9% (8.76 hours/year downtime) |
| Scalability | Concurrent users | 10,000+ |
| Latency | P99 query response | < 1 second |
| Throughput | Queries per second | 1,000+ |
| Security | Compliance | SOC 2 Type II ready |
| Data Durability | RPO/RTO | RPO < 1 hour, RTO < 4 hours |

### 1.2 MVP to Production Delta

| Component | MVP | Production |
|-----------|-----|------------|
| LLM Providers | Anthropic only | Anthropic, OpenAI, Ollama, local models |
| Interface | CLI only | CLI + REST API + WebSocket |
| Storage | In-memory | PostgreSQL + pgvector |
| Tenancy | Single-user | Multi-tenant with isolation |
| Auth | None | API keys + OAuth 2.0 + RBAC |
| Verification | Static thresholds | Policy DSL + custom models |
| Deployment | Local binary | Docker, Kubernetes, Helm |
| Observability | Traces + logs | Full stack (metrics, alerts, dashboards) |

### 1.3 Timeline

**Duration**: 8 weeks (post-MVP)

| Milestone | Weeks | Deliverable |
|-----------|-------|-------------|
| M1: Multi-Provider | 1-2 | Provider registry, fallback chains, cost tracking |
| M2: REST API | 2-3 | OpenAPI 3.0, auth, rate limiting |
| M3: Persistence | 3-4 | PostgreSQL, pgvector, migrations |
| M4: Multi-Tenancy | 4-5 | Row-level security, quotas, billing |
| M5: Policy Engine | 5-6 | Policy DSL, custom NLI, versioning |
| M6: Deployment | 6-7 | Docker, Kubernetes, Helm |
| M7: Security & Obs | 7-8 | Full security suite, Prometheus/Grafana |

---

## 2. Multi-Provider LLM Support

### 2.1 Provider Registry

**Requirement**: Support multiple LLM providers with hot-swappable configuration.

```go
// pkg/llm/registry.go

// ProviderRegistry manages multiple LLM providers
type ProviderRegistry struct {
    providers    map[string]LLMProvider
    primary      string
    fallbacks    []string
    costTracker  *CostTracker
    mu           sync.RWMutex
}

// NewRegistry creates a provider registry with configuration
func NewRegistry(cfg RegistryConfig) (*ProviderRegistry, error)

// Get retrieves a provider by name
func (r *ProviderRegistry) Get(name string) (LLMProvider, error)

// Primary returns the primary provider
func (r *ProviderRegistry) Primary() LLMProvider

// WithFallback executes with automatic failover
func (r *ProviderRegistry) WithFallback(ctx context.Context, fn func(LLMProvider) error) error

// RegistryConfig configures the registry
type RegistryConfig struct {
    Providers []ProviderConfig `yaml:"providers"`
    Primary   string           `yaml:"primary"`
    Fallbacks []string         `yaml:"fallbacks"`
}

// ProviderConfig configures a single provider
type ProviderConfig struct {
    Name     string            `yaml:"name"`
    Type     ProviderType      `yaml:"type"` // anthropic, openai, ollama
    APIKey   string            `yaml:"api_key,omitempty"`
    Endpoint string            `yaml:"endpoint,omitempty"`
    Model    string            `yaml:"model"`
    Options  map[string]any    `yaml:"options,omitempty"`
}

type ProviderType string
const (
    ProviderAnthropic ProviderType = "anthropic"
    ProviderOpenAI    ProviderType = "openai"
    ProviderOllama    ProviderType = "ollama"
    ProviderLocal     ProviderType = "local"
)
```

### 2.2 Provider Implementations

#### 2.2.1 OpenAI Provider

```go
// pkg/llm/openai/provider.go

import "github.com/openai/openai-go"

type OpenAIProvider struct {
    client *openai.Client
    model  string
    embeddingModel string
}

type Config struct {
    APIKey         string `env:"OPENAI_API_KEY"`
    Model          string `default:"gpt-4o"`
    EmbeddingModel string `default:"text-embedding-3-small"`
    MaxTokens      int    `default:"4096"`
    Organization   string `env:"OPENAI_ORG_ID,optional"`
}

func New(cfg Config) (*OpenAIProvider, error)
func (p *OpenAIProvider) Complete(ctx context.Context, prompt Prompt) Result[Response]
func (p *OpenAIProvider) Embed(ctx context.Context, text string) Result[Embedding]
func (p *OpenAIProvider) Stream(ctx context.Context, prompt Prompt) <-chan StreamChunk
func (p *OpenAIProvider) Info() ProviderInfo
```

#### 2.2.2 Ollama Provider

```go
// pkg/llm/ollama/provider.go

type OllamaProvider struct {
    baseURL string
    model   string
    embeddingModel string
}

type Config struct {
    BaseURL        string `default:"http://localhost:11434"`
    Model          string `default:"llama3.1:8b"`
    EmbeddingModel string `default:"nomic-embed-text"`
    Timeout        time.Duration `default:"120s"`
}

func New(cfg Config) (*OllamaProvider, error)
func (p *OllamaProvider) Complete(ctx context.Context, prompt Prompt) Result[Response]
func (p *OllamaProvider) Embed(ctx context.Context, text string) Result[Embedding]
func (p *OllamaProvider) Stream(ctx context.Context, prompt Prompt) <-chan StreamChunk
func (p *OllamaProvider) Info() ProviderInfo
```

#### 2.2.3 Local Model Provider (vLLM/llama.cpp)

```go
// pkg/llm/local/provider.go

type LocalProvider struct {
    endpoint string
    model    string
    protocol LocalProtocol // openai-compat, vllm, llamacpp
}

type LocalProtocol string
const (
    ProtocolOpenAICompat LocalProtocol = "openai-compat"
    ProtocolVLLM         LocalProtocol = "vllm"
    ProtocolLlamaCpp     LocalProtocol = "llamacpp"
)

type Config struct {
    Endpoint string `required:"true"`
    Model    string `required:"true"`
    Protocol LocalProtocol `default:"openai-compat"`
    EmbeddingEndpoint string
}
```

### 2.3 Fallback Chain

```go
// pkg/llm/fallback.go

// FallbackChain executes with automatic failover
type FallbackChain struct {
    providers []LLMProvider
    retries   int
    backoff   BackoffPolicy
}

// Execute attempts providers in order until success
func (f *FallbackChain) Execute(ctx context.Context, fn func(LLMProvider) Result[any]) Result[any] {
    var lastErr error
    for _, provider := range f.providers {
        result := fn(provider)
        if IsOk(result) {
            return result
        }
        lastErr = GetErr(result)
        // Check if error is retryable
        if !isRetryable(lastErr) {
            break
        }
        // Apply backoff
        f.backoff.Wait()
    }
    return Err[any](lastErr)
}

// isRetryable determines if an error warrants trying next provider
func isRetryable(err error) bool {
    var providerErr *ProviderError
    if errors.As(err, &providerErr) {
        return providerErr.IsTransient()
    }
    return false
}
```

### 2.4 Cost Tracking

```go
// pkg/llm/cost.go

// CostTracker tracks token usage and costs per provider
type CostTracker struct {
    usage map[string]*ProviderUsage
    mu    sync.RWMutex
}

// ProviderUsage tracks usage for a single provider
type ProviderUsage struct {
    Provider       string
    InputTokens    int64
    OutputTokens   int64
    EmbeddingTokens int64
    EstimatedCost  float64 // USD
    RequestCount   int64
    LastUpdated    time.Time
}

// Track records token usage for a request
func (c *CostTracker) Track(provider string, usage TokenUsage)

// GetUsage returns usage for a provider
func (c *CostTracker) GetUsage(provider string) *ProviderUsage

// GetTotalCost returns total estimated cost across all providers
func (c *CostTracker) GetTotalCost() float64

// Reset clears all usage tracking
func (c *CostTracker) Reset()

// Export returns usage as prometheus metrics
func (c *CostTracker) Export() []prometheus.Metric

// Pricing table (USD per 1M tokens, as of 2024)
var defaultPricing = map[string]Pricing{
    "claude-sonnet-4-20250514": {Input: 3.00, Output: 15.00},
    "claude-opus-4-20250514":   {Input: 15.00, Output: 75.00},
    "gpt-4o":                   {Input: 5.00, Output: 15.00},
    "gpt-4o-mini":              {Input: 0.15, Output: 0.60},
    "text-embedding-3-small":   {Input: 0.02, Output: 0.0},
}
```

---

## 3. REST API (OpenAPI 3.0)

### 3.1 API Overview

**Base URL**: `https://api.vera.example.com/v1`

```yaml
# openapi.yaml

openapi: 3.0.3
info:
  title: VERA API
  description: Verifiable Evidence-grounded Reasoning Architecture
  version: 1.0.0
  contact:
    name: VERA Support
    email: support@vera.example.com

servers:
  - url: https://api.vera.example.com/v1
    description: Production
  - url: https://api-staging.vera.example.com/v1
    description: Staging
  - url: http://localhost:8080/v1
    description: Local development

security:
  - BearerAuth: []
  - ApiKeyAuth: []

tags:
  - name: Documents
    description: Document ingestion and management
  - name: Queries
    description: Query execution with verification
  - name: Verification
    description: Verification policies and results
  - name: Admin
    description: Administrative operations
```

### 3.2 Endpoints

#### 3.2.1 Documents

```yaml
paths:
  /documents:
    post:
      tags: [Documents]
      summary: Ingest a document
      operationId: ingestDocument
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              required: [file]
              properties:
                file:
                  type: string
                  format: binary
                metadata:
                  type: object
                  properties:
                    name:
                      type: string
                    tags:
                      type: array
                      items:
                        type: string
                chunk_config:
                  $ref: '#/components/schemas/ChunkConfig'
      responses:
        '202':
          description: Document accepted for processing
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DocumentJob'
        '400':
          $ref: '#/components/responses/BadRequest'
        '413':
          description: Document too large
    get:
      tags: [Documents]
      summary: List documents
      operationId: listDocuments
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
        - name: offset
          in: query
          schema:
            type: integer
            default: 0
        - name: tags
          in: query
          schema:
            type: array
            items:
              type: string
      responses:
        '200':
          description: Document list
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DocumentList'

  /documents/{documentId}:
    get:
      tags: [Documents]
      summary: Get document details
      operationId: getDocument
      parameters:
        - name: documentId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Document details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Document'
        '404':
          $ref: '#/components/responses/NotFound'
    delete:
      tags: [Documents]
      summary: Delete document
      operationId: deleteDocument
      parameters:
        - name: documentId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '204':
          description: Document deleted
        '404':
          $ref: '#/components/responses/NotFound'
```

#### 3.2.2 Queries

```yaml
  /query:
    post:
      tags: [Queries]
      summary: Execute verified query
      operationId: executeQuery
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/QueryRequest'
      responses:
        '200':
          description: Query response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QueryResponse'
        '202':
          description: Query accepted (async mode)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QueryJob'
        '400':
          $ref: '#/components/responses/BadRequest'
        '429':
          $ref: '#/components/responses/RateLimited'

  /query/stream:
    post:
      tags: [Queries]
      summary: Execute streaming query
      operationId: executeQueryStream
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/QueryRequest'
      responses:
        '200':
          description: Streaming response
          content:
            text/event-stream:
              schema:
                $ref: '#/components/schemas/StreamEvent'

  /queries/{queryId}:
    get:
      tags: [Queries]
      summary: Get query result
      operationId: getQueryResult
      parameters:
        - name: queryId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Query result
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QueryResponse'
        '202':
          description: Query still processing
        '404':
          $ref: '#/components/responses/NotFound'
```

#### 3.2.3 Verification

```yaml
  /policies:
    get:
      tags: [Verification]
      summary: List verification policies
      operationId: listPolicies
      responses:
        '200':
          description: Policy list
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PolicyList'
    post:
      tags: [Verification]
      summary: Create verification policy
      operationId: createPolicy
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PolicyCreate'
      responses:
        '201':
          description: Policy created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Policy'

  /policies/{policyId}:
    get:
      tags: [Verification]
      summary: Get policy details
      operationId: getPolicy
      parameters:
        - name: policyId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Policy details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Policy'
```

### 3.3 Schemas

```yaml
components:
  schemas:
    QueryRequest:
      type: object
      required: [query]
      properties:
        query:
          type: string
          minLength: 1
          maxLength: 10000
        document_ids:
          type: array
          items:
            type: string
            format: uuid
        policy_id:
          type: string
          format: uuid
        options:
          $ref: '#/components/schemas/QueryOptions'
        async:
          type: boolean
          default: false

    QueryOptions:
      type: object
      properties:
        grounding_threshold:
          type: number
          minimum: 0
          maximum: 1
          default: 0.80
        max_retrieval_hops:
          type: integer
          minimum: 1
          maximum: 10
          default: 3
        include_citations:
          type: boolean
          default: true
        provider:
          type: string
          description: Override default provider

    QueryResponse:
      type: object
      required: [id, response, grounding_score, status]
      properties:
        id:
          type: string
          format: uuid
        response:
          type: string
        grounding_score:
          type: number
          minimum: 0
          maximum: 1
        grounding_status:
          type: string
          enum: [GROUNDED, PARTIAL, UNGROUNDED]
        citations:
          type: array
          items:
            $ref: '#/components/schemas/Citation'
        usage:
          $ref: '#/components/schemas/TokenUsage'
        duration_ms:
          type: integer
        status:
          type: string
          enum: [complete, partial, failed]
        metadata:
          $ref: '#/components/schemas/QueryMetadata'

    Citation:
      type: object
      properties:
        claim_text:
          type: string
        source_id:
          type: string
          format: uuid
        source_name:
          type: string
        page_number:
          type: integer
        source_text:
          type: string
        score:
          type: number

    Document:
      type: object
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        content_type:
          type: string
        size_bytes:
          type: integer
        chunk_count:
          type: integer
        status:
          type: string
          enum: [processing, ready, failed]
        tags:
          type: array
          items:
            type: string
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time

    Policy:
      type: object
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        version:
          type: integer
        rules:
          $ref: '#/components/schemas/PolicyRules'
        active:
          type: boolean
        created_at:
          type: string
          format: date-time

    PolicyRules:
      type: object
      properties:
        grounding_threshold:
          type: number
        required_citation_count:
          type: integer
        allowed_sources:
          type: array
          items:
            type: string
        forbidden_topics:
          type: array
          items:
            type: string
        custom_nli_model:
          type: string
        escalation_rules:
          $ref: '#/components/schemas/EscalationRules'

    StreamEvent:
      type: object
      properties:
        event:
          type: string
          enum: [delta, citation, metadata, done, error]
        data:
          oneOf:
            - type: string
            - $ref: '#/components/schemas/Citation'
            - $ref: '#/components/schemas/QueryMetadata'

    Error:
      type: object
      required: [code, message]
      properties:
        code:
          type: string
        message:
          type: string
        details:
          type: object
        trace_id:
          type: string
```

### 3.4 Authentication

#### 3.4.1 API Key Authentication

```go
// pkg/api/auth/apikey.go

type APIKeyAuth struct {
    store     KeyStore
    validator KeyValidator
}

// KeyStore persists API keys
type KeyStore interface {
    Get(ctx context.Context, keyHash string) (*APIKey, error)
    Create(ctx context.Context, key *APIKey) error
    Revoke(ctx context.Context, keyID string) error
    List(ctx context.Context, tenantID string) ([]*APIKey, error)
}

// APIKey represents an API key
type APIKey struct {
    ID          string
    TenantID    string
    Name        string
    KeyHash     string // SHA-256 hash, never store plaintext
    Permissions []Permission
    RateLimit   *RateLimit
    ExpiresAt   *time.Time
    LastUsedAt  *time.Time
    CreatedAt   time.Time
}

// Permission defines allowed operations
type Permission string
const (
    PermissionRead    Permission = "read"
    PermissionWrite   Permission = "write"
    PermissionAdmin   Permission = "admin"
    PermissionQuery   Permission = "query"
    PermissionIngest  Permission = "ingest"
    PermissionPolicy  Permission = "policy"
)

// Middleware validates API key from header
func (a *APIKeyAuth) Middleware() func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            key := r.Header.Get("X-API-Key")
            if key == "" {
                key = r.Header.Get("Authorization")
                key = strings.TrimPrefix(key, "Bearer ")
            }

            apiKey, err := a.validate(r.Context(), key)
            if err != nil {
                http.Error(w, "Unauthorized", http.StatusUnauthorized)
                return
            }

            ctx := context.WithValue(r.Context(), apiKeyContextKey, apiKey)
            next.ServeHTTP(w, r.WithContext(ctx))
        })
    }
}
```

#### 3.4.2 OAuth 2.0 Support

```go
// pkg/api/auth/oauth.go

type OAuthConfig struct {
    Issuer          string   `yaml:"issuer"`
    Audience        string   `yaml:"audience"`
    JWKSEndpoint    string   `yaml:"jwks_endpoint"`
    AllowedIssuers  []string `yaml:"allowed_issuers"`
    ClaimsMapping   map[string]string `yaml:"claims_mapping"`
}

// OAuthValidator validates JWT tokens
type OAuthValidator struct {
    config   OAuthConfig
    keyCache *JWKSCache
}

func (v *OAuthValidator) Validate(ctx context.Context, token string) (*Claims, error)

// Claims represents validated OAuth claims
type Claims struct {
    Subject    string
    TenantID   string
    Email      string
    Roles      []string
    Permissions []Permission
    ExpiresAt  time.Time
}
```

### 3.5 Rate Limiting

```go
// pkg/api/ratelimit/limiter.go

type RateLimiter struct {
    store    RateLimitStore
    defaults RateLimitConfig
}

// RateLimitStore persists rate limit state
type RateLimitStore interface {
    Get(ctx context.Context, key string) (*RateLimitState, error)
    Increment(ctx context.Context, key string, window time.Duration) (*RateLimitState, error)
}

// RateLimitConfig defines limits per tier
type RateLimitConfig struct {
    RequestsPerMinute int           `yaml:"requests_per_minute"`
    RequestsPerHour   int           `yaml:"requests_per_hour"`
    RequestsPerDay    int           `yaml:"requests_per_day"`
    TokensPerMinute   int           `yaml:"tokens_per_minute"`
    BurstSize         int           `yaml:"burst_size"`
}

// Tier-based rate limits
var DefaultRateLimits = map[string]RateLimitConfig{
    "free":       {RequestsPerMinute: 10, RequestsPerHour: 100, TokensPerMinute: 10000},
    "starter":    {RequestsPerMinute: 60, RequestsPerHour: 1000, TokensPerMinute: 100000},
    "pro":        {RequestsPerMinute: 600, RequestsPerHour: 10000, TokensPerMinute: 1000000},
    "enterprise": {RequestsPerMinute: 6000, RequestsPerHour: 100000, TokensPerMinute: 10000000},
}

// Middleware applies rate limiting
func (r *RateLimiter) Middleware() func(http.Handler) http.Handler

// Headers set on response
// X-RateLimit-Limit: 60
// X-RateLimit-Remaining: 45
// X-RateLimit-Reset: 1609459200
// Retry-After: 30 (on 429)
```

### 3.6 WebSocket Streaming

```go
// pkg/api/websocket/handler.go

type WebSocketHandler struct {
    upgrader websocket.Upgrader
    pipeline PipelineExecutor
}

// Message types
type WSMessage struct {
    Type    string          `json:"type"`
    ID      string          `json:"id"`
    Payload json.RawMessage `json:"payload"`
}

type WSMessageType string
const (
    WSTypeQuery      WSMessageType = "query"
    WSTypeDelta      WSMessageType = "delta"
    WSTypeCitation   WSMessageType = "citation"
    WSTypeMetadata   WSMessageType = "metadata"
    WSTypeError      WSMessageType = "error"
    WSTypeDone       WSMessageType = "done"
    WSTypePing       WSMessageType = "ping"
    WSTypePong       WSMessageType = "pong"
)

// Protocol
// Client -> Server: {"type": "query", "id": "123", "payload": {...}}
// Server -> Client: {"type": "delta", "id": "123", "payload": {"text": "..."}}
// Server -> Client: {"type": "citation", "id": "123", "payload": {...}}
// Server -> Client: {"type": "done", "id": "123", "payload": {"grounding_score": 0.87}}
```

---

## 4. Persistent Storage

### 4.1 PostgreSQL Schema

```sql
-- migrations/001_initial_schema.up.sql

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Tenants table
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    plan VARCHAR(50) NOT NULL DEFAULT 'free',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- API Keys table
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hash
    permissions TEXT[] NOT NULL DEFAULT '{}',
    rate_limit JSONB,
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    size_bytes BIGINT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'processing',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX idx_documents_tenant ON documents(tenant_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_tags ON documents USING GIN(tags);

-- Chunks table (with vector embeddings)
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536), -- OpenAI text-embedding-3-small dimension
    page_number INT,
    start_offset INT,
    end_offset INT,
    token_count INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_tenant ON chunks(tenant_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search index for hybrid search
ALTER TABLE chunks ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX idx_chunks_search ON chunks USING GIN(search_vector);

-- Queries table (audit log)
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    response_text TEXT,
    grounding_score FLOAT,
    grounding_status VARCHAR(20),
    citations JSONB DEFAULT '[]',
    document_ids UUID[] DEFAULT '{}',
    policy_id UUID,
    provider VARCHAR(50),
    input_tokens INT,
    output_tokens INT,
    duration_ms INT,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_queries_tenant ON queries(tenant_id);
CREATE INDEX idx_queries_created ON queries(created_at);
CREATE INDEX idx_queries_status ON queries(status);

-- Policies table
CREATE TABLE policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    version INT NOT NULL DEFAULT 1,
    rules JSONB NOT NULL,
    active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(tenant_id, name)
);

CREATE INDEX idx_policies_tenant ON policies(tenant_id);
CREATE INDEX idx_policies_active ON policies(active) WHERE active = true;

-- Policy versions (for audit trail)
CREATE TABLE policy_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_id UUID NOT NULL REFERENCES policies(id) ON DELETE CASCADE,
    version INT NOT NULL,
    rules JSONB NOT NULL,
    created_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(policy_id, version)
);

-- Usage tracking
CREATE TABLE usage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    provider VARCHAR(50) NOT NULL,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    embedding_tokens BIGINT NOT NULL DEFAULT 0,
    query_count INT NOT NULL DEFAULT 0,
    document_count INT NOT NULL DEFAULT 0,
    storage_bytes BIGINT NOT NULL DEFAULT 0,
    estimated_cost_usd DECIMAL(10, 4) NOT NULL DEFAULT 0,

    UNIQUE(tenant_id, date, provider)
);

CREATE INDEX idx_usage_tenant_date ON usage_records(tenant_id, date);

-- Row Level Security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE policies ENABLE ROW LEVEL SECURITY;

-- RLS Policies (tenant isolation)
CREATE POLICY tenant_isolation_documents ON documents
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_chunks ON chunks
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_queries ON queries
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_policies ON policies
    USING (tenant_id = current_setting('app.tenant_id')::uuid);
```

### 4.2 Migrations

```go
// internal/storage/migrations/migrator.go

type Migrator struct {
    db      *sql.DB
    dir     string
    version int
}

// Up applies all pending migrations
func (m *Migrator) Up(ctx context.Context) error

// Down reverts the last migration
func (m *Migrator) Down(ctx context.Context) error

// Version returns current migration version
func (m *Migrator) Version() (int, error)

// MigrationHistory returns applied migrations
func (m *Migrator) History() ([]Migration, error)
```

### 4.3 Backup and Restore

```go
// internal/storage/backup/backup.go

type BackupConfig struct {
    Schedule      string        `yaml:"schedule"` // cron expression
    Retention     time.Duration `yaml:"retention"`
    S3Bucket      string        `yaml:"s3_bucket"`
    EncryptionKey string        `yaml:"encryption_key"`
}

// Backup creates a database backup
func Backup(ctx context.Context, cfg BackupConfig) (*BackupResult, error)

// Restore restores from a backup
func Restore(ctx context.Context, backupID string) error

// ListBackups returns available backups
func ListBackups(ctx context.Context) ([]BackupInfo, error)

// Point-in-time recovery
func RestoreToPoint(ctx context.Context, timestamp time.Time) error
```

### 4.4 Connection Pooling

```go
// internal/storage/pool.go

type PoolConfig struct {
    MaxOpenConns     int           `yaml:"max_open_conns" default:"25"`
    MaxIdleConns     int           `yaml:"max_idle_conns" default:"10"`
    ConnMaxLifetime  time.Duration `yaml:"conn_max_lifetime" default:"5m"`
    ConnMaxIdleTime  time.Duration `yaml:"conn_max_idle_time" default:"1m"`
    HealthCheckPeriod time.Duration `yaml:"health_check_period" default:"30s"`
}

// NewPool creates a connection pool with pgxpool
func NewPool(ctx context.Context, dsn string, cfg PoolConfig) (*pgxpool.Pool, error)

// WithTenant sets tenant context for RLS
func WithTenant(ctx context.Context, pool *pgxpool.Pool, tenantID string) (*pgxpool.Conn, error) {
    conn, err := pool.Acquire(ctx)
    if err != nil {
        return nil, err
    }
    _, err = conn.Exec(ctx, "SET app.tenant_id = $1", tenantID)
    if err != nil {
        conn.Release()
        return nil, err
    }
    return conn, nil
}
```

---

## 5. Multi-Tenant Architecture

### 5.1 Tenant Isolation

```go
// pkg/tenant/tenant.go

type Tenant struct {
    ID       string
    Name     string
    Plan     TenantPlan
    Settings TenantSettings
    Quotas   TenantQuotas
}

type TenantPlan string
const (
    PlanFree       TenantPlan = "free"
    PlanStarter    TenantPlan = "starter"
    PlanPro        TenantPlan = "pro"
    PlanEnterprise TenantPlan = "enterprise"
)

// TenantSettings configurable per tenant
type TenantSettings struct {
    DefaultProvider       string         `json:"default_provider"`
    DefaultGroundingThreshold float64    `json:"default_grounding_threshold"`
    AllowedProviders      []string       `json:"allowed_providers"`
    MaxDocumentSizeMB     int           `json:"max_document_size_mb"`
    RetentionDays         int           `json:"retention_days"`
    WebhookURL            string         `json:"webhook_url,omitempty"`
    CustomBranding        map[string]any `json:"custom_branding,omitempty"`
}

// TenantQuotas defines resource limits
type TenantQuotas struct {
    MaxDocuments       int   `json:"max_documents"`
    MaxStorageGB       int   `json:"max_storage_gb"`
    MaxQueriesPerMonth int   `json:"max_queries_per_month"`
    MaxTokensPerMonth  int64 `json:"max_tokens_per_month"`
    MaxChunksTotal     int   `json:"max_chunks_total"`
}

// Default quotas per plan
var DefaultQuotas = map[TenantPlan]TenantQuotas{
    PlanFree:       {MaxDocuments: 10, MaxStorageGB: 1, MaxQueriesPerMonth: 100, MaxTokensPerMonth: 100000},
    PlanStarter:    {MaxDocuments: 100, MaxStorageGB: 10, MaxQueriesPerMonth: 1000, MaxTokensPerMonth: 1000000},
    PlanPro:        {MaxDocuments: 1000, MaxStorageGB: 100, MaxQueriesPerMonth: 10000, MaxTokensPerMonth: 10000000},
    PlanEnterprise: {MaxDocuments: -1, MaxStorageGB: -1, MaxQueriesPerMonth: -1, MaxTokensPerMonth: -1}, // Unlimited
}
```

### 5.2 Resource Quotas

```go
// pkg/tenant/quota.go

type QuotaEnforcer struct {
    store QuotaStore
}

// QuotaStore tracks usage against quotas
type QuotaStore interface {
    GetUsage(ctx context.Context, tenantID string) (*TenantUsage, error)
    IncrementUsage(ctx context.Context, tenantID string, delta UsageDelta) error
    ResetMonthlyUsage(ctx context.Context) error
}

// TenantUsage current usage metrics
type TenantUsage struct {
    DocumentCount      int
    StorageBytes       int64
    QueriesThisMonth   int
    TokensThisMonth    int64
    ChunkCount         int
    LastUpdated        time.Time
}

// CheckQuota returns error if quota exceeded
func (e *QuotaEnforcer) CheckQuota(ctx context.Context, tenant *Tenant, action QuotaAction) error {
    usage, err := e.store.GetUsage(ctx, tenant.ID)
    if err != nil {
        return err
    }

    quotas := tenant.Quotas
    switch action {
    case ActionIngestDocument:
        if quotas.MaxDocuments > 0 && usage.DocumentCount >= quotas.MaxDocuments {
            return ErrQuotaExceeded{Resource: "documents", Limit: quotas.MaxDocuments}
        }
    case ActionExecuteQuery:
        if quotas.MaxQueriesPerMonth > 0 && usage.QueriesThisMonth >= quotas.MaxQueriesPerMonth {
            return ErrQuotaExceeded{Resource: "queries", Limit: quotas.MaxQueriesPerMonth}
        }
    case ActionConsumeTokens:
        // Checked at consumption time
    }

    return nil
}

// ErrQuotaExceeded returned when quota is exceeded
type ErrQuotaExceeded struct {
    Resource string
    Limit    int
    Current  int
}
```

### 5.3 Billing Hooks

```go
// pkg/billing/hooks.go

type BillingHooks struct {
    provider    BillingProvider
    usageStore  UsageStore
}

// BillingProvider interface for payment processing
type BillingProvider interface {
    CreateCustomer(ctx context.Context, tenant *Tenant) (string, error)
    UpdateSubscription(ctx context.Context, customerID string, plan TenantPlan) error
    RecordUsage(ctx context.Context, customerID string, usage UsageRecord) error
    GetInvoices(ctx context.Context, customerID string) ([]Invoice, error)
}

// UsageRecord for metered billing
type UsageRecord struct {
    TenantID      string
    Timestamp     time.Time
    Metric        UsageMetric
    Quantity      int64
    UnitPriceUSD  float64
}

type UsageMetric string
const (
    MetricInputTokens    UsageMetric = "input_tokens"
    MetricOutputTokens   UsageMetric = "output_tokens"
    MetricEmbeddingTokens UsageMetric = "embedding_tokens"
    MetricStorageGB      UsageMetric = "storage_gb"
    MetricQueries        UsageMetric = "queries"
)

// RecordUsage hooks into query execution
func (b *BillingHooks) RecordUsage(ctx context.Context, tenantID string, usage TokenUsage) error {
    records := []UsageRecord{
        {TenantID: tenantID, Timestamp: time.Now(), Metric: MetricInputTokens, Quantity: int64(usage.InputTokens)},
        {TenantID: tenantID, Timestamp: time.Now(), Metric: MetricOutputTokens, Quantity: int64(usage.OutputTokens)},
    }

    for _, record := range records {
        if err := b.provider.RecordUsage(ctx, tenantID, record); err != nil {
            return err
        }
    }

    return nil
}
```

---

## 6. Extended Verification

### 6.1 Policy DSL

```go
// pkg/verify/policy/dsl.go

// Policy defines verification rules
type Policy struct {
    ID          string
    Name        string
    Version     int
    Rules       PolicyRules
    Active      bool
    CreatedAt   time.Time
}

// PolicyRules configurable verification rules
type PolicyRules struct {
    // Thresholds
    GroundingThreshold float64 `json:"grounding_threshold"`
    CitationThreshold  float64 `json:"citation_threshold"`
    CoverageThreshold  float64 `json:"coverage_threshold"`

    // Requirements
    MinCitations       int      `json:"min_citations"`
    MaxRetrievalHops   int      `json:"max_retrieval_hops"`
    RequiredSources    []string `json:"required_sources,omitempty"`

    // Restrictions
    ForbiddenTopics    []string `json:"forbidden_topics,omitempty"`
    AllowedDomains     []string `json:"allowed_domains,omitempty"`
    MaxResponseLength  int      `json:"max_response_length,omitempty"`

    // Custom verification
    CustomNLIModel     string   `json:"custom_nli_model,omitempty"`
    CustomPrompts      map[string]string `json:"custom_prompts,omitempty"`

    // Escalation
    EscalationRules    EscalationRules `json:"escalation_rules"`
}

// EscalationRules define when to escalate to human review
type EscalationRules struct {
    LowGroundingAction   EscalationAction `json:"low_grounding_action"`
    LowGroundingThreshold float64         `json:"low_grounding_threshold"`
    ForbiddenTopicAction EscalationAction `json:"forbidden_topic_action"`
    WebhookURL           string           `json:"webhook_url,omitempty"`
    SlackChannel         string           `json:"slack_channel,omitempty"`
}

type EscalationAction string
const (
    ActionAllow    EscalationAction = "allow"   // Continue with warning
    ActionBlock    EscalationAction = "block"   // Reject response
    ActionEscalate EscalationAction = "escalate" // Send to human review
    ActionRedact   EscalationAction = "redact"   // Remove ungrounded claims
)

// PolicyDSL grammar (YAML representation)
/*
policy:
  name: "strict-legal"
  version: 1
  rules:
    grounding_threshold: 0.90
    min_citations: 2
    required_sources:
      - "contract.pdf"
      - "amendment.pdf"
    forbidden_topics:
      - "personal_injury"
      - "criminal_matters"
    escalation_rules:
      low_grounding_action: escalate
      low_grounding_threshold: 0.75
      webhook_url: "https://hooks.example.com/vera"
*/
```

### 6.2 Custom NLI Models

```go
// pkg/verify/nli/custom.go

// NLIModel interface for custom NLI implementations
type NLIModel interface {
    // Verify checks if premise entails hypothesis
    Verify(ctx context.Context, premise, hypothesis string) (NLIResult, error)

    // BatchVerify for efficiency
    BatchVerify(ctx context.Context, pairs []PremiseHypothesis) ([]NLIResult, error)

    // Info returns model metadata
    Info() NLIModelInfo
}

// NLIResult from verification
type NLIResult struct {
    Entailment    float64 `json:"entailment"`
    Neutral       float64 `json:"neutral"`
    Contradiction float64 `json:"contradiction"`
}

// NLIModelInfo describes the model
type NLIModelInfo struct {
    Name      string
    Provider  string // huggingface, local, custom
    Endpoint  string
    MaxLength int
}

// CustomNLIProvider allows tenant-specific NLI models
type CustomNLIProvider struct {
    defaultModel NLIModel
    customModels map[string]NLIModel // keyed by model name
}

func (p *CustomNLIProvider) GetModel(name string) (NLIModel, error) {
    if name == "" {
        return p.defaultModel, nil
    }
    model, ok := p.customModels[name]
    if !ok {
        return nil, ErrModelNotFound{Name: name}
    }
    return model, nil
}

// LocalNLI runs inference on local model
type LocalNLI struct {
    modelPath string
    runtime   ONNXRuntime // or PyTorch
}

func (l *LocalNLI) Verify(ctx context.Context, premise, hypothesis string) (NLIResult, error)
```

### 6.3 Policy Versioning

```go
// pkg/verify/policy/versioning.go

type PolicyVersioning struct {
    store PolicyStore
}

// PolicyStore persists policies and versions
type PolicyStore interface {
    Create(ctx context.Context, policy *Policy) error
    Update(ctx context.Context, policy *Policy) error
    Get(ctx context.Context, id string) (*Policy, error)
    GetVersion(ctx context.Context, id string, version int) (*Policy, error)
    ListVersions(ctx context.Context, id string) ([]PolicyVersion, error)
    SetActive(ctx context.Context, id string, version int) error
}

// CreateVersion creates a new policy version
func (v *PolicyVersioning) CreateVersion(ctx context.Context, policyID string, rules PolicyRules) (*Policy, error) {
    current, err := v.store.Get(ctx, policyID)
    if err != nil {
        return nil, err
    }

    newVersion := &Policy{
        ID:        policyID,
        Name:      current.Name,
        Version:   current.Version + 1,
        Rules:     rules,
        Active:    false, // Not active until explicitly activated
        CreatedAt: time.Now(),
    }

    if err := v.store.Update(ctx, newVersion); err != nil {
        return nil, err
    }

    return newVersion, nil
}

// ActivateVersion makes a version active
func (v *PolicyVersioning) ActivateVersion(ctx context.Context, policyID string, version int) error {
    return v.store.SetActive(ctx, policyID, version)
}

// Rollback reverts to a previous version
func (v *PolicyVersioning) Rollback(ctx context.Context, policyID string, version int) error {
    return v.store.SetActive(ctx, policyID, version)
}
```

---

## 7. Deployment

### 7.1 Docker

```dockerfile
# Dockerfile

# Build stage
FROM golang:1.23-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build binaries
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /vera ./cmd/vera
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /vera-server ./cmd/vera-server

# Runtime stage
FROM alpine:3.19

RUN apk --no-cache add ca-certificates tzdata

WORKDIR /app

COPY --from=builder /vera /app/vera
COPY --from=builder /vera-server /app/vera-server

# Non-root user
RUN adduser -D -u 1000 vera
USER vera

EXPOSE 8080 9090

ENTRYPOINT ["/app/vera-server"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  vera:
    build: .
    ports:
      - "8080:8080"   # API
      - "9090:9090"   # Metrics
    environment:
      - VERA_DATABASE_URL=postgres://vera:vera@postgres:5432/vera?sslmode=disable
      - VERA_REDIS_URL=redis://redis:6379
      - VERA_ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - VERA_OTEL_ENDPOINT=http://jaeger:4317
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=vera
      - POSTGRES_PASSWORD=vera
      - POSTGRES_DB=vera
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  jaeger:
    image: jaegertracing/all-in-one:1.52
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP

  prometheus:
    image: prom/prometheus:v2.48.0
    volumes:
      - ./deploy/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9091:9090"

  grafana:
    image: grafana/grafana:10.2.2
    volumes:
      - ./deploy/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 7.2 Kubernetes Manifests

```yaml
# deploy/kubernetes/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: vera
  labels:
    app: vera
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vera
  template:
    metadata:
      labels:
        app: vera
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: vera
      containers:
        - name: vera
          image: vera:latest
          ports:
            - name: http
              containerPort: 8080
            - name: metrics
              containerPort: 9090
          env:
            - name: VERA_DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: vera-secrets
                  key: database-url
            - name: VERA_REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: vera-secrets
                  key: redis-url
            - name: VERA_ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: vera-secrets
                  key: anthropic-api-key
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: vera
                topologyKey: kubernetes.io/hostname
---
apiVersion: v1
kind: Service
metadata:
  name: vera
spec:
  selector:
    app: vera
  ports:
    - name: http
      port: 80
      targetPort: 8080
    - name: metrics
      port: 9090
      targetPort: 9090
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vera
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vera
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### 7.3 Helm Chart

```yaml
# charts/vera/Chart.yaml
apiVersion: v2
name: vera
description: VERA - Verifiable Evidence-grounded Reasoning Architecture
type: application
version: 1.0.0
appVersion: "1.0.0"

dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: "18.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
```

```yaml
# charts/vera/values.yaml
replicaCount: 3

image:
  repository: vera
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: api.vera.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: vera-tls
      hosts:
        - api.vera.example.com

resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    postgresPassword: ""  # Set via --set
    database: vera
  primary:
    persistence:
      size: 100Gi
    resources:
      requests:
        cpu: 500m
        memory: 1Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: ""  # Set via --set
  master:
    persistence:
      size: 10Gi

config:
  logLevel: info
  logFormat: json
  defaultProvider: anthropic
  groundingThreshold: 0.80
  maxRetrievalHops: 3

secrets:
  anthropicApiKey: ""  # Set via --set-file or external secrets
  openaiApiKey: ""
  databaseUrl: ""
  redisUrl: ""

observability:
  prometheus:
    enabled: true
  jaeger:
    enabled: true
    agent:
      host: jaeger-agent
      port: 6831
  serviceMonitor:
    enabled: true
    interval: 15s
```

### 7.4 Health Checks

```go
// pkg/api/health/health.go

type HealthChecker struct {
    db       *pgxpool.Pool
    redis    *redis.Client
    registry *ProviderRegistry
}

// LivenessProbe for Kubernetes liveness
func (h *HealthChecker) LivenessProbe(w http.ResponseWriter, r *http.Request) {
    // Simple check - if we can respond, we're alive
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]string{"status": "alive"})
}

// ReadinessProbe for Kubernetes readiness
func (h *HealthChecker) ReadinessProbe(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
    defer cancel()

    checks := []HealthCheck{
        {Name: "database", Check: h.checkDatabase},
        {Name: "redis", Check: h.checkRedis},
        {Name: "provider", Check: h.checkProvider},
    }

    results := make(map[string]HealthResult)
    healthy := true

    for _, check := range checks {
        result := check.Check(ctx)
        results[check.Name] = result
        if !result.Healthy {
            healthy = false
        }
    }

    status := http.StatusOK
    if !healthy {
        status = http.StatusServiceUnavailable
    }

    w.WriteHeader(status)
    json.NewEncoder(w).Encode(map[string]any{
        "status":  statusString(healthy),
        "checks":  results,
    })
}

type HealthResult struct {
    Healthy  bool          `json:"healthy"`
    Latency  time.Duration `json:"latency_ms"`
    Message  string        `json:"message,omitempty"`
}
```

---

## 8. Security

### 8.1 Encryption at Rest

```go
// internal/security/encryption.go

type Encryptor struct {
    key []byte // AES-256 key
}

// NewEncryptor creates encryptor from key or generates one
func NewEncryptor(keyBase64 string) (*Encryptor, error)

// Encrypt encrypts plaintext using AES-256-GCM
func (e *Encryptor) Encrypt(plaintext []byte) ([]byte, error)

// Decrypt decrypts ciphertext
func (e *Encryptor) Decrypt(ciphertext []byte) ([]byte, error)

// Column-level encryption for sensitive fields
// Documents: Original file bytes encrypted
// API Keys: Encrypted in transit, hashed at rest
// Audit logs: PII fields encrypted
```

### 8.2 Encryption in Transit

```yaml
# TLS configuration
server:
  tls:
    enabled: true
    cert_file: /etc/vera/tls/tls.crt
    key_file: /etc/vera/tls/tls.key
    min_version: TLS1.2
    cipher_suites:
      - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
      - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
      - TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
```

### 8.3 RBAC

```go
// pkg/api/auth/rbac.go

type RBAC struct {
    store RoleStore
}

// Role defines a set of permissions
type Role struct {
    ID          string
    Name        string
    Permissions []Permission
    TenantID    string // Empty for global roles
}

// Predefined roles
var DefaultRoles = map[string]Role{
    "admin": {
        Name: "admin",
        Permissions: []Permission{
            PermissionRead, PermissionWrite, PermissionAdmin,
            PermissionQuery, PermissionIngest, PermissionPolicy,
        },
    },
    "editor": {
        Name: "editor",
        Permissions: []Permission{
            PermissionRead, PermissionWrite,
            PermissionQuery, PermissionIngest,
        },
    },
    "viewer": {
        Name: "viewer",
        Permissions: []Permission{
            PermissionRead, PermissionQuery,
        },
    },
}

// Authorize checks if user has permission
func (r *RBAC) Authorize(ctx context.Context, userID string, permission Permission) error {
    user := UserFromContext(ctx)
    if user == nil {
        return ErrUnauthorized
    }

    role, err := r.store.GetUserRole(ctx, userID)
    if err != nil {
        return err
    }

    for _, p := range role.Permissions {
        if p == permission || p == PermissionAdmin {
            return nil
        }
    }

    return ErrForbidden{Required: permission}
}
```

### 8.4 Audit Logging

```go
// internal/audit/audit.go

type AuditLogger struct {
    store AuditStore
}

// AuditEvent represents an auditable action
type AuditEvent struct {
    ID         string         `json:"id"`
    Timestamp  time.Time      `json:"timestamp"`
    TenantID   string         `json:"tenant_id"`
    UserID     string         `json:"user_id"`
    Action     AuditAction    `json:"action"`
    Resource   string         `json:"resource"`
    ResourceID string         `json:"resource_id"`
    Outcome    AuditOutcome   `json:"outcome"`
    Details    map[string]any `json:"details,omitempty"`
    IPAddress  string         `json:"ip_address"`
    UserAgent  string         `json:"user_agent"`
}

type AuditAction string
const (
    AuditActionCreate      AuditAction = "create"
    AuditActionRead        AuditAction = "read"
    AuditActionUpdate      AuditAction = "update"
    AuditActionDelete      AuditAction = "delete"
    AuditActionQuery       AuditAction = "query"
    AuditActionAuth        AuditAction = "auth"
    AuditActionExport      AuditAction = "export"
    AuditActionPolicyChange AuditAction = "policy_change"
)

type AuditOutcome string
const (
    AuditOutcomeSuccess AuditOutcome = "success"
    AuditOutcomeFailure AuditOutcome = "failure"
    AuditOutcomeDenied  AuditOutcome = "denied"
)

// Log records an audit event
func (a *AuditLogger) Log(ctx context.Context, event AuditEvent) error

// Query retrieves audit events with filters
func (a *AuditLogger) Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error)

// Retention policy: 90 days default, configurable per tenant
```

### 8.5 Vulnerability Scanning

```yaml
# .github/workflows/security.yml

name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 0 * * *'

jobs:
  gosec:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: securego/gosec@master
        with:
          args: ./...

  trivy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t vera:scan .
      - name: Run Trivy
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: vera:scan
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
      - name: Upload results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

  dependency-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/dependency-review-action@v3
```

---

## 9. Performance

### 9.1 Horizontal Scaling

```go
// pkg/scaling/coordinator.go

// Scaling targets
const (
    TargetP99Latency      = 1 * time.Second
    TargetThroughput      = 1000 // QPS
    TargetCPUUtilization  = 70   // percent
    TargetMemUtilization  = 80   // percent
)

// WorkerPool for parallel processing
type WorkerPool struct {
    workers   int
    queue     chan Task
    results   chan Result
    wg        sync.WaitGroup
}

func NewWorkerPool(workers int) *WorkerPool

func (p *WorkerPool) Submit(task Task) <-chan Result

// Sharding strategy for documents
type ShardingStrategy interface {
    GetShard(tenantID string) int
    GetAllShards() []int
}

// ConsistentHashing for even distribution
type ConsistentHashSharding struct {
    ring     *hashring.HashRing
    replicas int
}
```

### 9.2 Redis Caching

```go
// internal/cache/redis.go

type RedisCache struct {
    client *redis.Client
    prefix string
}

// Cache configuration
type CacheConfig struct {
    EmbeddingTTL    time.Duration `default:"24h"`
    QueryResultTTL  time.Duration `default:"1h"`
    DocumentMetaTTL time.Duration `default:"6h"`
    MaxMemory       string        `default:"512mb"`
}

// CacheKey generates namespaced keys
func CacheKey(prefix, category, id string) string {
    return fmt.Sprintf("%s:%s:%s", prefix, category, id)
}

// Cached embedding lookup
func (c *RedisCache) GetEmbedding(ctx context.Context, text string) ([]float32, bool, error) {
    key := CacheKey(c.prefix, "embedding", hashText(text))
    data, err := c.client.Get(ctx, key).Bytes()
    if err == redis.Nil {
        return nil, false, nil
    }
    if err != nil {
        return nil, false, err
    }

    var embedding []float32
    if err := msgpack.Unmarshal(data, &embedding); err != nil {
        return nil, false, err
    }

    return embedding, true, nil
}

// SetEmbedding caches an embedding
func (c *RedisCache) SetEmbedding(ctx context.Context, text string, embedding []float32) error {
    key := CacheKey(c.prefix, "embedding", hashText(text))
    data, err := msgpack.Marshal(embedding)
    if err != nil {
        return err
    }
    return c.client.Set(ctx, key, data, c.config.EmbeddingTTL).Err()
}

// Query result caching with invalidation
func (c *RedisCache) GetQueryResult(ctx context.Context, queryHash string) (*QueryResponse, bool, error)
func (c *RedisCache) SetQueryResult(ctx context.Context, queryHash string, result *QueryResponse) error
func (c *RedisCache) InvalidateDocument(ctx context.Context, docID string) error
```

### 9.3 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query Latency P50 | < 500ms | End-to-end |
| Query Latency P99 | < 1s | End-to-end |
| Ingestion Throughput | 100 pages/min | Per instance |
| Query Throughput | 1000 QPS | Cluster |
| Embedding Cache Hit Rate | > 80% | Per tenant |
| Vector Search Latency | < 50ms | Per query |
| Time to First Token (streaming) | < 200ms | From request |

---

## 10. Observability

### 10.1 Prometheus Metrics

```go
// internal/observability/metrics.go

import "github.com/prometheus/client_golang/prometheus"

var (
    QueryDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "vera_query_duration_seconds",
            Help:    "Query execution duration in seconds",
            Buckets: []float64{0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0},
        },
        []string{"tenant", "provider", "status"},
    )

    GroundingScore = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "vera_grounding_score",
            Help:    "Distribution of grounding scores",
            Buckets: []float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
        },
        []string{"tenant"},
    )

    TokensUsed = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "vera_tokens_total",
            Help: "Total tokens consumed",
        },
        []string{"tenant", "provider", "type"}, // type: input, output, embedding
    )

    ActiveQueries = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "vera_active_queries",
            Help: "Number of currently executing queries",
        },
        []string{"tenant"},
    )

    CacheHitRate = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "vera_cache_hit_rate",
            Help: "Cache hit rate",
        },
        []string{"cache_type"}, // embedding, query
    )

    VerificationFailures = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "vera_verification_failures_total",
            Help: "Total verification failures",
        },
        []string{"tenant", "reason"}, // reason: low_grounding, forbidden_topic, etc.
    )
)
```

### 10.2 Grafana Dashboards

```json
{
  "dashboard": {
    "title": "VERA Overview",
    "panels": [
      {
        "title": "Query Latency (P99)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(vera_query_duration_seconds_bucket[5m]))",
            "legendFormat": "{{tenant}}"
          }
        ]
      },
      {
        "title": "Grounding Score Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(vera_grounding_score_bucket[1h])"
          }
        ]
      },
      {
        "title": "Token Usage by Provider",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(vera_tokens_total[5m])) by (provider, type)",
            "legendFormat": "{{provider}} - {{type}}"
          }
        ]
      },
      {
        "title": "Active Queries",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(vera_active_queries)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(vera_query_duration_seconds_count{status='error'}[5m])) / sum(rate(vera_query_duration_seconds_count[5m]))"
          }
        ]
      }
    ]
  }
}
```

### 10.3 Alerting Rules

```yaml
# deploy/prometheus/alerts.yaml

groups:
  - name: vera
    rules:
      - alert: HighQueryLatency
        expr: histogram_quantile(0.99, rate(vera_query_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High query latency detected"
          description: "P99 query latency is {{ $value }}s (threshold: 1s)"

      - alert: LowGroundingScores
        expr: avg(vera_grounding_score) < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low average grounding scores"
          description: "Average grounding score is {{ $value }} (threshold: 0.7)"

      - alert: HighErrorRate
        expr: sum(rate(vera_query_duration_seconds_count{status="error"}[5m])) / sum(rate(vera_query_duration_seconds_count[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: ProviderRateLimited
        expr: increase(vera_verification_failures_total{reason="rate_limited"}[5m]) > 10
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "LLM provider rate limiting detected"
          description: "{{ $labels.provider }} is rate limiting requests"

      - alert: HighTokenUsage
        expr: sum(rate(vera_tokens_total[1h])) by (tenant) > 1000000
        for: 30m
        labels:
          severity: info
        annotations:
          summary: "High token usage for tenant"
          description: "Tenant {{ $labels.tenant }} using {{ $value }} tokens/hour"

      - alert: DatabaseConnectionPoolExhausted
        expr: pg_stat_activity_count{datname="vera"} / pg_settings_max_connections > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "{{ $value | humanizePercentage }} connections in use"

      - alert: CacheMissRateHigh
        expr: vera_cache_hit_rate{cache_type="embedding"} < 0.5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Embedding cache hit rate is {{ $value | humanizePercentage }}"
```

---

## 11. Quality Gates

### 11.1 Test Coverage Requirements

| Package | Minimum Coverage | Focus Areas |
|---------|-----------------|-------------|
| `pkg/core` | 90% | Result, Pipeline, errors |
| `pkg/llm` | 85% | Provider interface, registry, fallback |
| `pkg/verify` | 90% | Grounding, NLI, policy engine |
| `pkg/pipeline` | 90% | Operators, middleware |
| `pkg/api` | 80% | Handlers, auth, rate limiting |
| `pkg/tenant` | 85% | Quota, isolation |
| `internal/storage` | 85% | Repository, migrations |
| `cmd/vera-server` | 70% | Server startup, health |
| **Overall** | **90%** | |

### 11.2 Performance Regression Tests

```go
// tests/benchmarks/regression_test.go

func BenchmarkQueryE2E(b *testing.B) {
    // Setup real environment
    ctx := context.Background()
    pipeline := setupTestPipeline(b)
    query := testQuery()

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        result := pipeline.Execute(ctx, query)
        if !result.IsOk() {
            b.Fatal("Query failed:", result.Error())
        }
    }
}

// Performance baselines (fail if exceeded)
var baselines = map[string]time.Duration{
    "BenchmarkQueryE2E":         2 * time.Second,
    "BenchmarkEmbedding":        100 * time.Millisecond,
    "BenchmarkVectorSearch":     50 * time.Millisecond,
    "BenchmarkGroundingScore":   500 * time.Millisecond,
    "BenchmarkIngestion100Pages": 30 * time.Second,
}

func TestPerformanceRegression(t *testing.T) {
    for name, baseline := range baselines {
        t.Run(name, func(t *testing.T) {
            result := testing.Benchmark(func(b *testing.B) {
                // Run benchmark
            })
            if result.NsPerOp() > baseline.Nanoseconds() {
                t.Errorf("Performance regression: %s took %v (baseline: %v)",
                    name, time.Duration(result.NsPerOp()), baseline)
            }
        })
    }
}
```

### 11.3 Security Audit Checklist

| Category | Check | Status | ADR |
|----------|-------|--------|-----|
| **Authentication** | | | |
| | API key hashing (SHA-256) | REQUIRED | ADR-0020 |
| | OAuth 2.0 JWT validation | REQUIRED | ADR-0021 |
| | Token expiration handling | REQUIRED | |
| | Brute force protection | REQUIRED | |
| **Authorization** | | | |
| | RBAC enforcement on all endpoints | REQUIRED | ADR-0022 |
| | Tenant isolation verification | REQUIRED | |
| | Resource-level access control | REQUIRED | |
| **Data Protection** | | | |
| | TLS 1.2+ for all connections | REQUIRED | |
| | AES-256 encryption at rest | REQUIRED | ADR-0023 |
| | PII data handling | REQUIRED | |
| | Secure key management | REQUIRED | |
| **Input Validation** | | | |
| | SQL injection prevention | REQUIRED | |
| | XSS prevention | REQUIRED | |
| | File upload validation | REQUIRED | |
| | Rate limiting | REQUIRED | |
| **Audit & Compliance** | | | |
| | Audit logging | REQUIRED | |
| | Log retention policy | REQUIRED | |
| | GDPR data deletion | REQUIRED | |
| | SOC 2 controls | RECOMMENDED | |
| **Infrastructure** | | | |
| | Container security scanning | REQUIRED | |
| | Dependency vulnerability scanning | REQUIRED | |
| | Secret management | REQUIRED | |
| | Network segmentation | REQUIRED | |

---

## 12. ADR References (Production)

| ADR | Title | Status |
|-----|-------|--------|
| ADR-0020 | API Key Security Model | Proposed |
| ADR-0021 | OAuth 2.0 Integration | Proposed |
| ADR-0022 | RBAC Permission Model | Proposed |
| ADR-0023 | Encryption Strategy | Proposed |
| ADR-0024 | Multi-Provider Fallback | Proposed |
| ADR-0025 | PostgreSQL + pgvector Schema | Proposed |
| ADR-0026 | Redis Caching Strategy | Proposed |
| ADR-0027 | Multi-Tenant Isolation | Proposed |
| ADR-0028 | Policy DSL Design | Proposed |
| ADR-0029 | Kubernetes Deployment | Proposed |
| ADR-0030 | Monitoring and Alerting | Proposed |
| ADR-0031 | Backup and Recovery | Proposed |
| ADR-0032 | Cost Tracking Implementation | Proposed |
| ADR-0033 | WebSocket Protocol | Proposed |
| ADR-0034 | Billing Integration | Proposed |

---

## 13. Constitution Compliance (Production)

| Article | Description | MVP Compliance | Production Compliance | Evidence |
|---------|-------------|----------------|----------------------|----------|
| I | Verification as First-Class | YES | YES | Policy DSL extends verification |
| II | Composition Over Configuration | YES | YES | Provider registry, policy composition |
| III | Provider Agnosticism | YES | YES | Multi-provider registry + fallback |
| IV | Human Ownership | YES | YES | < 10 min per file, modular design |
| V | Type Safety | YES | YES | Generics throughout, typed policies |
| VI | Categorical Correctness | YES | YES | Law tests + property tests |
| VII | No Mocks in MVP | YES | N/A | Production uses real services |
| VIII | Graceful Degradation | YES | YES | Fallback chains, circuit breakers |
| IX | Observable by Default | YES | YES | Full Prometheus/Grafana/Jaeger |

---

## 14. Timeline and Milestones

### Phase 1: Multi-Provider (Weeks 1-2)

| Task | Days | Dependencies |
|------|------|--------------|
| Provider Registry implementation | 2 | - |
| OpenAI provider | 1 | Registry |
| Ollama provider | 1 | Registry |
| Fallback chain | 2 | All providers |
| Cost tracking | 2 | Registry |
| Integration tests | 2 | All above |

### Phase 2: REST API (Weeks 2-3)

| Task | Days | Dependencies |
|------|------|--------------|
| OpenAPI spec | 1 | - |
| Core handlers | 2 | - |
| API key auth | 2 | - |
| OAuth integration | 2 | - |
| Rate limiting | 1 | Auth |
| WebSocket streaming | 2 | Handlers |

### Phase 3: Persistence (Weeks 3-4)

| Task | Days | Dependencies |
|------|------|--------------|
| PostgreSQL schema | 1 | - |
| pgvector integration | 2 | Schema |
| Repository layer | 3 | Schema |
| Migrations | 1 | Repository |
| Connection pooling | 1 | Repository |
| Backup/restore | 2 | All above |

### Phase 4: Multi-Tenancy (Weeks 4-5)

| Task | Days | Dependencies |
|------|------|--------------|
| Tenant model | 1 | Persistence |
| RLS policies | 2 | Schema |
| Quota enforcement | 2 | Tenant model |
| Billing hooks | 2 | Quotas |
| Integration tests | 3 | All above |

### Phase 5: Policy Engine (Weeks 5-6)

| Task | Days | Dependencies |
|------|------|--------------|
| Policy DSL | 2 | - |
| Policy storage | 1 | Persistence |
| Custom NLI support | 3 | DSL |
| Policy versioning | 2 | Storage |
| Escalation rules | 2 | DSL |

### Phase 6: Deployment (Weeks 6-7)

| Task | Days | Dependencies |
|------|------|--------------|
| Docker optimization | 1 | - |
| Kubernetes manifests | 2 | Docker |
| Helm chart | 2 | K8s |
| Health checks | 1 | All |
| CI/CD pipeline | 2 | All |
| Documentation | 2 | All |

### Phase 7: Security & Observability (Weeks 7-8)

| Task | Days | Dependencies |
|------|------|--------------|
| Encryption implementation | 2 | - |
| RBAC | 2 | Auth |
| Audit logging | 2 | - |
| Prometheus metrics | 2 | - |
| Grafana dashboards | 2 | Metrics |
| Alerting rules | 1 | Metrics |
| Security audit | 3 | All |

---

**Document Status**: Draft - Pending Review
**Next Action**: MERCURIO + MARS review before Human Gate
**Quality Targets**:
- MERCURIO >= 8.5/10 across all planes
- MARS >= 92% confidence
- Test coverage >= 90%
- Security audit: All REQUIRED items passed

---

*Generated by: Specification-Driven Development Expert*
*Date: 2025-12-29*
*Input: synthesis.md (0.89), MVP-SPEC.md (1.0)*
*Extends: MVP-SPEC.md v1.0.0*
