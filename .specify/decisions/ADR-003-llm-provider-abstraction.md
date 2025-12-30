# ADR-003: LLM Provider Abstraction Interface

**Status**: Proposed
**Date**: 2025-12-29
**Context**: VERA Categorical Verification System - Provider Agnosticism

## Context

VERA must work with multiple LLM providers:
- **MVP**: Anthropic Claude (single provider)
- **Production**: Anthropic, OpenAI, Ollama, local models

Core business logic must NEVER contain provider-specific code. Switching providers should require zero changes to verification logic.

## Decision

**Define a minimal LLM interface that ALL providers implement identically.**

```go
// pkg/llm/provider.go

// LLMProvider is the ONLY LLM dependency in VERA core
type LLMProvider interface {
    // Complete generates a response for a prompt
    Complete(ctx context.Context, prompt Prompt) Result[Response]

    // Embed generates embeddings for text
    Embed(ctx context.Context, text string) Result[Embedding]

    // Stream generates a streaming response
    Stream(ctx context.Context, prompt Prompt) <-chan StreamChunk

    // Info returns provider metadata
    Info() ProviderInfo
}

// Prompt represents input to the LLM
type Prompt struct {
    System   string
    Messages []Message
    Options  PromptOptions
}

// Response represents LLM output
type Response struct {
    Content   string
    Usage     TokenUsage
    Model     string
    StopReason StopReason
}

// Embedding is a vector representation
type Embedding struct {
    Vector    []float32
    Model     string
    Dimension int
}
```

### Implementation Strategy

```go
// pkg/llm/anthropic/provider.go
type AnthropicProvider struct {
    client *anthropic.Client
}

func (p *AnthropicProvider) Complete(ctx context.Context, prompt Prompt) Result[Response] {
    // Translate Prompt → Anthropic API
    // Call anthropic-sdk-go
    // Translate Anthropic response → Response
    // Return Result[Response]
}

// pkg/llm/openai/provider.go
type OpenAIProvider struct {
    client *openai.Client
}

func (p *OpenAIProvider) Complete(ctx context.Context, prompt Prompt) Result[Response] {
    // Same interface, different implementation
}
```

## Consequences

### Positive
- **Zero business logic changes**: Swap provider by changing one line
- **Test isolation**: Can test verification without real LLM
- **Cost optimization**: Route to cheaper providers for simple tasks
- **Fallback support**: Chain providers for reliability
- **Clear boundary**: LLM code contained in pkg/llm/

### Negative
- **Lowest common denominator**: Can't use provider-specific features easily
- **Translation overhead**: Convert to/from internal types
- **Feature parity**: Must maintain equivalent behavior across providers

### Neutral
- Provider-specific extensions possible via optional interfaces
- Observability must work identically across providers

## Alternatives Considered

### Alternative 1: Direct SDK Usage
- **Approach**: Use anthropic-sdk-go directly in business logic
- **Pros**: Full SDK features, no translation overhead
- **Cons**: Vendor lock-in, can't swap providers, testing harder
- **Why rejected**: Violates Article III (Provider Agnosticism)

### Alternative 2: LangChain Go Port
- **Approach**: Use or create LangChain-style abstraction
- **Pros**: Industry standard, many providers supported
- **Cons**: Heavy dependency, over-abstracted, doesn't fit categorical model
- **Why rejected**: Doesn't align with composition-over-configuration

### Alternative 3: Per-Operation Interfaces
- **Approach**: Separate interfaces for Complete, Embed, Stream
- **Pros**: More flexible, providers can implement subset
- **Cons**: More complex, harder to swap providers atomically
- **Why rejected**: Single interface is simpler and sufficient

## Interface Extensions

For provider-specific features:

```go
// Optional interface for providers that support tool use
type ToolCapableProvider interface {
    LLMProvider
    CompleteWithTools(ctx context.Context, prompt Prompt, tools []Tool) Result[ToolResponse]
}

// Check capability at runtime
if toolProvider, ok := provider.(ToolCapableProvider); ok {
    return toolProvider.CompleteWithTools(ctx, prompt, tools)
}
```

## References

- Anthropic Go SDK: https://github.com/anthropics/anthropic-sdk-go
- OpenAI Go SDK: https://github.com/sashabaranov/go-openai
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md

## Constitution Compliance

- [x] Article I: Verification as First-Class - Interface returns Result[T]
- [x] Article II: Composition Over Configuration - Interface, not config
- [x] Article III: Provider Agnosticism - **This is the article**
- [x] Article IV: Human Ownership - Simple 4-method interface
- [x] Article V: Type Safety - Typed Prompt, Response, Embedding
- [x] Article VI: Categorical Correctness - N/A
- [x] Article VII: No Mocks in MVP - Real providers in MVP
- [x] Article VIII: Graceful Degradation - Result[T] for all operations
- [x] Article IX: Observable by Default - Providers emit traces
