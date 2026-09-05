# Architecture Document: LLM Manager

## 1. System Overview

LLM Manager is a Python library that provides a unified interface for multiple LLM providers. It implements a provider abstraction layer with failover, caching, rate limiting, and cost tracking capabilities.

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                         │
└─────────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLMManager Class                           │
│                    (llm_manager.py)                          │
└─────────────────────────┬─────────────────────────────────────┘
                          │
      ┌───────────────────┼───────────────────┐
      ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Caching   │    │ Cost        │    │ Rate Limit  │
│   Layer     │    │ Tracking    │    │ Handler     │
│ (caching.py)│    │(cost_track) │    │(rate_limit) │
└─────────────┘    └─────────────┘    └─────────────┘
      │                   │                   │
      └───────────────────┼───────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Provider Interface                           │
│              (Abstract Provider Class)                       │
└─────────────────────────┬─────────────────────────────────────┘
                          │
      ┌───────────────────┼───────────────────┐
      ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   OpenAI    │    │  Anthropic  │    │   Gemini    │
│  Provider   │    │   Provider  │    │  Provider   │
└─────────────┘    └─────────────┘    └─────────────┘
      │                   │                   │
      └───────────────────┼───────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 External LLM APIs                            │
│         (OpenAI, Google Gemini, Groq, Ollama, local)       │
└─────────────────────────────────────────────────────────────
```

## 3. Core Components

### 3.1 LLMManager (llm_manager.py)
- Main entry point for the library
- Provider selection and failover logic
- Request/response handling
- Configuration management

### 3.2 Caching (caching.py)
- Response caching based on request hash
- In-memory and file-based cache options
- TTL (time-to-live) support

### 3.3 Cost Tracking (cost_tracking.py)
- Token usage tracking per provider
- Cost calculation based on provider pricing
- Usage analytics and reporting

### 3.4 Rate Limiting (rate_limit.py)
- Token bucket algorithm
- Exponential backoff on rate limit errors
- Request queuing

### 3.5 Streaming (streaming.py)
- Real-time response streaming
- Event-based processing
- Chunk handling

### 3.6 System Prompts (system_prompts.py)
- Pre-built prompt templates
- Prompt management
- Version tracking

### 3.7 Batch Processing (batch_processing.py)
- Multiple request handling
- Parallel processing
- Result aggregation

## 4. Provider Interface

```python
class BaseProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict]) -> str:
        pass
    
    @abstractmethod
    def stream(self, messages: List[Dict]) -> Iterator[str]:
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        pass
    
    @abstractmethod
    def get_cost(self, tokens: int) -> float:
        pass
```

## 5. File Structure

```
llm-manager/
├── llm_manager.py           # Main class
├── caching.py              # Caching layer
├── cost_tracking.py        # Cost analytics
├── rate_limit.py           # Rate limiting
├── streaming.py            # Streaming support
├── system_prompts.py       # Prompt templates
├── batch_processing.py     # Batch processing
├── requirements.txt        # Dependencies
├── .env.example           # Configuration
├── specs/                 # Documentation
└── README.md
```

---

*Document Version: 1.0*  
*Created: 2026-03-17*
