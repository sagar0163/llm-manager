# llm-manager

> **Resilient multi-provider LLM wrapper with automatic failover, load balancing, and cost optimization**

[![CI](https://github.com/sagar0163/llm-manager/workflows/CI/badge.svg)](https://github.com/sagar0163/llm-manager/actions/workflows/ci.yml)
[![Release](https://github.com/sagar0163/llm-manager/workflows/Release/badge.svg)](https://github.com/sagar0163/llm-manager/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org/)

---

## 🎯 Problem

Production LLM applications need reliability: provider outages, rate limits, cost spikes, and model deprecations break apps. Writing retry/fallback logic for every provider is error-prone.

## 💡 Solution

A **production-ready LLM gateway** that handles:

- **Automatic failover** — seamless switch on 429, 5xx, timeout, circuit open
- **Load balancing** — round-robin, weighted, latency-aware, cost-aware
- **Cost optimization** — route to cheapest capable model, budget enforcement
- **Unified interface** — one API for OpenAI, Groq, Ollama, local

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLM Manager                               │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Provider    │  Router      │  Circuit     │  Metrics           │
│  Registry    │  (strategies)│  Breaker     │  (Prometheus)      │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

## 🚀 Quick Start

```bash
pip install llm-manager
```

### Local-Only (No API Keys Required)

Works out-of-the-box with local Ollama! If no API keys are provided, it automatically routes to your local Ollama instance (defaults to http://localhost:11434).

```python
import asyncio
from llm_manager import LLMManager

async def main():
    manager = LLMManager()
    await manager.initialize()
    
    response = await manager.generate_text("Explain quantum computing")
    print(response.text)

asyncio.run(main())
```

### Multi-Provider Production Setup

```python
from llm_manager import LLMManager, ProviderConfig

manager = LLMManager([
    ProviderConfig(
        name="openai",
        provider="openai",
        models=["gpt-4o", "gpt-4o-mini"],
        api_key="sk-...",
        weight=0.5,
        cost_per_1k={"input": 0.005, "output": 0.015}
    ),
    ProviderConfig(
        name="anthropic",
        provider="anthropic",
        models=["claude-3-5-sonnet", "claude-3-5-haiku"],
        api_key="sk-ant-...",
        weight=0.3,
        cost_per_1k={"input": 0.003, "output": 0.015}
    ),
    ProviderConfig(
        name="nvidia-free",
        provider="nvidia",
        models=["nvidia/nemotron-3-ultra-550b-a55b"],
        api_key="nvapi-...",
        weight=0.2,
        cost_per_1k={"input": 0.0, "output": 0.0}  # Free tier
    ),
])

# Use it — automatic failover built in
response = await manager.chat.completions.create(
    model="auto",  # Let router pick best
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=500,
    budget_usd=0.01  # Optional cost cap
)
```

## ⚙️ Configuration

```yaml
# llm_manager.yaml
router:
  strategy: "cost_aware"  # round_robin, weighted, latency_aware, cost_aware
  fallback_order: ["nvidia-free", "openai", "anthropic"]

circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 60
  half_open_requests: 3

retry:
  max_attempts: 3
  base_delay: 1.0
  max_delay: 30.0
  exponential_base: 2

budget:
  daily_usd: 10.0
  monthly_usd: 200.0
  alert_at_percent: 80

providers:
  openai:
    base_url: "https://api.openai.com/v1"
    timeout: 30
  anthropic:
    base_url: "https://api.anthropic.com"
    timeout: 30
  nvidia:
    base_url: "https://integrate.api.nvidia.com/v1"
    timeout: 60
```

## 🎯 Routing Strategies

| Strategy | Behavior |
|---|---|
| `round_robin` | Even distribution across healthy providers |
| `weighted` | Distribute by configured weights |
| `latency_aware` | Route to fastest healthy provider |
| `cost_aware` | Route to cheapest capable model (default) |
| `priority` | Try in order until success |

## 📊 Metrics (Prometheus)

```python
# Exposes /metrics endpoint
from llm_manager import metrics

metrics.requests_total        # Total requests by provider/model/outcome
metrics.request_duration      # Latency histograms
metrics.tokens_used           # Input/output tokens by model
metrics.cost_usd              # Cumulative cost
metrics.circuit_state         # Circuit breaker states
metrics.fallback_total        # Failover events
```

## 🔌 Custom Providers

```python
from llm_manager import BaseProvider

class MyCustomProvider(BaseProvider):
    async def chat_completion(self, request):
        # Your implementation
        pass
    
    async def health_check(self):
        # Return True/False
        pass

manager.register_provider("my-custom", MyCustomProvider())
```

## 🧪 Testing

```bash
pytest tests/ -v
pytest tests/ --cov=llm_manager
```

## 📦 Release

```bash
poetry version patch
git push origin main --tags
# GitHub Actions: test → build → release → PyPI
```

## 📄 License

MIT License

---

**Build resilient AI apps that never go down**