# LLM Manager

A resilient multi-provider LLM wrapper with automatic failover, rate limiting, cost tracking, and streaming support.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)

## Features

- 🔄 **Multi-Provider Support** - OpenAI, Anthropic, Google Gemini, and more
- 🛡️ **Automatic Failover** - Automatically switch providers on failure
- 💰 **Cost Tracking** - Track usage and costs across providers
- ⚡ **Rate Limiting** - Built-in rate limit handling
- 📡 **Streaming Support** - Real-time response streaming
- 📝 **System Prompts** - Pre-built system prompts library

## Installation

```bash
pip install llm-manager
```

Or install from source:

```bash
git clone https://github.com/sagar0163/llm-manager.git
cd llm-manager
pip install -r requirements.txt
```

## Quick Start

```python
from llm_manager import LLMManager

# Initialize with multiple providers
manager = LLMManager(
    providers=["openai", "anthropic", "gemini"],
    default_provider="openai"
)

# Make a request (auto-failover on failure)
response = manager.chat("Hello! How are you?")
print(response)
```

## Configuration

See [.env.example](.env.example) for environment variables.

```bash
# Copy and configure
cp .env.example .env
# Edit .env with your API keys
```

## Modules

| Module | Description |
|--------|-------------|
| `llm_manager.py` | Main LLM wrapper |
| `caching.py` | Response caching |
| `cost_tracking.py` | Cost analytics |
| `rate_limit.py` | Rate limit handling |
| `streaming.py` | Streaming responses |
| `batch_processing.py` | Batch requests |
| `system_prompts.py` | Pre-built prompts |

## Documentation

See [API Documentation](API.md) for detailed usage.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE)

---

⭐ Star this repo if you find it useful!
# Update
