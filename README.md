# LLM Manager

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python">
</p>

A resilient, production-ready wrapper for multiple LLM providers with automatic failover.

## Features

- 🔄 **Automatic Failover** - Primary → Secondary → Local
- 🌐 **Multi-Provider** - Gemini, OpenAI, Ollama, Groq support
- ⚡ **Async/Await** - Full async support
- 📝 **Comprehensive Logging** - Every failure logged
- 🛡️ **Safe Fallbacks** - Local cache + hardcoded responses
- ⚙️ **Environment-Driven** - .env configuration
- 🔁 **Retry Logic** - Exponential backoff on 5xx errors

## Quick Start

```bash
# Install dependencies
pip install python-dotenv aiohttp google-generativeai openai groq

# Copy config
cp .env.example .env
# Add your API keys

# Use in code
from llm_manager import LLMManager

manager = LLMManager()
await manager.initialize()

response = await manager.generate_text("Hello!")
print(response.text)
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| PRIMARY_PROVIDER | First LLM to try | gemini |
| SECONDARY_PROVIDER | Fallback LLM | groq |
| LOCAL_PROVIDER_URL | Ollama URL | http://localhost:11434 |
| FALLBACK_STRATEGY | Failover order | ordered |

## Architecture

```
LLMManager
├── GeminiProvider    (Google Gemini)
├── OpenAIProvider   (GPT-4/3.5)
├── OllamaProvider   (Local models)
├── GroqProvider     (Fast inference)
└── SafeFallbackCache
```

## License

MIT
