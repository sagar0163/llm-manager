# LLM Manager - Advanced Features

## Provider Configuration

### OpenAI

```python
config = {
    'provider': 'openai',
    'api_key': os.getenv('OPENAI_API_KEY'),
    'models': ['gpt-4', 'gpt-3.5-turbo'],
    'temperature': 0.7
}
```

### Anthropic

```python
config = {
    'provider': 'anthropic',
    'api_key': os.getenv('ANTHROPIC_API_KEY'),
    'models': ['claude-3-opus', 'claude-3-sonnet']
}
```

## Cost Optimization

### Token Tracking

```python
tracker = CostTracker()
tracker.track_usage(prompt_tokens, completion_tokens)
print(tracker.get_total_cost())
```

### Caching Strategies

| Strategy | Use Case |
|----------|----------|
| No Cache | Real-time |
| Memory | Short TTL |
| Redis | Distributed |

## Error Handling

### Retry Logic

```python
manager = LLMManager(
    max_retries=3,
    backoff_factor=2,
    timeout=30
)
```

## Streaming

```python
for chunk in manager.stream("Prompt"):
    print(chunk, end="")
```
