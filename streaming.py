"""
Streaming Support for LLM Manager
==================================
Adds real-time streaming responses
"""

import asyncio
from typing import AsyncGenerator, Optional
from dataclasses import dataclass


@dataclass
class StreamChunk:
    """Streaming response chunk"""
    text: str
    provider: str
    model: str
    is_final: bool = False
    delta: str = ""


class StreamingMixin:
    """Mixin for streaming responses"""
    
    async def generate_stream(
        self, 
        prompt: str, 
        provider_name: str = "primary",
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate streaming response
        
        Usage:
            async for chunk in manager.generate_stream("Tell me a story"):
                print(chunk.delta, end="")
        """
        
        # Get the appropriate provider
        provider = self._get_provider(provider_name)
        
        if not provider:
            yield StreamChunk(
                text="",
                provider="fallback",
                model="safe-mode",
                is_final=True,
                delta="Streaming not available. Use generate_text() instead."
            )
            return
        
        # Stream based on provider
        if hasattr(provider, 'generate_stream'):
            async for chunk in provider.generate_stream(prompt, **kwargs):
                yield chunk
        else:
            # Fallback to regular generation
            response = await provider.generate(prompt, **kwargs)
            yield StreamChunk(
                text=response.text,
                provider=response.provider,
                model=response.model,
                is_final=True,
                delta=response.text
            )
    
    def _get_provider(self, name: str):
        """Get provider by name"""
        name = name.lower()
        
        if name == "primary" or name == self.config.get("primary_provider"):
            return self.primary
        elif name == "secondary" or name == self.config.get("secondary_provider"):
            return self.secondary
        elif name in ["local", "ollama"]:
            return self.local
        
        return self.primary


# Add streaming to Ollama provider
async def ollama_stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[StreamChunk, None]:
    """Generate streaming response from Ollama"""
    import aiohttp
    
    url = f"{self.base_url}/api/generate"
    
    payload = {
        "model": kwargs.get('model', self.config.model),
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": kwargs.get('temperature', self.config.temperature),
            "num_predict": kwargs.get('max_tokens', self.config.max_tokens)
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=self.config.timeout) as resp:
            if resp.status != 200:
                yield StreamChunk(
                    text="",
                    provider=self.config.name,
                    model=self.config.model,
                    is_final=True,
                    delta=f"Error: {resp.status}"
                )
                return
            
            async for line in resp.content:
                if line:
                    try:
                        data = json.loads(line)
                        text = data.get('response', '')
                        yield StreamChunk(
                            text=text,
                            provider=self.config.name,
                            model=self.config.model,
                            is_final=data.get('done', False),
                            delta=text
                        )
                    except:
                        continue


import json  # Added for JSON parsing in streaming


# Usage example for streaming
async def example_streaming():
    """Example of using streaming"""
    
    # Initialize
    manager = LLMManager()
    await manager.initialize()
    
    print("Streaming response:\n")
    
    full_response = ""
    
    # Stream the response
    async for chunk in manager.generate_stream(
        "Write a short poem about AI:",
        prompt_type="generate"
    ):
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
            full_response += chunk.delta
        
        if chunk.is_final:
            print(f"\n\n✓ Complete! Provider: {chunk.provider}")
    
    return full_response


if __name__ == "__main__":
    asyncio.run(example_streaming())
