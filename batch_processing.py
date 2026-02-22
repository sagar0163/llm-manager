"""
Batch Processing for LLM Manager
================================
Process multiple prompts efficiently
"""

import asyncio
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json


@dataclass
class BatchResult:
    """Result of batch processing"""
    index: int
    prompt: str
    response: str
    success: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class BatchProcessor:
    """
    Efficient batch processing for LLM prompts
    
    Features:
    - Parallel processing
    - Rate limiting
    - Error handling
    - Progress tracking
    
    Usage:
        processor = BatchProcessor()
        
        prompts = [
            "What is AI?",
            "What is ML?",
            "What is DL?"
        ]
        
        results = await processor.process(prompts)
        
        for result in results:
            print(result.response)
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,
        retry_failed: bool = True,
        max_retries: int = 2
    ):
        self.max_concurrent = max_concurrent
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process(
        self,
        prompts: List[str],
        process_func: Callable,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> List[BatchResult]:
        """
        Process multiple prompts
        
        Args:
            prompts: List of prompts to process
            process_func: Async function to call for each prompt
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments to pass to process_func
        
        Returns:
            List of BatchResult objects
        """
        results = []
        
        async def process_with_semaphore(index: int, prompt: str):
            async with self.semaphore:
                result = await self._process_single(
                    index, prompt, process_func, **kwargs
                )
                
                if progress_callback:
                    progress_callback(index + 1, len(prompts))
                
                return result
        
        # Create tasks
        tasks = [
            process_with_semaphore(i, prompt)
            for i, prompt in enumerate(prompts)
        ]
        
        # Execute all
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(BatchResult(
                    index=i,
                    prompt=prompts[i],
                    response="",
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        # Retry failed if enabled
        if self.retry_failed:
            final_results = await self._retry_failed(
                final_results, prompts, process_func, **kwargs
            )
        
        return final_results
    
    async def _process_single(
        self,
        index: int,
        prompt: str,
        process_func: Callable,
        **kwargs
    ) -> BatchResult:
        """Process a single prompt with timing"""
        import time
        start = time.time()
        
        try:
            response = await process_func(prompt, **kwargs)
            latency = (time.time() - start) * 1000
            
            return BatchResult(
                index=index,
                prompt=prompt,
                response=response,
                success=True,
                latency_ms=latency
            )
            
        except Exception as e:
            latency = (time.time() - start) * 1000
            
            return BatchResult(
                index=index,
                prompt=prompt,
                response="",
                success=False,
                error=str(e),
                latency_ms=latency
            )
    
    async def _retry_failed(
        self,
        results: List[BatchResult],
        prompts: List[str],
        process_func: Callable,
        **kwargs
    ) -> List[BatchResult]:
        """Retry failed prompts"""
        failed_indices = [
            i for i, r in enumerate(results)
            if not r.success
        ]
        
        if not failed_indices:
            return results
        
        # Retry each failed
        for idx in failed_indices:
            for retry in range(self.max_retries):
                try:
                    result = await self._process_single(
                        idx, prompts[idx], process_func, **kwargs
                    )
                    results[idx] = result
                    
                    if result.success:
                        break
                        
                except Exception:
                    pass
        
        return results


class PromptProcessor:
    """
    Specialized processor for common prompt patterns
    """
    
    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager
    
    async def summarize_batch(
        self,
        texts: List[str],
        max_concurrent: int = 3
    ) -> List[str]:
        """Summarize multiple texts"""
        
        def make_summary_prompt(text: str) -> str:
            return f"Summarize this in 2 sentences: {text}"
        
        processor = BatchProcessor(max_concurrent=max_concurrent)
        
        prompts = [make_summary_prompt(t) for t in texts]
        
        # Use LLM manager if available
        if self.llm_manager:
            async def process(prompt):
                result = await self.llm_manager.generate_text(prompt)
                return result.text
            
            results = await processor.process(prompts, process)
        
        return [r.response for r in results]
    
    async def translate_batch(
        self,
        texts: List[str],
        target_language: str,
        max_concurrent: int = 3
    ) -> List[str]:
        """Translate multiple texts"""
        
        def make_translate_prompt(text: str) -> str:
            return f"Translate to {target_language}: {text}"
        
        processor = BatchProcessor(max_concurrent=max_concurrent)
        
        prompts = [make_translate_prompt(t) for t in texts]
        
        if self.llm_manager:
            async def process(prompt):
                result = await self.llm_manager.generate_text(prompt)
                return result.text
            
            results = await processor.process(prompts, process)
        
        return [r.response for r in results]
    
    async def extract_batch(
        self,
        texts: List[str],
        extract_field: str,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Extract structured data from multiple texts"""
        
        def make_extract_prompt(text: str) -> str:
            return f"""Extract {extract_field} from this text. Return as JSON:
{text}

JSON:"""
        
        processor = BatchProcessor(max_concurrent=max_concurrent)
        
        prompts = [make_extract_prompt(t) for t in texts]
        
        results = []
        
        if self.llm_manager:
            async def process(prompt):
                result = await self.llm_manager.generate_text(prompt)
                try:
                    # Try to parse JSON
                    import json
                    return json.loads(result.text)
                except:
                    return {"error": "Failed to parse"}
            
            batch_results = await processor.process(prompts, process)
            results = [r.response for r in batch_results]
        
        return results


# Example usage
async def example_batch_processing():
    """Example of batch processing"""
    
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is React?",
        "What is Vue.js?",
        "What is Angular?"
    ]
    
    processor = BatchProcessor(max_concurrent=2)
    
    # Simulated process function
    async def mock_process(prompt):
        await asyncio.sleep(0.5)  # Simulate API call
        return f"Answer to: {prompt}"
    
    print(f"Processing {len(prompts)} prompts with max 2 concurrent...")
    
    results = await processor.process(
        prompts,
        mock_process,
        progress_callback=lambda current, total: print(f"Progress: {current}/{total}")
    )
    
    print("\n📊 Results:")
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} {result.prompt[:30]}...")
        if result.error:
            print(f"   Error: {result.error}")
    
    # Stats
    successful = sum(1 for r in results if r.success)
    print(f"\n📈 Success rate: {successful}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(example_batch_processing())
