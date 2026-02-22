"""
Cost Tracking for LLM Manager
==============================
Track API usage and costs across providers
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import json


# Pricing per 1M tokens (approximate)
PRICING = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "llama-3.1-70b-versatile": {"input": 0.00, "output": 0.00},  # Groq free
    "llama3": {"input": 0.00, "output": 0.00},  # Ollama free
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}


@dataclass
class UsageRecord:
    """Single usage record"""
    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: Optional[float] = None


@dataclass
class CostSummary:
    """Cost summary for a period"""
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    by_provider: Dict[str, float]
    by_model: Dict[str, float]


class CostTracker:
    """
    Track and analyze LLM usage costs
    
    Usage:
        tracker = CostTracker()
        
        # After each request
        tracker.record(
            provider="gemini",
            model="gemini-1.5-flash",
            input_tokens=100,
            output_tokens=500
        )
        
        # Get summary
        summary = tracker.get_summary()
        print(f"Total cost: ${summary.total_cost:.4f}")
    """
    
    def __init__(self, storage_path: str = ".llm_costs.json"):
        storage_path
        self.storage_path = self.records: List[UsageRecord] = []
        self._load()
    
    def _load(self):
        """Load records from disk"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.records = [
                    UsageRecord(
                        timestamp=datetime.fromisoformat(r['timestamp']),
                        provider=r['provider'],
                        model=r['model'],
                        input_tokens=r['input_tokens'],
                        output_tokens=r['output_tokens'],
                        cost=r['cost'],
                        latency_ms=r.get('latency_ms')
                    )
                    for r in data
                ]
        except FileNotFoundError:
            self.records = []
        except Exception:
            self.records = []
    
    def _save(self):
        """Save records to disk"""
        data = [
            {
                'timestamp': r.timestamp.isoformat(),
                'provider': r.provider,
                'model': r.model,
                'input_tokens': r.input_tokens,
                'output_tokens': r.output_tokens,
                'cost': r.cost,
                'latency_ms': r.latency_ms
            }
            for r in self.records
        ]
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: Optional[float] = None
    ):
        """Record a usage"""
        
        # Calculate cost
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency_ms=latency_ms
        )
        
        self.records.append(record)
        self._save()
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model pricing"""
        
        # Get pricing (default to free if unknown)
        pricing = PRICING.get(model, {"input": 0, "output": 0})
        
        input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
        
        return input_cost + output_cost
    
    def get_summary(self, since: Optional[datetime] = None) -> CostSummary:
        """Get cost summary"""
        
        # Filter records
        records = self.records
        if since:
            records = [r for r in records if r.timestamp >= since]
        
        if not records:
            return CostSummary(
                total_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost=0.0,
                by_provider={},
                by_model={}
            )
        
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_cost = sum(r.cost for r in records)
        
        # Group by provider
        by_provider = defaultdict(float)
        for r in records:
            by_provider[r.provider] += r.cost
        
        # Group by model
        by_model = defaultdict(float)
        for r in records:
            by_model[r.model] += r.cost
        
        return CostSummary(
            total_requests=len(records),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost=total_cost,
            by_provider=dict(by_provider),
            by_model=dict(by_model)
        )
    
    def get_daily_costs(self) -> Dict[str, float]:
        """Get costs by day"""
        daily = defaultdict(float)
        
        for r in self.records:
            day = r.timestamp.strftime("%Y-%m-%d")
            daily[day] += r.cost
        
        return dict(daily)
    
    def reset(self):
        """Reset all records"""
        self.records = []
        self._save()
    
    def export_csv(self, filepath: str):
        """Export to CSV"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'provider', 'model', 'input_tokens', 
                'output_tokens', 'cost', 'latency_ms'
            ])
            
            for r in self.records:
                writer.writerow([
                    r.timestamp.isoformat(),
                    r.provider,
                    r.model,
                    r.input_tokens,
                    r.output_tokens,
                    r.cost,
                    r.latency_ms or ''
                ])


# Decorator for automatic cost tracking
def track_cost(tracker: CostTracker):
    """Decorator to automatically track costs"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Extract info from result
            if hasattr(result, 'provider') and hasattr(result, 'tokens_used'):
                tracker.record(
                    provider=result.provider,
                    model=result.model,
                    input_tokens=kwargs.get('input_tokens', 0),
                    output_tokens=result.tokens_used or 0
                )
            
            return result
        return wrapper
    return decorator


# Usage with LLMManager
async def example_with_cost_tracking():
    """Example of using cost tracking"""
    
    tracker = CostTracker()
    
    # Initialize manager
    manager = LLMManager()
    await manager.initialize()
    
    # Generate (cost will be tracked manually)
    response = await manager.generate_text("Hello!")
    
    # Record the cost
    tracker.record(
        provider=response.provider,
        model=response.model,
        input_tokens=10,  # Approximate
        output_tokens=len(response.text.split()) * 1.3  # Rough estimate
    )
    
    # Get summary
    summary = tracker.get_summary()
    print(f"Total requests: {summary.total_requests}")
    print(f"Total cost: ${summary.total_cost:.4f}")
    print(f"By provider: {summary.by_provider}")


if __name__ == "__main__":
    tracker = CostTracker()
    summary = tracker.get_summary()
    print(f"Total cost: ${summary.total_cost:.4f}")
