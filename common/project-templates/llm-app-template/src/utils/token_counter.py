"""
Token Counting and Cost Management for AI Forge

Provides accurate token counting, cost estimation, and usage tracking
for different LLM providers and models.
"""

import re
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import tiktoken

from .logger import get_logger


@dataclass
class TokenUsage:
    """Token usage information."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    model: str = ""
    provider: str = ""


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"


class TokenCounter:
    """Advanced token counter with provider-specific logic."""
    
    # Token costs per 1K tokens (USD)
    PRICING = {
        ModelProvider.OPENAI: {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "text-embedding-ada-002": {"input": 0.0001, "output": 0.0001},
            "text-embedding-3-large": {"input": 0.00013, "output": 0.00013},
        },
        ModelProvider.ANTHROPIC: {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        },
        ModelProvider.GOOGLE: {
            "gemini-pro": {"input": 0.00025, "output": 0.0005},
            "gemini-pro-vision": {"input": 0.00025, "output": 0.0005},
        }
    }
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._encoders = {}
        self._initialize_encoders()
    
    def _initialize_encoders(self):
        """Initialize tiktoken encoders for OpenAI models."""
        try:
            self._encoders["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
            self._encoders["gpt-3.5-turbo"] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self._encoders["text-embedding-ada-002"] = tiktoken.encoding_for_model("text-embedding-ada-002")
        except Exception as e:
            self.logger.warning(f"Failed to initialize some encoders: {e}")
    
    def count_tokens(
        self, 
        text: str, 
        model: str = "gpt-3.5-turbo",
        provider: str = ModelProvider.OPENAI
    ) -> int:
        """Count tokens in text for specific model."""
        
        if provider == ModelProvider.OPENAI:
            return self._count_openai_tokens(text, model)
        elif provider == ModelProvider.ANTHROPIC:
            return self._count_anthropic_tokens(text)
        elif provider == ModelProvider.GOOGLE:
            return self._count_google_tokens(text)
        else:
            # Fallback estimation
            return self._estimate_tokens(text)
    
    def _count_openai_tokens(self, text: str, model: str) -> int:
        """Count tokens using tiktoken for OpenAI models."""
        try:
            if model in self._encoders:
                encoder = self._encoders[model]
            else:
                # Try to get encoder for this model
                encoder = tiktoken.encoding_for_model(model)
                self._encoders[model] = encoder
            
            return len(encoder.encode(text))
        
        except Exception as e:
            self.logger.warning(f"Failed to count tokens with tiktoken: {e}")
            return self._estimate_tokens(text)
    
    def _count_anthropic_tokens(self, text: str) -> int:
        """Estimate tokens for Anthropic models."""
        # Anthropic uses different tokenization, approximate with 4 chars per token
        return max(1, len(text) // 4)
    
    def _count_google_tokens(self, text: str) -> int:
        """Estimate tokens for Google models."""
        # Google tokenization approximation
        return max(1, len(text.split()) + len(text) // 10)
    
    def _estimate_tokens(self, text: str) -> int:
        """Fallback token estimation."""
        # Conservative estimation: 1 token ≈ 3.5 characters
        return max(1, len(text) // 4)
    
    def count_messages_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        provider: str = ModelProvider.OPENAI
    ) -> int:
        """Count tokens in a list of messages."""
        
        if provider == ModelProvider.OPENAI:
            return self._count_openai_messages_tokens(messages, model)
        else:
            # For other providers, sum individual message tokens
            total_tokens = 0
            for message in messages:
                content = message.get("content", "")
                role = message.get("role", "")
                total_tokens += self.count_tokens(f"{role}: {content}", model, provider)
            return total_tokens
    
    def _count_openai_messages_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str
    ) -> int:
        """Count tokens for OpenAI chat completion format."""
        try:
            # OpenAI-specific token counting for chat format
            if model in self._encoders:
                encoder = self._encoders[model]
            else:
                encoder = tiktoken.encoding_for_model(model)
                self._encoders[model] = encoder
            
            tokens_per_message = 3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = 1     # If there's a name, the role is omitted
            
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoder.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            
            num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
            return num_tokens
        
        except Exception as e:
            self.logger.warning(f"Failed to count message tokens: {e}")
            # Fallback: sum individual message tokens
            return sum(self.count_tokens(msg.get("content", ""), model) for msg in messages)
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        provider: str = ModelProvider.OPENAI
    ) -> float:
        """Estimate cost for token usage."""
        
        provider_pricing = self.PRICING.get(provider, {})
        model_pricing = provider_pricing.get(model, {"input": 0.001, "output": 0.002})
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def create_usage_report(
        self,
        input_text: str,
        output_text: str,
        model: str,
        provider: str = ModelProvider.OPENAI
    ) -> TokenUsage:
        """Create comprehensive token usage report."""
        
        input_tokens = self.count_tokens(input_text, model, provider)
        output_tokens = self.count_tokens(output_text, model, provider)
        total_tokens = input_tokens + output_tokens
        estimated_cost = self.estimate_cost(input_tokens, output_tokens, model, provider)
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            model=model,
            provider=provider
        )


class UsageTracker:
    """Track and aggregate token usage over time."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.daily_usage: Dict[str, Dict[str, Any]] = {}
        self.session_usage: Dict[str, List[TokenUsage]] = {}
        self.total_usage = TokenUsage()
    
    def record_usage(
        self,
        usage: TokenUsage,
        session_id: str = "default",
        user_id: str = "anonymous"
    ) -> None:
        """Record token usage."""
        
        # Update session usage
        if session_id not in self.session_usage:
            self.session_usage[session_id] = []
        self.session_usage[session_id].append(usage)
        
        # Update daily usage
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in self.daily_usage:
            self.daily_usage[today] = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "by_model": {},
                "by_user": {}
            }
        
        daily = self.daily_usage[today]
        daily["total_tokens"] += usage.total_tokens
        daily["total_cost"] += usage.estimated_cost
        
        # By model
        model_key = f"{usage.provider}:{usage.model}"
        if model_key not in daily["by_model"]:
            daily["by_model"][model_key] = {"tokens": 0, "cost": 0.0}
        daily["by_model"][model_key]["tokens"] += usage.total_tokens
        daily["by_model"][model_key]["cost"] += usage.estimated_cost
        
        # By user
        if user_id not in daily["by_user"]:
            daily["by_user"][user_id] = {"tokens": 0, "cost": 0.0}
        daily["by_user"][user_id]["tokens"] += usage.total_tokens
        daily["by_user"][user_id]["cost"] += usage.estimated_cost
        
        # Update total usage
        self.total_usage.input_tokens += usage.input_tokens
        self.total_usage.output_tokens += usage.output_tokens
        self.total_usage.total_tokens += usage.total_tokens
        self.total_usage.estimated_cost += usage.estimated_cost
    
    def get_session_usage(self, session_id: str) -> List[TokenUsage]:
        """Get usage for a specific session."""
        return self.session_usage.get(session_id, [])
    
    def get_daily_usage(self, date: str = None) -> Dict[str, Any]:
        """Get usage for a specific date."""
        if date is None:
            from datetime import datetime
            date = datetime.now().strftime("%Y-%m-%d")
        
        return self.daily_usage.get(date, {})
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get overall usage summary."""
        return {
            "total_usage": self.total_usage,
            "sessions": len(self.session_usage),
            "days_tracked": len(self.daily_usage),
            "average_daily_cost": (
                sum(day["total_cost"] for day in self.daily_usage.values()) /
                max(len(self.daily_usage), 1)
            )
        }
    
    def check_budget_alert(
        self,
        daily_budget: float = 100.0,
        monthly_budget: float = 1000.0
    ) -> List[str]:
        """Check for budget alerts."""
        alerts = []
        
        # Check daily budget
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        daily_usage = self.daily_usage.get(today, {})
        daily_cost = daily_usage.get("total_cost", 0.0)
        
        if daily_cost > daily_budget * 0.8:
            alerts.append(f"Daily budget alert: ${daily_cost:.2f} / ${daily_budget:.2f}")
        
        # Check monthly budget
        current_month = datetime.now().strftime("%Y-%m")
        monthly_cost = sum(
            day["total_cost"]
            for date, day in self.daily_usage.items()
            if date.startswith(current_month)
        )
        
        if monthly_cost > monthly_budget * 0.8:
            alerts.append(f"Monthly budget alert: ${monthly_cost:.2f} / ${monthly_budget:.2f}")
        
        return alerts


# Global instances
_token_counter = None
_usage_tracker = None


def get_token_counter() -> TokenCounter:
    """Get global token counter instance."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def get_usage_tracker() -> UsageTracker:
    """Get global usage tracker instance."""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker
