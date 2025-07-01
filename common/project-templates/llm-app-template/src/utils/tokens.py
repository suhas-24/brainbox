"""
Token Counting and Cost Estimation for AI Forge

Provides accurate token counting for various LLM providers and models,
cost estimation, and usage tracking with multiple encoding strategies.
"""

import re
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import math

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .logging import get_logger


logger = get_logger(__name__)


class ModelProvider(Enum):
    """LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


@dataclass
class TokenCount:
    """Token count result."""
    prompt_tokens: int
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class CostEstimate:
    """Cost estimation result."""
    prompt_cost: float
    completion_cost: float
    total_cost: float
    currency: str = "USD"
    model: str = ""
    provider: str = ""


@dataclass
class ModelPricing:
    """Model pricing information."""
    prompt_cost_per_1k: float  # Cost per 1K prompt tokens
    completion_cost_per_1k: float  # Cost per 1K completion tokens
    currency: str = "USD"
    max_tokens: int = 4096
    context_window: int = 4096


class TokenCounter:
    """Advanced token counter with multiple encoding strategies."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Model encoding mappings
        self.model_encodings = {
            # OpenAI models
            "gpt-4": "cl100k_base",
            "gpt-4-32k": "cl100k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-4-vision": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-3.5-turbo-16k": "cl100k_base",
            "text-davinci-003": "p50k_base",
            "text-davinci-002": "p50k_base",
            "code-davinci-002": "p50k_base",
            
            # Anthropic models (approximate with GPT-4 encoding)
            "claude-3-opus": "cl100k_base",
            "claude-3-sonnet": "cl100k_base",
            "claude-3-haiku": "cl100k_base",
            "claude-2": "cl100k_base",
            "claude-instant": "cl100k_base",
        }
        
        # Pricing information (updated as of 2024)
        self.model_pricing = {
            # OpenAI GPT-4 models
            "gpt-4": ModelPricing(
                prompt_cost_per_1k=0.03,
                completion_cost_per_1k=0.06,
                max_tokens=8192,
                context_window=8192
            ),
            "gpt-4-32k": ModelPricing(
                prompt_cost_per_1k=0.06,
                completion_cost_per_1k=0.12,
                max_tokens=32768,
                context_window=32768
            ),
            "gpt-4-turbo": ModelPricing(
                prompt_cost_per_1k=0.01,
                completion_cost_per_1k=0.03,
                max_tokens=4096,
                context_window=128000
            ),
            "gpt-4-vision": ModelPricing(
                prompt_cost_per_1k=0.01,
                completion_cost_per_1k=0.03,
                max_tokens=4096,
                context_window=128000
            ),
            
            # OpenAI GPT-3.5 models
            "gpt-3.5-turbo": ModelPricing(
                prompt_cost_per_1k=0.0015,
                completion_cost_per_1k=0.002,
                max_tokens=4096,
                context_window=16385
            ),
            "gpt-3.5-turbo-16k": ModelPricing(
                prompt_cost_per_1k=0.003,
                completion_cost_per_1k=0.004,
                max_tokens=4096,
                context_window=16385
            ),
            
            # Anthropic Claude models
            "claude-3-opus": ModelPricing(
                prompt_cost_per_1k=0.015,
                completion_cost_per_1k=0.075,
                max_tokens=4096,
                context_window=200000
            ),
            "claude-3-sonnet": ModelPricing(
                prompt_cost_per_1k=0.003,
                completion_cost_per_1k=0.015,
                max_tokens=4096,
                context_window=200000
            ),
            "claude-3-haiku": ModelPricing(
                prompt_cost_per_1k=0.00025,
                completion_cost_per_1k=0.00125,
                max_tokens=4096,
                context_window=200000
            ),
            "claude-2": ModelPricing(
                prompt_cost_per_1k=0.008,
                completion_cost_per_1k=0.024,
                max_tokens=4096,
                context_window=100000
            ),
            
            # Google models (approximate pricing)
            "gemini-pro": ModelPricing(
                prompt_cost_per_1k=0.001,
                completion_cost_per_1k=0.002,
                max_tokens=2048,
                context_window=32768
            ),
            "gemini-pro-vision": ModelPricing(
                prompt_cost_per_1k=0.001,
                completion_cost_per_1k=0.002,
                max_tokens=2048,
                context_window=32768
            ),
        }
        
        # Cache for tokenizers
        self._tokenizer_cache = {}
    
    def _get_tiktoken_encoding(self, model: str) -> Optional[Any]:
        """Get tiktoken encoding for a model."""
        if not TIKTOKEN_AVAILABLE:
            return None
        
        try:
            if model in self.model_encodings:
                encoding_name = self.model_encodings[model]
                return tiktoken.get_encoding(encoding_name)
            else:
                # Try to get encoding for the model directly
                return tiktoken.encoding_for_model(model)
        except Exception as e:
            self.logger.warning(f"Failed to get tiktoken encoding for {model}: {e}")
            return None
    
    def _get_transformers_tokenizer(self, model: str) -> Optional[Any]:
        """Get HuggingFace tokenizer for a model."""
        if not TRANSFORMERS_AVAILABLE:
            return None
        
        if model in self._tokenizer_cache:
            return self._tokenizer_cache[model]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            self._tokenizer_cache[model] = tokenizer
            return tokenizer
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer for {model}: {e}")
            return None
    
    def _estimate_tokens_simple(self, text: str) -> int:
        """Simple token estimation based on word count."""
        if not text:
            return 0
        
        # Rough approximation: 1 token ≈ 0.75 words for English
        words = len(text.split())
        tokens = int(words / 0.75)
        
        # Add some tokens for punctuation and special characters
        special_chars = len(re.findall(r'[^\w\s]', text))
        tokens += int(special_chars * 0.5)
        
        return max(1, tokens)
    
    def count_tokens(
        self,
        text: Union[str, List[Dict[str, str]]],
        model: str = "gpt-3.5-turbo",
        provider: Optional[ModelProvider] = None
    ) -> int:
        """
        Count tokens in text for a specific model.
        
        Args:
            text: Text string or list of chat messages
            model: Model name
            provider: LLM provider
            
        Returns:
            Number of tokens
        """
        try:
            # Handle different input formats
            if isinstance(text, list):
                # Chat messages format
                full_text = ""
                for message in text:
                    role = message.get("role", "")
                    content = message.get("content", "")
                    full_text += f"{role}: {content}\n"
                text = full_text.strip()
            
            if not text:
                return 0
            
            # Try tiktoken first (most accurate for OpenAI models)
            encoding = self._get_tiktoken_encoding(model)
            if encoding:
                return len(encoding.encode(text))
            
            # Try transformers for HuggingFace models
            if provider == ModelProvider.HUGGINGFACE:
                tokenizer = self._get_transformers_tokenizer(model)
                if tokenizer:
                    tokens = tokenizer.encode(text)
                    return len(tokens)
            
            # Fallback to simple estimation
            return self._estimate_tokens_simple(text)
            
        except Exception as e:
            self.logger.error(f"Token counting failed for model {model}: {e}")
            return self._estimate_tokens_simple(text if isinstance(text, str) else str(text))
    
    def count_message_tokens(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo"
    ) -> int:
        """
        Count tokens for chat completion messages.
        Includes overhead for message formatting.
        """
        try:
            # Get base token count
            total_tokens = self.count_tokens(messages, model)
            
            # Add overhead for message formatting
            # OpenAI adds ~3 tokens per message for formatting
            if model.startswith("gpt-"):
                overhead_per_message = 3
                total_tokens += len(messages) * overhead_per_message
                total_tokens += 3  # Every reply is primed with assistant
            
            return total_tokens
            
        except Exception as e:
            self.logger.error(f"Message token counting failed: {e}")
            return self.count_tokens(str(messages), model)
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int = 0,
        model: str = "gpt-3.5-turbo",
        provider: Optional[ModelProvider] = None
    ) -> CostEstimate:
        """
        Estimate cost for token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
            provider: LLM provider
            
        Returns:
            Cost estimate
        """
        try:
            # Get pricing info
            pricing = self.model_pricing.get(model)
            if not pricing:
                # Use default pricing based on provider
                if provider == ModelProvider.OPENAI or model.startswith("gpt-"):
                    pricing = self.model_pricing["gpt-3.5-turbo"]
                elif provider == ModelProvider.ANTHROPIC or model.startswith("claude"):
                    pricing = self.model_pricing["claude-3-haiku"]
                else:
                    # Default to GPT-3.5 pricing
                    pricing = self.model_pricing["gpt-3.5-turbo"]
            
            # Calculate costs
            prompt_cost = (prompt_tokens / 1000) * pricing.prompt_cost_per_1k
            completion_cost = (completion_tokens / 1000) * pricing.completion_cost_per_1k
            total_cost = prompt_cost + completion_cost
            
            return CostEstimate(
                prompt_cost=round(prompt_cost, 6),
                completion_cost=round(completion_cost, 6),
                total_cost=round(total_cost, 6),
                currency=pricing.currency,
                model=model,
                provider=provider.value if provider else "unknown"
            )
            
        except Exception as e:
            self.logger.error(f"Cost estimation failed: {e}")
            return CostEstimate(
                prompt_cost=0.0,
                completion_cost=0.0,
                total_cost=0.0,
                model=model,
                provider=provider.value if provider else "unknown"
            )
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a model."""
        pricing = self.model_pricing.get(model)
        encoding = self.model_encodings.get(model)
        
        return {
            "model": model,
            "encoding": encoding,
            "pricing": {
                "prompt_cost_per_1k": pricing.prompt_cost_per_1k if pricing else None,
                "completion_cost_per_1k": pricing.completion_cost_per_1k if pricing else None,
                "currency": pricing.currency if pricing else None,
            } if pricing else None,
            "limits": {
                "max_tokens": pricing.max_tokens if pricing else None,
                "context_window": pricing.context_window if pricing else None,
            } if pricing else None,
            "supported": encoding is not None or model in self.model_pricing
        }
    
    def validate_token_limits(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate if token usage is within model limits.
        
        Returns:
            (is_valid, details)
        """
        pricing = self.model_pricing.get(model)
        if not pricing:
            return True, {"warning": "Model limits unknown"}
        
        total_tokens = prompt_tokens + completion_tokens
        
        validation = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "max_tokens": pricing.max_tokens,
            "context_window": pricing.context_window,
            "within_completion_limit": completion_tokens <= pricing.max_tokens,
            "within_context_limit": total_tokens <= pricing.context_window
        }
        
        is_valid = (
            completion_tokens <= pricing.max_tokens and
            total_tokens <= pricing.context_window
        )
        
        if not is_valid:
            if completion_tokens > pricing.max_tokens:
                validation["error"] = f"Completion tokens ({completion_tokens}) exceed max_tokens ({pricing.max_tokens})"
            elif total_tokens > pricing.context_window:
                validation["error"] = f"Total tokens ({total_tokens}) exceed context window ({pricing.context_window})"
        
        return is_valid, validation
    
    def add_model_pricing(self, model: str, pricing: ModelPricing):
        """Add custom model pricing."""
        self.model_pricing[model] = pricing
        self.logger.info(f"Added pricing for model: {model}")
    
    def add_model_encoding(self, model: str, encoding: str):
        """Add custom model encoding."""
        self.model_encodings[model] = encoding
        self.logger.info(f"Added encoding for model: {model}")


class UsageTracker:
    """Track token usage and costs over time."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.usage_history: List[Dict[str, Any]] = []
        self.token_counter = TokenCounter()
    
    def track_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        provider: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track usage for a request."""
        import time
        
        cost_estimate = self.token_counter.estimate_cost(
            prompt_tokens, completion_tokens, model
        )
        
        usage_record = {
            "timestamp": time.time(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model,
            "provider": provider,
            "cost_estimate": cost_estimate.__dict__,
            "user_id": user_id,
            "session_id": session_id,
            "metadata": metadata or {}
        }
        
        self.usage_history.append(usage_record)
        
        # Keep only last 10000 records in memory
        if len(self.usage_history) > 10000:
            self.usage_history = self.usage_history[-10000:]
        
        return usage_record
    
    def get_usage_stats(
        self,
        time_range_hours: int = 24,
        user_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get usage statistics."""
        import time
        
        cutoff_time = time.time() - (time_range_hours * 3600)
        
        # Filter records
        filtered_records = [
            record for record in self.usage_history
            if record["timestamp"] >= cutoff_time
        ]
        
        if user_id:
            filtered_records = [
                record for record in filtered_records
                if record.get("user_id") == user_id
            ]
        
        if model:
            filtered_records = [
                record for record in filtered_records
                if record.get("model") == model
            ]
        
        if not filtered_records:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "time_range_hours": time_range_hours
            }
        
        # Calculate statistics
        total_requests = len(filtered_records)
        total_tokens = sum(record["total_tokens"] for record in filtered_records)
        total_cost = sum(record["cost_estimate"]["total_cost"] for record in filtered_records)
        
        # Model breakdown
        model_stats = {}
        for record in filtered_records:
            model_name = record["model"]
            if model_name not in model_stats:
                model_stats[model_name] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            
            model_stats[model_name]["requests"] += 1
            model_stats[model_name]["tokens"] += record["total_tokens"]
            model_stats[model_name]["cost"] += record["cost_estimate"]["total_cost"]
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "average_tokens_per_request": total_tokens / total_requests if total_requests > 0 else 0,
            "average_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "model_breakdown": model_stats,
            "time_range_hours": time_range_hours
        }


# Global instances
_token_counter: Optional[TokenCounter] = None
_usage_tracker: Optional[UsageTracker] = None


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


def count_tokens(text: Union[str, List[Dict[str, str]]], model: str = "gpt-3.5-turbo") -> int:
    """Convenience function to count tokens."""
    return get_token_counter().count_tokens(text, model)


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int = 0,
    model: str = "gpt-3.5-turbo"
) -> CostEstimate:
    """Convenience function to estimate cost."""
    return get_token_counter().estimate_cost(prompt_tokens, completion_tokens, model)
