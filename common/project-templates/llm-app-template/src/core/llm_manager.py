"""
LLM Management and Provider Abstraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum

import openai
from anthropic import Anthropic
import google.generativeai as genai

from .config import get_settings


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate text using the LLM."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """Generate text using the LLM with streaming."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def generate(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate text using OpenAI."""
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
    
    async def generate_stream(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """Generate text using OpenAI with streaming."""
        stream = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider implementation."""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    async def generate(
        self,
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate text using Anthropic."""
        response = await self.client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.content[0].text
    
    async def generate_stream(
        self,
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """Generate text using Anthropic with streaming."""
        # Implementation for streaming would go here
        # Anthropic may have different streaming API
        yield await self.generate(prompt, model, temperature, max_tokens, **kwargs)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel('gemini-pro')
    
    async def generate(
        self,
        prompt: str,
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate text using Gemini."""
        response = await self.client.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
        )
        return response.text
    
    async def generate_stream(
        self,
        prompt: str,
        model: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        """Generate text using Gemini with streaming."""
        # Implementation for streaming would go here
        yield await self.generate(prompt, model, temperature, max_tokens, **kwargs)


class LLMManager:
    """Manages multiple LLM providers and routing with advanced features."""
    
    def __init__(self):
        self.settings = get_settings()
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.fallback_chain: List[str] = []
        self.circuit_breakers: Dict[str, Dict] = {}
        self.usage_stats: Dict[str, Dict] = {}
        self._initialize_providers()
        self._setup_fallback_chain()
        self._initialize_circuit_breakers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers."""
        if self.settings.openai_api_key:
            self.providers[ModelProvider.OPENAI] = OpenAIProvider(
                self.settings.openai_api_key
            )
        
        if self.settings.anthropic_api_key:
            self.providers[ModelProvider.ANTHROPIC] = AnthropicProvider(
                self.settings.anthropic_api_key
            )
        
        if self.settings.gemini_api_key:
            self.providers[ModelProvider.GEMINI] = GeminiProvider(
                self.settings.gemini_api_key
            )
    
    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using specified or default provider."""
        provider = provider or self.settings.default_model_provider
        model = model or self.settings.default_model
        temperature = temperature or self.settings.default_temperature
        max_tokens = max_tokens or self.settings.default_max_tokens
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        return await self.providers[provider].generate(
            prompt, model, temperature, max_tokens, **kwargs
        )
    
    async def generate_stream(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate text with streaming using specified or default provider."""
        provider = provider or self.settings.default_model_provider
        model = model or self.settings.default_model
        temperature = temperature or self.settings.default_temperature
        max_tokens = max_tokens or self.settings.default_max_tokens
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        async for chunk in self.providers[provider].generate_stream(
            prompt, model, temperature, max_tokens, **kwargs
        ):
            yield chunk
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    def _setup_fallback_chain(self):
        """Setup fallback chain for provider failures."""
        available_providers = list(self.providers.keys())
        if ModelProvider.OPENAI in available_providers:
            self.fallback_chain.append(ModelProvider.OPENAI)
        if ModelProvider.ANTHROPIC in available_providers:
            self.fallback_chain.append(ModelProvider.ANTHROPIC)
        if ModelProvider.GEMINI in available_providers:
            self.fallback_chain.append(ModelProvider.GEMINI)
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for each provider."""
        for provider in self.providers.keys():
            self.circuit_breakers[provider] = {
                'failure_count': 0,
                'last_failure_time': None,
                'state': 'closed',  # closed, open, half_open
                'failure_threshold': 5,
                'timeout': 60  # seconds
            }
            self.usage_stats[provider] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_tokens': 0,
                'total_cost': 0.0
            }
    
    def _is_circuit_open(self, provider: str) -> bool:
        """Check if circuit breaker is open for a provider."""
        import time
        
        circuit = self.circuit_breakers.get(provider, {})
        if circuit.get('state') == 'open':
            if circuit.get('last_failure_time'):
                time_since_failure = time.time() - circuit['last_failure_time']
                if time_since_failure > circuit.get('timeout', 60):
                    circuit['state'] = 'half_open'
                    return False
                return True
        return False
    
    def _record_success(self, provider: str):
        """Record successful request for a provider."""
        circuit = self.circuit_breakers.get(provider, {})
        circuit['failure_count'] = 0
        circuit['state'] = 'closed'
        
        stats = self.usage_stats.get(provider, {})
        stats['total_requests'] = stats.get('total_requests', 0) + 1
        stats['successful_requests'] = stats.get('successful_requests', 0) + 1
    
    def _record_failure(self, provider: str):
        """Record failed request for a provider."""
        import time
        
        circuit = self.circuit_breakers.get(provider, {})
        circuit['failure_count'] = circuit.get('failure_count', 0) + 1
        circuit['last_failure_time'] = time.time()
        
        if circuit['failure_count'] >= circuit.get('failure_threshold', 5):
            circuit['state'] = 'open'
        
        stats = self.usage_stats.get(provider, {})
        stats['total_requests'] = stats.get('total_requests', 0) + 1
        stats['failed_requests'] = stats.get('failed_requests', 0) + 1
    
    async def generate_with_fallback(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text with automatic fallback to other providers."""
        last_error = None
        
        for provider in self.fallback_chain:
            if self._is_circuit_open(provider):
                continue
            
            try:
                result = await self.generate(
                    prompt, provider, model, temperature, max_tokens, **kwargs
                )
                self._record_success(provider)
                return result
            except Exception as e:
                self._record_failure(provider)
                last_error = e
                continue
        
        # If all providers failed, raise the last error
        if last_error:
            raise last_error
        else:
            raise RuntimeError("No available providers")
    
    def get_usage_stats(self) -> Dict[str, Dict]:
        """Get usage statistics for all providers."""
        return self.usage_stats.copy()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        status = {}
        for provider in self.providers.keys():
            circuit = self.circuit_breakers.get(provider, {})
            stats = self.usage_stats.get(provider, {})
            
            status[provider] = {
                'available': not self._is_circuit_open(provider),
                'circuit_state': circuit.get('state', 'unknown'),
                'failure_count': circuit.get('failure_count', 0),
                'success_rate': (
                    stats.get('successful_requests', 0) / 
                    max(stats.get('total_requests', 1), 1)
                ) * 100
            }
        
        return status
