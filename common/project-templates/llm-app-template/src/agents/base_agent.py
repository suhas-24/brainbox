"""
Base Agent Implementation for AI Forge

Provides the foundational agent class with advanced capabilities including
memory management, tool integration, and robust error handling.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from ..core.llm_manager import LLMManager
from ..utils.logger import get_logger


@dataclass
class AgentResponse:
    """Structured response from an agent."""
    
    content: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)


@dataclass
class AgentContext:
    """Context for agent execution."""
    
    session_id: str
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    tools_available: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all AI agents.
    
    Provides common functionality including:
    - LLM integration with fallback
    - Memory management
    - Tool integration
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_manager: Optional[LLMManager] = None,
        tools: Optional[List[Any]] = None,
        max_iterations: int = 10,
        timeout: float = 300.0
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm_manager = llm_manager or LLMManager()
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.timeout = timeout
        
        self.logger = get_logger(f"agent.{name.lower()}")
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_tokens_used": 0
        }
    
    async def execute(
        self,
        task: str,
        context: Optional[AgentContext] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Execute a task with the agent.
        
        Args:
            task: The task description or prompt
            context: Optional execution context
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse with results and metadata
        """
        start_time = time.time()
        session_id = context.session_id if context else str(uuid4())
        
        self.logger.info(f"Starting task execution", extra={
            "session_id": session_id,
            "task": task[:100] + "..." if len(task) > 100 else task
        })
        
        try:
            # Initialize context if not provided
            if context is None:
                context = AgentContext(session_id=session_id)
            
            # Validate inputs
            self._validate_inputs(task, context)
            
            # Execute the main task logic
            response = await self._execute_task(task, context, **kwargs)
            
            # Record success metrics
            execution_time = time.time() - start_time
            self._record_execution_success(execution_time, response.token_usage)
            
            response.execution_time = execution_time
            
            self.logger.info(f"Task completed successfully", extra={
                "session_id": session_id,
                "execution_time": execution_time,
                "token_usage": response.token_usage
            })
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_execution_failure(execution_time)
            
            error_msg = f"Task execution failed: {str(e)}"
            self.logger.error(error_msg, extra={
                "session_id": session_id,
                "execution_time": execution_time,
                "error": str(e)
            })
            
            return AgentResponse(
                content="",
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    @abstractmethod
    async def _execute_task(
        self,
        task: str,
        context: AgentContext,
        **kwargs
    ) -> AgentResponse:
        """
        Abstract method for task execution logic.
        Must be implemented by subclasses.
        """
        pass
    
    def _validate_inputs(self, task: str, context: AgentContext):
        """Validate inputs before execution."""
        if not task or not task.strip():
            raise ValueError("Task cannot be empty")
        
        if not context.session_id:
            raise ValueError("Session ID is required")
    
    async def _generate_response(
        self,
        prompt: str,
        context: AgentContext,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        use_fallback: bool = True
    ) -> str:
        """Generate response using LLM with optional fallback."""
        try:
            # Construct full prompt with system prompt and context
            full_prompt = self._build_full_prompt(prompt, context)
            
            if use_fallback:
                response = await self.llm_manager.generate_with_fallback(
                    full_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = await self.llm_manager.generate(
                    full_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            return response
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise
    
    def _build_full_prompt(self, prompt: str, context: AgentContext) -> str:
        """Build the full prompt including system prompt and context."""
        components = [self.system_prompt]
        
        # Add conversation history if available
        if context.conversation_history:
            components.append("\\n## Conversation History:")
            for msg in context.conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                components.append(f"{role.upper()}: {content}")
        
        # Add working memory if available
        if context.working_memory:
            components.append("\\n## Working Memory:")
            for key, value in context.working_memory.items():
                components.append(f"- {key}: {value}")
        
        # Add the current prompt
        components.append(f"\\n## Current Task:\\n{prompt}")
        
        return "\\n".join(components)
    
    def _record_execution_success(self, execution_time: float, token_usage: Dict[str, int]):
        """Record successful execution metrics."""
        self.execution_stats["total_executions"] += 1
        self.execution_stats["successful_executions"] += 1
        
        # Update average execution time
        total_time = (
            self.execution_stats["average_execution_time"] * 
            (self.execution_stats["total_executions"] - 1) + execution_time
        )
        self.execution_stats["average_execution_time"] = total_time / self.execution_stats["total_executions"]
        
        # Update token usage
        total_tokens = token_usage.get("total_tokens", 0)
        self.execution_stats["total_tokens_used"] += total_tokens
    
    def _record_execution_failure(self, execution_time: float):
        """Record failed execution metrics."""
        self.execution_stats["total_executions"] += 1
        self.execution_stats["failed_executions"] += 1
        
        # Update average execution time
        total_time = (
            self.execution_stats["average_execution_time"] * 
            (self.execution_stats["total_executions"] - 1) + execution_time
        )
        self.execution_stats["average_execution_time"] = total_time / self.execution_stats["total_executions"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent execution statistics."""
        stats = self.execution_stats.copy()
        stats["success_rate"] = (
            stats["successful_executions"] / max(stats["total_executions"], 1) * 100
        )
        stats["agent_name"] = self.name
        return stats
    
    def reset_stats(self):
        """Reset execution statistics."""
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_tokens_used": 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the agent."""
        try:
            # Test LLM connectivity
            test_response = await self._generate_response(
                "Respond with 'OK' to confirm you are working.",
                AgentContext(session_id="health_check"),
                max_tokens=10
            )
            
            llm_healthy = "ok" in test_response.lower()
            
            return {
                "agent_name": self.name,
                "status": "healthy" if llm_healthy else "degraded",
                "llm_connectivity": llm_healthy,
                "stats": self.get_stats(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "agent_name": self.name,
                "status": "unhealthy",
                "error": str(e),
                "stats": self.get_stats(),
                "timestamp": time.time()
            }
