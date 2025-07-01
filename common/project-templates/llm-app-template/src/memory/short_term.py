"""
Short-term Conversation Memory

Manages conversational context and working memory for ongoing interactions.
Provides efficient storage and retrieval of recent conversation history.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from ..utils.logger import get_logger


@dataclass
class Message:
    """Represents a single message in conversation history."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    role: str = "user"  # user, assistant, system
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


@dataclass
class ConversationSession:
    """Represents a conversation session."""
    
    session_id: str
    user_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    total_tokens: int = 0
    max_messages: int = 100


class ConversationMemory:
    """
    Manages short-term conversational memory and working context.
    
    Features:
    - Message history management with configurable limits
    - Working memory for temporary context
    - Token counting and management
    - Automatic cleanup of old conversations
    - Context summarization when limits are reached
    """
    
    def __init__(
        self,
        max_messages_per_session: int = 100,
        max_tokens_per_session: int = 8000,
        max_sessions: int = 1000,
        cleanup_interval: int = 3600  # 1 hour
    ):
        self.max_messages_per_session = max_messages_per_session
        self.max_tokens_per_session = max_tokens_per_session
        self.max_sessions = max_sessions
        self.cleanup_interval = cleanup_interval
        
        self.sessions: Dict[str, ConversationSession] = {}
        self.last_cleanup = time.time()
        
        self.logger = get_logger("memory.conversation")
    
    def get_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> ConversationSession:
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                max_messages=self.max_messages_per_session
            )
            self.logger.info(f"Created new conversation session", extra={
                "session_id": session_id,
                "user_id": user_id
            })
        
        # Perform cleanup if needed
        self._maybe_cleanup()
        
        return self.sessions[session_id]
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None
    ) -> Message:
        """Add a message to the conversation history."""
        session = self.get_or_create_session(session_id)
        
        # Estimate token count if not provided
        if token_count is None:
            token_count = self._estimate_tokens(content)
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            token_count=token_count
        )
        
        session.messages.append(message)
        session.total_tokens += token_count
        session.updated_at = time.time()
        
        # Check if we need to manage message limits
        self._manage_session_limits(session)
        
        self.logger.debug(f"Added message to session", extra={
            "session_id": session_id,
            "role": role,
            "token_count": token_count,
            "total_messages": len(session.messages),
            "total_tokens": session.total_tokens
        })
        
        return message
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        since: Optional[float] = None
    ) -> List[Message]:
        """Get messages from a session with optional filtering."""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        messages = session.messages
        
        # Filter by timestamp if provided
        if since:
            messages = [msg for msg in messages if msg.timestamp >= since]
        
        # Apply limit if provided
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_conversation_history(
        self,
        session_id: str,
        format_type: str = "messages"
    ) -> Union[List[Message], List[Dict[str, str]], str]:
        """Get conversation history in different formats."""
        messages = self.get_messages(session_id)
        
        if format_type == "messages":
            return messages
        elif format_type == "dict":
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in messages
            ]
        elif format_type == "text":
            return "\\n".join(
                f"{msg.role.upper()}: {msg.content}"
                for msg in messages
            )
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def set_working_memory(
        self,
        session_id: str,
        key: str,
        value: Any
    ):
        """Set a value in the session's working memory."""
        session = self.get_or_create_session(session_id)
        session.working_memory[key] = value
        session.updated_at = time.time()
        
        self.logger.debug(f"Set working memory", extra={
            "session_id": session_id,
            "key": key,
            "value_type": type(value).__name__
        })
    
    def get_working_memory(
        self,
        session_id: str,
        key: Optional[str] = None
    ) -> Union[Any, Dict[str, Any]]:
        """Get value(s) from the session's working memory."""
        if session_id not in self.sessions:
            return {} if key is None else None
        
        session = self.sessions[session_id]
        
        if key is None:
            return session.working_memory
        else:
            return session.working_memory.get(key)
    
    def clear_working_memory(self, session_id: str, key: Optional[str] = None):
        """Clear working memory for a session."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if key is None:
            session.working_memory.clear()
            self.logger.info(f"Cleared all working memory", extra={
                "session_id": session_id
            })
        else:
            session.working_memory.pop(key, None)
            self.logger.debug(f"Cleared working memory key", extra={
                "session_id": session_id,
                "key": key
            })
        
        session.updated_at = time.time()
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation session."""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "message_count": len(session.messages),
            "total_tokens": session.total_tokens,
            "working_memory_keys": list(session.working_memory.keys()),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "duration": session.updated_at - session.created_at
        }
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Deleted conversation session", extra={
                "session_id": session_id
            })
            return True
        return False
    
    def list_sessions(self, user_id: Optional[str] = None) -> List[str]:
        """List all session IDs, optionally filtered by user."""
        if user_id is None:
            return list(self.sessions.keys())
        else:
            return [
                session_id for session_id, session in self.sessions.items()
                if session.user_id == user_id
            ]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: 1 token ≈ 4 characters
        return max(1, len(text) // 4)
    
    def _manage_session_limits(self, session: ConversationSession):
        """Manage session limits by removing old messages or summarizing."""
        # Remove old messages if we exceed the limit
        while len(session.messages) > self.max_messages_per_session:
            removed_message = session.messages.pop(0)
            session.total_tokens -= removed_message.token_count
            
            self.logger.debug(f"Removed old message from session", extra={
                "session_id": session.session_id,
                "removed_tokens": removed_message.token_count
            })
        
        # Handle token limit (simple removal for now)
        while session.total_tokens > self.max_tokens_per_session and session.messages:
            removed_message = session.messages.pop(0)
            session.total_tokens -= removed_message.token_count
            
            self.logger.debug(f"Removed message due to token limit", extra={
                "session_id": session.session_id,
                "removed_tokens": removed_message.token_count
            })
    
    def _maybe_cleanup(self):
        """Perform periodic cleanup of old sessions."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        # Remove oldest sessions if we exceed max_sessions
        if len(self.sessions) > self.max_sessions:
            # Sort sessions by last update time
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].updated_at
            )
            
            # Remove the oldest sessions
            sessions_to_remove = len(self.sessions) - self.max_sessions
            for i in range(sessions_to_remove):
                session_id, _ = sorted_sessions[i]
                del self.sessions[session_id]
                self.logger.info(f"Cleaned up old session", extra={
                    "session_id": session_id
                })
        
        self.last_cleanup = current_time
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get overall memory statistics."""
        total_messages = sum(len(session.messages) for session in self.sessions.values())
        total_tokens = sum(session.total_tokens for session in self.sessions.values())
        
        return {
            "total_sessions": len(self.sessions),
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "average_messages_per_session": total_messages / max(len(self.sessions), 1),
            "average_tokens_per_session": total_tokens / max(len(self.sessions), 1),
            "last_cleanup": self.last_cleanup
        }
