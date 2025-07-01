"""
Unified Memory Manager for AI Forge

Integrates short-term, long-term, and vector memory systems
into a cohesive memory management solution.
"""

from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass

from .short_term import ConversationMemory
from ..utils import get_logger


@dataclass
class MemorySnapshot:
    """Snapshot of memory state for debugging/monitoring."""
    short_term_sessions: int
    short_term_messages: int
    long_term_entries: int
    vector_embeddings: int
    total_memory_mb: float


class MemoryManager:
    """Unified memory management system."""
    
    def __init__(self, enable_long_term: bool = True, enable_vector: bool = True):
        self.logger = get_logger(__name__)
        
        # Initialize memory systems
        self.short_term = ConversationMemory()
        
        # Initialize long-term memory if enabled
        self.long_term = None
        if enable_long_term:
            try:
                from .long_term import LongTermMemory
                self.long_term = LongTermMemory()
                self.logger.info("Long-term memory enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize long-term memory: {e}")
        
        # Initialize vector memory if enabled
        self.vector = None
        if enable_vector:
            try:
                from .vector_memory import VectorMemory
                self.vector = VectorMemory()
                self.logger.info("Vector memory enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize vector memory: {e}")
        
        self.logger.info("Memory manager initialized")
    
    async def store_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        user_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store a conversation exchange in memory."""
        try:
            # Store in short-term memory
            self.short_term.add_message(session_id, "user", user_message)
            self.short_term.add_message(session_id, "assistant", assistant_response)
            
            # Store additional metadata if provided
            if metadata:
                for key, value in metadata.items():
                    self.short_term.set_working_memory(session_id, key, value)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to store conversation: {e}")
            return False
    
    async def get_conversation_context(
        self,
        session_id: str,
        max_messages: int = 20
    ) -> List[Dict[str, Any]]:
        """Get conversation context for a session."""
        try:
            messages = self.short_term.get_messages(session_id, limit=max_messages)
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
        
        except Exception as e:
            self.logger.error(f"Failed to get conversation context: {e}")
            return []
    
    async def get_working_memory(
        self,
        session_id: str,
        key: str = None
    ) -> Any:
        """Get working memory for a session."""
        return self.short_term.get_working_memory(session_id, key)
    
    async def set_working_memory(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> bool:
        """Set working memory for a session."""
        try:
            self.short_term.set_working_memory(session_id, key, value)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set working memory: {e}")
            return False
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear all memory for a session."""
        try:
            return self.short_term.delete_session(session_id)
        except Exception as e:
            self.logger.error(f"Failed to clear session: {e}")
            return False
    
    async def store_conversation_enhanced(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        user_id: str = None,
        metadata: Dict[str, Any] = None,
        importance: float = 0.5,
        store_in_vector: bool = True
    ) -> bool:
        """Enhanced conversation storage across all memory systems."""
        try:
            # Store in short-term memory
            await self.store_conversation(session_id, user_message, assistant_response, user_id, metadata)
            
            # Store in long-term memory if available
            if self.long_term and user_id:
                try:
                    from .long_term import LongTermEntry
                    entry = LongTermEntry(
                        user_id=user_id,
                        session_id=session_id,
                        entry_type="conversation",
                        content=f"User: {user_message}\nAssistant: {assistant_response}",
                        importance=importance,
                        metadata=metadata or {}
                    )
                    await self.long_term.store_entry(entry)
                except Exception as e:
                    self.logger.warning(f"Long-term storage failed: {e}")
            
            # Store in vector memory if available
            if self.vector and store_in_vector and user_id:
                try:
                    await self.vector.store_conversation(
                        user_message, assistant_response, user_id, session_id, metadata
                    )
                except Exception as e:
                    self.logger.warning(f"Vector storage failed: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced conversation storage failed: {e}")
            return False
    
    async def search_similar_conversations(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar past conversations using vector memory."""
        if not self.vector:
            return []
        
        try:
            results = await self.vector.get_similar_conversations(
                query, user_id, limit, min_score
            )
            
            return [
                {
                    "content": result.entry.content,
                    "score": result.score,
                    "session_id": result.entry.session_id,
                    "timestamp": result.entry.timestamp,
                    "metadata": result.entry.metadata
                }
                for result in results
            ]
            
        except Exception as e:
            self.logger.error(f"Similar conversation search failed: {e}")
            return []
    
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 50,
        min_importance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Get user's conversation history from long-term memory."""
        if not self.long_term:
            return []
        
        try:
            entries = await self.long_term.search_entries(
                user_id=user_id,
                entry_type="conversation",
                min_importance=min_importance,
                limit=limit
            )
            
            return [
                {
                    "id": entry.id,
                    "content": entry.content,
                    "importance": entry.importance,
                    "created_at": entry.created_at,
                    "metadata": entry.metadata
                }
                for entry in entries
            ]
            
        except Exception as e:
            self.logger.error(f"User history retrieval failed: {e}")
            return []
    
    async def store_user_preference(
        self,
        user_id: str,
        preference_name: str,
        preference_value: Any,
        importance: float = 0.8
    ) -> bool:
        """Store user preference in long-term memory."""
        if not self.long_term:
            return False
        
        try:
            from .long_term import LongTermEntry
            entry = LongTermEntry(
                user_id=user_id,
                entry_type="preference",
                content=f"{preference_name}: {preference_value}",
                importance=importance,
                metadata={
                    "preference_name": preference_name,
                    "preference_value": preference_value
                }
            )
            
            await self.long_term.store_entry(entry)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store user preference: {e}")
            return False
    
    async def get_user_preferences(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get user preferences from long-term memory."""
        if not self.long_term:
            return {}
        
        try:
            entries = await self.long_term.search_entries(
                user_id=user_id,
                entry_type="preference"
            )
            
            preferences = {}
            for entry in entries:
                pref_name = entry.metadata.get("preference_name")
                pref_value = entry.metadata.get("preference_value")
                if pref_name:
                    preferences[pref_name] = pref_value
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Failed to get user preferences: {e}")
            return {}
    
    async def get_memory_stats(self) -> MemorySnapshot:
        """Get comprehensive memory usage statistics."""
        try:
            short_term_stats = self.short_term.get_memory_stats()
            
            # Get long-term stats
            long_term_entries = 0
            if self.long_term:
                try:
                    lt_stats = await self.long_term.get_stats()
                    long_term_entries = lt_stats.get("total_entries", 0)
                except Exception:
                    pass
            
            # Get vector stats
            vector_embeddings = 0
            if self.vector:
                try:
                    v_stats = await self.vector.get_stats()
                    vector_embeddings = v_stats.get("vector_backend", {}).get("total_vectors", 0)
                except Exception:
                    pass
            
            # Estimate total memory usage
            memory_mb = (
                short_term_stats.get("estimated_size_mb", 0) +
                (long_term_entries * 0.001) +  # Rough estimate
                (vector_embeddings * 0.006)   # Rough estimate for embeddings
            )
            
            return MemorySnapshot(
                short_term_sessions=short_term_stats.get("total_sessions", 0),
                short_term_messages=short_term_stats.get("total_messages", 0),
                long_term_entries=long_term_entries,
                vector_embeddings=vector_embeddings,
                total_memory_mb=memory_mb
            )
        
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return MemorySnapshot(0, 0, 0, 0, 0.0)
    
    async def cleanup_expired_entries(self) -> Dict[str, int]:
        """Clean up expired entries across all memory systems."""
        cleanup_stats = {
            "short_term": 0,
            "long_term": 0,
            "vector": 0
        }
        
        try:
            # Cleanup short-term memory
            cleanup_stats["short_term"] = self.short_term.cleanup_old_sessions(max_age_hours=24)
            
            # Cleanup long-term memory
            if self.long_term:
                try:
                    cleanup_stats["long_term"] = await self.long_term.cleanup_expired()
                except Exception as e:
                    self.logger.warning(f"Long-term cleanup failed: {e}")
            
            # Vector memory cleanup would be done based on usage patterns
            
            self.logger.info(f"Memory cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
            return cleanup_stats
