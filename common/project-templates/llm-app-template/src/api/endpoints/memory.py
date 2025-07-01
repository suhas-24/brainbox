"""
Memory Management API Endpoints

Provides endpoints for managing conversation memory including
session operations, working memory, and memory statistics.
"""

from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...utils import get_logger
from ..app import get_app_state


router = APIRouter()
logger = get_logger(__name__)


class MemoryStats(BaseModel):
    """Memory statistics model."""
    short_term_sessions: int = Field(..., description="Number of active sessions")
    short_term_messages: int = Field(..., description="Total messages in memory")
    long_term_entries: int = Field(..., description="Long-term memory entries")
    vector_embeddings: int = Field(..., description="Vector embeddings stored")
    total_memory_mb: float = Field(..., description="Total memory usage in MB")


class WorkingMemoryItem(BaseModel):
    """Working memory item model."""
    key: str = Field(..., description="Memory key")
    value: Any = Field(..., description="Memory value")


@router.get("/stats", response_model=MemoryStats)
async def get_memory_stats() -> MemoryStats:
    """
    Get memory usage statistics.
    
    Returns:
        Memory statistics
    """
    try:
        app_state = get_app_state()
        memory_manager = app_state["memory_manager"]
        
        stats = await memory_manager.get_memory_stats()
        
        return MemoryStats(
            short_term_sessions=stats.short_term_sessions,
            short_term_messages=stats.short_term_messages,
            long_term_entries=stats.long_term_entries,
            vector_embeddings=stats.vector_embeddings,
            total_memory_mb=stats.total_memory_mb
        )
        
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions() -> List[str]:
    """
    List all active memory sessions.
    
    Returns:
        List of session IDs
    """
    try:
        app_state = get_app_state()
        memory_manager = app_state["memory_manager"]
        
        # Get session list from short-term memory
        sessions = memory_manager.short_term.sessions.keys()
        
        return list(sessions)
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str) -> Dict[str, str]:
    """
    Clear a specific memory session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success confirmation
    """
    try:
        app_state = get_app_state()
        memory_manager = app_state["memory_manager"]
        
        success = await memory_manager.clear_session(session_id)
        
        if success:
            return {"status": "success", "message": f"Session {session_id} cleared"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions")
async def clear_all_sessions() -> Dict[str, str]:
    """
    Clear all memory sessions.
    
    Returns:
        Success confirmation
    """
    try:
        app_state = get_app_state()
        memory_manager = app_state["memory_manager"]
        
        # Clear all sessions
        sessions_cleared = 0
        for session_id in list(memory_manager.short_term.sessions.keys()):
            success = await memory_manager.clear_session(session_id)
            if success:
                sessions_cleared += 1
        
        return {
            "status": "success", 
            "message": f"Cleared {sessions_cleared} sessions"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear all sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/working")
async def get_working_memory(
    session_id: str,
    key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get working memory for a session.
    
    Args:
        session_id: Session identifier
        key: Specific memory key (optional)
        
    Returns:
        Working memory data
    """
    try:
        app_state = get_app_state()
        memory_manager = app_state["memory_manager"]
        
        if key:
            # Get specific key
            value = await memory_manager.get_working_memory(session_id, key)
            return {key: value}
        else:
            # Get all working memory
            working_memory = await memory_manager.get_working_memory(session_id)
            return working_memory if working_memory else {}
        
    except Exception as e:
        logger.error(f"Failed to get working memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/working")
async def set_working_memory(
    session_id: str,
    items: List[WorkingMemoryItem]
) -> Dict[str, str]:
    """
    Set working memory items for a session.
    
    Args:
        session_id: Session identifier
        items: Working memory items to set
        
    Returns:
        Success confirmation
    """
    try:
        app_state = get_app_state()
        memory_manager = app_state["memory_manager"]
        
        # Set each working memory item
        for item in items:
            success = await memory_manager.set_working_memory(
                session_id, 
                item.key, 
                item.value
            )
            if not success:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to set working memory item: {item.key}"
                )
        
        return {
            "status": "success", 
            "message": f"Set {len(items)} working memory items"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set working memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}/working")
async def clear_working_memory(
    session_id: str,
    key: Optional[str] = None
) -> Dict[str, str]:
    """
    Clear working memory for a session.
    
    Args:
        session_id: Session identifier
        key: Specific memory key to clear (optional)
        
    Returns:
        Success confirmation
    """
    try:
        app_state = get_app_state()
        memory_manager = app_state["memory_manager"]
        
        if key:
            # Clear specific key
            success = await memory_manager.set_working_memory(session_id, key, None)
            if success:
                return {"status": "success", "message": f"Cleared working memory key: {key}"}
            else:
                raise HTTPException(status_code=404, detail="Working memory key not found")
        else:
            # Clear all working memory for session
            # This would require additional implementation in memory manager
            return {"status": "success", "message": "Working memory clearing not fully implemented"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear working memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
