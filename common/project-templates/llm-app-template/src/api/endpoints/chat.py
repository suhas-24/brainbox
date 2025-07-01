"""
Chat Completions API Endpoints

Provides endpoints for LLM chat interactions including single completions,
streaming responses, and conversation management.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ...utils import get_logger
from ..app import get_app_state


router = APIRouter()
logger = get_logger(__name__)


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    provider: Optional[str] = Field(None, description="LLM provider to use")
    model: Optional[str] = Field(None, description="Model name")
    session_id: Optional[str] = Field(None, description="Session ID for memory")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    stream: bool = Field(False, description="Stream response")
    use_memory: bool = Field(True, description="Use conversation memory")


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str = Field(..., description="Response ID")
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    usage: Dict[str, Any] = Field(..., description="Token usage information")
    created: datetime = Field(..., description="Creation timestamp")
    session_id: Optional[str] = Field(None, description="Session ID")


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks
) -> ChatCompletionResponse:
    """
    Create a chat completion.
    
    Args:
        request: Chat completion request
        background_tasks: Background task manager
        
    Returns:
        Chat completion response
    """
    try:
        app_state = get_app_state()
        provider_manager = app_state["provider_manager"]
        memory_manager = app_state["memory_manager"]
        
        # Get conversation context if using memory
        messages = request.messages
        if request.use_memory and request.session_id:
            context = await memory_manager.get_conversation_context(
                request.session_id,
                max_messages=20
            )
            
            # Add context messages to the beginning
            context_messages = [
                ChatMessage(
                    role=msg["role"],
                    content=msg["content"],
                    metadata=msg.get("metadata")
                )
                for msg in context
            ]
            messages = context_messages + messages
        
        # Generate completion
        completion = await provider_manager.generate_completion(
            messages=[msg.dict() for msg in messages],
            provider=request.provider,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Store in memory if enabled
        if request.use_memory and request.session_id:
            background_tasks.add_task(
                memory_manager.store_conversation,
                request.session_id,
                request.messages[-1].content,  # Last user message
                completion["content"],
                metadata={"provider": completion["provider"], "model": completion["model"]}
            )
        
        return ChatCompletionResponse(
            id=completion["id"],
            content=completion["content"],
            model=completion["model"],
            provider=completion["provider"],
            usage=completion["usage"],
            created=datetime.now(),
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def create_chat_completion_stream(request: ChatCompletionRequest):
    """
    Create a streaming chat completion.
    
    Args:
        request: Chat completion request (must have stream=True)
        
    Returns:
        Server-sent events stream
    """
    if not request.stream:
        raise HTTPException(status_code=400, detail="Stream must be enabled")
    
    try:
        app_state = get_app_state()
        provider_manager = app_state["provider_manager"]
        memory_manager = app_state["memory_manager"]
        
        # Get conversation context if using memory
        messages = request.messages
        if request.use_memory and request.session_id:
            context = await memory_manager.get_conversation_context(
                request.session_id,
                max_messages=20
            )
            
            # Add context messages to the beginning
            context_messages = [
                ChatMessage(
                    role=msg["role"],
                    content=msg["content"],
                    metadata=msg.get("metadata")
                )
                for msg in context
            ]
            messages = context_messages + messages
        
        async def generate_stream():
            """Generate streaming response."""
            full_content = ""
            
            async for chunk in provider_manager.generate_completion_stream(
                messages=[msg.dict() for msg in messages],
                provider=request.provider,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ):
                if chunk.get("content"):
                    full_content += chunk["content"]
                
                yield {
                    "event": "data",
                    "data": chunk
                }
            
            # Store in memory after completion
            if request.use_memory and request.session_id and full_content:
                await memory_manager.store_conversation(
                    request.session_id,
                    request.messages[-1].content,
                    full_content,
                    metadata={"provider": request.provider, "model": request.model}
                )
            
            yield {
                "event": "done",
                "data": {"status": "completed"}
            }
        
        return EventSourceResponse(generate_stream())
        
    except Exception as e:
        logger.error(f"Streaming chat completion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    limit: int = 50
) -> List[ChatMessage]:
    """
    Get messages for a session.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of messages to return
        
    Returns:
        List of chat messages
    """
    try:
        app_state = get_app_state()
        memory_manager = app_state["memory_manager"]
        
        context = await memory_manager.get_conversation_context(session_id, limit)
        
        return [
            ChatMessage(
                role=msg["role"],
                content=msg["content"],
                metadata=msg.get("metadata")
            )
            for msg in context
        ]
        
    except Exception as e:
        logger.error(f"Failed to get session messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str) -> Dict[str, str]:
    """
    Clear a conversation session.
    
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
            
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
