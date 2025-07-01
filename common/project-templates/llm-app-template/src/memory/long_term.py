"""
Long-term Persistent Memory for AI Forge

Provides persistent storage for conversation history, user preferences,
and contextual information with SQLite and PostgreSQL backends.
"""

import json
import sqlite3
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False

from ..utils import get_logger


logger = get_logger(__name__)


@dataclass
class LongTermEntry:
    """Long-term memory entry."""
    id: Optional[str] = None
    user_id: str = ""
    session_id: str = ""
    entry_type: str = "conversation"  # conversation, preference, context, fact
    content: str = ""
    metadata: Dict[str, Any] = None
    tags: List[str] = None
    importance: float = 0.5  # 0.0 to 1.0
    created_at: datetime = None
    updated_at: datetime = None
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at


class LongTermMemory:
    """Long-term persistent memory with multiple backend support."""
    
    def __init__(
        self,
        backend: str = "sqlite",
        connection_string: str = "memory.db",
        table_prefix: str = "ai_forge_"
    ):
        self.backend = backend
        self.connection_string = connection_string
        self.table_prefix = table_prefix
        self.logger = get_logger(__name__)
        
        # Connection pools
        self._sqlite_pool = None
        self._postgres_pool = None
        
        # Initialize backend
        asyncio.create_task(self._initialize_backend())
    
    async def _initialize_backend(self):
        """Initialize the selected backend."""
        try:
            if self.backend == "sqlite":
                await self._init_sqlite()
            elif self.backend == "postgresql":
                await self._init_postgresql()
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
            
            self.logger.info(f"Long-term memory initialized with {self.backend} backend")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize long-term memory: {e}")
            raise
    
    async def _init_sqlite(self):
        """Initialize SQLite backend."""
        if not AIOSQLITE_AVAILABLE:
            raise ImportError("aiosqlite required for SQLite backend")
        
        # Ensure directory exists
        db_path = Path(self.connection_string)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables
        async with aiosqlite.connect(self.connection_string) as db:
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_prefix}long_term_memory (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    entry_type TEXT NOT NULL DEFAULT 'conversation',
                    content TEXT NOT NULL,
                    metadata TEXT,
                    tags TEXT,
                    importance REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            
            # Create indexes
            await db.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_user_id 
                ON {self.table_prefix}long_term_memory(user_id)
            """)
            
            await db.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_session_id 
                ON {self.table_prefix}long_term_memory(session_id)
            """)
            
            await db.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_entry_type 
                ON {self.table_prefix}long_term_memory(entry_type)
            """)
            
            await db.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON {self.table_prefix}long_term_memory(created_at)
            """)
            
            await db.commit()
    
    async def _init_postgresql(self):
        """Initialize PostgreSQL backend."""
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg required for PostgreSQL backend")
        
        self._postgres_pool = await asyncpg.create_pool(self.connection_string)
        
        async with self._postgres_pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_prefix}long_term_memory (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    entry_type TEXT NOT NULL DEFAULT 'conversation',
                    content TEXT NOT NULL,
                    metadata JSONB,
                    tags TEXT[],
                    importance REAL DEFAULT 0.5,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE
                )
            """)
            
            # Create indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_user_id 
                ON {self.table_prefix}long_term_memory(user_id)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_session_id 
                ON {self.table_prefix}long_term_memory(session_id)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_entry_type 
                ON {self.table_prefix}long_term_memory(entry_type)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON {self.table_prefix}long_term_memory(created_at)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_metadata 
                ON {self.table_prefix}long_term_memory USING GIN(metadata)
            """)
    
    def _serialize_entry(self, entry: LongTermEntry) -> Dict[str, Any]:
        """Serialize entry for database storage."""
        data = asdict(entry)
        
        if self.backend == "sqlite":
            # SQLite needs JSON serialization
            data["metadata"] = json.dumps(data["metadata"])
            data["tags"] = json.dumps(data["tags"])
            data["created_at"] = data["created_at"].isoformat()
            data["updated_at"] = data["updated_at"].isoformat()
            if data["expires_at"]:
                data["expires_at"] = data["expires_at"].isoformat()
        
        return data
    
    def _deserialize_entry(self, data: Dict[str, Any]) -> LongTermEntry:
        """Deserialize entry from database."""
        if self.backend == "sqlite":
            # SQLite needs JSON deserialization
            data["metadata"] = json.loads(data["metadata"]) if data["metadata"] else {}
            data["tags"] = json.loads(data["tags"]) if data["tags"] else []
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            if data["expires_at"]:
                data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        
        return LongTermEntry(**data)
    
    async def store_entry(self, entry: LongTermEntry) -> str:
        """Store a long-term memory entry."""
        try:
            # Generate ID if not provided
            if not entry.id:
                import uuid
                entry.id = str(uuid.uuid4())
            
            entry.updated_at = datetime.now()
            data = self._serialize_entry(entry)
            
            if self.backend == "sqlite":
                async with aiosqlite.connect(self.connection_string) as db:
                    await db.execute(f"""
                        INSERT OR REPLACE INTO {self.table_prefix}long_term_memory 
                        (id, user_id, session_id, entry_type, content, metadata, tags, 
                         importance, created_at, updated_at, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data["id"], data["user_id"], data["session_id"], 
                        data["entry_type"], data["content"], data["metadata"], 
                        data["tags"], data["importance"], data["created_at"], 
                        data["updated_at"], data["expires_at"]
                    ))
                    await db.commit()
            
            elif self.backend == "postgresql":
                async with self._postgres_pool.acquire() as conn:
                    await conn.execute(f"""
                        INSERT INTO {self.table_prefix}long_term_memory 
                        (id, user_id, session_id, entry_type, content, metadata, tags, 
                         importance, created_at, updated_at, expires_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        tags = EXCLUDED.tags,
                        importance = EXCLUDED.importance,
                        updated_at = EXCLUDED.updated_at,
                        expires_at = EXCLUDED.expires_at
                    """, 
                        entry.id, entry.user_id, entry.session_id, entry.entry_type,
                        entry.content, entry.metadata, entry.tags, entry.importance,
                        entry.created_at, entry.updated_at, entry.expires_at
                    )
            
            self.logger.debug(f"Stored long-term memory entry: {entry.id}")
            return entry.id
            
        except Exception as e:
            self.logger.error(f"Failed to store long-term memory entry: {e}")
            raise
    
    async def get_entry(self, entry_id: str) -> Optional[LongTermEntry]:
        """Get a specific long-term memory entry."""
        try:
            if self.backend == "sqlite":
                async with aiosqlite.connect(self.connection_string) as db:
                    db.row_factory = aiosqlite.Row
                    cursor = await db.execute(f"""
                        SELECT * FROM {self.table_prefix}long_term_memory 
                        WHERE id = ?
                    """, (entry_id,))
                    row = await cursor.fetchone()
                    
                    if row:
                        return self._deserialize_entry(dict(row))
            
            elif self.backend == "postgresql":
                async with self._postgres_pool.acquire() as conn:
                    row = await conn.fetchrow(f"""
                        SELECT * FROM {self.table_prefix}long_term_memory 
                        WHERE id = $1
                    """, entry_id)
                    
                    if row:
                        return LongTermEntry(**dict(row))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get long-term memory entry: {e}")
            return None
    
    async def search_entries(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        entry_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        content_search: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 100,
        offset: int = 0
    ) -> List[LongTermEntry]:
        """Search long-term memory entries."""
        try:
            conditions = []
            params = []
            
            # Build query conditions
            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            
            if entry_type:
                conditions.append("entry_type = ?")
                params.append(entry_type)
            
            if min_importance > 0:
                conditions.append("importance >= ?")
                params.append(min_importance)
            
            if content_search:
                conditions.append("content LIKE ?")
                params.append(f"%{content_search}%")
            
            # Add expiration filter
            conditions.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.now().isoformat())
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            if self.backend == "sqlite":
                async with aiosqlite.connect(self.connection_string) as db:
                    db.row_factory = aiosqlite.Row
                    cursor = await db.execute(f"""
                        SELECT * FROM {self.table_prefix}long_term_memory 
                        WHERE {where_clause}
                        ORDER BY importance DESC, created_at DESC
                        LIMIT ? OFFSET ?
                    """, params + [limit, offset])
                    
                    rows = await cursor.fetchall()
                    return [self._deserialize_entry(dict(row)) for row in rows]
            
            elif self.backend == "postgresql":
                # Adjust query for PostgreSQL
                where_clause = where_clause.replace("?", "${}").format(*range(1, len(params) + 1))
                query = f"""
                    SELECT * FROM {self.table_prefix}long_term_memory 
                    WHERE {where_clause}
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """
                
                async with self._postgres_pool.acquire() as conn:
                    rows = await conn.fetch(query, *params, limit, offset)
                    return [LongTermEntry(**dict(row)) for row in rows]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to search long-term memory: {e}")
            return []
    
    async def delete_entry(self, entry_id: str) -> bool:
        """Delete a long-term memory entry."""
        try:
            if self.backend == "sqlite":
                async with aiosqlite.connect(self.connection_string) as db:
                    cursor = await db.execute(f"""
                        DELETE FROM {self.table_prefix}long_term_memory 
                        WHERE id = ?
                    """, (entry_id,))
                    await db.commit()
                    return cursor.rowcount > 0
            
            elif self.backend == "postgresql":
                async with self._postgres_pool.acquire() as conn:
                    result = await conn.execute(f"""
                        DELETE FROM {self.table_prefix}long_term_memory 
                        WHERE id = $1
                    """, entry_id)
                    return result.split()[-1] != "0"
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete long-term memory entry: {e}")
            return False
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        try:
            now = datetime.now()
            
            if self.backend == "sqlite":
                async with aiosqlite.connect(self.connection_string) as db:
                    cursor = await db.execute(f"""
                        DELETE FROM {self.table_prefix}long_term_memory 
                        WHERE expires_at IS NOT NULL AND expires_at <= ?
                    """, (now.isoformat(),))
                    await db.commit()
                    return cursor.rowcount
            
            elif self.backend == "postgresql":
                async with self._postgres_pool.acquire() as conn:
                    result = await conn.execute(f"""
                        DELETE FROM {self.table_prefix}long_term_memory 
                        WHERE expires_at IS NOT NULL AND expires_at <= $1
                    """, now)
                    return int(result.split()[-1])
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired entries: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            if self.backend == "sqlite":
                async with aiosqlite.connect(self.connection_string) as db:
                    cursor = await db.execute(f"""
                        SELECT 
                            COUNT(*) as total_entries,
                            COUNT(DISTINCT user_id) as unique_users,
                            COUNT(DISTINCT session_id) as unique_sessions,
                            AVG(importance) as avg_importance
                        FROM {self.table_prefix}long_term_memory
                        WHERE expires_at IS NULL OR expires_at > ?
                    """, (datetime.now().isoformat(),))
                    
                    row = await cursor.fetchone()
                    stats = dict(row) if row else {}
            
            elif self.backend == "postgresql":
                async with self._postgres_pool.acquire() as conn:
                    row = await conn.fetchrow(f"""
                        SELECT 
                            COUNT(*) as total_entries,
                            COUNT(DISTINCT user_id) as unique_users,
                            COUNT(DISTINCT session_id) as unique_sessions,
                            AVG(importance) as avg_importance
                        FROM {self.table_prefix}long_term_memory
                        WHERE expires_at IS NULL OR expires_at > $1
                    """, datetime.now())
                    
                    stats = dict(row) if row else {}
            
            return {
                "backend": self.backend,
                "total_entries": stats.get("total_entries", 0),
                "unique_users": stats.get("unique_users", 0),
                "unique_sessions": stats.get("unique_sessions", 0),
                "average_importance": float(stats.get("avg_importance", 0)) if stats.get("avg_importance") else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {"backend": self.backend, "error": str(e)}
    
    async def close(self):
        """Close database connections."""
        try:
            if self._postgres_pool:
                await self._postgres_pool.close()
            
            self.logger.info("Long-term memory connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")


# Convenience functions
async def store_conversation(
    user_id: str,
    session_id: str,
    content: str,
    importance: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Store a conversation entry."""
    memory = LongTermMemory()
    entry = LongTermEntry(
        user_id=user_id,
        session_id=session_id,
        entry_type="conversation",
        content=content,
        importance=importance,
        metadata=metadata or {}
    )
    return await memory.store_entry(entry)


async def store_user_preference(
    user_id: str,
    preference_name: str,
    preference_value: Any,
    importance: float = 0.8
) -> str:
    """Store a user preference."""
    memory = LongTermMemory()
    entry = LongTermEntry(
        user_id=user_id,
        entry_type="preference",
        content=f"{preference_name}: {preference_value}",
        importance=importance,
        metadata={"preference_name": preference_name, "preference_value": preference_value}
    )
    return await memory.store_entry(entry)


async def get_user_history(
    user_id: str,
    limit: int = 50,
    min_importance: float = 0.3
) -> List[LongTermEntry]:
    """Get user's conversation history."""
    memory = LongTermMemory()
    return await memory.search_entries(
        user_id=user_id,
        entry_type="conversation",
        min_importance=min_importance,
        limit=limit
    )
