"""
Vector Memory with Embeddings for AI Forge

Provides semantic search capabilities using vector embeddings with support
for multiple backends including Chroma, Pinecone, and in-memory storage.
"""

import json
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..utils import get_logger


logger = get_logger(__name__)


@dataclass
class VectorEntry:
    """Vector memory entry with embeddings."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SearchResult:
    """Vector search result."""
    entry: VectorEntry
    score: float
    distance: float


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        raise NotImplementedError
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required for OpenAI embeddings")
        
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self._dimension = 1536  # Default for ada-002
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers package required")
        
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using sentence transformers."""
        try:
            # Run in thread pool since sentence transformers is not async
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                self.model.encode, 
                text
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Sentence transformer embedding generation failed: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        return self._dimension


class VectorBackend:
    """Base class for vector storage backends."""
    
    async def store(self, entries: List[VectorEntry]) -> bool:
        """Store vector entries."""
        raise NotImplementedError
    
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        raise NotImplementedError
    
    async def delete(self, entry_ids: List[str]) -> bool:
        """Delete vector entries."""
        raise NotImplementedError
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        raise NotImplementedError


class ChromaBackend(VectorBackend):
    """ChromaDB vector backend."""
    
    def __init__(
        self,
        collection_name: str = "ai_forge_vectors",
        persist_directory: str = "./chroma_db",
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb package required for Chroma backend")
        
        self.collection_name = collection_name
        
        if host and port:
            # Remote Chroma server
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Local persistent client
            self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ChromaDB backend initialized: {collection_name}")
    
    async def store(self, entries: List[VectorEntry]) -> bool:
        """Store entries in ChromaDB."""
        try:
            ids = [entry.id for entry in entries]
            embeddings = [entry.embedding for entry in entries]
            documents = [entry.content for entry in entries]
            metadatas = []
            
            for entry in entries:
                metadata = entry.metadata.copy()
                metadata.update({
                    "user_id": entry.user_id,
                    "session_id": entry.session_id,
                    "timestamp": entry.timestamp.isoformat()
                })
                metadatas.append(metadata)
            
            # Run in thread pool since ChromaDB is not async
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.collection.upsert,
                ids,
                embeddings,
                metadatas,
                documents
            )
            
            logger.debug(f"Stored {len(entries)} entries in ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store entries in ChromaDB: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in ChromaDB."""
        try:
            # Build where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = filter_metadata
            
            # Run search in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=where_clause,
                    include=["documents", "metadatas", "distances", "embeddings"]
                )
            )
            
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    entry = VectorEntry(
                        id=doc_id,
                        content=results["documents"][0][i],
                        embedding=results["embeddings"][0][i] if "embeddings" in results else None,
                        metadata=results["metadatas"][0][i],
                        user_id=results["metadatas"][0][i].get("user_id"),
                        session_id=results["metadatas"][0][i].get("session_id"),
                        timestamp=datetime.fromisoformat(results["metadatas"][0][i].get("timestamp"))
                    )
                    
                    distance = results["distances"][0][i]
                    score = 1.0 - distance  # Convert distance to similarity score
                    
                    search_results.append(SearchResult(
                        entry=entry,
                        score=score,
                        distance=distance
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    async def delete(self, entry_ids: List[str]) -> bool:
        """Delete entries from ChromaDB."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.collection.delete,
                entry_ids
            )
            
            logger.debug(f"Deleted {len(entry_ids)} entries from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete entries from ChromaDB: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(
                None,
                self.collection.count
            )
            
            return {
                "backend": "chromadb",
                "collection_name": self.collection_name,
                "total_vectors": count,
                "status": "connected"
            }
            
        except Exception as e:
            logger.error(f"Failed to get ChromaDB stats: {e}")
            return {"backend": "chromadb", "error": str(e)}


class InMemoryBackend(VectorBackend):
    """In-memory vector backend with cosine similarity."""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.entries: Dict[str, VectorEntry] = {}
        self.embeddings_matrix = None
        self.entry_ids = []
        
        logger.info(f"In-memory vector backend initialized (max: {max_entries})")
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a_np = np.array(a)
            b_np = np.array(b)
            
            dot_product = np.dot(a_np, b_np)
            norm_a = np.linalg.norm(a_np)
            norm_b = np.linalg.norm(b_np)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception:
            return 0.0
    
    def _rebuild_matrix(self):
        """Rebuild embeddings matrix for efficient search."""
        if not self.entries:
            self.embeddings_matrix = None
            self.entry_ids = []
            return
        
        embeddings = []
        self.entry_ids = []
        
        for entry_id, entry in self.entries.items():
            if entry.embedding:
                embeddings.append(entry.embedding)
                self.entry_ids.append(entry_id)
        
        if embeddings:
            self.embeddings_matrix = np.array(embeddings)
        else:
            self.embeddings_matrix = None
    
    async def store(self, entries: List[VectorEntry]) -> bool:
        """Store entries in memory."""
        try:
            for entry in entries:
                self.entries[entry.id] = entry
                
                # Evict oldest entries if over limit
                if len(self.entries) > self.max_entries:
                    # Sort by timestamp and remove oldest
                    sorted_entries = sorted(
                        self.entries.items(),
                        key=lambda x: x[1].timestamp
                    )
                    entries_to_remove = len(self.entries) - self.max_entries
                    
                    for i in range(entries_to_remove):
                        del self.entries[sorted_entries[i][0]]
            
            self._rebuild_matrix()
            
            logger.debug(f"Stored {len(entries)} entries in memory")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store entries in memory: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in memory."""
        try:
            if not self.embeddings_matrix or len(self.entry_ids) == 0:
                return []
            
            # Calculate similarities
            query_np = np.array(query_embedding)
            
            # Normalize query vector
            query_norm = np.linalg.norm(query_np)
            if query_norm == 0:
                return []
            query_np = query_np / query_norm
            
            # Normalize stored embeddings
            norms = np.linalg.norm(self.embeddings_matrix, axis=1)
            normalized_embeddings = self.embeddings_matrix / norms[:, np.newaxis]
            
            # Calculate cosine similarities
            similarities = np.dot(normalized_embeddings, query_np)
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:limit * 2]  # Get more for filtering
            
            search_results = []
            
            for idx in top_indices:
                if len(search_results) >= limit:
                    break
                
                entry_id = self.entry_ids[idx]
                entry = self.entries[entry_id]
                similarity = similarities[idx]
                
                # Apply metadata filtering
                if filter_metadata:
                    match = True
                    for key, value in filter_metadata.items():
                        if key == "user_id" and entry.user_id != value:
                            match = False
                            break
                        elif key == "session_id" and entry.session_id != value:
                            match = False
                            break
                        elif entry.metadata.get(key) != value:
                            match = False
                            break
                    
                    if not match:
                        continue
                
                distance = 1.0 - similarity
                
                search_results.append(SearchResult(
                    entry=entry,
                    score=float(similarity),
                    distance=float(distance)
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"In-memory search failed: {e}")
            return []
    
    async def delete(self, entry_ids: List[str]) -> bool:
        """Delete entries from memory."""
        try:
            deleted_count = 0
            
            for entry_id in entry_ids:
                if entry_id in self.entries:
                    del self.entries[entry_id]
                    deleted_count += 1
            
            if deleted_count > 0:
                self._rebuild_matrix()
            
            logger.debug(f"Deleted {deleted_count} entries from memory")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete entries from memory: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get in-memory backend statistics."""
        return {
            "backend": "memory",
            "total_vectors": len(self.entries),
            "max_entries": self.max_entries,
            "memory_usage_mb": len(self.entries) * 1536 * 4 / (1024 * 1024)  # Rough estimate
        }


class VectorMemory:
    """Vector memory with embeddings and semantic search."""
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_backend: Optional[VectorBackend] = None
    ):
        self.logger = get_logger(__name__)
        
        # Initialize embedding provider
        if embedding_provider:
            self.embedding_provider = embedding_provider
        else:
            # Default to sentence transformers if available, otherwise OpenAI
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_provider = SentenceTransformerProvider()
            elif OPENAI_AVAILABLE:
                self.embedding_provider = OpenAIEmbeddingProvider()
            else:
                raise ImportError("No embedding provider available")
        
        # Initialize vector backend
        if vector_backend:
            self.vector_backend = vector_backend
        else:
            # Default to ChromaDB if available, otherwise in-memory
            if CHROMADB_AVAILABLE:
                self.vector_backend = ChromaBackend()
            else:
                self.vector_backend = InMemoryBackend()
        
        self.logger.info(f"Vector memory initialized with {type(self.embedding_provider).__name__} and {type(self.vector_backend).__name__}")
    
    def _generate_id(self, content: str, user_id: str = None) -> str:
        """Generate unique ID for content."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if user_id:
            return f"{user_id}_{content_hash}"
        return content_hash
    
    async def store_text(
        self,
        content: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None
    ) -> str:
        """Store text with generated embeddings."""
        try:
            # Generate embedding
            embedding = await self.embedding_provider.generate_embedding(content)
            
            # Generate ID if not provided
            if not entry_id:
                entry_id = self._generate_id(content, user_id)
            
            # Create entry
            entry = VectorEntry(
                id=entry_id,
                content=content,
                embedding=embedding,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {}
            )
            
            # Store in backend
            success = await self.vector_backend.store([entry])
            
            if success:
                self.logger.debug(f"Stored vector entry: {entry_id}")
                return entry_id
            else:
                raise Exception("Backend storage failed")
                
        except Exception as e:
            self.logger.error(f"Failed to store text: {e}")
            raise
    
    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        min_score: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for semantically similar content."""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_provider.generate_embedding(query)
            
            # Build metadata filter
            filter_dict = {}
            if user_id:
                filter_dict["user_id"] = user_id
            if session_id:
                filter_dict["session_id"] = session_id
            if metadata_filter:
                filter_dict.update(metadata_filter)
            
            # Search in backend
            results = await self.vector_backend.search(
                query_embedding=query_embedding,
                limit=limit * 2,  # Get more for score filtering
                filter_metadata=filter_dict if filter_dict else None
            )
            
            # Filter by minimum score
            filtered_results = [
                result for result in results
                if result.score >= min_score
            ]
            
            # Return top results
            return filtered_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    async def delete_entries(self, entry_ids: List[str]) -> bool:
        """Delete vector entries."""
        try:
            return await self.vector_backend.delete(entry_ids)
        except Exception as e:
            self.logger.error(f"Failed to delete entries: {e}")
            return False
    
    async def get_similar_conversations(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        min_score: float = 0.7
    ) -> List[SearchResult]:
        """Get similar past conversations for a user."""
        return await self.search_similar(
            query=query,
            limit=limit,
            user_id=user_id,
            min_score=min_score,
            metadata_filter={"type": "conversation"}
        )
    
    async def store_conversation(
        self,
        user_message: str,
        assistant_response: str,
        user_id: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Store a conversation exchange."""
        try:
            base_metadata = {"type": "conversation"}
            if metadata:
                base_metadata.update(metadata)
            
            # Store user message
            user_entry_id = await self.store_text(
                content=f"User: {user_message}",
                user_id=user_id,
                session_id=session_id,
                metadata={**base_metadata, "role": "user"}
            )
            
            # Store assistant response
            assistant_entry_id = await self.store_text(
                content=f"Assistant: {assistant_response}",
                user_id=user_id,
                session_id=session_id,
                metadata={**base_metadata, "role": "assistant"}
            )
            
            return user_entry_id, assistant_entry_id
            
        except Exception as e:
            self.logger.error(f"Failed to store conversation: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector memory statistics."""
        try:
            backend_stats = await self.vector_backend.get_stats()
            
            stats = {
                "embedding_provider": type(self.embedding_provider).__name__,
                "embedding_dimension": self.embedding_provider.dimension,
                "vector_backend": backend_stats
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get vector memory stats: {e}")
            return {"error": str(e)}


# Global vector memory instance
_vector_memory: Optional[VectorMemory] = None


def get_vector_memory(
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_backend: Optional[VectorBackend] = None
) -> VectorMemory:
    """Get global vector memory instance."""
    global _vector_memory
    
    if _vector_memory is None:
        _vector_memory = VectorMemory(embedding_provider, vector_backend)
    
    return _vector_memory


async def setup_vector_memory(
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_backend: Optional[VectorBackend] = None
) -> VectorMemory:
    """Setup global vector memory."""
    global _vector_memory
    _vector_memory = VectorMemory(embedding_provider, vector_backend)
    return _vector_memory
