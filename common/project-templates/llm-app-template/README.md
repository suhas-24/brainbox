# BrainBox - Complete AI Intelligence Framework

A production-ready, feature-rich framework for building sophisticated AI applications with intelligent memory, multi-provider support, and enterprise-grade infrastructure. Your complete AI brain in a box.

## 🎯 Overview

BrainBox provides a robust foundation for developing AI applications with enterprise-grade features including intelligent memory systems, multi-agent coordination, semantic search, rate limiting, caching, and comprehensive API infrastructure. Built with FastAPI and designed for production deployment.

## 🏗️ Architecture

```
src/
├── api/                    # FastAPI application layer
│   ├── app.py             # Application factory with lifecycle management
│   ├── middleware.py      # CORS, compression, logging, security
│   ├── routes.py          # Route configuration
│   └── endpoints/         # API endpoint implementations
│       ├── chat.py        # Chat completions with streaming
│       ├── health.py      # Health checks and monitoring
│       ├── providers.py   # Provider management
│       ├── memory.py      # Memory operations
│       └── stats.py       # Usage statistics
├── providers/             # Multi-provider LLM integration
│   ├── base.py           # Abstract provider interface
│   ├── openai.py         # OpenAI GPT integration
│   ├── anthropic.py      # Anthropic Claude integration
│   ├── google.py         # Google Gemini integration
│   └── manager.py        # Provider management and routing
├── memory/               # Advanced memory management
│   ├── short_term.py     # Conversational memory
│   ├── long_term.py      # Persistent memory (SQLite/PostgreSQL)
│   ├── vector_memory.py  # Semantic search with embeddings
│   └── memory_manager.py # Unified memory orchestration
├── agents/               # Multi-agent systems
│   ├── base_agent.py     # Base agent implementation
│   ├── coordinator.py    # Agent coordination
│   └── specialized/      # Specialized agent implementations
├── utils/                # Core utilities
│   ├── logging.py        # Advanced logging system
│   ├── cache.py          # Multi-backend caching
│   ├── rate_limit.py     # Rate limiting strategies
│   └── tokens.py         # Token counting and cost estimation
├── config/               # Configuration management
└── examples/             # Usage examples and demos
```

## 🌟 Features

### Core Infrastructure
- **FastAPI Web Framework**: Production-ready REST API with automatic OpenAPI documentation
- **Multi-Provider LLM Support**: OpenAI, Anthropic, Google with unified interface and fallbacks
- **Advanced Memory Systems**: Short-term, long-term persistent, and vector-based semantic memory
- **Multi-Agent Architecture**: Coordinated AI agents with specialized roles and workflows

### Production Features
- **Caching System**: Memory, Redis, and file-based caching with TTL, compression, and LRU eviction
- **Rate Limiting**: Token bucket, sliding window strategies with Redis and memory backends
- **Token Management**: Accurate counting and cost estimation for all LLM providers
- **Health Monitoring**: Comprehensive health checks, readiness probes, and performance metrics
- **Request Middleware**: CORS, GZip compression, security headers, request logging with UUIDs

### Advanced Capabilities
- **Vector Memory**: Semantic search using ChromaDB, Pinecone, or in-memory with embeddings
- **Long-term Storage**: Persistent conversation history and user preferences with SQLite/PostgreSQL
- **Usage Analytics**: Detailed tracking of token usage, costs, and performance metrics
- **Streaming Support**: Server-sent events for real-time LLM response streaming
- **Memory Optimization**: Intelligent cleanup, expiration, and memory management

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd brainbox-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
```

### Configuration

Edit `.env` with your API keys and settings:

```bash
# LLM Provider APIs
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Application Settings
BRAINBOX_ENVIRONMENT=development
BRAINBOX_LOG_LEVEL=info

# Database (optional)
DATABASE_URL=sqlite:///./memory.db
# DATABASE_URL=postgresql://user:pass@localhost/brainbox

# Redis (optional, for distributed caching/rate limiting)
REDIS_URL=redis://localhost:6379

# CORS Settings
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]
```

### Running the Application

```bash
# Start the FastAPI server
python server.py

# Or with custom settings
python server.py --host 0.0.0.0 --port 8000 --env production

# For development with auto-reload
python server.py --reload --log-level debug
```

The API will be available at:
- **Application**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **Alternative Docs**: http://127.0.0.1:8000/redoc

## 📖 Usage Examples

### Basic Chat Completion

```python
import httpx
import asyncio

async def chat_example():
    async with httpx.AsyncClient() as client:
        response = await client.post("http://127.0.0.1:8000/api/v1/chat/completions", json={
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "provider": "openai",
            "model": "gpt-4",
            "session_id": "user-123",
            "use_memory": True
        })
        
        result = response.json()
        print(f"Response: {result['content']}")

asyncio.run(chat_example())
```

### Streaming Chat

```python
import httpx
import asyncio

async def streaming_example():
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "http://127.0.0.1:8000/api/v1/chat/stream", json={
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "stream": True,
            "session_id": "user-123"
        }) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    print(line[6:])  # Remove "data: " prefix

asyncio.run(streaming_example())
```

### Using Memory Features

```python
from src.memory import MemoryManager

async def memory_example():
    memory = MemoryManager()
    
    # Store enhanced conversation with all memory systems
    await memory.store_conversation_enhanced(
        session_id="session-123",
        user_message="I love pizza",
        assistant_response="That's great! Pizza is delicious.",
        user_id="user-456",
        importance=0.8,
        metadata={"topic": "food preferences"}
    )
    
    # Search for similar conversations
    similar = await memory.search_similar_conversations(
        query="food I like",
        user_id="user-456",
        limit=5,
        min_score=0.7
    )
    
    # Get user preferences
    preferences = await memory.get_user_preferences("user-456")
    print(f"User preferences: {preferences}")

asyncio.run(memory_example())
```

### Multi-Agent System

```python
from src.agents import AgentCoordinator, SpecializedAgent

async def agent_example():
    # Create specialized agents
    researcher = SpecializedAgent(
        name="researcher",
        role="Research and gather information",
        instructions="You are a research specialist..."
    )
    
    writer = SpecializedAgent(
        name="writer",
        role="Write and edit content",
        instructions="You are a content writer..."
    )
    
    # Create coordinator
    coordinator = AgentCoordinator([researcher, writer])
    
    # Execute coordinated workflow
    result = await coordinator.execute_workflow(
        task="Write a blog post about AI",
        workflow=[
            {"agent": "researcher", "task": "Research AI trends"},
            {"agent": "writer", "task": "Write blog post using research"}
        ]
    )
    
    print(f"Final result: {result}")

asyncio.run(agent_example())
```

## 🔧 Configuration

### Application Configuration (`config/app.yaml`)

```yaml
app:
  name: "BrainBox Application"
  version: "1.0.0"
  environment: "development"
  
api:
  host: "127.0.0.1"
  port: 8000
  workers: 1
  
memory:
  short_term:
    max_sessions: 1000
    max_messages_per_session: 100
  long_term:
    backend: "sqlite"  # or "postgresql"
    connection_string: "memory.db"
  vector:
    backend: "memory"  # or "chromadb"
    embedding_provider: "sentence_transformers"
    
cache:
  backend: "memory"  # or "redis", "file"
  default_ttl: 3600
  max_size: 1000
  
rate_limiting:
  enabled: true
  strategy: "token_bucket"
  api_limit: 100  # requests per minute
  llm_limit: 10   # requests per minute
```

### Model Configuration (`config/models.yaml`)

```yaml
providers:
  openai:
    api_key_env: "OPENAI_API_KEY"
    base_url: "https://api.openai.com/v1"
    default_model: "gpt-4"
    max_tokens: 4096
    temperature: 0.7
    timeout: 30
    
  anthropic:
    api_key_env: "ANTHROPIC_API_KEY"
    default_model: "claude-3-sonnet-20240229"
    max_tokens: 4096
    temperature: 0.7
    
models:
  fast:
    provider: "openai"
    model: "gpt-3.5-turbo"
    temperature: 0.5
    
  powerful:
    provider: "anthropic"
    model: "claude-3-opus-20240229"
    temperature: 0.7
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/api/

# Run performance tests
pytest tests/performance/ -v
```

## 📊 API Documentation

### Core Endpoints

- `POST /api/v1/chat/completions` - Generate chat completions
- `POST /api/v1/chat/stream` - Stream chat completions
- `GET /api/v1/chat/sessions/{session_id}/messages` - Get conversation history
- `DELETE /api/v1/chat/sessions/{session_id}` - Clear session

### Health & Monitoring

- `GET /api/v1/health/` - Comprehensive health check
- `GET /api/v1/health/ready` - Readiness probe
- `GET /api/v1/health/live` - Liveness probe

### Provider Management

- `GET /api/v1/providers/` - List all providers
- `GET /api/v1/providers/{name}` - Get provider details
- `GET /api/v1/providers/{name}/health` - Provider health check

### Memory Operations

- `GET /api/v1/memory/stats` - Memory usage statistics
- `GET /api/v1/memory/sessions` - List active sessions
- `DELETE /api/v1/memory/sessions/{session_id}` - Clear session memory

### Statistics

- `GET /api/v1/stats/usage` - Usage statistics
- `GET /api/v1/stats/performance` - Performance metrics
- `GET /api/v1/stats/memory` - Memory statistics

For complete API documentation, visit `/docs` when running the application.

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000", "--env", "production"]
```

```bash
# Build and run
docker build -t brainbox-app .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key brainbox-app
```

### Production Considerations

1. **Environment Variables**: Set production API keys and database URLs
2. **Database**: Use PostgreSQL for better performance and concurrency
3. **Caching**: Configure Redis for distributed caching and rate limiting
4. **Logging**: Set up centralized logging (ELK stack, CloudWatch, etc.)
5. **Monitoring**: Implement APM and metrics collection
6. **Security**: Add authentication, HTTPS, and input validation
7. **Scaling**: Use multiple workers and load balancers

### Example Production Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - BRAINBOX_ENVIRONMENT=production
      - DATABASE_URL=postgresql://user:pass@db:5432/brainbox
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
      
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: brainbox
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7-alpine
    
volumes:
  postgres_data:
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Format code: `black src/` and `flake8 src/`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📋 Roadmap

- [ ] **Security**: Authentication, authorization, and API keys
- [ ] **Guardrails**: Content filtering, PII detection, safety checks
- [ ] **Multimodal**: Vision and audio processing capabilities
- [ ] **Plugins**: Extensible plugin system
- [ ] **Dashboard**: Web-based management interface
- [ ] **Observability**: Distributed tracing and advanced metrics
- [ ] **Auto-scaling**: Dynamic resource allocation
- [ ] **MLOps**: Model deployment and monitoring pipelines

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of excellent open-source projects including FastAPI, ChromaDB, and sentence-transformers
- Inspired by the LangChain ecosystem and modern AI engineering practices
- Community feedback and contributions that help improve the framework

## 📞 Support

- **Documentation**: Comprehensive guides in `/docs`
- **Examples**: Working examples in `/examples`
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community support

---

**BrainBox** - Building the future of AI applications, one brain at a time. 🧠
