# AI Forge API Documentation

This document provides comprehensive documentation for the AI Forge FastAPI application, including endpoints, authentication, request/response formats, and usage examples.

## Quick Start

### Starting the Server

```bash
# Development mode with auto-reload
python server.py --reload --log-level debug

# Production mode
python server.py --host 0.0.0.0 --port 8000 --env production

# Custom configuration
python server.py --host 127.0.0.1 --port 3000 --log-level info
```

### Server Options

- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 8000)  
- `--reload`: Enable auto-reload for development
- `--log-level`: Set logging level (debug, info, warning, error, critical)
- `--env`: Environment (development, staging, production)

## API Overview

The AI Forge API provides RESTful endpoints for:

- **Chat Completions**: LLM chat interactions with streaming support
- **Health Monitoring**: Application and component health checks
- **Provider Management**: LLM provider configuration and status
- **Memory Operations**: Conversation memory and working memory management
- **Statistics**: Usage, performance, and memory statistics

## Base URL

- Development: `http://127.0.0.1:8000`
- API Base: `/api/v1`
- Documentation: `/docs` (Swagger UI)
- Alternative Docs: `/redoc` (ReDoc)

## Authentication

Currently, the API does not implement authentication. In production deployments, you should add:

- API key authentication
- OAuth 2.0 / JWT tokens
- Rate limiting
- CORS configuration

## Endpoints

### Root Endpoint

```http
GET /
```

Returns basic API information.

**Response:**
```json
{
  "name": "AI Forge",
  "description": "Advanced LLM Application Framework", 
  "version": "1.0.0",
  "docs": "/docs",
  "api": "/api/v1"
}
```

### Chat Completions

#### Create Chat Completion

```http
POST /api/v1/chat/completions
```

Generate a chat completion using configured LLM providers.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?",
      "metadata": {}
    }
  ],
  "provider": "openai",
  "model": "gpt-4",
  "session_id": "session-123",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false,
  "use_memory": true
}
```

**Response:**
```json
{
  "id": "response-456",
  "content": "Hello! I'm doing well, thank you for asking.",
  "model": "gpt-4",
  "provider": "openai",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 11,
    "total_tokens": 23
  },
  "created": "2024-01-15T10:30:00Z",
  "session_id": "session-123"
}
```

#### Create Streaming Chat Completion

```http
POST /api/v1/chat/stream
```

Generate a streaming chat completion with Server-Sent Events.

**Request:** Same as above with `"stream": true`

**Response:** Server-Sent Events stream
```
event: data
data: {"content": "Hello", "delta": true}

event: data  
data: {"content": "! I'm", "delta": true}

event: done
data: {"status": "completed"}
```

#### Get Session Messages

```http
GET /api/v1/chat/sessions/{session_id}/messages?limit=50
```

Retrieve conversation history for a session.

**Response:**
```json
[
  {
    "role": "user",
    "content": "Hello",
    "metadata": {}
  },
  {
    "role": "assistant", 
    "content": "Hi there!",
    "metadata": {"provider": "openai", "model": "gpt-4"}
  }
]
```

#### Clear Session

```http
DELETE /api/v1/chat/sessions/{session_id}
```

Clear conversation history for a session.

**Response:**
```json
{
  "status": "success",
  "message": "Session session-123 cleared"
}
```

### Health Monitoring

#### Health Check

```http
GET /api/v1/health/
```

Get comprehensive application health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime": 3600.0,
  "components": {
    "providers": {
      "status": "healthy",
      "details": {
        "healthy": true,
        "providers": {
          "openai": {"healthy": true, "latency": 120.5}
        }
      }
    },
    "memory": {
      "status": "healthy", 
      "details": {
        "sessions": 5,
        "messages": 150
      }
    }
  }
}
```

#### Readiness Check

```http
GET /api/v1/health/ready
```

Check if application is ready to serve requests.

**Response:**
```json
{
  "ready": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "checks": {
    "app_state": true,
    "provider_manager": true,
    "memory_manager": true,
    "config": true,
    "providers_ready": true
  }
}
```

#### Liveness Check

```http
GET /api/v1/health/live
```

Simple liveness probe.

**Response:**
```json
{
  "status": "alive",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Provider Management

#### List Providers

```http
GET /api/v1/providers/
```

Get information about all configured providers.

**Response:**
```json
[
  {
    "name": "openai",
    "enabled": true,
    "models": ["gpt-4", "gpt-3.5-turbo"],
    "health": "healthy",
    "config": {
      "max_tokens": 4096,
      "timeout": 30,
      "rate_limit": 60
    }
  }
]
```

#### Get Provider

```http
GET /api/v1/providers/{provider_name}
```

Get detailed information about a specific provider.

#### Get Provider Health

```http
GET /api/v1/providers/{provider_name}/health
```

**Response:**
```json
{
  "name": "openai",
  "healthy": true,
  "latency": 120.5,
  "error": null
}
```

#### Get Provider Models

```http
GET /api/v1/providers/{provider_name}/models
```

**Response:**
```json
["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
```

#### Enable/Disable Provider

```http
POST /api/v1/providers/{provider_name}/enable
POST /api/v1/providers/{provider_name}/disable
```

### Memory Management

#### Get Memory Statistics

```http
GET /api/v1/memory/stats
```

**Response:**
```json
{
  "short_term_sessions": 10,
  "short_term_messages": 500,
  "long_term_entries": 0,
  "vector_embeddings": 0,
  "total_memory_mb": 45.2
}
```

#### List Sessions

```http
GET /api/v1/memory/sessions
```

**Response:**
```json
["session-123", "session-456", "session-789"]
```

#### Clear Session

```http
DELETE /api/v1/memory/sessions/{session_id}
```

#### Clear All Sessions

```http
DELETE /api/v1/memory/sessions
```

#### Working Memory Operations

```http
GET /api/v1/memory/sessions/{session_id}/working
GET /api/v1/memory/sessions/{session_id}/working?key=specific_key
POST /api/v1/memory/sessions/{session_id}/working
DELETE /api/v1/memory/sessions/{session_id}/working?key=specific_key
```

### Statistics

#### Usage Statistics

```http
GET /api/v1/stats/usage
```

**Response:**
```json
{
  "total_requests": 1000,
  "total_tokens": 250000,
  "average_latency": 120.5,
  "provider_breakdown": {
    "openai": {
      "requests": 600,
      "tokens": 150000
    }
  }
}
```

#### Performance Statistics

```http
GET /api/v1/stats/performance
```

#### Memory Statistics

```http
GET /api/v1/stats/memory
```

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include details:

```json
{
  "error": "Internal server error",
  "detail": "An unexpected error occurred"
}
```

## Request Headers

All requests should include:

```
Content-Type: application/json
Accept: application/json
```

For streaming endpoints:
```
Accept: text/event-stream
```

## Rate Limiting

Rate limiting is not currently implemented but can be added using middleware. Recommended limits:

- Chat completions: 60 requests/minute per IP
- Other endpoints: 300 requests/minute per IP

## CORS

CORS is configured based on the `CORS_ORIGINS` environment variable. In development, all origins are allowed.

## Monitoring and Observability

The API includes:

- Request logging with unique request IDs
- Performance metrics collection
- Health check endpoints for monitoring
- Structured logging with correlation IDs

## Development

### Running in Development

```bash
python server.py --reload --log-level debug
```

### API Documentation

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run API tests
pytest tests/test_api.py -v
```

## Deployment

### Production Considerations

1. **Security**: Add authentication, HTTPS, rate limiting
2. **Scaling**: Use multiple workers, load balancer
3. **Monitoring**: Add APM, metrics collection
4. **Configuration**: Use environment-specific configs

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Key environment variables for production:

```bash
AI_FORGE_ENVIRONMENT=production
AI_FORGE_LOG_LEVEL=info
CORS_ORIGINS=https://yourdomain.com
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## SDK and Client Libraries

Consider creating client SDKs for popular languages:

- Python client using `httpx`
- JavaScript/TypeScript client  
- Go client
- REST API postman collection

## Changelog

- **v1.0.0**: Initial API implementation with chat, health, providers, memory, and stats endpoints
