# 🚀 Getting Started with BrainBox

Welcome to BrainBox! This guide will get you up and running in under 10 minutes.

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- At least one AI API key (OpenAI, Anthropic, or Google AI)

## 🏁 Quick Start Options

Choose your adventure:

### Option 1: Try the Demo First (Recommended)
**Perfect for:** First-time users who want to see BrainBox in action

### Option 2: Start the API Server
**Perfect for:** Developers who want to integrate BrainBox into their apps

### Option 3: Create a New Project
**Perfect for:** Building a custom AI application from scratch

---

## 🎯 Option 1: Try the Demo (Personal Code Reviewer)

**Step 1: Clone the repository**
```bash
git clone https://github.com/suhas-24/brainbox.git
cd brainbox
```

**Step 2: Set up the code reviewer**
```bash
cd projects/personal-code-reviewer
pip install -r requirements.txt
cp .env.example .env
```

**Step 3: Add your API key**
Edit the `.env` file and add your OpenAI API key:
```bash
# Open .env in your favorite editor
nano .env

# Add your API key:
OPENAI_API_KEY=sk-your-key-here
```

**Step 4: Run the demo**
```bash
python demo.py
```

**What you'll see:**
- BrainBox analyzing Python code
- Detailed feedback on code quality, bugs, and improvements
- Suggestions for better practices

**Try it with your own code:**
```bash
python review.py path/to/your/python/file.py
```

---

## 🏭 Option 2: Start the API Server

**Step 1: Navigate to the server template**
```bash
cd common/project-templates/llm-app-template
pip install -r requirements.txt
cp .env.example .env
```

**Step 2: Configure your API keys**
Edit `.env` and add your preferred AI provider keys:
```bash
# Primary provider
OPENAI_API_KEY=sk-your-openai-key

# Fallback providers (optional but recommended)
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_API_KEY=your-google-key
```

**Step 3: Start the server**
```bash
python server.py
```

**Step 4: Test the API**
- Visit http://localhost:8000/docs for interactive API documentation
- Try the `/health` endpoint to verify everything is working
- Send a chat message to `/api/v1/chat`

**Example API call:**
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, BrainBox!",
    "user_id": "test-user"
  }'
```

---

## 🛠️ Option 3: Create a New Project

**Step 1: Use the project creation script**
```bash
# From the root directory
./scripts/create_project.sh my-ai-app
```

**Step 2: Navigate to your new project**
```bash
cd projects/my-ai-app
pip install -r requirements.txt
cp .env.example .env
```

**Step 3: Configure and customize**
```bash
# Add your API keys to .env
nano .env

# Start building your features
python main.py
```

---

## 🔧 Configuration Details

### Environment Variables
Create a `.env` file in your project directory:

```bash
# AI Provider Keys (add at least one)
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-ai-key

# Optional: Advanced Settings
REDIS_URL=redis://localhost:6379  # For advanced caching
DEFAULT_MODEL=gpt-4o-mini         # Default model to use
MAX_TOKENS=4000                   # Max tokens per request
TEMPERATURE=0.7                   # Response creativity (0.0-1.0)

# Optional: Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000

# Optional: Memory Settings
MEMORY_PROVIDER=file              # or 'redis' for production
MEMORY_MAX_MESSAGES=100           # Max messages to remember
```

### API Key Setup

**OpenAI:**
1. Go to https://platform.openai.com/api-keys
2. Create a new secret key
3. Add it to your `.env` file

**Anthropic:**
1. Go to https://console.anthropic.com/
2. Create an API key
3. Add it to your `.env` file

**Google AI:**
1. Go to https://ai.google.dev/
2. Get an API key
3. Add it to your `.env` file

---

## 🎯 Next Steps

### For Learning:
1. **Explore the code structure** - Check out how the LLM manager works
2. **Try different models** - Change the model in your config
3. **Experiment with prompts** - Modify the prompt templates
4. **Add memory** - See how conversations are remembered

### For Building:
1. **Customize the API** - Add your own endpoints
2. **Add new agents** - Create specialized AI agents
3. **Integrate databases** - Connect to your data sources
4. **Deploy to production** - Use the Docker configurations

### For Advanced Users:
1. **Multi-agent orchestration** - Coordinate multiple AI agents
2. **Vector memory** - Add semantic search capabilities
3. **Custom caching** - Implement Redis caching
4. **Monitoring** - Add usage analytics and logging

---

## 🆘 Troubleshooting

### Common Issues:

**❌ "Module not found" errors**
```bash
pip install -r requirements.txt
# Make sure you're in the right directory
```

**❌ "API key not found" errors**
```bash
# Check your .env file exists and has the right format
cat .env
# Make sure there are no extra spaces around the = sign
```

**❌ "Port already in use" errors**
```bash
# Kill the process using the port
lsof -ti:8000 | xargs kill -9
# Or use a different port
python server.py --port 8001
```

**❌ Rate limiting errors**
```bash
# You're hitting API limits, try:
# 1. Using a different model (cheaper/faster)
# 2. Adding delays between requests
# 3. Upgrading your API plan
```

### Getting Help:

1. **Check the logs** - Look for error messages in the console
2. **Verify your API keys** - Test them with a simple curl command
3. **Check the documentation** - Look at the specific component docs
4. **Open an issue** - Create a GitHub issue with your error details

---

## 📚 Learn More

- **[API Documentation](common/project-templates/llm-app-template/API.md)** - Complete API reference
- **[Architecture Guide](docs/architecture.md)** - How BrainBox works internally
- **[Use Cases](docs/use-cases-and-applications.md)** - Real-world applications
- **[Contributing](CONTRIBUTING.md)** - How to contribute to BrainBox

---

## 🎉 You're Ready!

You now have BrainBox running! Here's what you can do next:

- **Build something cool** - Use the templates as a starting point
- **Share your creation** - Show the community what you built
- **Contribute back** - Help make BrainBox even better

Happy building! 🚀
