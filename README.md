# 🧠 BrainBox

**Production-ready AI application framework that acts like a complete AI brain in a box.**

[![GitHub](https://img.shields.io/github/license/suhas-24/brainbox)](https://github.com/suhas-24/brainbox)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

## 🚀 What Is BrainBox?

BrainBox is a **complete AI application framework** that provides:
- 🧠 **Intelligent Memory** - Remembers conversations across sessions
- 🔄 **Multi-Provider Support** - OpenAI, Anthropic, Google with automatic fallback
- 🏭 **Production-Ready API** - FastAPI server with comprehensive endpoints
- ⚡ **Advanced Caching** - Memory and Redis support for performance
- 🛡️ **Rate Limiting** - Built-in protection and usage control
- 💰 **Real-time Cost Tracking** - Monitor token usage and costs
- 🔍 **Semantic Search** - Vector memory for intelligent context retrieval
- 📊 **Usage Analytics** - Track performance and optimize usage
- 🤖 **Multi-Agent Coordination** - Orchestrate multiple AI agents

## ⚡ Quick Start (2 minutes)

**1. Clone and Setup:**
```bash
git clone https://github.com/suhas-24/brainbox.git
cd brainbox
pip install -r common/project-templates/llm-app-template/requirements.txt
```

**2. Try the Personal Code Reviewer:**
```bash
cd projects/personal-code-reviewer
cp .env.example .env
# Add your OpenAI API key to .env
python demo.py
```

**3. Start the Production API Server:**
```bash
# From the root directory
cd common/project-templates/llm-app-template
cp .env.example .env
# Add your API keys to .env
python server.py
# Visit http://localhost:8000/docs for API documentation
```

## What You Get When You Clone This

### 1. **Project Templates** (Ready to Use)
- **Basic LLM App**: Simple chatbot/AI assistant setup
- **Advanced LLM App**: Multi-agent system with memory
- Pre-configured for OpenAI, Anthropic, Google AI

### 2. **Working Examples**
- **Personal Code Reviewer**: Actually useful tool that reviews your code
- Shows you how all the pieces fit together
- You can use it immediately or learn from it

### 3. **Common Components** (Copy-Paste Ready)
- LLM integration with fallback (when one API fails, try another)
- Memory management (AI remembers previous conversations)
- Configuration management (easy API key setup)
- Error handling and logging

## Real Use Cases

### For Individual Developers:
- **Build a personal AI assistant** for your coding workflow
- **Create custom chatbots** for specific tasks
- **Add AI features** to existing projects quickly

### For Learning:
- See how production AI apps are structured
- Learn patterns for multi-agent systems
- Understand context engineering and memory management

### For Work Projects:
- **Customer support bots** that remember context
- **Document analysis tools** for your industry
- **Code review automation** for your team

## Quick Start (5 minutes)

1. **Clone it:**
   ```bash
   git clone [this-repo]
   cd ai-forge-workspace
   ```

2. **Try the working example:**
   ```bash
   cd projects/personal-code-reviewer
   python3 demo.py
   ```

3. **Create your own project:**
   ```bash
   cd ../../
   ./scripts/create_project.sh my-ai-app
   cd projects/my-ai-app
   # Add your API key to .env
   # Start building your features
   ```

## What Makes This Different

**Not just tutorials or docs** - these are actual working templates you can build on.

**Pre-solved common problems:**
- ✅ Multiple LLM providers with fallback
- ✅ Memory management for conversations
- ✅ Error handling and monitoring
- ✅ Configuration management
- ✅ Project structure that scales

**You focus on your unique features, not infrastructure.**

## Example: What You Can Build Quickly

With these templates, you can build:

- **Smart document analyzer** (upload PDF, ask questions about it)
- **Code review bot** (analyze code for bugs and improvements)  
- **Personal research assistant** (search web, remember findings)
- **Customer support automation** (context-aware responses)
- **Data analysis chatbot** (upload CSV, ask questions in plain English)

Each of these would take 1-2 days instead of 1-2 weeks.

## Who This Is For

- **Python developers** who want to add AI to their projects
- **Anyone building chatbots** or AI assistants
- **Teams** who need consistent AI integration patterns
- **Students/learners** who want to see real AI app architecture

## Requirements

- Python 3.9+
- API key from OpenAI, Anthropic, or Google AI
- Basic familiarity with Python and REST APIs

## Contributing

Found a bug? Have a useful template to add? PRs welcome!

Common contributions:
- New project templates for specific use cases
- Additional LLM provider integrations
- Better example applications
- Documentation improvements

---

**Bottom line: This saves you weeks of setup time so you can focus on building cool AI features.**
