# 🔍 Personal Code Review Assistant

**Real-time project demonstrating AI Forge capabilities**

## What This Does

An intelligent code review assistant that:
- **Analyzes your code** for bugs, security issues, and best practices
- **Learns your coding style** and preferences over time
- **Provides context-aware suggestions** based on your project type
- **Integrates with your Git workflow** for seamless reviews
- **Tracks code quality metrics** across your projects

## Real-World Scenario

You're working on multiple projects:
- A FastAPI backend service
- A React frontend application  
- Some Python data analysis scripts

Instead of manually reviewing code or waiting for team reviews, this assistant:
1. **Instantly reviews** your commits before you push
2. **Remembers** your project contexts and coding standards
3. **Adapts** suggestions based on whether it's backend, frontend, or data code
4. **Learns** from your feedback to improve future reviews

## How It Helps You Personally

### Before AI Forge:
```bash
# You write code
git add .
git commit -m "Added new feature"
# Hope you didn't miss any issues
git push
# Find bugs in production 😬
```

### With AI Forge:
```bash
# You write code
./review-code.py --files src/api.py
# AI analyzes: "FastAPI project detected. Found 2 security issues, 1 performance optimization"
# Fix issues based on intelligent suggestions
git add .
git commit -m "Added new feature with security improvements"
git push
# Deploy with confidence ✅
```

## Live Demo Features

1. **Instant Code Analysis**
   ```bash
   python review.py --file your_file.py
   ```

2. **Git Hook Integration**
   ```bash
   ./setup-git-hooks.sh
   # Now every commit gets auto-reviewed
   ```

3. **Project Context Learning**
   ```bash
   python setup-project.py --type fastapi --name my-api
   # AI learns your project structure and standards
   ```

4. **Personal Metrics Dashboard**
   ```bash
   python dashboard.py
   # See your code quality trends over time
   ```

## Real Benefits You'll See

- **Catch bugs early**: Find issues before they reach production
- **Learn best practices**: Get personalized suggestions for improvement
- **Save time**: No waiting for human code reviews for simple issues
- **Improve code quality**: Track and improve your coding metrics
- **Context awareness**: Different suggestions for different types of projects

## Getting Started (5 minutes)

1. **Setup**:
   ```bash
   cd /Users/suhas/ai-forge-workspace/projects/personal-code-reviewer
   pip install -r requirements.txt
   cp .env.example .env
   # Add your OpenAI API key to .env
   ```

2. **Try it out**:
   ```bash
   python review.py --file example.py
   ```

3. **Integrate with your workflow**:
   ```bash
   ./setup-git-hooks.sh
   ```

4. **Start coding** - the assistant will help you in real-time!

## Technical Architecture

Built using AI Forge components:
- **Multi-Agent System**: Separate agents for security, performance, style analysis
- **Context Engineering**: Understands your project type and coding patterns
- **Memory Management**: Remembers your feedback and preferences
- **LLM Integration**: Uses multiple AI models for comprehensive analysis

This is a practical example of how AI Forge turns AI concepts into tools you can actually use in your daily development work.
