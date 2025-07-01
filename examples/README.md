# 🎯 BrainBox Examples

**Start here if you're new to BrainBox!** These examples go from super simple to more advanced.

## 📚 Learning Path

### 1. 🤖 [Simple Chatbot](simple-chatbot/) - **Start Here!**
- **Time**: 1 minute setup
- **Lines of code**: 25
- **What it does**: Basic Q&A chatbot
- **Perfect for**: Understanding the absolute basics

```bash
cd simple-chatbot && python main.py
```

### 2. 🧠 [Chatbot with Memory](simple-chatbot-with-memory/)
- **Time**: 2 minutes setup  
- **Lines of code**: 30
- **What it does**: Chatbot that remembers your conversation
- **Perfect for**: Understanding how AI memory works

```bash
cd simple-chatbot-with-memory && python main.py
```

### 3. 🔧 [Context Engineering Demo](../examples/context-engineering-demo.py)
- **Time**: 3 minutes setup
- **Lines of code**: 50
- **What it does**: Shows advanced prompt engineering
- **Perfect for**: Learning how to make AI responses better

### 4. 🏭 [Production API Server](../common/project-templates/llm-app-template/)
- **Time**: 5 minutes setup
- **Features**: Full REST API, multiple providers, memory, caching
- **Perfect for**: Building real applications

### 5. 🛠️ [Personal Code Reviewer](../projects/personal-code-reviewer/)
- **Time**: 3 minutes setup
- **What it does**: Actually useful tool that reviews your Python code
- **Perfect for**: Seeing a complete, practical application

## 🎯 Which Example Should I Try?

**👋 Complete beginner?** → Start with [Simple Chatbot](simple-chatbot/)

**🤔 Want to understand memory?** → Try [Chatbot with Memory](simple-chatbot-with-memory/)

**🚀 Building a real app?** → Jump to [Production API Server](../common/project-templates/llm-app-template/)

**🔍 Want something useful now?** → Try [Personal Code Reviewer](../projects/personal-code-reviewer/)

## 🛠️ Setup Requirements

All examples need:
- Python 3.9+
- OpenAI API key (get one at https://platform.openai.com/api-keys)
- 2 minutes of your time

## 💰 Cost Estimate

- **Simple examples**: ~$0.001-0.01 per conversation
- **Production examples**: ~$0.01-0.10 per session
- **All examples combined**: Under $1 to try everything

## 🆘 Troubleshooting

**❌ "Module not found"**
```bash
pip install -r requirements.txt
```

**❌ "API key not found"**
```bash
# Make sure you copied .env.example to .env and added your key
cp .env.example .env
nano .env  # Add OPENAI_API_KEY=sk-your-key-here
```

**❌ "Permission denied"** 
```bash
chmod +x main.py
```

## 🎉 What's Next?

After trying these examples:

1. **Customize them** - Change the personality, add features
2. **Combine concepts** - Mix memory with specialized tasks  
3. **Build something new** - Use the templates to create your own AI app
4. **Share your creation** - Show the community what you built!

**Happy coding!** 🚀
