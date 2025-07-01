# 🧠 Simple BrainBox Chatbot WITH MEMORY

**A chatbot that remembers your conversation - just 30 lines of code!**

## What's Different?

Unlike the basic chatbot, this one **remembers everything** you've said in the conversation. Ask it to reference something from earlier - it will remember!

## Example Conversation

```
🧠 BrainBox Chatbot with Memory
===================================
I'll remember our entire conversation!
Type 'quit' to exit

You: Hi! My name is Sarah
🤖 BrainBox: Hello Sarah! Nice to meet you. How can I help you today?

You: I like pizza
🤖 BrainBox: That's great! Pizza is delicious. What's your favorite type of pizza?

You: What did I just tell you about food?
🤖 BrainBox: You told me that you like pizza! Do you have a favorite topping or style?

You: What's my name again?
🤖 BrainBox: Your name is Sarah! Is there anything else I can help you with?
```

See how it remembers your name AND the pizza conversation? That's the power of memory!

## Quick Start

```bash
cd examples/simple-chatbot-with-memory
pip install -r requirements.txt
cp .env.example .env
nano .env  # Add your OpenAI API key
python main.py
```

## How Memory Works (The Secret!)

```python
# The key is this conversation list:
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi! My name is Sarah"},
    {"role": "assistant", "content": "Hello Sarah! Nice to meet you."},
    {"role": "user", "content": "I like pizza"},
    {"role": "assistant", "content": "That's great! Pizza is delicious."},
    # ... and so on
]

# Each time you send a message, we:
# 1. Add your message to the list
# 2. Send the ENTIRE list to the AI
# 3. Add the AI's response to the list
# 4. Repeat!
```

That's it! The AI sees the full conversation history every time, so it can remember everything.

## Cost vs Basic Chatbot

- **Basic chatbot**: Each message costs ~$0.001
- **Memory chatbot**: Costs increase as conversation gets longer
  - Message 1: ~$0.001
  - Message 10: ~$0.005 (because it's sending 10 messages worth of context)
  - Message 20: ~$0.010

**Pro tip**: The code automatically keeps only the last 20 messages to control costs!

## What's Next?

1. **Try both chatbots** to see the difference
2. **Experiment with the system message** - change the personality
3. **Check out the production templates** for persistent memory across sessions
4. **Add multiple AI providers** for better reliability

## The Learning Path

1. ✅ **Basic chatbot** (`../simple-chatbot/`) - Understanding the basics
2. ✅ **Memory chatbot** (this example) - Understanding conversation flow  
3. 🔥 **Production API** (`../../common/project-templates/llm-app-template/`) - Building real apps
4. 🚀 **Custom projects** - Your own AI applications!

Happy chatting! 🤖
