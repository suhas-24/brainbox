# 🤖 Simple BrainBox Chatbot

**The simplest possible AI chatbot example - just 25 lines of code!**

## What This Does

A basic command-line chatbot that:
- Takes your questions
- Sends them to OpenAI
- Shows you the AI's response
- Remembers nothing (starts fresh each message)

Perfect for understanding the basics before diving into the advanced features.

## Quick Start (1 minute)

```bash
# 1. Navigate to this directory
cd examples/simple-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
cp .env.example .env
nano .env  # Add your OpenAI API key

# 4. Run it!
python main.py
```

## Example Conversation

```
🧠 BrainBox Simple Chatbot
==============================
Type 'quit' to exit

You: Hello!
🤖 BrainBox: Hello! How can I help you today?

You: What's 2+2?
🤖 BrainBox: 2 + 2 equals 4.

You: Tell me a joke
🤖 BrainBox: Why don't scientists trust atoms? Because they make up everything!

You: quit
Goodbye! 👋
```

## Get Your API Key

1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key and paste it in your `.env` file

## What's Next?

This is just the beginning! Once you've tried this simple example:

1. **Add Memory**: Check out the full templates to see conversation memory
2. **Add Multiple Providers**: See how to use Anthropic, Google AI as backups
3. **Build an API**: Turn this into a web service
4. **Add Agents**: Create specialized AI assistants

## The Code Explained

```python
# This is literally all you need for a basic AI chatbot:

from openai import OpenAI
client = OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

That's it! BrainBox just makes this pattern reusable and production-ready.

## Cost

Using `gpt-4o-mini` (the fastest, cheapest model):
- **Input**: ~$0.00015 per 1K tokens (~750 words)
- **Output**: ~$0.0006 per 1K tokens (~750 words)
- **Typical conversation**: $0.001-0.01 per exchange

So you can chat for hours for just pennies! 💰
