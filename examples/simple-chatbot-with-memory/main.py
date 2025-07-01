#!/usr/bin/env python3
"""
Simple BrainBox Chatbot WITH MEMORY
===================================

A chatbot that remembers your conversation!
Each message builds on the previous ones.

Usage:
    python main.py

Requirements:
    - OpenAI API key in .env file
    - pip install openai python-dotenv
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def main():
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # This list will store our conversation history
    conversation = [
        {"role": "system", "content": "You are a helpful assistant with a great memory."}
    ]
    
    print("🧠 BrainBox Chatbot with Memory")
    print("=" * 35)
    print("I'll remember our entire conversation!")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye! I'll remember our chat! 👋")
            break
            
        if not user_input:
            continue
        
        # Add user message to conversation history
        conversation.append({"role": "user", "content": user_input})
            
        try:
            # Send entire conversation to AI (this gives it memory!)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,  # Send full conversation history
                max_tokens=150,
                temperature=0.7
            )
            
            # Get AI response
            ai_response = response.choices[0].message.content
            print(f"🤖 BrainBox: {ai_response}\n")
            
            # Add AI response to conversation history
            conversation.append({"role": "assistant", "content": ai_response})
            
            # Keep conversation from getting too long (optional)
            if len(conversation) > 20:  # Keep last 20 messages
                conversation = conversation[:1] + conversation[-19:]  # Keep system message + last 19
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure your OPENAI_API_KEY is set in .env file\n")

if __name__ == "__main__":
    main()
