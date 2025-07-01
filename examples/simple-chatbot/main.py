#!/usr/bin/env python3
"""
Simple BrainBox Chatbot Example
===============================

The simplest possible AI chatbot using BrainBox.
Just ask questions and get answers!

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
    
    print("🧠 BrainBox Simple Chatbot")
    print("=" * 30)
    print("Type 'quit' to exit\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye! 👋")
            break
            
        if not user_input:
            continue
            
        try:
            # Send to AI
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fastest, cheapest model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            # Print AI response
            ai_response = response.choices[0].message.content
            print(f"🤖 BrainBox: {ai_response}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure your OPENAI_API_KEY is set in .env file\n")

if __name__ == "__main__":
    main()
