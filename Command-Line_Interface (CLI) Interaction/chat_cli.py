import argparse
import sys
import time
import os
from openai import OpenAI

def simulate_typing(text, delay=0.05):
    """Simulate typing effect: print text character by character."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()  # New line

def query_large_model(user_input, api_key):
    """
    Send user input to the large model
    Use the passed api_key for authentication.
    """
    if not api_key:
        return "Error: OpenAI API Key not found. Please set it via --apikey parameter or environment variable."
    
    try:
        # Instantiate the client with the provided API key.
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Change model if needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error occurred while calling the model: {e}"

def main():
    parser = argparse.ArgumentParser(
        description="A CLI tool with large model responses using the new OpenAI Python SDK."
    )
    parser.add_argument('-v', '--version', action='version', help='Show the tool version', version='chat_cli 1.1')
    parser.add_argument('-p', '--prompt', type=str, default='> ', help='Set the input prompt')
    parser.add_argument('-a', '--apikey', type=str, default=None, help='Set the OpenAI API key')
    
    args = parser.parse_args()
    prompt = args.prompt
    api_key = args.apikey if args.apikey is not None else os.getenv("OPENAI_API_KEY")

    repeat_mode = False  # Flag to enable repeat user input mode

    try:
        while True:
            user_input = input(prompt)
            if repeat_mode:
                simulate_typing(user_input)  # Display user input with typing simulation
            else:
                model_reply = query_large_model(user_input, api_key)  # Send user input to the large model
            
                # Check if the response is an error message
                if model_reply.startswith("Error:") or model_reply.startswith("Error occurred"):
                    print(model_reply)  # Print error message directly
                
                    # Ask the user if they want to enable repeat mode
                    choice = input("Do you want to switch to repeat mode repeat mode? [Y/n]: ").strip().lower()
                    if choice in ['y', 'yes', '']:
                        repeat_mode = True
                        print("Repeat mode enabled.")
                    else:
                        repeat_mode = False
                        print("Repeat mode not enabled.")
                else:
                    simulate_typing(model_reply)  # Display the model's response with typing simulation
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
