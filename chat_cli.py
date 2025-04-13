import argparse
import sys
import time

def simulate_typing(text, delay=0.1):
    """Simulate typing effect by printing one character at a time."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()  # Move to the next line after typing

def main():
    parser = argparse.ArgumentParser(description="A CLI tool with typing simulation.")
    parser.add_argument('-v', '--version', action='version',help='Show the version of the tool', version='chat_cli 1.0')
    parser.add_argument('-p', '--prompt', type=str, default='> ', help='Set the input prompt')
    
    args = parser.parse_args()

    prompt = args.prompt

    try:
        while True:
            user_input = input(prompt)
            simulate_typing(user_input)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()