# Command-Line Interactive Tool

This is a command-line tool written in Python that simulates a typewriter effect and provides simple interactive functionality.

## Features

- Supports viewing the tool version with the `-v` or `--version` option.
- Supports customizing the input prompt with the `-p` or `--prompt` option.
- Simulates a typewriter effect by displaying user input one character at a time.
- Handles user interruptions (via `Ctrl+C`) gracefully.

## Usage

```bash
python chat_cli.py [options]
```

### Available Options

- `-v`, `--version`: Displays the tool's version.
- `-p`, `--prompt`: Sets the input prompt (default is `> `).

## Examples

- To display the version:

  ```bash
  python chat_cli.py --version
  ```

- To customize the prompt:

  ```bash
  python chat_cli.py --prompt "Enter text: "
  ```

## Implementation Details

The core functionalities of this tool include:

1. Parsing command-line arguments using the `argparse` module.
2. Implementing a typewriter effect through a `simulate_typing` function.
3. Handling the `KeyboardInterrupt` exception to exit the program gracefully.

## Notes

Ensure that Python is installed on your system, and run the commands in a terminal.

---

© 2025 Command-Line Interactive Tool Documentation
```