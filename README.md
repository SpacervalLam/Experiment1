# Experiment Repository

This repository contains a command-line interactive tool and an analysis report for the Iris dataset.

---

## Directory Structure

```
├── README.md
├── Command-Line_Interface (CLI) Interaction/
│   ├── chat_cli.py
│   ├── index.html
├── Data_Processing/
│   ├── eda_iris.ipynb
│   ├── iris_analysis_report.html
│   ├── iris_feature_distribution.png
├── Large_Model/
│   ├── Reverse_Chat_Demo.html
```

---

## Command-Line Interactive Tool

### Overview

`chat_cli.py` is a Python-based command-line tool that provides simple interactive functionality and simulates a typewriter effect.

### Features

- View the tool version using the `-v` or `--version` option.
- Customize the input prompt using the `-p` or `--prompt` option (default: `> `).
- Simulate a typewriter effect by displaying user input one character at a time.
- Gracefully handle user interruptions (via `Ctrl+C`).

### Usage

```bash
python chat_cli.py [options]
```

#### Available Options

- `-v`, `--version`: Displays the tool's version.
- `-p`, `--prompt`: Sets the input prompt.

### Examples

- Display the version:

  ```bash
  python chat_cli.py --version
  ```

- Customize the prompt:

  ```bash
  python chat_cli.py --prompt "Enter text: "
  ```

### Implementation Details

1. Command-line arguments are parsed using the `argparse` module.
2. The typewriter effect is implemented via the `simulate_typing` function.
3. The `KeyboardInterrupt` exception is handled to exit the program gracefully.

---

## Data Analysis Report

### Overview

`iris_analysis_report.html` is an analysis report based on the Iris dataset, including descriptive statistics, category distribution, data types, and feature distribution visualizations.

### Key Insights

1. **Dataset Overview**
   - The dataset contains 150 rows and 5 columns.
2. **Category Distribution**
   - Each category (0, 1, 2) has 50 samples.
3. **Data Types**
   - All features are numerical (`float64` or `int64`).
4. **Missing Values**
   - No missing values in the dataset.
5. **Descriptive Statistics**
   - Includes minimum, maximum, mean, and standard deviation for each feature.
6. **Feature Distribution**
   - Visualized feature distributions are saved as `iris_feature_distribution.png`.

### Viewing the Report

Open the `Data_Processing/iris_analysis_report.html` file in a browser to view the full report.

---

## Experience of large models

`Large_Model/Reverse_Chat_Demo.html` is adapted from HTML code written in GPT-4o demonstrating a reverse chat interaction.

### Features

- User messages are reversed and displayed as bot responses.
- Includes real-time input box resizing, message animation effects, and a typing indicator.

---

## Notes

1. Ensure Python is installed on your system to run the command-line tool.
2. Use a browser to open the analysis report and large model demo files.

---

© 2025 Experiment 1: Command-Line Interactive Tool and Data Analysis Report