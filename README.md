# EmotiSense: Journal Emotion Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akshay0649/Emotion-Analysis/blob/Phase-1/EMotion%20Analysis.ipynb)

> **Now available as an installable Python package, a command-line tool, and a Streamlit web app** — not just a notebook. See [Quick Start](#quick-start-local) below.

## Project Overview

EmotiSense is a Python-based tool designed for the analysis of emotional content within text, particularly tailored for applications like personal journaling, sentiment tracking, or any scenario requiring nuanced emotion detection. It harnesses the capabilities of the Hugging Face Inference API for cutting-edge, transformer-based emotion classification. For scenarios where API access is unavailable or for local execution, it includes a robust keyword-based fallback mechanism.

This project offers:
* A modular `EmotiSenseEngine` class for performing emotion analysis.
* Structured data storage (`EmotionResult` dataclass) to capture detailed analysis outcomes, including the primary emotion, its confidence score, all detected emotion probabilities, a timestamp, and word count.
* Functions to efficiently process multiple text entries in batches.
* Comprehensive data visualization tools (utilizing Matplotlib and Seaborn) to provide insights into emotion distribution, confidence levels, and their correlation with text length.
* Capabilities to export analysis results to a CSV file for further examination.

EmotiSense is an excellent starting point for developers and data enthusiasts who wish to integrate emotion analysis into their applications or simply gain deeper insights into the emotional tone of written content.

## Features

* **Hybrid Emotion Detection:**
    * **Primary (API-driven):** Leverages the Hugging Face Inference API, specifically the `j-hartmann/emotion-english-distilroberta-base` model, for advanced, context-aware emotion classification.
    * **Fallback (Keyword-based):** Implements a reliable keyword-based detection system that serves as a robust alternative if the API is not utilized, an API key is absent, or the API call encounters issues.
* **Detailed Analysis Results:** Each analysis captures granular data, including the predicted primary emotion, its associated confidence score, a dictionary of all emotion labels with their respective probabilities, the timestamp of the analysis, and the word count of the input text.
* **Batch Processing:** Efficiently processes and analyzes a collection of text entries, returning a list of `EmotionResult` objects.
* **Intuitive Data Visualization:** Generates insightful plots to summarize the emotional landscape of the analyzed texts:
    * A **Pie Chart** illustrating the distribution of primary emotions.
    * A **Histogram** displaying the distribution of confidence scores.
    * A **Box Plot** visualizing confidence levels across different emotion types.
    * A **Scatter Plot** exploring the relationship between word count and confidence scores.
* **Data Export:** All computed analysis results are conveniently compiled into a Pandas DataFrame and saved as `emotisense_results.csv` for easy download and further statistical analysis or integration.
* **Text Pre-processing:** Includes basic text cleaning functionalities (lowercase conversion, removal of extra whitespace and non-alphanumeric characters) to ensure consistent input for the emotion models.
* **Google Colab Optimized:** The project is structured and includes setup instructions to run seamlessly within Google Colab, facilitating quick experimentation, development, and sharing.

## How It Works

The `EmotiSenseEngine` class serves as the core component of this emotion analysis project.

1.  **Engine Initialization:**
    * When an `EmotiSenseEngine` instance is created, you can specify whether to `use_api` (defaults to `True`) and provide your Hugging Face API key.
    * A comprehensive dictionary of `emotion_keywords` is pre-loaded to serve as the basis for the fallback detection mechanism.

2.  **`analyze_emotion(text)` Method:**
    * The method first attempts to perform emotion analysis using the Hugging Face Inference API. It constructs an HTTP POST request to the specified model endpoint (`https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base`) with the input text.
    * Upon a successful API response (HTTP 200), it parses the JSON data to extract the emotion labels and their corresponding scores, identifying the emotion with the highest score as the `primary_emotion`.
    * In cases where `use_api` is `False`, the `api_key` is missing, or the API call fails (e.g., due to network issues, invalid token, or rate limiting), the engine gracefully falls back to the `keyword_based_detection` method.

3.  **`keyword_based_detection(text)` Method:**
    * This method first preprocesses the input text by converting it to lowercase, removing excess whitespace, and stripping non-alphanumeric characters.
    * It then iterates through the pre-defined `emotion_keywords` sets. For each word in the preprocessed text, it increments a score for any matching emotion category.
    * The raw scores are then normalized to produce confidence-like probabilities across all emotion categories. If no keywords are found, 'neutral' is assigned a confidence of 1.0.

4.  **`EmotionResult` Dataclass:** The outcomes of each `analyze_emotion` call are encapsulated in an `EmotionResult` dataclass. This ensures that the results are consistently structured, making them easy to access and process programmatically.

5.  **`visualize_results` Function:** This function takes a list of `EmotionResult` objects and generates a 2x2 grid of plots to visually summarize the findings.

6.  **`run_emotisense_test` Function:** This serves as the main execution entry point for the Colab notebook. It defines sample journal entries, initializes the `EmotiSenseEngine`, performs batch analysis, prints a detailed summary of individual results and overall performance metrics, generates the visualizations, and exports the complete dataset to `emotisense_results.csv`.

<a name="quick-start-local"></a>
## Quick Start (Local)

EmotiSense runs anywhere Python 3.8+ is available. No Hugging Face token is
required — without one it automatically uses the built-in keyword fallback.

```bash
# 1. Clone
git clone https://github.com/Akshay0649/Emotion-Analysis.git
cd Emotion-Analysis

# 2. Install dependencies
pip install -r requirements.txt
# ...or install the package itself (engine only): pip install -e .

# 3. (Optional) enable the Hugging Face API for best accuracy
export HF_TOKEN="hf_your_token_here"      # Linux/macOS
# $env:HF_TOKEN="hf_your_token_here"      # Windows PowerShell
```

### Use the command-line tool

```bash
# Analyse a single entry
python -m emotisense "I had an amazing day and I feel so grateful!"

# Analyse a file (one entry per line) and export a CSV
python -m emotisense --file entries.txt --csv results.csv

# Run the bundled demo on sample journal entries
python -m emotisense --demo

# Force the offline keyword fallback (skip the API)
python -m emotisense --no-api "I am furious about this"
```

### Launch the web app

```bash
streamlit run app.py
```

The app provides a single-entry analyzer with a per-emotion breakdown and a
batch mode that charts the emotion distribution and lets you download a CSV.

### Use it as a library

```python
from emotisense import EmotiSenseEngine

engine = EmotiSenseEngine(use_api=True)   # falls back to keywords if no HF_TOKEN
result = engine.analyze_emotion("I'm so excited and grateful today!")
print(result.primary_emotion, result.confidence)   # -> joy 0.98 (via API) / 1.0 (keyword)
```

### Run the demo script (with plots + CSV)

```bash
python demo.py     # prints a summary, writes emotisense_results.csv and emotisense_plots.png
```

### Run the tests

```bash
pytest
```

## Setup and Usage (Google Colab)

The project is specifically designed for ease of use within the Google Colab environment.

1.  **Open in Google Colab:**
    Click the badge below to open the notebook directly in Google Colab:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akshay0649/Emotion-Analysis/blob/Phase-1/EMotion%20Analysis.ipynb)

2.  **Hugging Face API Token (Recommended for Best Accuracy):**
    While the project includes a robust keyword-based fallback, leveraging the Hugging Face API provides significantly more accurate and nuanced emotion detection.
    * To obtain a token, sign up or log in at [Hugging Face](https://huggingface.co/join) and then navigate to your [Access Tokens settings](https://huggingface.co/settings/tokens). Create a new token with "read" access.
    * **Securely Add Your Token in Colab:**
        * In your Colab notebook, on the left sidebar, click the **key icon** (Secrets).
        * Click `+ New secret`.
        * For `Name`, enter: `HF_TOKEN` (This is the name the code expects).
        * For `Value`, paste your actual Hugging Face API token (e.g., `hf_YOUR_ACTUAL_TOKEN_HERE`).
        * Ensure `Notebook access` is checked for this secret.
        * The notebook is pre-configured to attempt to load the `HF_TOKEN` from Colab Secrets.

3.  **Run All Cells:**
    * Once the notebook is open in Colab, go to `Runtime` -> `Run all`.
    * The notebook will execute the `run_emotisense_test()` function, which will:
        * Initialize the EmotiSense Engine.
        * Analyze a predefined set of sample journal entries.
        * Print detailed results for each entry and a performance summary.
        * Generate interactive data visualizations.
        * Save the complete analysis data to a CSV file named `emotisense_results.csv` in your Colab environment (which you can then download).

> **Prefer local?** See [Quick Start (Local)](#quick-start-local) above for the
> package, CLI, and web-app workflow. The token is read from the `HF_TOKEN`
> environment variable — never hardcode it into source for public distribution.

## Project Structure

```
Emotion-Analysis/
├── emotisense/                # Installable Python package
│   ├── __init__.py            # Public API exports
│   ├── engine.py             # EmotiSenseEngine + EmotionResult (API + keyword fallback)
│   ├── visualize.py          # Matplotlib/Seaborn plots + CSV/DataFrame export
│   ├── sample_data.py        # Sample journal entries for demos/tests
│   ├── cli.py                # Command-line interface (python -m emotisense)
│   └── __main__.py           # Enables `python -m emotisense`
├── app.py                     # Streamlit web app
├── demo.py                    # Standalone demo (summary + CSV + plots)
├── tests/
│   └── test_engine.py        # Unit tests (keyword path, no network needed)
├── requirements.txt           # Runtime + dev dependencies
├── pyproject.toml             # Packaging metadata + console entry point
├── EMotion Analysis.ipynb     # Original Colab notebook (on the `Phase-1` branch)
├── emotisense_results.csv     # (Generated on run) detailed analysis results
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## Potential Enhancements

* ✅ **Interactive Web Application (done):** A Streamlit web app (`app.py`) provides single-entry and batch analysis with charts and CSV download. Future work could add Flask/Gradio variants or richer interactive visualizations.
* **Advanced Text Preprocessing:** Incorporate more sophisticated natural language processing (NLP) techniques such as stemming, lemmatization, stop word removal, and robust handling of emojis, slang, and common internet acronyms.
* **Customizable Keyword Sets:** Enable users to easily define, load, or modify their own sets of keywords for the fallback emotion detection, adapting it to specific domains or personal vocabularies.
* **Multilingual Support:** Extend the `EmotiSenseEngine` to support emotion analysis in multiple languages by integrating different Hugging Face models or multilingual keyword sets.
* **Time-Series Emotion Tracking:** For sequential journal entries, implement features to track and visualize emotional trends over time, providing insights into mood fluctuations.
* **Sentiment Analysis Integration:** Augment the current emotion detection with broader sentiment analysis (positive, negative, neutral) for a more complete understanding of text tone.
* **Robust Error Handling & Logging:** Enhance the robustness of the API calls with more comprehensive error handling mechanisms and implement logging for easier debugging and monitoring in production environments.
* **Model Fine-tuning Pipeline:** Provide tools or instructions for users to fine-tune the emotion detection model on their specific datasets, improving accuracy for niche applications.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Akshay0649/Emotion-Analysis/issues) if you want to contribute.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

