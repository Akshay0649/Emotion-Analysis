# Phase 1: EmotiSense Core Engine & Analysis

This directory contains the initial implementation of the EmotiSense project, focusing on the core emotion analysis engine, data processing, and visualization capabilities. This phase establishes the fundamental functionalities for detecting and understanding emotions in text, utilizing both advanced API integration and a robust keyword-based fallback.

## Contents

* `emotisense_notebook.ipynb`: The primary Google Colab notebook containing the complete source code for the EmotiSense engine, test cases, and visualization functions.
* `(emotisense_results.csv)`: (Generated upon execution) A CSV file that will be created after running the notebook, containing the detailed analysis results of the sample texts.

## Key Features Implemented in Phase 1

* **Hybrid Emotion Detection:**
    * **Hugging Face API Integration:** Connects to the `j-hartmann/emotion-english-distilroberta-base` model for high-accuracy, context-aware emotion classification.
    * **Keyword-Based Fallback:** A robust, local keyword matching system ensuring emotion detection even without API access.
* **Structured Output:** Results are encapsulated in an `EmotionResult` dataclass, providing clean access to primary emotion, confidence, all emotion scores, timestamp, and word count.
* **Text Preprocessing:** Includes basic text normalization (lowercasing, punctuation removal) for consistent analysis.
* **Batch Analysis:** Ability to process multiple text entries efficiently.
* **Comprehensive Visualizations:** Generates a set of plots to understand the emotional data:
    * Distribution of primary emotions (Pie Chart).
    * Confidence score distribution (Histogram).
    * Confidence levels across different emotion types (Box Plot).
    * Relationship between text length (word count) and confidence (Scatter Plot).
* **CSV Export:** Analysis results are saved to a CSV file for further quantitative analysis.

## How to Run This Phase (Google Colab)

The `emotisense_notebook.ipynb` is designed for seamless execution in Google Colab.

1.  **Open the Notebook:**
    Click the badge below to open this specific Phase 1 notebook directly in Google Colab:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/phase-1/emotisense_notebook.ipynb)

    *(**Important:** Make sure the path `phase-1/emotisense_notebook.ipynb` in the URL above correctly points to the notebook's location within your repository.)*

2.  **Hugging Face API Token (Highly Recommended):**
    For optimal performance and accuracy using the API, it's advised to provide your Hugging Face API token.
    * **Obtain a Token:** Sign up/log in to [Hugging Face](https://huggingface.co/join) and create a new "Access Token" (with "read" role) in your [settings](https://huggingface.co/settings/tokens).
    * **Add to Colab Secrets:**
        * In your Colab notebook, use the "Secrets" icon (a key) on the left sidebar.
        * Add a new secret named `HF_TOKEN` and paste your Hugging Face token as its value. Ensure "Notebook access" is enabled.
        * The notebook code is configured to automatically attempt to load this secret.

3.  **Execute the Notebook:**
    * Once the notebook is open and your API token is configured (if desired), go to `Runtime` -> `Run all` in the Colab menu.
    * The notebook will then run through the sample analysis, print a summary of results, display the generated plots, and save `emotisense_results.csv` to your Colab environment.

## Code Structure

The core logic resides within the `EmotiSenseEngine` class, which encapsulates the emotion detection methods and manages API interactions. The `EmotionResult` dataclass provides a clear structure for holding analysis outcomes.

* `EmotionResult`: Dataclass for structured output of emotion analysis.
* `EmotiSenseEngine`:
    * `__init__`: Initializes with API usage preference and API key, sets up keyword fallback.
    * `preprocess_text`: Cleans input text.
    * `keyword_based_detection`: Fallback method using keyword matching.
    * `analyze_emotion`: Main method for single text analysis (API first, then fallback).
    * `batch_analyze`: Processes a list of texts.
* `visualize_results`: Generates various plots for emotion distribution and confidence.
* `run_emotisense_test`: Orchestrates the testing process, runs analysis on sample data, prints results, and triggers visualizations/CSV export.

## Future Development (Leading to Phase 2, etc.)

This Phase 1 forms the foundational analysis layer. Future phases of the EmotiSense project could build upon this, including:

* **Phase 2: Web Interface Development:** Creating a user-friendly web application to interact with the `EmotiSenseEngine`.
* 
* **Phase 3: Advanced Features:** Implementing user-specific models, time-series analysis for journal trends, or integration with external data sources.

---
