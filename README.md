# EmotiSense: Journal Emotion Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/emotisense_notebook.ipynb)

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

## Setup and Usage (Google Colab Recommended)

The project is specifically designed for ease of use within the Google Colab environment.

1.  **Open in Google Colab:**
    Click the badge below to open the notebook directly in Google Colab:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/emotisense_notebook.ipynb)

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

### Local Setup (Optional)

If you prefer to run this project in a local Python environment, follow these steps:

1.  **Prerequisites:** Ensure you have Python 3.8 or higher installed on your system.

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

3.  **Install Dependencies:**
    ```bash
    pip install pandas matplotlib seaborn requests
    ```

4.  **Set Hugging Face API Token (Optional but Recommended):**
    * **Do NOT hardcode your token directly into the Python file for public distribution.**
    * You can set it as an environment variable before running the script:
        ```bash
        export HF_TOKEN="hf_YOUR_ACTUAL_TOKEN_HERE" # For Linux/macOS
        # On Windows (Command Prompt): set HF_TOKEN="hf_YOUR_ACTUAL_TOKEN_HERE"
        # On Windows (PowerShell): $env:HF_TOKEN="hf_YOUR_ACTUAL_TOKEN_HERE"
        ```
    * Alternatively, you can modify the `EmotiSenseEngine` initialization in your `emotisense_notebook.ipynb` (if running in Jupyter/JupyterLab) or a converted `.py` script to prompt for the token or read it from a local `.env` file (requires `python-dotenv`).

5.  **Run the Notebook/Script:**
    * If running as a Jupyter/JupyterLab notebook: `jupyter lab emotisense_notebook.ipynb`
    * If you convert it to a Python script (e.g., `emotisense.py`): `python emotisense.py`

## Project Structure

├── emotisense_notebook.ipynb  # The main Google Colab notebook containing all the code

├── emotisense_results.csv     # (Generated upon execution) CSV file with detailed analysis results

└── README.md                  # This project description file

├── LICENSE                    # The project's license file (e.g., MIT License)

## Potential Enhancements

* **Interactive Web Application:** Develop a user-friendly web interface (e.g., using Streamlit, Flask, or Gradio) to allow users to input text, receive real-time emotion analysis, and view interactive visualizations.
* **Advanced Text Preprocessing:** Incorporate more sophisticated natural language processing (NLP) techniques such as stemming, lemmatization, stop word removal, and robust handling of emojis, slang, and common internet acronyms.
* **Customizable Keyword Sets:** Enable users to easily define, load, or modify their own sets of keywords for the fallback emotion detection, adapting it to specific domains or personal vocabularies.
* **Multilingual Support:** Extend the `EmotiSenseEngine` to support emotion analysis in multiple languages by integrating different Hugging Face models or multilingual keyword sets.
* **Time-Series Emotion Tracking:** For sequential journal entries, implement features to track and visualize emotional trends over time, providing insights into mood fluctuations.
* **Sentiment Analysis Integration:** Augment the current emotion detection with broader sentiment analysis (positive, negative, neutral) for a more complete understanding of text tone.
* **Robust Error Handling & Logging:** Enhance the robustness of the API calls with more comprehensive error handling mechanisms and implement logging for easier debugging and monitoring in production environments.
* **Model Fine-tuning Pipeline:** Provide tools or instructions for users to fine-tune the emotion detection model on their specific datasets, improving accuracy for niche applications.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/issues) if you want to contribute.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

