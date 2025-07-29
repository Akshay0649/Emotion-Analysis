# Phase 1: EmotiSense Core Engine & Analysis

This directory contains the complete implementation of **Phase 1** of the EmotiSense project — a hybrid emotion detection engine built for journal entries, chat logs, and emotionally rich narratives. This phase focuses on foundational features like API integration, fallback detection, structured outputs, visualization, and CSV export.

---

## 📁 Contents

- `emotisense_notebook.ipynb` – Google Colab notebook containing the full codebase (engine, test cases, visualization, and CSV export).
- `emotisense_results.csv` – Generated after notebook execution; contains emotion analysis results for sample or custom inputs.

---

## 🔑 Key Features Implemented in Phase 1

### 🎯 Hybrid Emotion Detection
- **Hugging Face API Integration:** Uses `j-hartmann/emotion-english-distilroberta-base` for contextual, high-accuracy classification.
- **Keyword-Based Fallback:** Handles offline/local detection through keyword mapping with intensity modifiers.

### 📊 Analysis Capabilities
- **Batch Analysis:** Efficiently process multi-entry datasets (chats, journal logs).
- **Structured Output:** Results stored using the `EmotionResult` dataclass.
- **Preprocessing:** Normalizes input text (e.g., lowercasing, punctuation cleanup).

### 📈 Visualizations
- Pie Chart – Distribution of detected primary emotions  
- Histogram – Confidence score distribution  
- Box Plot – Confidence by emotion type  
- Scatter Plot – Word count vs confidence

### 📤 Export Functionality
- Outputs saved to `emotisense_results.csv` for external use (e.g., Power BI, frontend apps).

---

## 🧪 Special Test Case Logged (Phase 2 Bridge)
6. Complex Emotion Scenario Testing
>
>  **“Grief–Love–Guilt” Emotional Paragraph**
>
> I stood at the doorway of the hospital room, the scent of antiseptic still clinging to the folds of my clothes. Her fingers, once strong, now barely curled around mine. I smiled so she wouldn't see the storm behind my eyes, pretending I hadn’t already said goodbye a thousand times in my heart. Part of me begged for one more day, one more laugh, while another whispered that maybe it was time to let her go. The guilt of wanting relief battled with the love that refused to loosen its grip. I remembered her singing to me as a child — the same lullaby echoing in my mind now like a haunting farewell. And yet, I didn’t cry. Not because I wasn’t breaking, but because some grief is so sacred, it sits in silence..
>
> 🧠 Result:
> - Primary Emotion: `FEAR (0.528)`
> - Top Emotions: `fear: 0.53`, `sadness: 0.35`, `neutral: 0.07`

This test case demonstrated EmotiSense’s competence in handling nuanced emotional contradictions and sets a strong foundation for Phase 2 features like **truth vs tone detection**.

---

## 🚀 How to Run (Colab Recommended)

### 🔗 Open Notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/phase-1/emotisense_notebook.ipynb)

> Ensure the file path matches your repo folder structure.

### 🔑 Configure Hugging Face API Token:
1. Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens).
2. In Colab: Click 🔑 (Secrets), then add `HF_TOKEN` and paste your token.
3. The notebook will auto-detect and use this token if available.

### ▶️ Run Notebook:
Click `Runtime > Run All` to execute end-to-end: test cases → results → visualizations → export.

---

## 🧱 Code Architecture Overview

📦 Core Components
> EmotionResult (Dataclass) Encapsulates the structured output of emotion analysis:
> 
> Text: Original input string
>
>    - primary_emotion: Dominant emotion label (e.g., joy, sadness)
>
>    - confidence: Score from 0 to 1 for how confident the model is
>
>    - all_emotions: Dictionary of all emotion probabilities
>
>    - timestamp: Datetime stamp when analysis was performed
>
>    - word_count: Token count for the input text

- `EmotionResult`: Structured output with emotion, confidence, timestamp, word count.

- `EmotiSenseEngine` class:
>   - `__init__`                    – Set mode/API
>   - `preprocess_text`             – Normalize input
>   - `keyword_based_detection`     – Local logic
>   - `analyze_emotion`             – API or fallback
>   - `batch_analyze`               – Full dataset analysis
>   - `visualize_results()`             – Charting logic
>   - `run_emotisense_test()`           – Sample test orchestrator
>
   #### 📊 Visualization

> visualize_results(results)
>
> Creates a 2×2 grid of the following charts:
>
>    - Pie Chart – Distribution of primary emotions
>    - Histogram – Confidence score distribution
>    - Boxplot – Confidence vs emotion type
>    - Scatterplot – Word count vs confidence score

#### 🧪 Testing & Orchestration

run_emotisense_test()

A standalone function that:

> - Runs sample test entries
> - Analyzes emotion via API or fallback
> - Prints top 3 emotions per entry
> - Summarizes confidence accuracy
> - Triggers visualizations and CSV export

---

## 🔮 Next Phase: Road to MVP (Phase 2)

### Upcoming Features:
- 🗂 Upload and analyze `.txt`/`.csv` chat logs
- 🧠 Emotional contradiction detection (“I'm fine” = sad)
- 📈 Timeline-based trend analysis
- 💾 Save sessions to Firestore/Supabase
- 🌐 Build Streamlit/React frontend

### ⌚ Future Direction

> ✍ Add UI (Streamlit or React frontend)
> 
>🔢 Integrate relationship between truth vs tone
>
>📊 Timeline-based emotion progression
>
>☁️ Store logs in Supabase/Firestore
>
>✅ Ready MVP for user testing by end of Phase 2
---

## 📎 Reference

This README aligns with the [Phase 1 Completion Summary](../emotisense_phase1_summary.md), which logs all technical and emotional test milestones, including ChatGPT-based analysis and complex emotion benchmarking.

---
