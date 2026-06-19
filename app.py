"""EmotiSense Streamlit web app.

Run locally with::

    streamlit run app.py

Set ``HF_TOKEN`` in your environment (or paste it in the sidebar) to use the
Hugging Face API; otherwise the keyword fallback is used automatically.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from emotisense.engine import EmotiSenseEngine
from emotisense.sample_data import SAMPLE_ENTRIES
from emotisense.visualize import results_to_dataframe

EMOJI = {
    "joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨",
    "surprise": "😲", "love": "❤️", "neutral": "😐", "disgust": "🤢",
}

st.set_page_config(page_title="EmotiSense", page_icon="🧠", layout="wide")

st.title("🧠 EmotiSense — Journal Emotion Analysis")
st.caption("Detect the emotional tone of journal entries and other text.")

# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    use_api = st.toggle("Use Hugging Face API", value=bool(os.getenv("HF_TOKEN")))
    token_input = st.text_input(
        "HF_TOKEN (optional)",
        value="",
        type="password",
        help="Overrides the HF_TOKEN environment variable for this session.",
    )
    st.markdown(
        "Without a token, EmotiSense uses a built-in **keyword fallback**. "
        "Get a free token at [huggingface.co](https://huggingface.co/settings/tokens)."
    )


@st.cache_resource(show_spinner=False)
def get_engine(use_api: bool, api_key: str) -> EmotiSenseEngine:
    return EmotiSenseEngine(use_api=use_api, api_key=api_key or None)


engine = get_engine(use_api, token_input)
mode = "Hugging Face API" if engine.use_api else "Keyword fallback"
st.sidebar.success(f"Active mode: {mode}")

# ---------------------------------------------------------------------------
# Tabs: single entry vs. batch
# ---------------------------------------------------------------------------
single_tab, batch_tab = st.tabs(["✍️ Single entry", "📚 Batch analysis"])

with single_tab:
    text = st.text_area(
        "Write or paste a journal entry",
        value="I had an amazing day today and I feel so grateful!",
        height=160,
    )
    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            result = engine.analyze_emotion(text)
            emoji = EMOJI.get(result.primary_emotion, "🔎")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(
                    f"{emoji} {result.primary_emotion.title()}",
                    f"{result.confidence:.0%}",
                )
                st.caption(f"Source: {result.source} · {result.word_count} words")
            with col2:
                breakdown = pd.DataFrame(
                    sorted(result.all_emotions.items(), key=lambda kv: kv[1], reverse=True),
                    columns=["emotion", "score"],
                ).set_index("emotion")
                st.bar_chart(breakdown)

with batch_tab:
    st.write("One entry per line. Load the sample journal to try it out.")
    if st.button("Load sample entries"):
        st.session_state["batch_text"] = "\n".join(SAMPLE_ENTRIES)

    batch_text = st.text_area(
        "Entries",
        key="batch_text",
        height=220,
        placeholder="I had a great day...\nWork was stressful today...",
    )
    if st.button("Analyze batch", type="primary"):
        entries = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not entries:
            st.warning("Please enter at least one entry.")
        else:
            with st.spinner(f"Analyzing {len(entries)} entries..."):
                results = engine.batch_analyze(entries)
            df = results_to_dataframe(results)

            c1, c2, c3 = st.columns(3)
            c1.metric("Entries", len(results))
            c2.metric("Avg confidence", f"{df['confidence'].mean():.0%}")
            c3.metric("Distinct emotions", df["primary_emotion"].nunique())

            left, right = st.columns(2)
            with left:
                st.subheader("Emotion distribution")
                st.bar_chart(df["primary_emotion"].value_counts())
            with right:
                st.subheader("Confidence by entry")
                st.line_chart(df["confidence"])

            st.subheader("Detailed results")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="emotisense_results.csv",
                mime="text/csv",
            )
