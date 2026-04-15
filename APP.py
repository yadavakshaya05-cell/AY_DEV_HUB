import streamlit as st
from transformers import pipeline

# Load a dedicated summarization model
# facebook/bart-large-cnn is the standard for high-quality text summarization
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Streamlit UI
st.set_page_config(page_title="AI Text Summarizer", page_icon="📝")
st.title("📝 AI Text Summarizer")
st.write("Enter a long text below, and get a concise summary!")

# Text Input
long_text = st.text_area("Enter text to summarize:", height=250, placeholder="Paste your article or paragraph here...")

# Summary Parameters
col1, col2 = st.columns(2)
with col1:
    max_len = st.slider("Max Summary Length", min_value=50, max_value=500, value=130)
with col2:
    min_len = st.slider("Min Summary Length", min_value=10, max_value=100, value=30)

if st.button("Summarize ✨"):
    if long_text.strip():
        # Basic validation to ensure max > min
        if max_len <= min_len:
            st.error("Error: Max length must be greater than Min length.")
        else:
            with st.spinner("Generating summary... ⏳"):
                try:
                    summary = summarizer(long_text, max_length=max_len, 
                                         min_length=min_len, do_sample=False)
                    st.subheader("📌 Summary:")
                    st.success(summary[0]['summary_text'])
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter some text to summarize.")
