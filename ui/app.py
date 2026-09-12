import os
import streamlit as st
import requests
import fitz  # PyMuPDF
import plotly.graph_objects as go
import spacy
import numpy as np

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI vs Human Fingerprint Engine", layout="wide")

# Fallback to local 8001 if no environment variable is provided
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001").rstrip("/")

def extract_features(text: str):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    sentences = list(doc.sents)
    if not tokens or not sentences:
        return 0, 0, 0, 0
    ttr = len(set(tokens)) / len(tokens)
    avg_len = len(tokens) / len(sentences)
    lengths = [len([t for t in s if t.is_alpha]) for s in sentences]
    burstiness = float(np.std(lengths)) if len(lengths) > 1 else 0.0
    punct = sum(1 for t in doc if t.is_punct) / len(doc)
    return round(ttr, 3), round(avg_len, 2), round(burstiness, 2), round(punct, 3)

st.title("📄 AI-vs-Human Stylometric Attribution Engine")
st.markdown("Multi-stage forensic platform analyzing vector semantics and stylometric signatures.")

tab1, tab2 = st.tabs(["🔍 Document Verification", "📥 Ingest Seed PDF"])

with tab1:
    st.subheader("Verify Authorship & Fingerprint Breakdown")
    input_type = st.radio("Input Source:", ["Text Prompt", "PDF Upload"])
    
    query_text = ""
    if input_type == "Text Prompt":
        query_text = st.text_area("Paste snippet here:", height=140)
    else:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            query_text = "".join([page.get_text() for page in doc])

    if st.button("Run Verification", type="primary"):
        if not query_text.strip():
            st.warning("Please provide valid text or PDF input.")
        else:
            with st.spinner("Analyzing stylometrics and vector space..."):
                try:
                    res = requests.post(f"{API_URL}/search", json={"text": query_text, "top_k": 3})
                    if res.status_code == 200:
                        verdict = res.json().get("verdict", {}).get("final_verdict", {})
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Predicted Author", verdict.get("author", "Unknown"))
                        col2.metric("Hybrid Score", f"{verdict.get('score', 0.0):.2%}")
                        col3.metric("Verification Status", verdict.get("status", "N/A"))

                        st.markdown("---")
                        
                        ttr, avg_len, burstiness, punct = extract_features(query_text)
                        st.subheader("📊 Stylometric Feature Signatures")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Vocabulary Diversity (TTR)", ttr)
                        m2.metric("Avg Sentence Length", f"{avg_len} wps")
                        m3.metric("Burstiness (Variance)", burstiness)
                        m4.metric("Punctuation Density", punct)

                        categories = ['Vocabulary (TTR)', 'Sentence Length', 'Burstiness', 'Punctuation']
                        fig = go.Figure()

                        fig.add_trace(go.Scatterpolar(
                            r=[ttr * 100, min(avg_len * 3, 100), min(burstiness * 2.5, 100), punct * 500],
                            theta=categories,
                            fill='toself',
                            name='Uploaded Document'
                        ))

                        fig.add_trace(go.Scatterpolar(
                            r=[45, 54, 15, 40],
                            theta=categories,
                            fill='toself',
                            name='Standard LLM Benchmark (Low Variance)',
                            line=dict(dash='dash', color='orange')
                        ))

                        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Could not connect to API at {API_URL}: {str(e)}")

with tab2:
    st.subheader("Ingest Reference Document")
    author_name = st.text_input("Author Label:", value="Human_Architect")
    doc_id = st.text_input("Document ID:", value="doc_001")
    pdf_to_ingest = st.file_uploader("Upload Seed PDF", type=["pdf"], key="ingest")
    
    if st.button("Ingest Document"):
        if pdf_to_ingest:
            doc = fitz.open(stream=pdf_to_ingest.read(), filetype="pdf")
            extracted_text = "".join([page.get_text() for page in doc])
            payload = {"id": doc_id, "author": author_name, "text": extracted_text}
            res = requests.post(f"{API_URL}/ingest", json=payload)
            if res.status_code == 200:
                st.success("Successfully indexed in Qdrant Vector Store!")
