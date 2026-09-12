import os
import streamlit as st
import requests
import fitz  # PyMuPDF
import plotly.graph_objects as go
import spacy
import numpy as np

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import en_core_web_sm
    nlp = en_core_web_sm.load()

st.set_page_config(
    page_title="AI vs Human Authorship Authentication",
    layout="wide"
)

API_URL = os.getenv("API_URL", "https://ai-fingerprint-mvp-1.onrender.com").rstrip("/")

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

# App Title & Subtitle
st.title("AI vs Human Authorship Authentication Platform")
st.markdown("Forensic Engine for Hybrid Vector Semantic Matching & Stylometric Profiling")

tab1, tab2 = st.tabs(["Analyze & Detect PDF/Text", "Ingest Seed PDF Data"])

with tab1:
    st.subheader("Document Verification & Authentication")
    input_type = st.radio("Choose Input Type:", ["PDF Document Upload", "Direct Text Snippet"], horizontal=True)
    
    query_text = ""
    if input_type == "PDF Document Upload":
        uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"], key="verify_pdf")
        if uploaded_file:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            query_text = "".join([page.get_text() for page in doc])
            with st.expander("Preview Extracted PDF Text"):
                st.write(query_text[:1500] + ("..." if len(query_text) > 1500 else ""))
    else:
        query_text = st.text_area("Paste text content here:", height=160, placeholder="Paste article or PDF text...")

    if st.button("Run Authentication Scan", type="primary", use_container_width=True):
        if not query_text.strip():
            st.warning("Please upload a valid PDF or paste text to perform analysis.")
        else:
            with st.spinner("Analyzing stylometrics and comparing vector signatures..."):
                try:
                    res = requests.post(
                        f"{API_URL}/search", 
                        json={"text": query_text, "top_k": 3},
                        timeout=120
                    )
                    if res.status_code == 200:
                        data = res.json().get("verdict", {}).get("final_verdict", {})
                        
                        ai_prob = data.get("ai_probability", 0.0)
                        status = data.get("status", "Analyzed")
                        author = data.get("author", "Unknown")

                        st.markdown("---")
                        
                        v_col1, v_col2 = st.columns([1, 2])
                        with v_col1:
                            if "AI" in author or ai_prob >= 0.50:
                                st.error("### Verdict: AI Generated Content")
                            else:
                                st.success("### Verdict: Authentic Human Writing")
                            st.write(f"**Classification:** {status}")

                        with v_col2:
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=ai_prob * 100,
                                title={'text': "AI Score (%)"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#ff4b4b" if ai_prob >= 0.5 else "#00c853"},
                                    'steps': [
                                        {'range': [0, 45], 'color': "lightgreen"},
                                        {'range': [45, 65], 'color': "yellow"},
                                        {'range': [65, 100], 'color': "salmon"}
                                    ]
                                }
                            ))
                            fig_gauge.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
                            st.plotly_chart(fig_gauge, use_container_width=True)

                        ttr, avg_len, burstiness, punct = extract_features(query_text)
                        st.markdown("### Stylometric Metrics Breakdown")
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Vocabulary Diversity (TTR)", f"{ttr}")
                        m2.metric("Avg Sentence Length", f"{avg_len} words")
                        m3.metric("Burstiness (Variance)", f"{burstiness}")
                        m4.metric("Punctuation Density", f"{punct}")

                        categories = ['Vocabulary Diversity', 'Sentence Length', 'Burstiness Variance', 'Punctuation']
                        fig_radar = go.Figure()

                        fig_radar.add_trace(go.Scatterpolar(
                            r=[ttr * 100, min(avg_len * 3, 100), min(burstiness * 10, 100), punct * 500],
                            theta=categories,
                            fill='toself',
                            name='Your Uploaded Document'
                        ))

                        fig_radar.add_trace(go.Scatterpolar(
                            r=[40, 55, 20, 35],
                            theta=categories,
                            fill='toself',
                            name='GPT Baseline Signature',
                            line=dict(dash='dash', color='orange')
                        ))

                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            showlegend=True,
                            height=380
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                    else:
                        st.error(f"API Error ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Connection Failed: {str(e)}")

with tab2:
    st.subheader("Ingest Reference Data")
    author_name = st.text_input("Author Label:", value="LLM_ChatGPT")
    doc_id = st.text_input("Document ID:", value="doc_001")
    pdf_to_ingest = st.file_uploader("Upload Seed PDF", type=["pdf"], key="ingest_pdf")
    
    if st.button("Ingest into Vector Store", use_container_width=True):
        if pdf_to_ingest:
            doc = fitz.open(stream=pdf_to_ingest.read(), filetype="pdf")
            extracted_text = "".join([page.get_text() for page in doc])
            payload = {"id": doc_id, "author": author_name, "text": extracted_text}
            try:
                res = requests.post(f"{API_URL}/ingest", json=payload, timeout=120)
                if res.status_code == 200:
                    st.success("Successfully indexed reference vector in Qdrant!")
                else:
                    st.error(f"Ingest failed ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")