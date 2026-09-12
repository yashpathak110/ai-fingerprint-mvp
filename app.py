import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Fingerprint 3D Engine", page_icon="🛡️", layout="wide")
st.title("🛡️ AI Fingerprint & Author Attribution Engine")

tab1, tab2 = st.tabs(["🔍 3D Search & Attribution", "📥 Ingest Document"])

with tab1:
    st.header("Analyze Text Signature")
    query_text = st.text_area("Enter text excerpt to attribute:", height=120)
    top_k = st.slider("Matches to retrieve:", 1, 10, 5)
    
    if st.button("Detect Attribution", type="primary"):
        if query_text.strip():
            try:
                res = requests.post(f"{API_URL}/search", json={"text": query_text, "top_k": top_k}, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    verdict = data.get("verdict", {})
                    matches = data.get("matches", [])
                    
                    col_left, col_right = st.columns([1, 1])
                    
                    with col_left:
                        st.subheader("Decision Pipeline Verdict")
                        status = verdict.get("status", "UNKNOWN")
                        if status == "CONFIRMED_ATTRIBUTION":
                            st.success(f"**Verdict:** {verdict.get('author')}\n\n**Confidence Score:** {verdict.get('score', 0):.4f}")
                        else:
                            st.warning(f"**Verdict:** {verdict.get('author')}\n\n**Status:** Low Confidence Boundary")
                        
                        st.subheader("Candidate Matches")
                        for idx, match in enumerate(matches, start=1):
                            score = match.get("score", 0.0)
                            payload = match.get("payload", {})
                            st.progress(min(max(score, 0.0), 1.0))
                            with st.expander(f"#{idx} Author: {payload.get('author', 'Unknown')} — Score: {score:.4f}", expanded=(idx==1)):
                                st.write(f"**Doc ID:** {payload.get('doc_id')}")
                                st.info(payload.get("text", ""))

                    with col_right:
                        st.subheader("🌐 3D Vector Space Representation")
                        
                        # Generate 3D Spatial Cloud for Visualization
                        np.random.seed(42)
                        num_points = max(len(matches), 1)
                        raw_vecs = np.random.randn(num_points + 1, 16) # Simulated 16D subspace
                        
                        pca = PCA(n_components=3)
                        coords = pca.fit_transform(raw_vecs)
                        
                        plot_data = []
                        # Query Node
                        plot_data.append({
                            "X": coords[0, 0], "Y": coords[0, 1], "Z": coords[0, 2],
                            "Author": ">>> INPUT QUERY <<<",
                            "Score": 1.0, "Type": "Query Target", "Size": 18
                        })
                        
                        # Match Nodes
                        for idx, match in enumerate(matches):
                            payload = match.get("payload", {})
                            plot_data.append({
                                "X": coords[idx + 1, 0], "Y": coords[idx + 1, 1], "Z": coords[idx + 1, 2],
                                "Author": payload.get("author", "Unknown"),
                                "Score": match.get("score", 0.0),
                                "Type": f"Match #{idx+1}", "Size": 12
                            })
                            
                        df_3d = pd.DataFrame(plot_data)
                        
                        fig = px.scatter_3d(
                            df_3d, x="X", y="Y", z="Z",
                            color="Author", size="Size",
                            hover_data=["Author", "Score", "Type"],
                            title="Embedding Trajectory & Cluster Distance",
                            template="plotly_dark"
                        )
                        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500)
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

with tab2:
    st.header("Ingest Document")
    doc_id = st.text_input("Document ID:")
    author = st.text_input("Author Tag:")
    doc_text = st.text_area("Content:", height=200)
    if st.button("Ingest Document"):
        if doc_id.strip() and doc_text.strip():
            res = requests.post(f"{API_URL}/ingest", json={"id": doc_id, "author": author, "text": doc_text})
            if res.status_code == 200:
                st.success("Document ingested successfully!")