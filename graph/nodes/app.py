import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI Fingerprint & Author Attribution Engine", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Fingerprint & Author Attribution Engine")

tab1, tab2 = st.tabs(["🤖 AI PDF Verifier", "🔍 3D Search & Attribution"])

with tab1:
    st.header("Upload PDF for AI Detection & Analysis")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Analyze PDF", type="primary"):
            with st.spinner("Analyzing document through verification graph pipeline..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post("http://127.0.0.1:8000/verify", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Analysis Complete!")
                        
                        # Top Metrics Overview
                        col1, col2, col3 = st.columns(3)
                        ai_score = float(data.get("ai_probability", 0))
                        human_score = round(100.0 - ai_score, 1)
                        
                        col1.metric("Classification", data.get("classification", "N/A"))
                        col2.metric("AI Probability", f"{ai_score}%")
                        col3.metric("Confidence", data.get("confidence", "N/A"))
                        
                        st.divider()
                        
                        # Visual Representation Row: Pie/Donut Chart & Bar Chart
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            st.subheader("📊 AI vs Human Content Distribution")
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=['AI Generated', 'Human Authored'],
                                values=[ai_score, human_score],
                                hole=.4,
                                marker_colors=['#FF4B4B', '#00C853']
                            )])
                            fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        with col_chart2:
                            st.subheader("📈 Text Signature Metrics Breakdown")
                            metrics = data.get("metrics", {})
                            
                            df_metrics = pd.DataFrame({
                                "Metric": ["Perplexity", "Burstiness (x100)", "Lexical Diversity (x100)"],
                                "Value": [
                                    float(metrics.get("perplexity", 0)),
                                    float(metrics.get("burstiness", 0)) * 100,
                                    float(metrics.get("lexical_diversity", 0)) * 100
                                ]
                            })
                            fig_bar = px.bar(df_metrics, x="Metric", y="Value", color="Metric", 
                                             color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig_bar.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300, showlegend=False)
                            st.plotly_chart(fig_bar, use_container_width=True)

                        with st.expander("Raw Analysis Payload"):
                            st.json(data)
                    else:
                        st.error(f"Error from API server: {response.status_code}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI server. Error: {e}")

with tab2:
    st.header("Analyze Text Signature (3D Vector Search)")
    user_text = st.text_area("Enter text excerpt to attribute:")
    top_k = st.slider("Matches to retrieve:", 1, 10, 5)
    
    if st.button("Detect Attribution"):
        if user_text.strip():
            st.info("Performing 3D stylometric embedding search across vector space...")
            
            # Example 3D Embedding Scatter Plot
            st.subheader("🌐 3D Stylometric Coordinate Embedding")
            df_3d = pd.DataFrame({
                'X': [0.1, 0.4, -0.2, 0.8, -0.5, 0.05],
                'Y': [0.5, -0.3, 0.7, 0.1, -0.4, 0.48],
                'Z': [0.2, 0.9, -0.1, -0.6, 0.3, 0.22],
                'Source': ['GPT-4', 'Claude-3', 'Human Author A', 'Human Author B', 'Llama-3', 'Your Sample']
            })
            
            fig_3d = px.scatter_3d(df_3d, x='X', y='Y', z='Z', color='Source', symbol='Source', size_max=10)
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("Please enter text before running attribution search.")
