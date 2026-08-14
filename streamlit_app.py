"""
streamlit_app.py  —  LLM ETF Ensemble Dashboard
================================================

Displays results from HuggingFace dataset.
"""

import streamlit as st
import pandas as pd
import json
import requests
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="P2-LLM-ETF-ENSEMBLE",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .ticker-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
        transition: transform 0.2s;
    }
    .ticker-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .confidence-high { color: #27ae60; font-weight: 600; }
    .confidence-medium { color: #f39c12; font-weight: 600; }
    .confidence-low { color: #e74c3c; font-weight: 600; }
    .stMetric {
        background: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)


def load_data():
    """Load the latest results from HuggingFace dataset."""
    try:
        repo_id = "P2SAMAPA/p2-llm-etf-ensemble-results"
        
        # First, list files in the repo
        api_url = f"https://huggingface.co/api/datasets/{repo_id}"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            # Get the latest JSON file
            files_url = f"https://huggingface.co/api/datasets/{repo_id}/refs/main"
            files_response = requests.get(files_url, timeout=10)
            
            if files_response.status_code == 200:
                # The response is a list of file names
                files = files_response.json()
                # Filter JSON files
                json_files = [f for f in files if f.endswith('.json') and f.startswith('llm_etf_ensemble_')]
                
                if json_files:
                    latest = sorted(json_files)[-1]
                    # Download the file
                    data_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{latest}"
                    data_response = requests.get(data_url, timeout=10)
                    
                    if data_response.status_code == 200:
                        data = data_response.json()
                        return data, latest
                    else:
                        st.warning(f"Could not download {latest}: {data_response.status_code}")
                else:
                    st.warning("No JSON files found in the dataset")
            else:
                st.warning(f"Could not list files: {files_response.status_code}")
        else:
            st.warning(f"Could not access dataset: {response.status_code}")
    
    except Exception as e:
        st.warning(f"Error loading from HuggingFace: {str(e)}")
    
    return None, None


def get_confidence_color(confidence):
    if confidence.lower() == "high":
        return "confidence-high"
    elif confidence.lower() == "medium":
        return "confidence-medium"
    else:
        return "confidence-low"


def create_probability_chart(selections):
    """Create a bar chart of expected returns."""
    if not selections:
        return None
    
    df = pd.DataFrame(selections)
    df = df.sort_values('expected_return', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['expected_return'],
        y=df['ticker'],
        orientation='h',
        text=df['expected_return'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        marker_color=['#27ae60' if r > 1.0 else '#f39c12' if r > 0.5 else '#e74c3c' 
                      for r in df['expected_return']],
        hovertemplate='<b>%{y}</b><br>Expected Return: %{x:.1f}%<br>Confidence: %{customdata}<br>Votes: %{text}<extra></extra>',
        customdata=df['confidence']
    ))
    
    fig.update_layout(
        title="Expected Return by ETF",
        xaxis_title="Expected Return (%)",
        yaxis_title="",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        xaxis=dict(gridcolor='#f0f0f0')
    )
    
    return fig


def main():
    st.markdown('<div class="main-header">🤖 P2-LLM-ETF-ENSEMBLE</div>', unsafe_allow_html=True)
    st.markdown("*AI-Powered ETF Selection via Ensemble Voting*")
    
    # Load data
    data, filename = load_data()
    
    if not data:
        st.error("⚠️ No data available. The daily run may not have completed yet.")
        st.info("⏳ Results are typically available by 01:00 UTC daily.")
        if st.button("🔄 Retry", use_container_width=True):
            st.rerun()
        return
    
    # Show last update time
    run_date = data.get('run_date', 'Unknown')
    st.caption(f"📊 Results from: **{run_date}** | File: {filename}")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Dashboard")
        
        universes = list(data.get('universes', {}).keys())
        if universes:
            selected_universe = st.selectbox(
                "Select Universe",
                ["All Universes"] + universes
            )
        
        st.markdown("---")
        
        # Stats
        total_picks = sum(len(u.get('top_picks', [])) for u in data.get('universes', {}).values())
        st.metric("Total Top Picks", total_picks)
        
        total_llms = data.get('ensemble_summary', {}).get('total_llm_calls', 0)
        st.metric("Total LLM Calls", total_llms)
        
        st.markdown("---")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Display content
    if selected_universe == "All Universes":
        for universe_name, universe_data in data.get('universes', {}).items():
            st.markdown(f"## {universe_name}")
            display_universe(universe_data)
            st.markdown("---")
    else:
        universe_data = data.get('universes', {}).get(selected_universe, {})
        st.markdown(f"## {selected_universe}")
        display_universe(universe_data)
    
    # Cross-universe summary
    st.markdown("## 🌟 Cross-Universe Top Picks")
    top_cross = data.get('ensemble_summary', {}).get('top_cross_universe_picks', [])
    if top_cross:
        df_cross = pd.DataFrame(top_cross)
        if 'expected_return' in df_cross.columns:
            df_cross['expected_return'] = df_cross['expected_return'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(
            df_cross,
            use_container_width=True,
            hide_index=True,
            column_config={
                'ticker': 'Ticker',
                'universe': 'Universe',
                'expected_return': 'Expected Return',
                'confidence': 'Confidence',
                'votes': 'Votes'
            }
        )
    else:
        st.info("No cross-universe picks available")
    
    # Footer
    st.markdown("---")
    st.caption(f"Data as of {run_date} | Powered by Ensemble LLM Analysis | Auto-updates daily")


def display_universe(universe_data):
    top_picks = universe_data.get('top_picks', [])
    if not top_picks:
        st.warning("No recommendations available")
        return
    
    # Display as cards
    cols = st.columns(min(len(top_picks), 3))
    for idx, pick in enumerate(top_picks):
        col = cols[idx % len(cols)]
        with col:
            confidence_class = get_confidence_color(pick['confidence'])
            
            # Get model names
            models = pick.get('models', ['unknown'])
            if isinstance(models, list):
                model_names = ', '.join([m.replace('openai/', '').replace('meta-llama/', '').replace('mistralai/', '')[:15] for m in models[:3]])
            else:
                model_names = str(models)[:20]
            
            st.markdown(f"""
            <div class="ticker-card">
                <h3 style="margin:0; font-size:1.8rem;">{pick['ticker']}</h3>
                <div style="font-size:2.2rem; font-weight:700; margin:0.5rem 0; color:#2c3e50;">
                    {pick['expected_return']:.1f}%
                </div>
                <div class="{confidence_class}" style="font-size:1.1rem;">
                    Confidence: {pick['confidence']}
                </div>
                <div style="font-size:0.8rem; color:#666; margin-top:0.5rem;">
                    Votes: {pick.get('votes', 0)} | Models: {model_names}
                </div>
                <div style="font-size:0.85rem; margin-top:0.5rem; color:#444; background:white; padding:0.5rem; border-radius:5px;">
                    {pick.get('rationale', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chart
    fig = create_probability_chart(top_picks)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Show ensemble stats
    stats = universe_data.get('ensemble_stats', {})
    if stats:
        with st.expander("📊 Ensemble Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analysers", stats.get('total_analyzers', 0))
            with col2:
                st.metric("Total Votes", stats.get('total_votes', 0))
            with col3:
                models_used = stats.get('models_used', [])
                st.metric("Models Used", len(models_used) if models_used else 0)
            
            # Show vote distribution
            ticker_votes = stats.get('ticker_votes', {})
            if ticker_votes:
                df_votes = pd.DataFrame([
                    {'Ticker': t, 'Votes': v} 
                    for t, v in ticker_votes.items()
                ]).sort_values('Votes', ascending=False).head(10)
                
                fig_votes = px.bar(
                    df_votes, 
                    x='Ticker', 
                    y='Votes',
                    title='Vote Distribution',
                    color='Votes',
                    color_continuous_scale='Blues',
                    height=300
                )
                fig_votes.update_layout(showlegend=False)
                st.plotly_chart(fig_votes, use_container_width=True)


if __name__ == "__main__":
    main()
