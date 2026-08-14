"""
streamlit_app.py  —  LLM ETF Ensemble Dashboard
================================================

Visualizes LLM recommendations and consensus rankings.
"""

import streamlit as st
import pandas as pd
import json
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

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
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .ticker-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
    }
    .confidence-high {
        color: #27ae60;
        font-weight: 600;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: 600;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


def load_data():
    """Load the latest results from HuggingFace or local file."""
    
    # Try to load from HuggingFace first
    try:
        repo_id = "P2SAMAPA/p2-llm-etf-ensemble-results"
        files = requests.get(f"https://huggingface.co/api/datasets/{repo_id}/refs/main").json()
        
        # Get the most recent JSON file
        json_files = [f for f in files if f.endswith('.json') and f.startswith('llm_etf_ensemble_')]
        if json_files:
            latest = sorted(json_files)[-1]
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{latest}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json(), latest
    except Exception as e:
        st.warning(f"Could not load from HuggingFace: {e}")
    
    # Fallback: try local file
    try:
        import glob
        files = glob.glob("llm_etf_ensemble_*.json")
        if files:
            latest = sorted(files)[-1]
            with open(latest, 'r') as f:
                return json.load(f), latest
    except:
        pass
    
    return None, None


def get_confidence_color(confidence):
    """Return color for confidence level."""
    if confidence.lower() == "high":
        return "confidence-high"
    elif confidence.lower() == "medium":
        return "confidence-medium"
    else:
        return "confidence-low"


def create_probability_chart(selections):
    """Create a bar chart of probabilities."""
    if not selections:
        return None
    
    df = pd.DataFrame(selections)
    df = df.sort_values('probability', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['probability'],
        y=df['ticker'],
        orientation='h',
        text=df['probability'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        marker_color=['#27ae60' if p > 70 else '#f39c12' if p > 50 else '#e74c3c' 
                      for p in df['probability']],
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.1f}%<br>Confidence: %{customdata}<extra></extra>',
        customdata=df['confidence']
    ))
    
    fig.update_layout(
        title="Probability of Positive Return",
        xaxis_title="Probability (%)",
        yaxis_title="ETF",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">🤖 P2-LLM-ETF-ENSEMBLE</div>', unsafe_allow_html=True)
    st.markdown("*LLM-Powered ETF Selection via Ensemble Voting*")
    
    # Load data
    data, filename = load_data()
    
    if not data:
        st.error("No data available. Please run the trainer first.")
        st.info("Run `python trainer.py` to generate results.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Dashboard Controls")
        
        run_date = data.get('run_date', 'Unknown')
        st.markdown(f"**Run Date:** {run_date}")
        
        if filename:
            st.markdown(f"**File:** {filename}")
        
        st.markdown("---")
        
        # Universe filter
        universes = list(data.get('universes', {}).keys())
        if universes:
            selected_universe = st.selectbox(
                "Select Universe",
                ["All Universes"] + universes
            )
        
        st.markdown("---")
        st.markdown("### 📈 Stats")
        
        total_picks = sum(len(u.get('top_picks', [])) for u in data.get('universes', {}).values())
        st.metric("Total Top Picks", total_picks)
        
        total_llms = data.get('ensemble_summary', {}).get('total_llm_calls', 0)
        st.metric("Total LLM Calls", total_llms)
    
    # Main content
    if selected_universe == "All Universes":
        # Show all universes
        for universe_name, universe_data in data.get('universes', {}).items():
            st.markdown(f"## {universe_name}")
            
            top_picks = universe_data.get('top_picks', [])
            if not top_picks:
                st.warning(f"No recommendations for {universe_name}")
                continue
            
            # Create columns for display
            cols = st.columns(min(len(top_picks), 3))
            
            for idx, pick in enumerate(top_picks):
                col = cols[idx % len(cols)]
                with col:
                    confidence_class = get_confidence_color(pick['confidence'])
                    st.markdown(f"""
                    <div class="ticker-card">
                        <h3 style="margin:0;">{pick['ticker']}</h3>
                        <div style="font-size:1.8rem; font-weight:700; margin:0.5rem 0;">
                            {pick['probability']:.1f}%
                        </div>
                        <div class="{confidence_class}">
                            Confidence: {pick['confidence']}
                        </div>
                        <div style="font-size:0.8rem; color:#666; margin-top:0.5rem;">
                            Votes: {pick.get('votes', 0)} | Models: {', '.join(pick.get('models', ['unknown'])[:3])}
                        </div>
                        <div style="font-size:0.85rem; margin-top:0.5rem; color:#444;">
                            {pick.get('rationale', '')[:100]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Chart
            fig = create_probability_chart(top_picks)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
    else:
        # Show single universe
        universe_data = data.get('universes', {}).get(selected_universe, {})
        
        st.markdown(f"## {selected_universe}")
        st.markdown(f"**ETFs in Universe:** {', '.join(universe_data.get('all_tickers', []))}")
        
        top_picks = universe_data.get('top_picks', [])
        if not top_picks:
            st.warning(f"No recommendations for {selected_universe}")
            return
        
        # Display top picks as cards
        cols = st.columns(min(len(top_picks), 3))
        for idx, pick in enumerate(top_picks):
            col = cols[idx % len(cols)]
            with col:
                confidence_class = get_confidence_color(pick['confidence'])
                st.markdown(f"""
                <div class="ticker-card">
                    <h3 style="margin:0;">{pick['ticker']}</h3>
                    <div style="font-size:1.8rem; font-weight:700; margin:0.5rem 0;">
                        {pick['probability']:.1f}%
                    </div>
                    <div class="{confidence_class}">
                        Confidence: {pick['confidence']}
                    </div>
                    <div style="font-size:0.8rem; color:#666; margin-top:0.5rem;">
                        Votes: {pick.get('votes', 0)} | Models: {', '.join(pick.get('models', ['unknown'])[:3])}
                    </div>
                    <div style="font-size:0.85rem; margin-top:0.5rem; color:#444;">
                        {pick.get('rationale', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Chart
        fig = create_probability_chart(top_picks)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Ensemble stats
        stats = universe_data.get('ensemble_stats', {})
        if stats:
            st.markdown("### Ensemble Voting Details")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Analyzers", stats.get('total_analyzers', 0))
            with col2:
                st.metric("Total Votes", stats.get('total_votes', 0))
            
            # Show vote distribution
            ticker_votes = stats.get('ticker_votes', {})
            if ticker_votes:
                df_votes = pd.DataFrame([
                    {'Ticker': t, 'Votes': v} 
                    for t, v in ticker_votes.items()
                ]).sort_values('Votes', ascending=False)
                
                fig_votes = px.bar(
                    df_votes, 
                    x='Ticker', 
                    y='Votes',
                    title='Vote Distribution by Ticker',
                    color='Votes',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_votes, use_container_width=True)
    
    # Cross-universe summary
    st.markdown("## 🌟 Cross-Universe Top Picks")
    
    top_cross = data.get('ensemble_summary', {}).get('top_cross_universe_picks', [])
    if top_cross:
        df_cross = pd.DataFrame(top_cross)
        df_cross = df_cross[['ticker', 'universe', 'probability', 'confidence', 'votes']]
        df_cross['probability'] = df_cross['probability'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            df_cross,
            use_container_width=True,
            hide_index=True,
            column_config={
                'ticker': 'Ticker',
                'universe': 'Universe',
                'probability': 'Probability',
                'confidence': 'Confidence',
                'votes': 'Votes'
            }
        )
    else:
        st.info("No cross-universe picks available")
    
    # Footer
    st.markdown("---")
    st.caption(f"Data as of {run_date} | Powered by Ensemble LLM Analysis")


if __name__ == "__main__":
    main()
