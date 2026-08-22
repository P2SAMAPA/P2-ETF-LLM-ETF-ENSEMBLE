"""
streamlit_app.py  —  LLM ETF Ensemble Dashboard
================================================

Displays results from HuggingFace dataset with fallback to local files.
"""

import streamlit as st
import pandas as pd
import json
import requests
import os
import glob
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
    .llm-tag {
        display: inline-block;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.65rem;
        margin: 2px;
        white-space: nowrap;
    }
    .llm-tag-openai { background: #10a37f; }
    .llm-tag-anthropic { background: #d97757; }
    .llm-tag-meta { background: #3168e0; }
    .llm-tag-mistral { background: #f7b731; color: #1a1a1a; }
    .llm-tag-ollama { background: #764ba2; }
    .llm-tag-google { background: #4285f4; }
    .llm-tag-deepseek { background: #1e293b; }
    .llm-tag-qwen { background: #d32f2f; }
    .llm-tag-other { background: #6c757d; }
    .provider-group {
        background: white;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)


def get_llm_provider(model_name):
    """Get the provider of an LLM model."""
    if not model_name:
        return 'Unknown'
    model_name = model_name.lower()
    if 'openai' in model_name or 'gpt' in model_name:
        return 'OpenAI'
    elif 'anthropic' in model_name or 'claude' in model_name:
        return 'Anthropic'
    elif 'meta' in model_name or 'llama' in model_name:
        return 'Meta'
    elif 'mistral' in model_name or 'mixtral' in model_name:
        return 'Mistral'
    elif 'ollama' in model_name:
        return 'Ollama'
    elif 'google' in model_name or 'gemini' in model_name:
        return 'Google'
    elif 'deepseek' in model_name:
        return 'DeepSeek'
    elif 'qwen' in model_name:
        return 'Qwen'
    elif 'cohere' in model_name:
        return 'Cohere'
    else:
        return 'Other'


def get_llm_color(model_name):
    """Get color class for an LLM model."""
    provider = get_llm_provider(model_name)
    color_map = {
        'OpenAI': 'llm-tag-openai',
        'Anthropic': 'llm-tag-anthropic',
        'Meta': 'llm-tag-meta',
        'Mistral': 'llm-tag-mistral',
        'Ollama': 'llm-tag-ollama',
        'Google': 'llm-tag-google',
        'DeepSeek': 'llm-tag-deepseek',
        'Qwen': 'llm-tag-qwen',
        'Other': 'llm-tag-other',
        'Unknown': 'llm-tag-other'
    }
    return color_map.get(provider, 'llm-tag-other')


def get_short_model_name(model_name):
    """Get short display name for a model."""
    if not model_name:
        return 'unknown'
    if '/' in model_name:
        return model_name.split('/')[-1][:15]
    return model_name[:15]


def load_from_huggingface():
    """Load the latest results from HuggingFace dataset."""
    try:
        repo_id = "P2SAMAPA/p2-llm-etf-ensemble-results"
        
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        for date in [today, yesterday]:
            filename = f"llm_etf_ensemble_{date}.json"
            data_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
            response = requests.get(data_url, timeout=10)
            if response.status_code == 200:
                return response.json(), filename
        
        return None, None
    except Exception as e:
        st.warning(f"Error loading from HuggingFace: {str(e)}")
        return None, None


def load_from_local():
    """Fallback: load from local files."""
    try:
        json_files = glob.glob("llm_etf_ensemble_*.json")
        if json_files:
            latest = sorted(json_files)[-1]
            with open(latest, 'r') as f:
                return json.load(f), latest
    except Exception as e:
        st.warning(f"Error loading local file: {str(e)}")
    
    return None, None


def load_data():
    """Load data from HuggingFace with fallback to local."""
    data, filename = load_from_huggingface()
    if data:
        return data, f"HF: {filename}"
    
    data, filename = load_from_local()
    if data:
        return data, f"Local: {filename}"
    
    return None, None


def get_confidence_color(confidence):
    if confidence is None:
        return "confidence-medium"
    if confidence.lower() == "high":
        return "confidence-high"
    elif confidence.lower() == "medium":
        return "confidence-medium"
    else:
        return "confidence-low"


def create_probability_chart(selections):
    """Bar chart of each pick's consensus predicted next-1-month return
    (average of each model's own forward-looking estimate)."""
    if not selections:
        return None

    df = pd.DataFrame(selections)
    return_col = 'predicted_return_1m' if 'predicted_return_1m' in df.columns else 'expected_return'
    df[return_col] = pd.to_numeric(df.get(return_col), errors='coerce').fillna(0)
    df = df.sort_values(return_col, ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[return_col],
        y=df['ticker'],
        orientation='h',
        text=df[return_col].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        marker_color=['#27ae60' if r > 1.0 else '#f39c12' if r > 0.5 else '#e74c3c'
                      for r in df[return_col]],
        hovertemplate='<b>%{y}</b><br>Predicted 1M Return: %{x:.1f}%<br>Confidence: %{customdata}<extra></extra>',
        customdata=df['confidence']
    ))

    fig.update_layout(
        title="Predicted Next-1-Month Return by ETF (AI consensus estimate, not guaranteed)",
        xaxis_title="Predicted 1M Return (%)",
        yaxis_title="",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        xaxis=dict(gridcolor='#f0f0f0')
    )

    return fig


def get_all_models_from_data(data):
    """Extract all unique models from the data."""
    all_models = set()
    
    for universe_name, universe_data in data.get('universes', {}).items():
        for pick in universe_data.get('top_picks', []):
            models = pick.get('models', [])
            if isinstance(models, list):
                for m in models:
                    if m and m != 'unknown':
                        all_models.add(m)
        
        stats = universe_data.get('ensemble_stats', {})
        models_used = stats.get('models_used', [])
        if isinstance(models_used, list):
            for m in models_used:
                if m and m != 'unknown':
                    all_models.add(m)
    
    return sorted(list(all_models))


def display_llm_sidebar(all_models):
    """Display LLM models in the sidebar with provider grouping."""
    if not all_models:
        st.markdown("### 🤖 LLMs Used")
        st.info("No LLM data available in the results")
        return
    
    providers = {}
    for model in all_models:
        provider = get_llm_provider(model)
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(model)
    
    st.markdown("### 🤖 LLMs Used")
    st.markdown(f"**Total Models:** {len(all_models)}")
    
    for provider, models in sorted(providers.items()):
        st.markdown(f"""
        <div class="provider-group">
            <div style="font-weight:600; font-size:0.8rem; color:#2c3e50; margin-bottom:4px;">{provider} ({len(models)})</div>
            <div style="display:flex; flex-wrap:wrap; gap:3px;">
        """, unsafe_allow_html=True)
        
        for model in models[:15]:
            short_name = get_short_model_name(model)
            color_class = get_llm_color(model)
            st.markdown(f'<span class="llm-tag {color_class}">{short_name}</span>', unsafe_allow_html=True)
        
        if len(models) > 15:
            st.markdown(f'<span class="llm-tag llm-tag-other">+{len(models)-15} more</span>', unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)


def display_universe(universe_data, universe_name):
    """Display a single universe's results."""
    top_picks = universe_data.get('top_picks', [])
    if not top_picks:
        st.warning(f"No recommendations available for {universe_name}")
        return
    
    cols = st.columns(min(len(top_picks), 3))
    for idx, pick in enumerate(top_picks):
        col = cols[idx % len(cols)]
        with col:
            confidence_class = get_confidence_color(pick.get('confidence'))
            
            models = pick.get('models', [])
            if isinstance(models, list) and models:
                model_tags = []
                for m in models[:5]:
                    if m and m != 'unknown':
                        short_name = get_short_model_name(m)
                        color_class = get_llm_color(m)
                        model_tags.append(f'<span class="llm-tag {color_class}">{short_name}</span>')
                model_html = ' '.join(model_tags)
                if len(models) > 5:
                    model_html += f' <span class="llm-tag llm-tag-other">+{len(models)-5} more</span>'
            else:
                model_html = '<span class="llm-tag llm-tag-other">No model data</span>'
            
            predicted_return = pick.get('predicted_return_1m', pick.get('expected_return'))
            return_display = f"{predicted_return:.1f}%" if isinstance(predicted_return, (int, float)) else "n/a"

            pred_range = pick.get('predicted_return_1m_range')
            range_display = f" (range {pred_range[0]:.1f}% to {pred_range[1]:.1f}%)" if pred_range else ""

            trailing = pick.get('trailing_return_1m')
            trailing_display = f"{trailing:.1f}%" if isinstance(trailing, (int, float)) else "n/a"
            vol = pick.get('annualized_volatility_pct')
            vol_display = f"{vol:.1f}%" if isinstance(vol, (int, float)) else "n/a"

            points = pick.get('points')
            points_display = f" · {points} pts" if points is not None else ""

            st.markdown(f"""
            <div class="ticker-card">
                <h3 style="margin:0; font-size:1.8rem;">{pick.get('ticker', 'N/A')}</h3>
                <div style="font-size:2.2rem; font-weight:700; margin:0.5rem 0; color:#2c3e50;">
                    {return_display}
                </div>
                <div style="font-size:0.75rem; color:#888; margin-top:-0.4rem;">
                    predicted next-1-month return (AI consensus estimate, not guaranteed){range_display}
                </div>
                <div class="{confidence_class}" style="font-size:1.1rem; margin-top:0.5rem;">
                    Confidence: {pick.get('confidence', 'Medium')}
                </div>
                <div style="font-size:0.9rem; color:#666; margin-top:0.3rem;">
                    🗳️ {pick.get('votes', 0)} model(s) selected this ETF{points_display}
                </div>
                <div style="font-size:0.75rem; color:#888; margin-top:0.2rem;">
                    trailing 1M: {trailing_display} · ann. volatility: {vol_display}
                </div>
                <div style="margin-top:0.3rem; font-size:0.7rem; color:#888;">
                    Models that voted:
                </div>
                <div style="margin-top:0.3rem;">
                    {model_html}
                </div>
                <div style="font-size:0.85rem; margin-top:0.5rem; color:#444; background:white; padding:0.5rem; border-radius:5px;">
                    {pick.get('rationale', '')[:200]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    fig = create_probability_chart(top_picks)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    stats = universe_data.get('ensemble_stats', {})
    if stats:
        with st.expander("📊 Ensemble Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Queried", stats.get('models_queried', 0))
            with col2:
                st.metric("Models Responded", stats.get('models_responded', 0))
            with col3:
                models_used = stats.get('models_used', [])
                st.metric("Distinct Models Used", len(models_used) if models_used else 0)

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


def main():
    st.markdown('<div class="main-header">🤖 P2-LLM-ETF-ENSEMBLE</div>', unsafe_allow_html=True)
    st.markdown("*AI-Powered ETF Selection via Ensemble Voting*")
    
    data, source = load_data()
    
    if not data:
        st.error("⚠️ No data available. The daily run may not have completed yet.")
        st.info("⏳ Results are typically available by 01:00 UTC daily.")
        
        st.markdown("### To generate results:")
        st.code("""
# Install dependencies
pip install -r requirements.txt

# Set your API key(s) — at least one required
export OPENROUTER_API_KEY="your-openrouter-key"
export OLLAMA_API_KEY="your-ollama-cloud-key"

# Run the trainer
python trainer.py
        """, language="bash")
        
        if st.button("🔄 Retry", use_container_width=True):
            st.rerun()
        return
    
    all_models = get_all_models_from_data(data)
    
    with st.sidebar:
        st.markdown("## 📊 Dashboard")
        
        universes = list(data.get('universes', {}).keys())
        if universes:
            selected_universe = st.selectbox(
                "Select Universe",
                ["All Universes"] + universes
            )
        
        st.markdown("---")
        
        total_picks = sum(len(u.get('top_picks', [])) for u in data.get('universes', {}).values())
        st.metric("Total Top Picks", total_picks)
        st.metric("Unique Models", len(all_models))

        models_meta = data.get('models', {})
        if models_meta:
            st.caption(
                f"Discovered this run: {len(models_meta.get('openrouter', []))} free "
                f"OpenRouter models, {len(models_meta.get('ollama', []))} Ollama models"
            )

        st.markdown("---")
        display_llm_sidebar(all_models)
        
        st.markdown("---")
        st.caption(f"📊 Source: {source}")
        run_date = data.get('run_date', 'Unknown')
        st.caption(f"📅 Date: {run_date}")
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    if selected_universe == "All Universes":
        for universe_name, universe_data in data.get('universes', {}).items():
            st.markdown(f"## {universe_name}")
            display_universe(universe_data, universe_name)
            st.markdown("---")
    else:
        universe_data = data.get('universes', {}).get(selected_universe, {})
        st.markdown(f"## {selected_universe}")
        display_universe(universe_data, selected_universe)
    
    st.markdown("## 🌟 Cross-Universe Top Picks")
    top_cross = data.get('ensemble_summary', {}).get('top_cross_universe_picks', [])
    if top_cross:
        df_cross = pd.DataFrame(top_cross)
        if 'predicted_return_1m' in df_cross.columns:
            df_cross['predicted_return_1m'] = df_cross['predicted_return_1m'].apply(
                lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else "n/a"
            )
        st.dataframe(
            df_cross,
            use_container_width=True,
            hide_index=True,
            column_config={
                'ticker': 'Ticker',
                'universe': 'Universe',
                'predicted_return_1m': 'Predicted 1M Return',
                'confidence': 'Confidence',
                'votes': 'Votes'
            }
        )
    else:
        st.info("No cross-universe picks available")
    
    st.markdown("---")
    st.caption(f"Data as of {run_date} | Powered by Ensemble LLM Analysis | Auto-updates daily")


if __name__ == "__main__":
    main()
