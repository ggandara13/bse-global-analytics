"""
🏀 BSE Global Analytics Dashboard
Brooklyn Nets & NY Liberty Data Analysis
Senior Data Scientist Interview Project
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page config
st.set_page_config(
    page_title="BSE Global Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #000000;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #000000, #FFFFFF, #000000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .insight-box {
        background-color: #e8f4ea;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_data():
    """Load all BSE data files"""
    data = {}
    data_path = "./bse_data"
    
    # Try to load each file
    files_to_load = {
        'nets_prices': 'nets_pricing_analysis_REAL.csv',
        'knicks_prices': 'knicks_ticket_prices_REAL.csv',
        'comparison': 'nets_vs_knicks_comparison_REAL.csv',
        'attendance': 'nba_attendance_rankings_2024_25.csv',
        'predictions': 'nets_attendance_predictions.csv',
        'reddit': 'reddit_gonets_experience_REAL.csv',
        'weather': 'nyc_weather_historical.csv',
        'games': 'nets_games_all_seasons.csv',
    }
    
    for key, filename in files_to_load.items():
        try:
            data[key] = pd.read_csv(f"{data_path}/{filename}")
        except:
            data[key] = None
    
    return data

# Load data
data = load_data()

# Sidebar navigation - BSE Global Logo
st.sidebar.image("https://media.licdn.com/dms/image/v2/C5622AQGlk2E3fhKiBA/feedshare-shrink_800/feedshare-shrink_800/0/1654477792148?e=2147483647&v=beta&t=TWJuLtxooQURrBz9U2GTG7X9t5iufy2YcuWpyhkcwM4", use_container_width=True)
st.sidebar.title("🏀 BSE Analytics")
st.sidebar.markdown("*Brooklyn Nets • NY Liberty • Barclays Center*")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Executive Summary", "💰 Pricing Analysis", "🎟️ Attendance", "💬 Sentiment Analysis", "🔮 Predictive Model", "💡 Recommendations"]
)

# ============================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================
if page == "📊 Executive Summary":
    st.markdown("<h1 class='main-header'>🏀 BSE Global Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("### Senior Data Scientist Interview Project | December 2025")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📁 Data Files",
            value="48",
            delta="Real API + Research"
        )
    
    with col2:
        st.metric(
            label="📊 Total Rows",
            value="3,236",
            delta="Analyzed"
        )
    
    with col3:
        st.metric(
            label="🎫 Pricing Gap",
            value="6.4x",
            delta="Knicks vs Nets",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="😊 Fan Sentiment",
            value="32%",
            delta="Positive"
        )
    
    st.markdown("---")
    
    # Two columns for key findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔑 Key Findings")
        st.markdown("""
        <div class='insight-box'>
        <strong>1. The Attendance Paradox</strong><br>
        Nets fill 98.1% capacity while ranked #21 with only 14 home wins. 
        For a rebuilding team, this is impressive!
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
        <strong>2. Pricing Gap Opportunity</strong><br>
        Nets: $57 avg | Knicks: $231 avg<br>
        Same opponent (Pistons): $21 vs $258 = 1,106% difference
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
        <strong>3. Sentiment Analysis</strong><br>
        100 Reddit posts analyzed with AI:<br>
        32% Positive | 28% Negative | 30% Neutral | 10% Mixed
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Data Sources")
        st.markdown("""
        | Source | Type | Rows |
        |--------|------|------|
        | NBA API | ✅ Real | 1,400+ |
        | Weather API | ✅ Real | 481 |
        | RapidAPI/SeatGeek | ✅ Real | 35 |
        | Reddit API | ✅ Real | 215 |
        | Research Data | Curated | 1,100+ |
        """)
        
        st.markdown("""
        <div class='warning-box'>
        <strong>⚠️ Context Matters</strong><br>
        The Nets are in a planned REBUILD (#21/30).<br>
        The Knicks are CONTENDERS (#2/30).<br>
        Fair comparison: Nets vs other rebuilding teams.
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PAGE 2: PRICING ANALYSIS
# ============================================
elif page == "💰 Pricing Analysis":
    st.title("💰 Ticket Pricing Analysis")
    st.markdown("*Real-time data from SeatGeek via RapidAPI*")
    
    if data['nets_prices'] is not None:
        nets = data['nets_prices']
        
        # Pricing overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nets Average", f"${nets['seatgeek_price'].mean():.0f}")
        with col2:
            st.metric("Nets Median", f"${nets['seatgeek_price'].median():.0f}")
        with col3:
            st.metric("Price Range", f"${nets['seatgeek_price'].min():.0f} - ${nets['seatgeek_price'].max():.0f}")
        
        st.markdown("---")
        
        # Chart: Price by Opponent
        st.subheader("📊 Ticket Prices by Opponent")
        
        fig = px.bar(
            nets.sort_values('seatgeek_price', ascending=True),
            x='seatgeek_price',
            y='opponent',
            orientation='h',
            color='seatgeek_price',
            color_continuous_scale='RdYlGn_r',
            labels={'seatgeek_price': 'Average Price ($)', 'opponent': 'Opponent'}
        )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tier Analysis
        st.subheader("🏀 Price by Opponent Tier")
        
        if 'tier' in nets.columns:
            tier_avg = nets.groupby('tier')['seatgeek_price'].mean().reset_index()
            
            fig2 = px.bar(
                tier_avg,
                x='tier',
                y='seatgeek_price',
                color='tier',
                color_discrete_map={
                    'Tier 1 (Marquee)': '#FF4136',
                    'Tier 2 (Playoff)': '#FF851B',
                    'Tier 3 (Standard)': '#2ECC40'
                }
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.info(f"**420% Premium**: Tier 1 games (${tier_avg[tier_avg['tier']=='Tier 1 (Marquee)']['seatgeek_price'].values[0]:.0f}) vs Tier 3 (${tier_avg[tier_avg['tier']=='Tier 3 (Standard)']['seatgeek_price'].values[0]:.0f})")
        
        # Nets vs Knicks Comparison
        if data['comparison'] is not None:
            st.markdown("---")
            st.subheader("🏀 Nets vs Knicks: Same Opponent Comparison")
            
            comp = data['comparison']
            
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                name='Nets',
                x=comp['opponent'],
                y=comp['nets_price'],
                marker_color='#000000'
            ))
            fig3.add_trace(go.Bar(
                name='Knicks',
                x=comp['opponent'],
                y=comp['knicks_price'],
                marker_color='#006BB6'
            ))
            fig3.update_layout(barmode='group', height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            st.error(f"**Average Knicks Premium: {comp['knicks_premium_%'].mean():.0f}%** - Knicks charge 6.4x more for identical opponents!")
    else:
        st.warning("Pricing data not loaded. Please ensure CSV files are in ./bse_data/")

# ============================================
# PAGE 3: ATTENDANCE
# ============================================
elif page == "🎟️ Attendance":
    st.title("🎟️ Attendance Analysis")
    
    if data['attendance'] is not None:
        att = data['attendance']
        
        # Nets position
        nets_row = att[att['team'].str.contains('Nets', case=False, na=False)]
        
        if len(nets_row) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("League Rank", f"#{nets_row['rank'].values[0]}/30")
            with col2:
                st.metric("Avg Attendance", f"{nets_row['avg_attendance'].values[0]:,}")
            with col3:
                st.metric("Capacity %", f"{nets_row['pct_capacity'].values[0]}%")
            with col4:
                st.metric("Home Wins", nets_row['home_wins'].values[0])
        
        st.markdown("---")
        
        # Full rankings chart
        st.subheader("📊 NBA Attendance Rankings 2024-25")
        
        fig = px.bar(
            att.sort_values('avg_attendance', ascending=True),
            x='avg_attendance',
            y='team',
            orientation='h',
            color='pct_capacity',
            color_continuous_scale='RdYlGn',
            labels={'avg_attendance': 'Avg Attendance', 'team': 'Team', 'pct_capacity': 'Capacity %'}
        )
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)
        
        # Fair comparison
        st.markdown("---")
        st.subheader("⚖️ Fair Comparison: Nets vs Other Rebuilding Teams")
        
        rebuilding = att[(att['home_wins'] >= 10) & (att['home_wins'] <= 20)]
        
        fig2 = px.scatter(
            rebuilding,
            x='home_wins',
            y='pct_capacity',
            size='avg_attendance',
            color='team',
            hover_data=['rank', 'avg_attendance'],
            labels={'home_wins': 'Home Wins', 'pct_capacity': 'Capacity %'}
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.success("**Insight:** Nets fill 98.1% capacity with only 14 home wins - impressive for a rebuilding team!")

# ============================================
# PAGE 4: SENTIMENT ANALYSIS
# ============================================
elif page == "💬 Sentiment Analysis":
    st.title("💬 Fan Sentiment Analysis")
    st.markdown("*AI-powered analysis of 100+ Reddit posts from r/GoNets*")
    
    # Sentiment breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral', 'Mixed'],
            'Percentage': [32, 28, 30, 10],
            'Color': ['#28a745', '#dc3545', '#6c757d', '#ffc107']
        })
        
        fig = px.pie(
            sentiment_data,
            values='Percentage',
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={
                'Positive': '#28a745',
                'Negative': '#dc3545',
                'Neutral': '#6c757d',
                'Mixed': '#ffc107'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔴 Pain Point Classification")
        
        pain_points = pd.DataFrame({
            'Category': ['External (Kyrie)', 'Fanbase Issues', 'Atmosphere', 'Team Performance', 'Operations', 'Management'],
            'Percentage': [39.5, 21.1, 15.8, 10.5, 7.9, 5.3]
        })
        
        fig2 = px.bar(
            pain_points.sort_values('Percentage', ascending=True),
            x='Percentage',
            y='Category',
            orientation='h',
            color='Percentage',
            color_continuous_scale='Reds'
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Top complaints
    st.subheader("📝 Top 10 Specific Complaints")
    
    complaints = pd.DataFrame({
        'Rank': range(1, 11),
        'Complaint': [
            'Kyrie Irving controversies and antisemitism drama',
            'Quiet/lacking home crowd energy',
            'Fair-weather fans leaving early or giving up',
            'Opposing fans taking over Barclays Center',
            'COVID vaccine mandate preventing Kyrie from playing',
            'Concession workers auto-tipping 30%',
            'Extremely explicit music at games',
            'Poor roster construction decisions',
            'Players having mental lapses and ejections',
            'Team giving up draft picks in trades'
        ],
        'Mentions': [8, 6, 5, 4, 3, 2, 2, 2, 2, 2]
    })
    
    st.dataframe(complaints, use_container_width=True, hide_index=True)
    
    # Positive themes
    st.subheader("✅ Top 5 Positive Themes")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.success("**Player Appreciation**\n\nPraising performances")
    with col2:
        st.success("**Playoff Excitement**\n\nPostseason enthusiasm")
    with col3:
        st.success("**Team Loyalty**\n\nPride in fanbase")
    with col4:
        st.success("**Barclays Atmosphere**\n\nWhen crowd is engaged")
    with col5:
        st.success("**Franchise Progress**\n\nAppreciation for growth")

# ============================================
# PAGE 5: PREDICTIVE MODEL
# ============================================
elif page == "🔮 Predictive Model":
    st.title("🔮 Attendance Prediction Model")
    st.markdown("*Prototype using ticket price as demand proxy*")
    
    if data['predictions'] is not None:
        pred = data['predictions']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        sellouts = len(pred[pred['predicted_attendance'] == 'SELLOUT (99%+)'])
        high = len(pred[pred['predicted_attendance'] == 'HIGH (95-99%)'])
        medium = len(pred[pred['predicted_attendance'] == 'MEDIUM (85-95%)'])
        low = len(pred[pred['predicted_attendance'] == 'LOW (75-85%)'])
        
        with col1:
            st.metric("Predicted Sellouts", sellouts, "Raise prices!")
        with col2:
            st.metric("High Attendance", high)
        with col3:
            st.metric("Medium Attendance", medium)
        with col4:
            st.metric("Low Attendance", low, "Need promos!", delta_color="inverse")
        
        st.markdown("---")
        
        # Prediction chart
        st.subheader("📊 Predicted Attendance by Game")
        
        color_map = {
            'SELLOUT (99%+)': '#28a745',
            'HIGH (95-99%)': '#17a2b8',
            'MEDIUM (85-95%)': '#ffc107',
            'LOW (75-85%)': '#dc3545'
        }
        
        fig = px.bar(
            pred.sort_values('seatgeek_price', ascending=False),
            x='opponent',
            y='seatgeek_price',
            color='predicted_attendance',
            color_discrete_map=color_map,
            labels={'seatgeek_price': 'Ticket Price ($)', 'opponent': 'Opponent'}
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.subheader("📋 Full Predictions")
        st.dataframe(
            pred[['date', 'opponent', 'day_of_week', 'seatgeek_price', 'predicted_attendance']].sort_values('date'),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        st.info("""
        **Model Explanation:**
        - Uses ticket price as proxy for demand
        - Higher prices indicate higher demand → higher attendance
        - For production: Add weather, team performance, day of week as features
        """)

# ============================================
# PAGE 6: RECOMMENDATIONS
# ============================================
elif page == "💡 Recommendations":
    st.title("💡 Strategic Recommendations")
    
    st.markdown("---")
    
    # Immediate
    st.subheader("🔴 Immediate Actions (0-3 months)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Audit Concessions**
        - Investigate auto-tipping
        - Ensure transparent pricing
        - Train staff on customer service
        """)
    
    with col2:
        st.markdown("""
        **Content Filtering**
        - Family-friendly PA music
        - Review playlist standards
        - Create atmosphere guidelines
        """)
    
    with col3:
        st.markdown("""
        **Fan Sections**
        - Create "Brooklyn Loyal" section
        - Incentivize engaged fans
        - Build home-court advantage
        """)
    
    st.markdown("---")
    
    # Short-term
    st.subheader("🟡 Short-Term Actions (3-12 months)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Dynamic Pricing 2.0**
        - Raise Tier 1 prices (below market)
        - Already seeing 420% tier premium
        - Still 6x below Knicks
        """)
    
    with col2:
        st.markdown("""
        **Experience Bundles**
        - Ticket + food + merch packages
        - Target weak games (Tier 3)
        - Convert price-sensitive fans
        """)
    
    with col3:
        st.markdown("""
        **Import Liberty Tactics**
        - Study Liberty atmosphere
        - Apply engagement strategies
        - Cross-promote events
        """)
    
    st.markdown("---")
    
    # Long-term
    st.subheader("🟢 Long-Term Actions (1+ years)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Loyalty Rewards**
        - Reward full-game attendance
        - Season ticket perks
        - Build lifetime value
        """)
    
    with col2:
        st.markdown("""
        **Barclays Basketball**
        - Cross-promote Nets + Liberty
        - Unified venue identity
        - Year-round engagement
        """)
    
    with col3:
        st.markdown("""
        **Brooklyn Identity**
        - Distinct from Knicks
        - Community connection
        - Authentic local brand
        """)
    
    st.markdown("---")
    
    # ROI Estimate
    st.subheader("💰 Revenue Opportunity")
    
    st.markdown("""
    **If Nets capture just 10% of the Knicks pricing gap:**
    
    - Current gap: $174 per ticket
    - 10% capture: $17.40 per ticket
    - × 17,000 seats × 41 home games
    - = **$12.1 million** additional annual revenue
    """)
    
    st.success("**The data shows the opportunity is real. The Liberty proves BSE can execute. The question is: how fast can we move?**")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🏀 BSE Global Analytics Dashboard | Built for Senior Data Scientist Interview<br>
    Data: NBA API, Weather API, RapidAPI/SeatGeek, Reddit API, Claude AI<br>
    December 2025
</div>
""", unsafe_allow_html=True)
