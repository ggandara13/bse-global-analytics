"""
REPLACE your existing "⭐ Barclays: Fan Experience" section in app.py with this code.
This displays Claude-powered intelligent analysis instead of basic review table.
"""

# ============================================
# PAGE: BARCLAYS FAN EXPERIENCE (CLAUDE-POWERED)
# ============================================
elif page == "⭐ Barclays: Fan Experience":
    st.markdown("<h1 class='barclays-header'>⭐ Barclays Center: AI-Powered Fan Experience Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>🤖 Powered by Claude AI - Intelligent Issue Detection & Root Cause Analysis</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Try to load Claude analysis
    claude_analysis_exists = False
    try:
        issues_claude = pd.read_csv('bse_data/barclays_issues_claude_analysis.csv')
        positives_claude = pd.read_csv('bse_data/barclays_positives_claude.csv')
        priority_claude = pd.read_csv('bse_data/barclays_priority_matrix_claude.csv')
        segments_claude = pd.read_csv('bse_data/barclays_fan_segments_claude.csv')
        summary_claude = pd.read_csv('bse_data/barclays_executive_summary_claude.csv')
        claude_analysis_exists = True
    except:
        pass
    
    if claude_analysis_exists:
        # EXECUTIVE SUMMARY
        st.subheader("📋 AI Executive Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sentiment = summary_claude['overall_sentiment_score'].values[0]
            st.metric("🎯 AI Sentiment Score", f"{sentiment}/5.0", "Claude Analysis")
        with col2:
            st.metric("📊 Reviews Analyzed", "25", "TripAdvisor + Yelp")
        with col3:
            urgent = summary_claude['urgent_action_needed'].values[0]
            color = "🔴" if "yes" in str(urgent).lower() else "🟢"
            st.metric(f"{color} Urgent Action", urgent.split(':')[0] if ':' in str(urgent) else urgent[:20])
        
        st.markdown(f"""
        <div class='insight-box'>
        <h4>🔑 Key Finding</h4>
        <p>{summary_claude['key_finding'].values[0]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ISSUES WITH ROOT CAUSE ANALYSIS
        st.subheader("🔴 AI-Identified Issues with Root Cause Analysis")
        
        for i, row in issues_claude.iterrows():
            severity_color = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(row['severity'], "⚪")
            
            with st.expander(f"{severity_color} {row['category']} ({row['mention_count']} mentions) - {row['severity']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **🔍 Root Cause Analysis:**  
                    {row['root_cause']}
                    
                    **💰 Revenue Impact:**  
                    - Type: {row['revenue_impact_type']}
                    - Estimate: {row['revenue_impact_estimate']}
                    - Confidence: {row['revenue_impact_confidence']}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **✅ Recommended Action:**  
                    {row['recommended_action']}
                    
                    **⚙️ Implementation:**  
                    - Difficulty: {row['implementation_difficulty']}
                    - Expected ROI: {row['expected_roi']}
                    """)
        
        st.markdown("---")
        
        # PRIORITY ACTION MATRIX
        st.subheader("🎯 AI-Generated Priority Matrix")
        
        # Visual priority chart
        fig = px.bar(
            priority_claude.head(8),
            x='priority_score',
            y='action',
            orientation='h',
            color='quick_win',
            color_discrete_map={True: '#28a745', False: '#1D428A'},
            labels={'priority_score': 'Priority Score (1-10)', 'action': 'Action Item', 'quick_win': 'Quick Win'}
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick wins callout
        quick_wins = priority_claude[priority_claude['quick_win'] == True]
        if len(quick_wins) > 0:
            st.markdown("""
            <div class='insight-box'>
            <h4>⚡ Quick Wins (High Impact, Easy Implementation)</h4>
            </div>
            """, unsafe_allow_html=True)
            for _, win in quick_wins.iterrows():
                st.markdown(f"• **{win['action']}** - Est. Cost: {win['estimated_cost']} → {win['expected_benefit']}")
        
        st.markdown("---")
        
        # POSITIVE HIGHLIGHTS
        st.subheader("✅ AI-Identified Strengths to Leverage")
        
        cols = st.columns(len(positives_claude))
        for i, (col, (_, row)) in enumerate(zip(cols, positives_claude.iterrows())):
            with col:
                st.markdown(f"""
                <div class='barclays-box' style='text-align: center;'>
                <h4>💪 {row['strength']}</h4>
                <p><b>{row['mention_count']}</b> mentions</p>
                <p style='font-size: 0.9rem;'>{row['leverage_opportunity']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # FAN SEGMENTS
        st.subheader("👥 AI-Identified Fan Segments")
        
        for _, segment in segments_claude.iterrows():
            with st.expander(f"🎯 {segment['segment']}"):
                st.markdown(f"**Profile:** {segment['characteristics']}")
                st.markdown(f"**Pain Points:** {segment['pain_points']}")
                st.markdown(f"**Opportunities:** {segment['opportunities']}")
        
        st.markdown("---")
        
        # COMPETITIVE INSIGHTS
        st.subheader("🏟️ Competitive Positioning")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='warning-box'>
            <h4>vs Madison Square Garden</h4>
            <p>{summary_claude['vs_msg'].values[0]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='insight-box'>
            <h4>🏆 Barclays Unique Advantage</h4>
            <p>{summary_claude['unique_advantage'].values[0]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <p class='data-source'>Analysis powered by Claude AI (Anthropic) | 25 reviews analyzed | 
        Root cause analysis, revenue impact estimation, and recommendations generated via LLM reasoning</p>
        """, unsafe_allow_html=True)
    
    else:
        # Fallback to basic display if Claude analysis not run
        st.warning("⚠️ Claude AI analysis not found. Run `python analyze_reviews_claude.py` first!")
        
        if 'barclays_reviews' in data:
            reviews_df = data['barclays_reviews']
            st.subheader("📝 Raw Fan Reviews")
            st.dataframe(reviews_df, use_container_width=True, hide_index=True)
            
            st.info("💡 To enable AI-powered analysis, run: `python analyze_reviews_claude.py`")
