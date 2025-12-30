"""
Claude API-Powered Intelligent Review Analysis
BSE Global - Barclays Center Fan Experience

This script demonstrates advanced NLP using Claude API:
- Intelligent issue extraction (not keyword matching)
- Root cause analysis
- Revenue impact estimation
- Actionable recommendations with priority scoring

Author: Gerardo Gandara
"""

import anthropic
import pandas as pd
import json
import os

# Initialize Claude client
client = anthropic.Anthropic()

def analyze_reviews_with_claude(reviews_df):
    """
    Send all reviews to Claude for intelligent analysis
    Returns structured insights
    """
    
    # Prepare reviews text
    reviews_text = "\n\n".join([
        f"Review {i+1} (Rating: {row['rating']}/5, Source: {row['source']}):\n{row['text']}"
        for i, row in reviews_df.iterrows()
    ])
    
    prompt = f"""You are a senior data scientist analyzing fan experience data for BSE Global (Brooklyn Nets, NY Liberty, Barclays Center).

Analyze these {len(reviews_df)} fan reviews from Barclays Center and provide a comprehensive analysis.

<reviews>
{reviews_text}
</reviews>

Provide your analysis in the following JSON structure. Be specific and quantitative where possible:

{{
    "executive_summary": {{
        "overall_sentiment_score": <float 1-5>,
        "key_finding": "<one sentence>",
        "urgent_action_needed": "<yes/no with reason>"
    }},
    
    "issue_categories": [
        {{
            "category": "<issue name>",
            "mention_count": <int>,
            "severity": "<Critical/High/Medium/Low>",
            "sample_quotes": ["<quote1>", "<quote2>"],
            "root_cause": "<why this happens>",
            "revenue_impact": {{
                "type": "<Lost tickets/Lower concessions/Reduced loyalty/Brand damage>",
                "estimated_annual_impact": "<$X-Y range>",
                "confidence": "<High/Medium/Low>"
            }},
            "recommended_action": "<specific fix>",
            "implementation_difficulty": "<Easy/Medium/Hard>",
            "expected_roi": "<description>"
        }}
    ],
    
    "positive_highlights": [
        {{
            "strength": "<what fans love>",
            "mention_count": <int>,
            "leverage_opportunity": "<how to amplify this>"
        }}
    ],
    
    "competitive_insights": {{
        "vs_msg": "<how Barclays compares to Madison Square Garden based on reviews>",
        "vs_ubs_arena": "<how Barclays compares to UBS Arena>",
        "unique_advantage": "<Barclays differentiator>"
    }},
    
    "priority_matrix": [
        {{
            "action": "<specific action>",
            "priority_score": <1-10>,
            "quick_win": <true/false>,
            "estimated_cost": "<$X-Y>",
            "expected_benefit": "<description>"
        }}
    ],
    
    "fan_segments_identified": [
        {{
            "segment": "<segment name>",
            "characteristics": "<description>",
            "pain_points": ["<point1>", "<point2>"],
            "opportunities": ["<opp1>", "<opp2>"]
        }}
    ],
    
    "data_quality_notes": {{
        "sample_size_assessment": "<adequate/limited/insufficient>",
        "potential_biases": ["<bias1>", "<bias2>"],
        "recommended_additional_data": ["<data1>", "<data2>"]
    }}
}}

Return ONLY valid JSON, no other text."""

    # Call Claude API
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse response
    response_text = message.content[0].text
    
    # Clean up response if needed
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    
    return json.loads(response_text.strip())


def save_analysis_to_csvs(analysis, output_dir="bse_data"):
    """Convert Claude analysis to CSV files for dashboard"""
    
    # 1. Issue Categories
    issues_df = pd.DataFrame(analysis['issue_categories'])
    issues_df['revenue_impact_type'] = issues_df['revenue_impact'].apply(lambda x: x['type'])
    issues_df['revenue_impact_estimate'] = issues_df['revenue_impact'].apply(lambda x: x['estimated_annual_impact'])
    issues_df['revenue_impact_confidence'] = issues_df['revenue_impact'].apply(lambda x: x['confidence'])
    issues_df = issues_df.drop(columns=['revenue_impact', 'sample_quotes'])
    issues_df.to_csv(f"{output_dir}/barclays_issues_claude_analysis.csv", index=False)
    print(f"✅ Saved: {output_dir}/barclays_issues_claude_analysis.csv")
    
    # 2. Positive Highlights
    positives_df = pd.DataFrame(analysis['positive_highlights'])
    positives_df.to_csv(f"{output_dir}/barclays_positives_claude.csv", index=False)
    print(f"✅ Saved: {output_dir}/barclays_positives_claude.csv")
    
    # 3. Priority Matrix
    priority_df = pd.DataFrame(analysis['priority_matrix'])
    priority_df = priority_df.sort_values('priority_score', ascending=False)
    priority_df.to_csv(f"{output_dir}/barclays_priority_matrix_claude.csv", index=False)
    print(f"✅ Saved: {output_dir}/barclays_priority_matrix_claude.csv")
    
    # 4. Fan Segments
    segments_df = pd.DataFrame(analysis['fan_segments_identified'])
    segments_df['pain_points'] = segments_df['pain_points'].apply(lambda x: '; '.join(x))
    segments_df['opportunities'] = segments_df['opportunities'].apply(lambda x: '; '.join(x))
    segments_df.to_csv(f"{output_dir}/barclays_fan_segments_claude.csv", index=False)
    print(f"✅ Saved: {output_dir}/barclays_fan_segments_claude.csv")
    
    # 5. Executive Summary (single row)
    summary_df = pd.DataFrame([analysis['executive_summary']])
    summary_df['vs_msg'] = analysis['competitive_insights']['vs_msg']
    summary_df['unique_advantage'] = analysis['competitive_insights']['unique_advantage']
    summary_df.to_csv(f"{output_dir}/barclays_executive_summary_claude.csv", index=False)
    print(f"✅ Saved: {output_dir}/barclays_executive_summary_claude.csv")
    
    # 6. Full JSON for reference
    with open(f"{output_dir}/barclays_full_analysis_claude.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✅ Saved: {output_dir}/barclays_full_analysis_claude.json")
    
    return analysis


def main():
    print("=" * 60)
    print("🤖 Claude-Powered Barclays Review Analysis")
    print("=" * 60)
    
    # Load reviews
    reviews_path = "bse_data/barclays_reviews_expanded.csv"
    if not os.path.exists(reviews_path):
        print(f"❌ Reviews file not found: {reviews_path}")
        return
    
    reviews_df = pd.read_csv(reviews_path)
    print(f"\n📊 Loaded {len(reviews_df)} reviews from {reviews_path}")
    print(f"   Sources: {reviews_df['source'].value_counts().to_dict()}")
    print(f"   Avg Rating: {reviews_df['rating'].mean():.2f}/5")
    
    # Run Claude analysis
    print("\n🔄 Sending to Claude API for intelligent analysis...")
    print("   (This may take 10-20 seconds)")
    
    try:
        analysis = analyze_reviews_with_claude(reviews_df)
        print("\n✅ Claude analysis complete!")
        
        # Display key findings
        print("\n" + "=" * 60)
        print("📋 EXECUTIVE SUMMARY")
        print("=" * 60)
        exec_summary = analysis['executive_summary']
        print(f"   Overall Sentiment: {exec_summary['overall_sentiment_score']}/5")
        print(f"   Key Finding: {exec_summary['key_finding']}")
        print(f"   Urgent Action: {exec_summary['urgent_action_needed']}")
        
        print("\n" + "=" * 60)
        print("🔴 TOP ISSUES IDENTIFIED")
        print("=" * 60)
        for i, issue in enumerate(analysis['issue_categories'][:5], 1):
            print(f"\n   {i}. {issue['category']} ({issue['severity']})")
            print(f"      Mentions: {issue['mention_count']}")
            print(f"      Root Cause: {issue['root_cause'][:80]}...")
            print(f"      Revenue Impact: {issue['revenue_impact']['estimated_annual_impact']}")
            print(f"      Fix: {issue['recommended_action'][:60]}...")
        
        print("\n" + "=" * 60)
        print("✅ TOP POSITIVES")
        print("=" * 60)
        for pos in analysis['positive_highlights'][:3]:
            print(f"   • {pos['strength']} ({pos['mention_count']} mentions)")
        
        # Save to CSVs
        print("\n" + "=" * 60)
        print("💾 SAVING ANALYSIS TO CSV FILES")
        print("=" * 60)
        save_analysis_to_csvs(analysis)
        
        print("\n" + "=" * 60)
        print("🎯 READY FOR DASHBOARD!")
        print("=" * 60)
        print("Run: streamlit run app.py")
        print("Navigate to Barclays: Fan Experience page")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
