"""
🏟️ Barclays Center Review Analysis - PROPER SENTIMENT ANALYSIS
Uses Claude API with rigorous SA methodology

Key improvements:
1. Sentiment polarity per topic (not just mention count)
2. Clear severity criteria based on frequency + sentiment + rating correlation
3. No fake revenue estimates
4. Distinguishes positive vs negative mentions of same topic

Author: Gerardo Gandara
"""

import anthropic
import pandas as pd
import json
import os

client = anthropic.Anthropic()


def analyze_reviews_with_claude(reviews_df):
    """
    Proper sentiment analysis using Claude API
    """
    
    # Calculate rating distribution for context
    rating_dist = reviews_df['rating'].value_counts().sort_index().to_dict()
    avg_rating = reviews_df['rating'].mean()
    
    # Stratified sample if too large
    MAX_REVIEWS = 100
    if len(reviews_df) > MAX_REVIEWS:
        print(f"   📊 Stratified sampling {MAX_REVIEWS} from {len(reviews_df)} reviews")
        # Sample proportionally by rating to preserve distribution
        sampled = reviews_df.groupby('rating', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(MAX_REVIEWS * len(x) / len(reviews_df)))))
        )
        reviews_df = sampled.head(MAX_REVIEWS)
        print(f"   → Using {len(reviews_df)} strategically sampled reviews")
    
    # Prepare reviews with rating context
    reviews_text = "\n\n".join([
        f"[Review {i+1}] Rating: {row['rating']}/5 | Source: {row['source']}\n\"{str(row['text'])[:500]}\""
        for i, row in reviews_df.iterrows()
    ])
    
    actual_count = len(reviews_df)
    
    prompt = f"""You are a senior data scientist performing rigorous sentiment analysis on venue reviews.

TASK: Analyze {actual_count} reviews for Barclays Center (Brooklyn Nets, NY Liberty venue).

CONTEXT:
- Total reviews: {actual_count}
- Rating distribution: {rating_dist}
- Average rating: {avg_rating:.2f}/5
- Sources: Google Maps, Yelp (2024-2025)

<reviews>
{reviews_text}
</reviews>

ANALYSIS REQUIREMENTS:

1. TOPIC EXTRACTION: Identify distinct topics mentioned in reviews
2. SENTIMENT PER TOPIC: For EACH topic, count:
   - Negative mentions (complaints, criticism)
   - Positive mentions (praise, satisfaction)
   - Neutral mentions (factual, no sentiment)
3. SEVERITY CLASSIFICATION must follow this logic:
   - Critical: negative_mentions >= 25 AND avg_rating_when_mentioned < 3.0
   - High: negative_mentions >= 15 OR (negative_mentions >= 10 AND negative_ratio > 0.7)
   - Medium: negative_mentions >= 8
   - Low: negative_mentions < 8
4. DO NOT estimate revenue impact - we don't have financial data

Return this exact JSON structure:

{{
    "methodology": {{
        "total_reviews_analyzed": {actual_count},
        "rating_distribution": {json.dumps(rating_dist)},
        "average_rating": {avg_rating:.2f}
    }},
    
    "overall_sentiment": {{
        "score": <float 1-5 based on actual ratings and text sentiment>,
        "positive_review_pct": <percentage of reviews that are primarily positive>,
        "negative_review_pct": <percentage of reviews that are primarily negative>,
        "neutral_review_pct": <percentage>,
        "key_insight": "<one sentence summary based on data>"
    }},
    
    "topic_sentiment_analysis": [
        {{
            "topic": "<topic name>",
            "total_mentions": <int>,
            "negative_mentions": <int - complaints about this topic>,
            "positive_mentions": <int - praise for this topic>,
            "neutral_mentions": <int>,
            "sentiment_ratio": <float - negative/total, higher = more problematic>,
            "avg_rating_when_mentioned": <float - average review rating when this topic appears>,
            "severity": "<Critical/High/Medium/Low - use criteria above>",
            "severity_justification": "<brief explanation of why this severity>",
            "example_negative": "<one real quote showing the issue>",
            "example_positive": "<one real quote if any positive mentions, else null>",
            "root_cause": "<analysis of why this issue exists>",
            "recommended_action": "<specific actionable fix>"
        }}
    ],
    
    "strengths": [
        {{
            "topic": "<strength area>",
            "positive_mentions": <int>,
            "example_quote": "<real quote>",
            "leverage_opportunity": "<how to amplify>"
        }}
    ],
    
    "fan_segments": [
        {{
            "segment": "<segment name>",
            "estimated_pct": <percentage of reviewers>,
            "defining_characteristics": "<what defines this segment>",
            "primary_concerns": ["<concern1>", "<concern2>"],
            "satisfaction_level": "<High/Medium/Low>"
        }}
    ],
    
    "analysis_confidence": {{
        "sample_size_adequate": <true/false>,
        "potential_biases": ["<bias1>", "<bias2>"],
        "confidence_level": "<High/Medium/Low>",
        "limitations": "<honest assessment of analysis limitations>"
    }}
}}

IMPORTANT:
- Count ACTUAL mentions, don't estimate
- Severity must match the criteria defined above
- Include real quotes from reviews as evidence
- Be honest about confidence and limitations
- DO NOT include revenue estimates

Return ONLY valid JSON."""

    # Call Claude API
    print("   🤖 Calling Claude API...")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    response_text = message.content[0].text
    
    # Clean JSON
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    
    return json.loads(response_text.strip())


def save_analysis(analysis, output_dir="bse_data"):
    """Save analysis to CSVs for dashboard"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Topic Sentiment Analysis (main issues table)
    topics_df = pd.DataFrame(analysis['topic_sentiment_analysis'])
    # Sort by severity then negative mentions
    severity_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    topics_df['severity_rank'] = topics_df['severity'].map(severity_order)
    topics_df = topics_df.sort_values(['severity_rank', 'negative_mentions'], ascending=[True, False])
    topics_df = topics_df.drop(columns=['severity_rank'])
    
    # Rename for dashboard compatibility
    topics_df = topics_df.rename(columns={
        'topic': 'category',
        'negative_mentions': 'mention_count'  # Dashboard shows negative mentions as the key metric
    })
    topics_df.to_csv(f"{output_dir}/barclays_issues_claude_analysis.csv", index=False)
    print(f"✅ Saved: {output_dir}/barclays_issues_claude_analysis.csv")
    
    # 2. Strengths
    strengths_df = pd.DataFrame(analysis['strengths'])
    strengths_df = strengths_df.rename(columns={
        'topic': 'strength',
        'positive_mentions': 'mention_count'
    })
    strengths_df.to_csv(f"{output_dir}/barclays_positives_claude.csv", index=False)
    print(f"✅ Saved: {output_dir}/barclays_positives_claude.csv")
    
    # 3. Executive Summary
    summary = analysis['overall_sentiment']
    summary['methodology'] = json.dumps(analysis['methodology'])
    summary_df = pd.DataFrame([summary])
    summary_df = summary_df.rename(columns={
        'score': 'overall_sentiment_score',
        'key_insight': 'key_finding'
    })
    summary_df.to_csv(f"{output_dir}/barclays_executive_summary_claude.csv", index=False)
    print(f"✅ Saved: {output_dir}/barclays_executive_summary_claude.csv")
    
    # 4. Fan Segments
    segments_df = pd.DataFrame(analysis['fan_segments'])
    segments_df['pain_points'] = segments_df['primary_concerns'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
    segments_df['characteristics'] = segments_df['defining_characteristics']
    segments_df['opportunities'] = ''  # Simplified
    segments_df.to_csv(f"{output_dir}/barclays_fan_segments_claude.csv", index=False)
    print(f"✅ Saved: {output_dir}/barclays_fan_segments_claude.csv")
    
    # 5. Full JSON
    with open(f"{output_dir}/barclays_full_analysis_claude.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✅ Saved: {output_dir}/barclays_full_analysis_claude.json")
    
    # 6. Priority Matrix (derived from topics)
    priority_df = topics_df[['category', 'severity', 'mention_count', 'recommended_action']].copy()
    priority_df['priority_score'] = priority_df['severity'].map({'Critical': 10, 'High': 8, 'Medium': 5, 'Low': 3})
    priority_df['quick_win'] = priority_df['severity'].isin(['Medium', 'Low'])
    priority_df['action'] = priority_df['recommended_action']
    priority_df = priority_df.sort_values('priority_score', ascending=False)
    priority_df.to_csv(f"{output_dir}/barclays_priority_matrix_claude.csv", index=False)
    print(f"✅ Saved: {output_dir}/barclays_priority_matrix_claude.csv")


def main():
    print("=" * 60)
    print("🏟️ BARCLAYS CENTER - PROPER SENTIMENT ANALYSIS")
    print("=" * 60)
    
    # Load reviews
    reviews_path = "bse_data/barclays_reviews_expanded.csv"
    if not os.path.exists(reviews_path):
        reviews_path = "/Users/gerardo.gandara/GERARDO/BSE_GLOBAL/bse_data/barclays_reviews_expanded.csv"
    
    if not os.path.exists(reviews_path):
        print(f"❌ Reviews not found: {reviews_path}")
        return
    
    reviews_df = pd.read_csv(reviews_path)
    print(f"\n📊 Loaded {len(reviews_df)} reviews")
    print(f"   Sources: {reviews_df['source'].value_counts().to_dict()}")
    print(f"   Avg Rating: {reviews_df['rating'].mean():.2f}/5")
    print(f"   Rating Distribution: {reviews_df['rating'].value_counts().sort_index().to_dict()}")
    
    # Run analysis
    print("\n🔄 Running sentiment analysis...")
    analysis = analyze_reviews_with_claude(reviews_df)
    
    print("\n" + "=" * 60)
    print("📋 RESULTS")
    print("=" * 60)
    
    # Show overall sentiment
    overall = analysis['overall_sentiment']
    print(f"\n🎯 Overall Sentiment: {overall['score']}/5")
    print(f"   Positive: {overall['positive_review_pct']}%")
    print(f"   Negative: {overall['negative_review_pct']}%")
    print(f"   Key Insight: {overall['key_insight']}")
    
    # Show topic analysis
    print(f"\n🔴 ISSUES BY SEVERITY:")
    for topic in analysis['topic_sentiment_analysis']:
        emoji = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}.get(topic['severity'], '⚪')
        print(f"   {emoji} {topic['topic']}")
        print(f"      Negative mentions: {topic['negative_mentions']} | Positive: {topic.get('positive_mentions', 0)}")
        print(f"      Severity: {topic['severity']} - {topic['severity_justification']}")
    
    # Show strengths
    print(f"\n✅ STRENGTHS:")
    for strength in analysis['strengths']:
        print(f"   💪 {strength['topic']} ({strength['positive_mentions']} mentions)")
    
    # Save
    print("\n" + "=" * 60)
    print("💾 SAVING ANALYSIS")
    print("=" * 60)
    save_analysis(analysis)
    
    # Also save to app location
    save_analysis(analysis, "/Users/gerardo.gandara/GERARDO/BSE_GLOBAL/bse-global-analytics/bse_data")
    
    print("\n✅ Done! Restart streamlit to see updated analysis.")


if __name__ == "__main__":
    main()
