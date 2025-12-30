"""
🏟️ Barclays Center Review Collector via Outscraper API
Collects 300-500 reviews from Google Maps + Yelp

Author: Gerardo Gandara
"""

import requests
import pandas as pd
import time
import os

# Outscraper API Configuration
API_KEY = os.environ.get('OUTSCRAPER_API_KEY', 'YOUR_API_KEY_HERE')
BASE_URL = "https://api.outscraper.com"

HEADERS = {
    "X-API-KEY": API_KEY
}


def get_google_maps_reviews(place_query="Barclays Center Brooklyn NY", limit=200):
    """
    Fetch Google Maps reviews for Barclays Center
    Cost: ~$0.002 per review
    """
    print("\n" + "="*60)
    print("🔍 FETCHING GOOGLE MAPS REVIEWS")
    print("="*60)
    
    endpoint = f"{BASE_URL}/maps/reviews-v3"
    
    params = {
        "query": place_query,
        "reviewsLimit": limit,
        "sort": "newest",  # Get most recent reviews
        "language": "en"
    }
    
    print(f"   Query: {place_query}")
    print(f"   Limit: {limit} reviews")
    print(f"   Sorting: newest first")
    
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parse reviews from response
            reviews = []
            
            if isinstance(data, list) and len(data) > 0:
                place_data = data[0]
                reviews_data = place_data.get('reviews_data', [])
                
                print(f"\n   ✅ Found {len(reviews_data)} Google reviews")
                
                for review in reviews_data:
                    reviews.append({
                        'source': 'Google Maps',
                        'rating': review.get('review_rating', 3),
                        'title': '',
                        'text': review.get('review_text', '')[:1500],
                        'date': review.get('review_datetime_utc', ''),
                        'author': review.get('author_title', ''),
                        'review_id': review.get('review_id', '')
                    })
            
            return reviews
        else:
            print(f"   ❌ Error: Status {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return []
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return []


def get_yelp_reviews(business_url="https://www.yelp.com/biz/barclays-center-brooklyn-2", limit=200):
    """
    Fetch Yelp reviews for Barclays Center
    Cost: ~$0.002 per review
    """
    print("\n" + "="*60)
    print("🔍 FETCHING YELP REVIEWS")
    print("="*60)
    
    endpoint = f"{BASE_URL}/yelp/reviews"
    
    params = {
        "query": business_url,
        "limit": limit,
        "sort": "date_desc"  # Most recent first
    }
    
    print(f"   URL: {business_url}")
    print(f"   Limit: {limit} reviews")
    
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            
            reviews = []
            
            if isinstance(data, list):
                reviews_data = data[0].get('reviews_data', data) if len(data) > 0 else data
                
                # Handle different response formats
                if isinstance(reviews_data, dict):
                    reviews_data = reviews_data.get('reviews', [])
                
                print(f"\n   ✅ Found {len(reviews_data)} Yelp reviews")
                
                for review in reviews_data:
                    if isinstance(review, dict):
                        reviews.append({
                            'source': 'Yelp',
                            'rating': review.get('rating', review.get('review_rating', 3)),
                            'title': '',
                            'text': review.get('text', review.get('review_text', ''))[:1500],
                            'date': review.get('date', review.get('review_datetime_utc', '')),
                            'author': review.get('user', {}).get('name', review.get('author_title', '')),
                            'review_id': review.get('id', review.get('review_id', ''))
                        })
            
            return reviews
        else:
            print(f"   ❌ Error: Status {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return []
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return []


def main():
    print("="*60)
    print("🏟️ BARCLAYS CENTER REVIEW COLLECTOR")
    print("   via Outscraper API")
    print("="*60)
    
    # Check API key
    if API_KEY == 'YOUR_API_KEY_HERE' or not API_KEY:
        print("\n⚠️  Please set your API key!")
        print("   Option 1: export OUTSCRAPER_API_KEY='your-key'")
        print("   Option 2: Edit this file and replace YOUR_API_KEY_HERE")
        return
    
    all_reviews = []
    
    # 1. Google Maps Reviews (200)
    google_reviews = get_google_maps_reviews(
        place_query="Barclays Center, 620 Atlantic Ave, Brooklyn, NY",
        limit=200
    )
    all_reviews.extend(google_reviews)
    print(f"\n   📊 Google total: {len(google_reviews)}")
    
    # Small delay between API calls
    time.sleep(2)
    
    # 2. Yelp Reviews (200)
    yelp_reviews = get_yelp_reviews(
        business_url="https://www.yelp.com/biz/barclays-center-brooklyn-2",
        limit=200
    )
    all_reviews.extend(yelp_reviews)
    print(f"   📊 Yelp total: {len(yelp_reviews)}")
    
    # Combine and clean
    if len(all_reviews) == 0:
        print("\n❌ No reviews collected. Check API key and credits.")
        return
    
    df = pd.DataFrame(all_reviews)
    
    # Remove duplicates and empty reviews
    df = df.drop_duplicates(subset=['text'])
    df = df[df['text'].str.len() > 20]
    
    # Convert ratings to numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(3)
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"Total unique reviews: {len(df)}")
    print(f"\nBy source:")
    print(df['source'].value_counts())
    print(f"\nRating distribution:")
    print(df['rating'].value_counts().sort_index())
    print(f"\nAverage rating: {df['rating'].mean():.2f}/5")
    
    # Parse dates for recency check
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        recent = df[df['date_parsed'] >= '2024-01-01']
        print(f"\n2024 reviews: {len(recent)} ({len(recent)/len(df)*100:.1f}%)")
    
    # Save
    output_path = 'bse_data/barclays_reviews_outscraper.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    # Also save as expanded for Claude analysis
    df.to_csv('bse_data/barclays_reviews_expanded.csv', index=False)
    print(f"💾 Saved to: bse_data/barclays_reviews_expanded.csv")
    
    # Sample reviews
    print("\n" + "="*60)
    print("📝 SAMPLE REVIEWS")
    print("="*60)
    for i, row in df.head(5).iterrows():
        print(f"\n[{row['source']}] ⭐{row['rating']} - {row['date']}")
        print(f"   {row['text'][:200]}...")
    
    print("\n" + "="*60)
    print("✅ NEXT STEP: Run Claude analysis")
    print("="*60)
    print("python analyze_reviews_claude.py")
    
    return df


if __name__ == "__main__":
    main()
