import sys
sys.path.append(r'D:\thumbnailiq')

import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime, timezone
from config import YOUTUBE_API_KEY

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

SEARCH_QUERIES = ["tech review 2024", "cooking recipe", "gaming highlights"]
MAX_VIDEOS_PER_QUERY = 50

def search_videos(query):
    print(f"Searching: {query}")
    video_ids = []

    request = youtube.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=50,
        relevanceLanguage="en"
    )
    response = request.execute()

    for item in response["items"]:
        video_ids.append(item["id"]["videoId"])

    print(f"  Found {len(video_ids)} videos")
    return video_ids

def get_video_details(video_ids):
    videos = []

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        request = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch)
        )
        response = request.execute()

        for item in response["items"]:
            try:
                snippet = item["snippet"]
                stats = item.get("statistics", {})

                published_at = snippet["publishedAt"]
                published_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                days_old = (datetime.now(timezone.utc) - published_date).days
                if days_old == 0:
                    days_old = 1

                videos.append({
                    "video_id": item["id"],
                    "title": snippet["title"],
                    "published_at": published_at,
                    "days_old": days_old,
                    "view_count": int(stats.get("viewCount", 0)),
                    "like_count": int(stats.get("likeCount", 0)),
                    "thumbnail_url": snippet["thumbnails"]["high"]["url"],
                    "category": snippet.get("categoryId", "unknown")
                })
            except Exception as e:
                print(f"  Skipping video: {e}")

    return videos

all_video_ids = []
for query in SEARCH_QUERIES:
    ids = search_videos(query)
    all_video_ids.extend(ids)

all_video_ids = list(set(all_video_ids))
print(f"\nTotal unique videos found: {len(all_video_ids)}")

print("Fetching video details...")
all_videos = get_video_details(all_video_ids)

df = pd.DataFrame(all_videos)
df.to_csv(r'D:\thumbnailiq\data\videos.csv', index=False)
print(f"\nSaved {len(df)} videos to data/videos.csv")
print(df.head())