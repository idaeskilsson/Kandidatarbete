import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

def get_youtube_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY saknas i .env-filen.")
    return build("youtube", "v3", developerKey=api_key)
