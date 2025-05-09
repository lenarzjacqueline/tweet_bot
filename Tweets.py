import os
import json
import asyncio
import random
from pathlib import Path
from difflib import SequenceMatcher
from dotenv import load_dotenv
from smolagents import load_tool, CodeAgent, HfApiModel, DuckDuckGoSearchTool, VisitWebpageTool
import tweepy

# Load .env variables
load_dotenv()

# === CONFIGURATION ===
HF_TOKEN = os.getenv('HF_TOKEN')
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Twitter credentials from .env
class Config:
    TWITTER_ACCESS_KEY = os.getenv("TWITTER_ACCESS_KEY")
    TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
    TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")

# Initialize Twitter client
class MessageService:
    def __init__(self):
        self.twitter = tweepy.Client(
            bearer_token=Config.TWITTER_BEARER_TOKEN,
            access_token=Config.TWITTER_ACCESS_KEY,
            access_token_secret=Config.TWITTER_ACCESS_SECRET,
            consumer_key=Config.TWITTER_CONSUMER_KEY,
            consumer_secret=Config.TWITTER_CONSUMER_SECRET,
        )

    async def send_tweet(self, message: str) -> bool:
        return await asyncio.to_thread(
            self.twitter.create_tweet,
            text=message
        )

# === Hugging Face model & agent setup ===
model = HfApiModel(model_id=MODEL_ID, token=HF_TOKEN)
web_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    additional_authorized_imports=['requests', 'bs4', 'numpy', 'pandas'],
)

# === News scraping ===
async def get_crypto_news():
    prompt = """
Search the web for today's top 3 news stories strictly related to cryptocurrency: projects, regulations, market moves, or scams.
Ignore general finance or tech news unless it's directly connected to crypto.
Return a list of dictionaries: [{"headline": ..., "summary": ..., "url": ...}]
"""
    response = web_agent.run(prompt.strip())
    return response.output  # Adjust if necessary

# === Hook generation ===
def generate_hook(headline: str, summary: str) -> str:
    prompt = f"""
You are a Twitter content strategist for a crypto news brand called \"Reachd\".
Given the headline and summary below, write a single catchy tweet hook in the same style as these examples:

Examples:
1. Heard about {{headline}} yet? If you were on Reachd, you'd already know.
2. Crypto moves fast. {{headline}} is the latest.
3. Stay ahead with Reachd: {{headline}}
4. Another one for the books: {{headline}} — details below.
5. ICYMI: {{headline}} just dropped. Catch it early with Reachd.

Now, write a new hook for:

Headline: "{headline}"
Summary: "{summary}"

Respond with only the tweet hook, no hashtags or explanations.
"""
    result = model.run(prompt.strip())
    return result.output.strip()

# === History handling ===
HISTORY_FILE = Path("recent_tweets.json")

def load_tweet_history() -> list:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_tweet_history(history: list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-50:], f, indent=2)

def is_similar_topic(headline1: str, headline2: str) -> bool:
    return SequenceMatcher(None, headline1.lower(), headline2.lower()).ratio() > 0.7

def should_skip(headline: str, hook: str, history: list) -> bool:
    if hook in [h["hook"] for h in history[-19:]]:
        return True
    for old in history:
        if is_similar_topic(headline, old["headline"]):
            return True
    return False

# === Tweet formatting ===
def format_tweet(hook: str, summary: str, url: str) -> str:
    tweet = f"{hook}\n\n{summary}\n\n🔗 {url}"
    if len(tweet) > 280:
        tweet = f"{hook}\n🔗 {url}"
    return tweet

# === Main execution ===
async def main():
    news_items = await get_crypto_news()
    history = load_tweet_history()
    service = MessageService()

    for item in news_items:
        headline = item.get("headline")
        summary = item.get("summary")
        url = item.get("url")

        if not all([headline, summary, url]):
            continue

        hook = generate_hook(headline, summary)

        if should_skip(headline, hook, history):
            continue

        tweet = format_tweet(hook, summary, url)
        await service.send_tweet(tweet)

        history.append({"hook": hook, "headline": headline})
        save_tweet_history(history)

        await asyncio.sleep(random.randint(5, 10))

if __name__ == "__main__":
    asyncio.run(main())
