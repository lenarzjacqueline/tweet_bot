#!/usr/bin/env python3
"""
Crypto Twitter Bot for Reachd Platform
Generates engaging tweets based on crypto news and sends them via Telegram for approval.
"""

# === Imports ===
import os
import json
import re
import asyncio
import aiohttp
import requests
import sys
from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import List
import random

# Third-party imports
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool, HfApiModel
from newsapi import NewsApiClient
import tweepy

# === Environment Setup ===
load_dotenv()

# Validate required environment variables
REQUIRED_ENV_VARS = [
    "TWITTER_ACCESS_KEY",
    "TWITTER_ACCESS_SECRET", 
    "TWITTER_BEARER_TOKEN",
    "TWITTER_CONSUMER_KEY",
    "TWITTER_CONSUMER_SECRET",
    "HF_TOKEN",
    "NEWSAPI_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID"
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    print(f"[!] Missing environment variables: {', '.join(missing_vars)}")
    exit(1)

# === Configuration ===
class Config:
    """Application configuration from environment variables"""
    TWITTER_ACCESS_KEY = os.getenv("TWITTER_ACCESS_KEY")
    TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
    TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")
    HF_TOKEN = os.getenv("HF_TOKEN")
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === Constants ===
MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
HISTORY_FILE = "tweet_history.json"
LAST_RUN_FILE = "last_run.json"
TWEET_CHAR_LIMIT = 280
MIN_HOURS_BETWEEN_RUNS = 9
MAX_HOURS_BETWEEN_RUNS = 20

# === Setup Models and APIs ===
model = HfApiModel(
    model_id=MODEL_ID, 
    token=Config.HF_TOKEN, 
    provider="hf-inference"
)

web_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    additional_authorized_imports=["requests", "bs4", "json", "newsapi"]
)

newsapi = NewsApiClient(api_key=Config.NEWSAPI_KEY)

# === History Management ===
def load_history() -> dict:
    """Load tweet history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Error loading {HISTORY_FILE}. Starting fresh.")
            return {"hooks": [], "topics": []}
    return {"hooks": [], "topics": []}

def save_history(history: dict) -> None:
    """Save tweet history to JSON file"""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[!] Error saving history: {e}")

# === Run Time Management ===
def load_last_run_info() -> dict:
    """Load information about the last successful run"""
    if os.path.exists(LAST_RUN_FILE):
        try:
            with open(LAST_RUN_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Error loading {LAST_RUN_FILE}. Starting fresh.")
            return {"last_run_time": None, "last_success_time": None}
    return {"last_run_time": None, "last_success_time": None}

def save_run_info(run_info: dict) -> None:
    """Save information about the current run"""
    try:
        with open(LAST_RUN_FILE, "w") as f:
            json.dump(run_info, f, indent=2)
    except Exception as e:
        print(f"[!] Error saving run info: {e}")

def get_next_run_time(last_run_info: dict) -> datetime:
    """Calculate when the bot should next run"""
    now = datetime.now()
    last_success = last_run_info.get("last_success_time")
    
    if last_success:
        # Convert string to datetime
        if isinstance(last_success, str):
            last_success = datetime.fromisoformat(last_success)
        
        # Random hours between MIN and MAX
        hours_to_add = random.uniform(MIN_HOURS_BETWEEN_RUNS, MAX_HOURS_BETWEEN_RUNS)
        next_run = last_success + timedelta(hours=hours_to_add)
        
        # If next_run is in the past, use now
        if next_run < now:
            return now
        return next_run
    
    # If no last success, run now
    return now

def get_article_timeframe(last_run_info: dict) -> int:
    """Calculate how many hours back to look for articles"""
    now = datetime.now()
    last_success = last_run_info.get("last_success_time")
    
    if last_success:
        # Convert string to datetime
        if isinstance(last_success, str):
            last_success = datetime.fromisoformat(last_success)
        
        # Hours since last success
        hours_since = (now - last_success).total_seconds() / 3600
        
        # Round up to nearest hour, minimum 1
        return max(1, int(hours_since) + 1)
    
    # Default to 24 hours if no last success
    return 24

# === Twitter Service ===
class MessageService:
    """Handles Twitter API interactions"""
    
    def __init__(self):
        self.twitter = tweepy.Client(
            bearer_token=Config.TWITTER_BEARER_TOKEN,
            access_token=Config.TWITTER_ACCESS_KEY,
            access_token_secret=Config.TWITTER_ACCESS_SECRET,
            consumer_key=Config.TWITTER_CONSUMER_KEY,
            consumer_secret=Config.TWITTER_CONSUMER_SECRET,
        )

    async def send_tweet(self, message: str) -> bool:
        """Send a tweet via Twitter API"""
        try:
            await asyncio.to_thread(self.twitter.create_tweet, text=message)
            print("[+] Tweet sent successfully!")
            return True
        except Exception as e:
            print(f"[!] Failed to send tweet: {e}")
            return False

# === Telegram Bot ===
class TelegramBot:
    """Handles Telegram bot interactions for approval workflow"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.last_update_id = 0
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.approval_future = None
        self.processed_callback_ids = set()  # Track processed callback IDs

    async def send_approval_request(self, tweet_content: str) -> str:
        """Send tweet content for approval with Yes/No buttons"""
        request_id = f"req_{int(datetime.now().timestamp())}"
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "Yes ✅", "callback_data": f"yes_{request_id}"},
                    {"text": "No ❌", "callback_data": f"no_{request_id}"}
                ]
            ]
        }
        data = {
            "chat_id": self.chat_id,
            "text": f"Approve this tweet?\n\n{tweet_content}",
            "reply_markup": json.dumps(keyboard)
        }
        
        url = f"{self.base_url}/sendMessage"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                result = await response.json()
                if result.get("ok", False):
                    print("[+] Sent approval request to Telegram.")
                    return request_id
                else:
                    print(f"[!] Failed to send approval request: {result}")
                    return None

    async def process_updates(self) -> None:
        """Process Telegram updates to handle button callbacks"""
        while True:
            try:
                url = f"{self.base_url}/getUpdates"
                params = {"offset": self.last_update_id + 1, "timeout": 30}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        result = await response.json()
                        
                        if not result.get("ok", False):
                            print(f"[!] Failed to get updates: {result}")
                            await asyncio.sleep(5)
                            continue
                            
                        updates = result.get("result", [])
                        for update in updates:
                            self.last_update_id = max(self.last_update_id, update.get("update_id", 0))
                            if "callback_query" in update:
                                callback_query = update["callback_query"]
                                callback_id = callback_query.get("id")
                                
                                # Only process each callback once
                                if callback_id not in self.processed_callback_ids:
                                    self.processed_callback_ids.add(callback_id)
                                    await self.handle_callback_query(callback_query)
                                
                                # Clean up old callback IDs (keep only last 100)
                                if len(self.processed_callback_ids) > 100:
                                    self.processed_callback_ids = set(list(self.processed_callback_ids)[-50:])
                                
            except Exception as e:
                print(f"[!] Error processing updates: {e}")
                await asyncio.sleep(5)

    async def handle_callback_query(self, callback_query: dict) -> None:
        """Handle Yes/No button presses for tweet approval"""
        data = callback_query.get("data", "")
        message = callback_query.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        message_id = message.get("message_id")
        callback_id = callback_query.get("id")

        print(f"[DEBUG] Processing callback query: {data} (ID: {callback_id})")
        print(f"[DEBUG] From chat ID: {chat_id}")
        print(f"[DEBUG] Expected chat ID: {self.chat_id}")

        if chat_id != int(self.chat_id):
            print(f"[!] Ignoring callback from unauthorized chat {chat_id}")
            return

        # Answer the callback query to remove loading state
        url = f"{self.base_url}/answerCallbackQuery"
        callback_data = {
            "callback_query_id": callback_id,
            "text": "Processing your response..."
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=callback_data)

        if data.startswith("yes_"):
            await self.update_message(chat_id, message_id, "✅ Tweet approved! Posting to X.")
            print("[+] User approved the tweet.")
            if self.approval_future and not self.approval_future.done():
                print("[DEBUG] Setting approval_future to True")
                self.approval_future.set_result(True)
                
        elif data.startswith("no_"):
            await self.update_message(chat_id, message_id, "❌ Tweet rejected. Not posting.")
            print("[+] User rejected the tweet.")
            if self.approval_future and not self.approval_future.done():
                print("[DEBUG] Setting approval_future to False")
                self.approval_future.set_result(False)
        else:
            print(f"[WARNING] Unknown callback data: {data}")

    async def update_message(self, chat_id: str, message_id: int, text: str) -> None:
        """Update an existing message"""
        data = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text
        }
        
        url = f"{self.base_url}/editMessageText"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                result = await response.json()
                if result.get("ok", False):
                    print(f"[+] Updated message {message_id} in chat {chat_id}")
                else:
                    print(f"[!] Failed to update message: {result}")
                    # Don't raise an error if the message is the same

# === Hook Generator ===
async def generate_hook(title: str, summary: str, past_hooks: List[str], article_url: str) -> str:
    """Generate an engaging tweet formatted as an influencer tip from Reachd"""
    
    # Hook examples for variety
    HOOK_EXAMPLES = [
        "Pro tip:", "Big brain move:", "Smart money knows:", "Alpha alert:",
        "Insider track:", "Next level play:", "Stay ahead:", "Game changer:", "Heads up:",
        "Level up:", "Power move:", "Top signal:", "Winning strategy:", "Future-proof tip:",
        "Elite insight:", "Market whisper:", "Crypto wisdom:", "Advanced play:", "Boss move:"
    ]
    
    # Keywords for news categorization
    BAD_KEYWORDS = [
        'crash', 'dump', 'hack', 'exploit', 'scam', 'warning', 'risk', 'loss', 
        'problem', 'issue', 'ban', 'regulation', 'lawsuit', 'plunge', 'collapse', 
        'down', 'drop', 'sink', 'tank', 'fall', 'decline', 'negative'
    ]
    
    GOOD_KEYWORDS = [
        'pump', 'moon', 'breakthrough', 'launch', 'opportunity', 'growth', 'profit', 
        'win', 'bullish', 'surge', 'breakthrough', 'adoption', 'milestone', 'target', 'rally',
        'up', 'rise', 'climb', 'spike', 'soar', 'jump', 'boost', 'gain', 'increase'
    ]
    
    INTERESTING_KEYWORDS = [
        'depegs', 'volatility', 'institutional', 'banks', 'etf', 'sec', 'halving', 
        'network', 'upgrade', 'merger', 'acquisition', 'regulation', 'movement', 'price',
        'market', 'volume', 'trend', 'breakout', 'support', 'resistance', 'bullish', 'bearish'
    ]
    
    # High priority price movement keywords for super interesting news
    PRICE_MOVEMENT_KEYWORDS = [
        'moon', 'dump', 'crash', 'surge', 'pump', 'rally', 'plunge', 'spike', 'soar',
        'breakout', 'breakdown', 'all-time high', 'ath', 'all-time low', 'atl',
        'up %', 'down %', 'gains', 'losses', 'doubled', 'halved'
    ]
    
    # Categorize news
    title_lower = title.lower()
    summary_lower = summary.lower() if summary else ""
    
    # Check for price movements - these get highest priority
    has_price_movement = any(word in title_lower or word in summary_lower for word in PRICE_MOVEMENT_KEYWORDS)
    
    is_bad_news = any(word in title_lower or word in summary_lower for word in BAD_KEYWORDS)
    is_good_news = any(word in title_lower or word in summary_lower for word in GOOD_KEYWORDS)
    is_interesting = any(word in title_lower or word in summary_lower for word in INTERESTING_KEYWORDS) or is_bad_news or is_good_news
    
    # Override categorization if price movement detected
    if has_price_movement:
        is_interesting = True  # Always consider price movements interesting
        # Determine if price movement is positive or negative
        negative_movement = any(word in title_lower or word in summary_lower for word in ['crash', 'dump', 'plunge', 'down', 'drop', 'fall', 'decline', 'losses', 'halved', 'collapse'])
        positive_movement = any(word in title_lower or word in summary_lower for word in ['moon', 'pump', 'surge', 'rally', 'spike', 'soar', 'up', 'gains', 'doubled', 'breakout'])
        
        if negative_movement:
            is_bad_news = True
        elif positive_movement:
            is_good_news = True
    
    # Track recently used hooks
    recent_hooks = past_hooks[-19:] if len(past_hooks) > 19 else past_hooks
    used_hook_words = [hook.split()[0].lower().rstrip(':') for hook in recent_hooks if hook]
    
    # Generate prompt based on news category
    if has_price_movement and is_bad_news:
        prompt = f"""Create an engaging tweet about cryptocurrency price movement that's concerning:

1. Start with a catchy hook (avoid these recently used: {', '.join(used_hook_words)})
2. EMPHASIZE the price movement - include specific details about how much it moved, what price/percentage
3. Include a subtle reference indicating this is based on recent market data
4. Frame it as protection knowledge that could save people from losses
5. Suggest this is the kind of insight people get from crypto influencers on Reachd
6. Use 1-2 relevant hashtags
7. Keep it under {TWEET_CHAR_LIMIT} characters total

Article title: {title}
Article summary: {summary}

Examples:
"Don't get rekt: Market data confirms [COIN] just dumped 30% - exactly what we've been warning about since $X. Connect with the right circle on Reachd. #crypto #priceaction"
"Smart money saw this coming: [COIN] crash to $X proves why we exited at $Y. Sources showing more downside ahead. #dyor #marketanalysis"

Include specific price movement details AND source reference. Return ONLY the tweet text.
"""
    elif has_price_movement and is_good_news:
        prompt = f"""Create an engaging tweet about exciting cryptocurrency price movement:

1. Start with a catchy hook (avoid these recently used: {', '.join(used_hook_words)})
2. EMPHASIZE the price movement - include specific details about how much it moved, what price/percentage
3. Include a subtle reference indicating this is based on recent market data
4. Frame it as an opportunity that smart money caught early
5. Suggest this is the kind of alpha people get from crypto influencers on Reachd
6. Use 1-2 relevant hashtags
7. Keep it under {TWEET_CHAR_LIMIT} characters total

Article title: {title}
Article summary: {summary}

Examples:
"Alpha alert: [COIN] just pumped 50% to $X - we've been accumulating since $Y 💰 Market data shows more upside ahead. Ask the pros on Reachd. #moonshot #crypto"
"Big brain move: Breaking analysis shows [COIN] breakout to new ATH at $X! Early access = early profits. This is why we stay connected on Reachd. #priceaction"

Include specific price movement details AND source reference. Return ONLY the tweet text.
"""
    elif is_interesting and is_bad_news:
        prompt = f"""Create an engaging tweet about cryptocurrency news formatted as an influencer tip. This news is important and concerning:

1. Start with a catchy hook (avoid these recently used: {', '.join(used_hook_words)})
2. Include the actual insight from the news, making it clear what happened and why it matters
3. IMPORTANT: Include a subtle reference indicating this is based on recent news/reports/data (e.g., "just saw reports", "data shows", "breaking news", "source confirmed")
4. Frame it as insider knowledge that could protect people from losses
5. Suggest this is the kind of insight people get from crypto influencers on Reachd
6. Use 1-2 relevant hashtags
7. Keep it under {TWEET_CHAR_LIMIT} characters total

Article title: {title}
Article summary: {summary}

Examples:
"Smart money saw this coming 👀 Breaking reports: [Specific event] exposed security flaw. The red flags were there for weeks. #dyor #crypto"
"Don't get rekt: Just confirmed - [Specific news] proves what our circle was saying about [specific concern]. Connect with the right people on Reachd. #alpha"
"Data shows [Specific insight] - exactly why we've been warning about this since last week. Source: latest market reports. #cryptowisdom"

Include the actual news insight AND source reference, not just generic advice. Return ONLY the tweet text.
"""
    elif is_interesting and is_good_news:
        prompt = f"""Create an engaging tweet about cryptocurrency news formatted as an influencer tip. This news presents a real opportunity:

1. Start with a catchy hook (avoid these recently used: {', '.join(used_hook_words)})
2. Include the actual insight from the news, making it clear what happened and why it's valuable
3. IMPORTANT: Include a subtle reference indicating this is based on recent news/reports/data (e.g., "breaking news", "reports confirm", "data just dropped", "sources saying")
4. Frame it as an opportunity people could have known about early
5. Suggest this is the kind of alpha people get from crypto influencers on Reachd
6. Use 1-2 relevant hashtags
7. Keep it under {TWEET_CHAR_LIMIT} characters total

Article title: {title}
Article summary: {summary}

Examples:
"Alpha alert: Breaking - [Specific event] confirms what we've been accumulating since [price]. Early access = early profits 💰 #alpha #crypto"
"Big brain move: Latest reports show [Specific news] is exactly what our circle predicted. The smart money got in before this announcement. Ask the pros on Reachd. #nextlevel"
"Just in: Market data confirms [Specific insight]. This is why we stay ahead of the news cycle. #earlybird"

Include the actual news insight AND source reference, not just generic advice. Return ONLY the tweet text.
"""
    elif is_interesting:
        prompt = f"""Create an engaging tweet about cryptocurrency news formatted as an influencer tip. This is interesting market insight:

1. Start with a catchy hook (avoid these recently used: {', '.join(used_hook_words)})
2. Include the actual insight from the news, making it clear what happened and its implications
3. IMPORTANT: Include a subtle reference indicating this is based on recent news/analysis/data (e.g., "new data shows", "reports indicate", "analysis confirms", "market intel")
4. Frame it as market wisdom that separates pros from amateurs
5. Position it as the kind of insight people get from top crypto influencers on Reachd
6. Use 1-2 relevant hashtags
7. Keep it under {TWEET_CHAR_LIMIT} characters total

Article title: {title}
Article summary: {summary}

Examples:
"Level up: New analysis shows [Specific event] reveals how retail misses the bigger picture. Understanding [specific insight] is what sets pros apart. #cryptowisdom"
"Market whisper: Fresh data confirms [Specific development] signals a shift most won't notice till too late. This is why we stay connected on Reachd. #insidertrack"
"Intel coming in: [Specific insight] from today's reports. Connect the dots before everyone else. #marketanalysis"

Include the actual news insight AND source reference, not just generic advice. Return ONLY the tweet text.
"""
    else:
        # For less interesting news or generic advertising content
        prompt = f"""Create an engaging tweet that promotes the value of Reachd platform without referencing specific news:

1. Start with a catchy hook (avoid these recently used: {', '.join(used_hook_words)})
2. Focus on the value of connecting with crypto influencers and getting insider knowledge
3. Frame it as what separates successful crypto traders from the rest
4. Promote Reachd as the platform where the smart money connects
5. Emphasize general market awareness, not specific news
6. Use 1-2 relevant hashtags
7. Keep it under {TWEET_CHAR_LIMIT} characters total

Examples:
"Pro tip: The difference between retail and smart money? Access to the right information at the right time. Join the inner circle on Reachd. #alpha"
"Next level play: While others chase pumps, we get early insights. Connect with crypto influencers who actually move markets. Try Reachd. #cryptowisdom"
"Smart money principle: Information > timing. Connect with influencers who see moves before they happen. #reachd #crypto"

Return ONLY the tweet text focused on general value proposition.
"""

    try:
        print("[+] Generating influencer tip tweet...")
        
        # Call model to generate content
        result = await asyncio.to_thread(lambda: model.generate(prompt))
        
        # Handle different result types
        if isinstance(result, dict):
            tweet_content = result.get('text', '') or result.get('output', '') or result.get('generated_text', '')
        elif isinstance(result, str):
            tweet_content = result
        else:
            tweet_content = str(result)
        
        if tweet_content:
            # Clean up result
            tweet_content = tweet_content.strip()
            
            # Remove code block markers
            if tweet_content.startswith("```") and tweet_content.endswith("```"):
                tweet_content = tweet_content[3:-3].strip()
            
            # Remove any accidentally included URLs
            tweet_content = re.sub(r'https?://\S+', '', tweet_content)
            
            # Ensure hashtag presence
            if "#" not in tweet_content:
                tweet_content += " #crypto"
            
            print(f"[+] Generated tweet content: {tweet_content}")
            return tweet_content
        else:
            print("[!] Empty result from model")
            raise ValueError("Empty result from model")
            
    except Exception as e:
        print(f"[!] Error in generate_hook: {e}")
        print(f"[!] Error type: {type(e)}")
        
        # Fallback content generation
        unused_hooks = [hook for hook in HOOK_EXAMPLES if hook.lower().rstrip(':') not in used_hook_words]
        random_hook = random.choice(unused_hooks if unused_hooks else HOOK_EXAMPLES)
        
        # Source reference phrases for fallback content
        source_phrases = [
            "Breaking reports show",
            "Latest data confirms",
            "Market intel suggests", 
            "New analysis reveals",
            "Fresh reports indicate",
            "Sources confirm",
            "Just in from the wire",
            "Market data shows"
        ]
        
        if is_interesting and is_bad_news:
            key_info = title.split(':')[0] if ':' in title else title.split('-')[0]
            source_phrase = random.choice(source_phrases)
            return f"{random_hook} {source_phrase} {key_info.strip().lower()} was predictable. Connect with the right crypto circle on Reachd. #cryptowisdom"
        elif is_interesting and is_good_news:
            key_info = title.split(':')[0] if ':' in title else title.split('-')[0]
            source_phrase = random.choice(source_phrases)
            return f"{random_hook} {source_phrase} {key_info.strip().lower()} - exactly what we've been watching. Early access = early profits on Reachd 💰 #alpha #crypto"
        elif is_interesting:
            key_info = title.split(':')[0] if ':' in title else title.split('-')[0]
            source_phrase = random.choice(source_phrases)
            return f"{random_hook} {source_phrase} {key_info.strip().lower()} shows patterns most miss. Level up your crypto game on Reachd. #insidertrack"
        else:
            # Generic advertising fallback (no news reference)
            return f"{random_hook} Smart money connects differently. Join the crypto influencers who called every major move on Reachd. #nextlevel"

# === Main Function ===
async def run_tweet_bot() -> None:
    """Main function to run the tweet bot"""
    print("[+] Starting tweet bot...")

    # Load tweet history
    history = load_history()
    past_hooks = history["hooks"]
    used_topics = history["topics"]

    # Fetch recent crypto articles
    from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    articles = newsapi.get_everything(
        q='crypto',
        domains='coindesk.com,decrypt.co,cointelegraph.com',
        from_param=from_date,
        language='en',
        sort_by='relevancy'
    ).get('articles', [])

    if not articles:
        print("[!] No articles found.")
        return

    print(f"[+] Found {len(articles)} articles")

    # Track attempted articles
    attempted_articles = set()
    
    # Initialize Telegram bot once for reuse
    telegram_bot = TelegramBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)
    update_task = None
    
    try:
        # Keep trying until we get an approved tweet or run out of articles
        while len(attempted_articles) < len(articles):
            # Select best unattempted article
            best_score = -1
            best_article = None
            best_article_idx = None
            
            for idx, article in enumerate(articles):
                if idx in attempted_articles:
                    continue
                    
                title = article.get("title", "")
                summary = article.get("description", "")
                url = article.get("url", "")

                # Score articles based on length and novelty
                score = len(title) + len(summary or "")
                repeated = any(topic in title.lower() for topic in used_topics)

                print(f"\nArticle #{idx + 1}:")
                print(f"Score: {score}, Repeated topic: {repeated}")
                print(f"Title: {title}")
                print(f"Summary: {summary}")
                print(f"URL: {url}")

                if title and url and not repeated and score > best_score:
                    best_article = article
                    best_score = score
                    best_article_idx = idx

            if not best_article:
                print("[!] No more new articles available.")
                break

            # Mark this article as attempted
            attempted_articles.add(best_article_idx)

            title = best_article["title"]
            summary = best_article.get("description", "")
            url = best_article["url"]

            print(f"\n[+] Selected article #{best_article_idx + 1}:")
            print(f"Title: {title}\nSummary: {summary}\nURL: {url}")

            # Generate tweet content
            tweet_hook = await generate_hook(title, summary, past_hooks, url)
            if not tweet_hook:
                print("[!] No hook generated. Trying next article...")
                continue

            # Final tweet is just the hook content (no URL)
            tweet = tweet_hook

            # Ensure tweet fits character limit
            if len(tweet) > TWEET_CHAR_LIMIT:
                print(f"[!] Tweet exceeds {TWEET_CHAR_LIMIT} characters. Truncating...")
                cutoff = tweet[:TWEET_CHAR_LIMIT-3].rfind(' ')
                if cutoff == -1:
                    cutoff = TWEET_CHAR_LIMIT-3
                tweet = f"{tweet[:cutoff]}..."

            print(f"\n[+] Final tweet to be sent:")
            print(tweet)

            # Start Telegram update processing if needed
            if update_task is None or update_task.done():
                update_task = asyncio.create_task(telegram_bot.process_updates())

            # Send approval request
            request_id = await telegram_bot.send_approval_request(tweet)
            if not request_id:
                print("[!] Failed to send approval request. Trying next article...")
                continue

            # Wait for approval decision
            print(f"[DEBUG] Creating new approval_future for request {request_id}")
            telegram_bot.approval_future = asyncio.Future()
            
            try:
                print(f"[+] Waiting for approval (timeout: 300s)...")
                decision = await asyncio.wait_for(telegram_bot.approval_future, timeout=300)
                print(f"[+] Received decision: {'Approved' if decision else 'Declined'}")
            except asyncio.TimeoutError:
                print("[!] Approval timed out. Trying next article...")
                decision = False
            except Exception as e:
                print(f"[!] Error waiting for approval: {e}")
                print(f"[DEBUG] approval_future state: {telegram_bot.approval_future}")
                decision = False
            finally:
                # Reset approval_future for next request
                print("[DEBUG] Resetting approval_future to None")
                telegram_bot.approval_future = None

            # Handle approval decision
            if decision:
                messenger = MessageService()
                success = await messenger.send_tweet(tweet)
                if success:
                    # Save to history
                    history["hooks"].append(tweet_hook.split('\n')[0].split('.')[0])
                    topic_keywords = [word.lower() for word in title.split() if len(word) > 3]
                    if topic_keywords:
                        history["topics"].append(topic_keywords[0])
                    save_history(history)
                    print("\n[+] Tweet successfully approved and posted!")
                    return  # Exit after successful posting
                else:
                    print("[!] Failed to post tweet. Trying next article...")
                    continue
            else:
                print(f"[!] Tweet not approved. Trying another article... ({len(attempted_articles)}/{len(articles)} articles attempted)")
                # Add a small delay before trying the next article
                await asyncio.sleep(1)
                continue  # Explicitly continue to next article
        # If all articles have been attempted
        if len(attempted_articles) == len(articles):
            print("\n[!] All articles have been attempted but none were approved. Exiting.")
    
    finally:
        # Clean up Telegram update task
        if update_task and not update_task.done():
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass

# === Scheduler ===
async def run_scheduler():
    """Main scheduler function that runs the tweet bot at appropriate intervals"""
    print(f"[+] Starting crypto tweet bot scheduler at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    while True:
        try:
            # Get last run info
            last_run_info = load_last_run_info()
            
            # Calculate when to next run
            next_run_time = get_next_run_time(last_run_info)
            now = datetime.now()
            
            if next_run_time <= now:
                print(f"[+] Time to run the bot ({now.strftime('%Y-%m-%d %H:%M:%S')})")
                
                # Run the tweet bot
                success = await run_tweet_bot()
                
                # Update last run info
                last_run_info["last_run_time"] = now.isoformat()
                if success:
                    last_run_info["last_success_time"] = now.isoformat()
                save_run_info(last_run_info)
                
                # Calculate next run time after this run
                next_run_time = get_next_run_time(last_run_info)
                
            # Wait until next run time
            wait_seconds = max(1, (next_run_time - datetime.now()).total_seconds())
            
            # Format the wait time nicely
            if wait_seconds < 60:
                wait_str = f"{int(wait_seconds)} seconds"
            elif wait_seconds < 3600:
                wait_str = f"{int(wait_seconds / 60)} minutes"
            else:
                hours = int(wait_seconds / 3600)
                minutes = int((wait_seconds % 3600) / 60)
                wait_str = f"{hours} hours and {minutes} minutes"
                
            print(f"[+] Next run scheduled for {next_run_time.strftime('%Y-%m-%d %H:%M:%S')} (in {wait_str})")
            
            # Sleep for 60 seconds at a time to allow for clean exits
            for _ in range(int(wait_seconds / 60) + 1):
                await asyncio.sleep(min(60, wait_seconds))
                wait_seconds -= 60
                if wait_seconds <= 0:
                    break
                    
        except KeyboardInterrupt:
            print("\n[!] Scheduler interrupted by user")
            break
        except Exception as e:
            print(f"[!] Scheduler error: {e}")
            # Wait 5 minutes before retrying
            await asyncio.sleep(300)

# === Entry Point ===
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--once":
            # Run once without scheduling
            print("[+] Running tweet bot once without scheduling...")
            asyncio.run(run_tweet_bot())
        else:
            # Run with scheduler
            print("[+] Running tweet bot with scheduler...")
            asyncio.run(run_scheduler())
    except KeyboardInterrupt:
        print("\n[!] Program interrupted by user")
        exit(0)
    except Exception as e:
        print(f"\n[!] Program error: {e}")
        exit(1)
