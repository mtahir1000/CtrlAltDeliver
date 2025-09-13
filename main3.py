import requests
import base64
from mistralai import Mistral
from dotenv import load_dotenv
import os
import webbrowser
import urllib.parse
import re

# LangChain imports for GPT-4
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -------------------------
# Load API Keys
# -------------------------
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not mistral_api_key or not openai_api_key:
    raise RuntimeError("Missing API keys in .env")

# -------------------------
# Initialize Clients
# -------------------------
mistral_client = Mistral(api_key=mistral_api_key)
gpt4_llm = ChatOpenAI(model="gpt-4", temperature=0.6, openai_api_key=openai_api_key)

# -------------------------
# Image Handling
# -------------------------
def getimages():
    img_url = 'https://t4.ftcdn.net/jpg/03/61/86/91/360_F_361869194_7JGmIOSj2iUNi0AYoVhVyhKvaN6PkOah.jpg'
    img_data = requests.get(img_url).content
    return base64.b64encode(img_data).decode('utf-8')

def getproductdescription(image_data, prompt=None):
    if not prompt:
        prompt = "Give me a list of ingredients for this image. Only return the ingredients, no other text."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}
            ]
        }
    ]

    chat_response = mistral_client.chat.complete(
        model="pixtral-12b-2409",
        messages=messages
    )

    output = chat_response.choices[0].message.content
    print("🧾 Extracted Ingredients:\n", output)
    return output

# -------------------------
# GPT-4 Playlist Generator
# -------------------------
def get_playlist_from_ingredients(ingredients: str):
    prompt_template = PromptTemplate(
        input_variables=["ingredients"],
        template="""
        You are a contemporary music curator.
        Based on these food ingredients: {ingredients}

        Suggest a playlist of 8–10 real hindi family non romantic songs that match the mood/flavors.
        Format as a simple numbered list with "Song – Artist".
        """
    )

    chain = LLMChain(llm=gpt4_llm, prompt=prompt_template)
    return chain.run(ingredients=ingredients)

# -------------------------
# Helpers
# -------------------------
def _parse_song_line(line: str):
    """
    Accepts lines like:
      "1. Blinding Lights – The Weeknd"
      "Blinding Lights - The Weeknd"
    Returns (song, artist) or (None, None) if not parseable.
    """
    # Remove a leading number/period if present
    line = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", line).strip()
    if "–" in line:  # en-dash
        parts = line.split("–", 1)
    elif "-" in line:  # hyphen fallback
        parts = line.split("-", 1)
    else:
        return None, None

    song = parts[0].strip()
    artist = parts[1].strip()
    if not song or not artist:
        return None, None
    return song, artist

def _build_url(song: str, artist: str, platform: str):
    query = f"{song} {artist}"
    encoded_query = urllib.parse.quote_plus(query)
    if platform.lower() == "spotify":
        return f"https://open.spotify.com/search/{encoded_query}"
    # default to YouTube
    return f"https://www.youtube.com/results?search_query={encoded_query}"

# -------------------------
# Open Playlist One-by-One (Interactive)
# -------------------------
def open_playlist_interactive(playlist_text: str, platform: str = "youtube"):
    """
    Opens one song at a time in the chosen platform (youtube|spotify).
    Controls:
      - Press Enter: open next
      - Type 's' + Enter: skip this song
      - Type 'q' + Enter: quit
    """
    platform = platform.lower().strip()
    if platform not in {"youtube", "spotify"}:
        print("Unknown platform. Using YouTube.")
        platform = "youtube"

    print(f"\n🎵 Opening songs in {platform.title()} one by one...")
    print("➡ Press Enter for next, 's' then Enter to skip, 'q' then Enter to quit.\n")

    lines = [ln for ln in (playlist_text or "").splitlines() if ln.strip()]
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        song, artist = _parse_song_line(line)

        if not song:
            # Not a song line, move on
            index += 1
            continue

        url = _build_url(song, artist, platform)
        print(f"\n▶️ {index+1}. {song} — {artist}")
        print(f"   🔗 {url}")

        # Open current song
        webbrowser.open_new_tab(url)

        # Wait for user input
        choice = input("👉 Enter = next | s = skip | q = quit: ").strip().lower()
        if choice == "q":
            print("⏹ Stopping playlist.")
            break
        # 's' just skips; Enter moves to next — both advance index
        index += 1

# -------------------------
# Run Everything
# -------------------------
if __name__ == "__main__":
    ingredients = getproductdescription(getimages())
    playlist = get_playlist_from_ingredients(ingredients)

    print("\n🎧 Recommended Playlist:\n")
    print(playlist)

    # Choose "youtube" or "spotify"
    open_playlist_interactive(playlist, platform="spotify")
