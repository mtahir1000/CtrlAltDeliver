import requests
import base64
from mistralai import Mistral
from dotenv import load_dotenv
import os

# Import LangChain for OpenAI GPT-4
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -------------------------
# Load API keys from .env
# -------------------------
load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise RuntimeError("MISTRAL_API_KEY not set. Put it in .env or environment.")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Put it in .env or environment.")

# -------------------------
# Initialize Clients
# -------------------------
# Mistral client (for image analysis)
mistral_client = Mistral(api_key=mistral_api_key)

# OpenAI GPT-4 client via LangChain
gpt4_llm = ChatOpenAI(
    model="gpt-4", 
    temperature=0.6,
    openai_api_key=openai_api_key
)

# -------------------------
# Image Handling Functions
# -------------------------
def getimages():
    img_url = 'https://t4.ftcdn.net/jpg/03/61/86/91/360_F_361869194_7JGmIOSj2iUNi0AYoVhVyhKvaN6PkOah.jpg'
    img_data = requests.get(img_url).content
    img_data_encoded = base64.b64encode(img_data).decode('utf-8')
    return img_data_encoded


def getproductdescription(image_data, prompt=None):
    """Uses Mistral Pixtral model to extract ingredients from an image"""

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
    print("Extracted Ingredients:", output)
    return output

# -------------------------
# Playlist Function (GPT-4)
# -------------------------
def get_playlist_from_ingredients(ingredients: str):
    """Uses GPT-4 to create a playlist based on ingredients"""

    prompt_template = PromptTemplate(
        input_variables=["ingredients"],
        template="""
        You are a contemporary music curator.
        Based on these food ingredients: {ingredients}

        Suggest a playlist of 8–10 real songs that match the mood/flavors.
        Format as a simple numbered list with "Song – Artist".
        """
    )

    chain = LLMChain(llm=gpt4_llm, prompt=prompt_template)

    playlist = chain.run(ingredients=ingredients)
    return playlist

# -------------------------
# Run Full Pipeline
# -------------------------
if __name__ == "__main__":
    ingredients = getproductdescription(getimages())
    playlist = get_playlist_from_ingredients(ingredients)

    print("\nRecommended Playlist:\n")
    print(playlist)
