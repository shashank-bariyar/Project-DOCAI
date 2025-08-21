import os
import base64
from groq import Groq
from dotenv import load_dotenv

# Step 1: Initialize Groq client with API key

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Step 2: Convert image to base64
# image_path = r"C:\Users\shash\Downloads\acne.jpg"  # Use raw string to avoid unicode error


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Step 3: Set up multimodal LLM
query = "Is there something wrong with my face?"
# Verify this model supports multimodal input
model = "meta-llama/llama-4-maverick-17b-128e-instruct"


def analyze_image_with_query(query, model, encoded_image):
    client = Groq()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]
    try:
        chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
        )
        # return response
        return (chat_completion.choices[0].message.content)
    except Exception as e:
        return (f"An error occurred: {e}")


# Step 4: Make API call

