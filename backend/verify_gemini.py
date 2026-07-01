# Verifies Gemini 2.5 Flash API key and basic generation
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(Path(__file__).parent / ".env")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("In one sentence, what is the Ruy Lopez chess opening?")

print(f"Gemini response: {response.text}")
print("GEMINI OK")
