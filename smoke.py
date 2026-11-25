import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=key)

for m in ["gemini-3.5-flash-lite", "gemini-3.6-flash"]:
    try:
        r = client.models.generate_content(model=m, contents="say ok")
        print(m, "->", r.text)
    except Exception as e:
        print(m, "FAILED:", type(e).__name__, str(e)[:150])