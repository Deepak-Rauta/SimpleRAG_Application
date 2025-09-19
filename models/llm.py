import google.generativeai as genai
from config.config import GOOGLE_API_KEY

# Initialize Gemini Model
def get_gemini_response(prompt: str) -> str: # takes a text prompt and return a string response from Gemini
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([prompt])

        # Safely extract output text
        if response and response.candidates:
            parts = response.candidates[0].content.parts
            return "".join(p.text for p in parts if hasattr(p, "text"))
        else:
            return "⚠️ No text generated."

    except Exception as e:
        return f"⚠️ Error from LLM: {str(e)}"
