# main.py - Core Logic Engine (Refactored for RAG and Vector DB)
import os
import uuid
import google.generativeai as genai
from gtts import gTTS
from dotenv import load_dotenv

# Import functions from the new RAG and VectorDB modules
from rag import get_enhanced_prompt
from vectordb import add_user_prompt_to_db, setup_vector_db

# Load environment variables from .env file
load_dotenv()

# One-time setup for the vector database
# This will initialize the DB and populate it with initial examples if it's empty.
setup_vector_db()

# This prompt remains as it's for the final audio summary, not the main calculation.
PROMPT_SUMMARY = """
Analyze the final result from the following detailed text. Create a single, concise summary sentence in Hindi
that is perfect for a voice assistant.

Example:
Detailed Text: "Trip ka Hisaab: ... Ravi ko aapko ₹200 aur dene hain."
Your Summary: "Hisaab ke anusaar, Ravi ko aapko ₹200 aur dene hain."
"""

def process_query_stream(api_key: str, user_story: str):
    """
    Processes the user's query using a RAG-enhanced prompt and streams the response.
    Saves the user's prompt to the vector database for continuous learning.
    """
    if not api_key:
        yield "❌ Error: Google API Key missing. Please set the GOOGLE_API_KEY environment variable."
        return
    
    if not user_story:
        yield "⚠️ Kripya apni kahani likhein ya bolein."
        return

    try:
        # 1. Get the dynamically generated, few-shot prompt from the RAG module.
        # This module will find relevant examples from the vector DB.
        enhanced_prompt = get_enhanced_prompt(user_story)

        # 2. Configure the generative model and get the response.
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response_stream = model.generate_content(enhanced_prompt, stream=True)

        # 3. Stream the response back to the user interface.
        full_response_text = ""
        for chunk in response_stream:
            if hasattr(chunk, "text") and chunk.text:
                full_response_text += chunk.text
                yield chunk.text
        
        # 4. After a successful response, save the user's original prompt to the Vector DB.
        # This helps the system get smarter over time.
        if full_response_text:
            add_user_prompt_to_db(user_story)

    except Exception as e:
        yield f"⚠️ Hisaab lagate samay error aaya: {e}"

# ---------------------------
# Generate Audio Summary (No changes needed here)
# ---------------------------
def generate_audio_summary(api_key: str, detailed_text: str, slow: bool = False, lang: str = "hi"):
    """
    Generates a short audio summary from the detailed text output.
    """
    if not api_key:
        print("Audio Gen Error: Google API Key missing.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        full_request = PROMPT_SUMMARY + f"\nDetailed Text: \"{detailed_text}\""

        response = model.generate_content(full_request)
        audio_text = response.text.strip() if response and hasattr(response, "text") else "Hisaab taiyaar hai."

        # Generate a unique filename to avoid browser caching issues
        audio_file_path = f"response_{uuid.uuid4().hex}.mp3"

        # Generate audio using gTTS
        tts = gTTS(text=audio_text, lang=lang, slow=slow)
        tts.save(audio_file_path)

        # Clean up older audio files to save space
        cleanup_old_audio_files(keep=3)

        return audio_file_path

    except Exception as e:
        print(f"Audio summary generate karte samay error aaya: {e}")
        return None

# ---------------------------
# Helper: Cleanup old audio files (No changes needed here)
# ---------------------------
def cleanup_old_audio_files(keep=3):
    """
    Deletes older audio files, keeping only the most recent ones.
    """
    try:
        files = [f for f in os.listdir('.') if f.startswith("response_") and f.endswith(".mp3")]
        files.sort(key=os.path.getmtime, reverse=True)
        
        for f in files[keep:]:
            os.remove(f)
    except Exception as e:
        print(f"Purani audio files delete karte samay error aaya: {e}")