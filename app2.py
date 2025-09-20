import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import os
import io
from pydub import AudioSegment

# main2.py ko 'main' naam se import kar rahe hain
import main2 as main

# ---------------------------
# Streamlit App UI Setup
# ---------------------------
st.set_page_config(page_title="💰 Hissab Assistant", layout="centered")

st.title("💰 Hissab Assistant (Smart RAG Version)")
st.write("Apni kahani bolkar ya likhkar bhejiye, main aapka hisaab nikal dunga.")

# Input mode selection
mode = st.radio("Aap input kaise dena chahte hain:", ["🎤 Voice", "⌨️ Text"], horizontal=True)

user_story = None

# --- Voice Input Logic (Definitive Fix with Format Conversion) ---
if mode == "🎤 Voice":
    st.write("Niche diye gaye button par click karke apni aawaz record karein.")
    
    audio_info = mic_recorder(start_prompt="▶️ Record", stop_prompt="⏹️ Stop", key='recorder')
    
    if audio_info and audio_info['bytes']:
        st.info("Audio record ho gaya hai. Ab process kiya ja raha hai...")
        st.audio(audio_info['bytes']) # Browser can play the original format
        
        recognizer = sr.Recognizer()
        converted_audio_path = "audio_converted.wav"
        try:
            # 1. Audio bytes ko memory mein load karein
            # Browser se mila audio WebM/Opus format mein ho sakta hai
            sound = AudioSegment.from_file(io.BytesIO(audio_info['bytes']))
            
            # 2. Sahi WAV format mein convert karke save karein
            sound.export(converted_audio_path, format="wav")
            
            # 3. Ab convert hui file ko speech recognition ke liye istemal karein
            with sr.AudioFile(converted_audio_path) as source:
                audio_data = recognizer.record(source)
            
            # Google Speech Recognition se audio ko text mein badlein
            recognized_text = recognizer.recognize_google(audio_data, language='hi-IN')
            st.success(f"📝 Aapne kaha: {recognized_text}")
            user_story = recognized_text

        except sr.UnknownValueError:
            st.warning("Maaf kijiye, main aapki aawaz samajh nahin paya.")
        except sr.RequestError as e:
            st.error(f"Google Speech Recognition service se connect nahin ho paya; {e}")
        except Exception as e:
            st.error(f"Audio process karte samay error aaya: {e}")
        finally:
            # Temporary converted file ko delete karein
            if os.path.exists(converted_audio_path):
                os.remove(converted_audio_path)

# --- Text Input Logic ---
else:
    user_story = st.text_area(
        "Apni kahani yahan likhiye:", 
        placeholder="Example: Hum 3 dost Goa gaye..."
    )

# ---------------------------
# Process Query and Display Results
# ---------------------------
if user_story:
    st.divider()
    st.subheader("📊 Aapka Detailed Hisaab")
    
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("❌ GOOGLE_API_KEY set nahi hai.")
    else:
        with st.spinner('Smart RAG system hisaab laga raha hai...'):
            try:
                response_generator = main.process_query_stream(api_key, user_story)
                detailed_text = st.write_stream(response_generator)
                
                if detailed_text and detailed_text.strip():
                    st.divider()
                    st.subheader("🔊 Audio Summary")
                    with st.spinner('Audio summary banaya ja raha hai...'):
                        audio_file = main.generate_audio_summary(api_key, detailed_text, slow=False)
                        if audio_file and os.path.exists(audio_file):
                            st.audio(audio_file, format="audio/mp3")
                        else:
                            st.warning("Audio summary generate nahi ho paya.")

            except Exception as e:
                st.error(f"Hisaab process karte samay ek anjaan error aayi: {e}")
