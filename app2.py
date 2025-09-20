import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import os
import tempfile

# main2.py ko 'main' naam se import kar rahe hain taaki code saaf rahe
import main2 as main

# ---------------------------
# Streamlit App UI Setup
# ---------------------------
st.set_page_config(page_title="💰 Hissab Assistant", layout="centered")

st.title("💰 Hissab Assistant (Smart RAG Version)")
st.write("Apni kahani bolkar ya likhkar bhejiye, main aapka hisaab nikal dunga. Yeh version aapke sawaalon se seekhta hai!")

# Input mode selection
mode = st.radio("Aap input kaise dena chahte hain:", ["🎤 Voice", "⌨️ Text"], horizontal=True)

user_story = None

# --- Voice Input Logic ---
if mode == "🎤 Voice":
    st.write("Niche diye gaye button par click karke apni aawaz record karein.")
    
    # streamlit_mic_recorder widget ka istemal karein
    audio_info = mic_recorder(start_prompt="▶️ Record", stop_prompt="⏹️ Stop", key='recorder')
    
    if audio_info and audio_info['bytes']:
        st.info("Audio record ho gaya hai. Ab process kiya ja raha hai...")
        # User ko sunane ke liye audio play karein
        st.audio(audio_info['bytes'], format="audio/wav")
        
        # Audio ko process karke text mein badlein
        recognizer = sr.Recognizer()
        try:
            # Audio bytes ko temporary file mein save karein
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
                tmp_audio_file.write(audio_info['bytes'])
                audio_filename = tmp_audio_file.name

            # Temporary file ko audio source ke roop mein istemal karein
            with sr.AudioFile(audio_filename) as source:
                audio_data = recognizer.record(source)
            
            # Google Speech Recognition se audio ko text mein badlein (Hindi)
            recognized_text = recognizer.recognize_google(audio_data, language='hi-IN')
            st.success(f"📝 Aapne kaha: {recognized_text}")
            user_story = recognized_text

        except sr.UnknownValueError:
            st.warning("Maaf kijiye, main aapki aawaz samajh nahin paya. Kripya dobara koshish karein ya type karein.")
        except sr.RequestError as e:
            st.error(f"Google Speech Recognition service se connect nahin ho paya; {e}")
        except Exception as e:
            st.error(f"Audio process karte samay error aaya: {e}")
        finally:
            # Temporary file ko delete karein
            if 'audio_filename' in locals() and os.path.exists(audio_filename):
                os.remove(audio_filename)

# --- Text Input Logic ---
else:
    user_story = st.text_area(
        "Apni kahani yahan likhiye:", 
        placeholder="Example: Hum 3 dost, main, Rohit aur Suman, Goa gaye. Maine hotel ke 6000 diye, Rohit ne khaane ke 3000 kharch kiye."
    )

# ---------------------------
# Process Query and Display Results
# ---------------------------
if user_story:
    st.divider()
    st.subheader("📊 Aapka Detailed Hisaab")
    
    # Environment variable se API key lein (Streamlit secrets ke liye behtar hai)
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("❌ GOOGLE_API_KEY set nahi hai. Kripya environment variable set karein.")
    else:
        # Display spinner while processing
        with st.spinner('Smart RAG system hisaab laga raha hai...'):
            try:
                # `st.write_stream` ka istemal karein streaming response ke liye
                response_generator = main.process_query_stream(api_key, user_story)
                detailed_text = st.write_stream(response_generator)
                
                # ---------------------------
                # Audio Summary Generation
                # ---------------------------
                if detailed_text:
                    st.divider()
                    st.subheader("🔊 Audio Summary")
                    with st.spinner('Audio summary banaya ja raha hai...'):
                        audio_file = main.generate_audio_summary(api_key, detailed_text)
                        if audio_file and os.path.exists(audio_file):
                            st.audio(audio_file, format="audio/mp3")
                            # Aap audio file ko delete bhi kar sakte hain agar zaroorat na ho
                            # os.remove(audio_file)
                        else:
                            st.warning("Audio summary generate nahi ho paya.")

            except Exception as e:
                st.error(f"Hisaab process karte samay ek anjaan error aayi: {e}")
