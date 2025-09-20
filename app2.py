import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import os

# main2.py ko 'main' naam se import kar rahe hain taaki code saaf rahe
# Yeh aapke naye RAG system ko istemal karega
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

# --- Voice Input Logic (Error Corrected) ---
if mode == "🎤 Voice":
    st.write("Niche diye gaye button par click karke apni aawaz record karein.")
    
    # Library ek dictionary return karti hai, jisme audio data 'bytes' key ke andar hota hai.
    audio_info = mic_recorder(start_prompt="▶️ Record", stop_prompt="⏹️ Stop", key='recorder')
    
    # Check karein ki dictionary aur 'bytes' key dono maujood hain.
    if audio_info and audio_info['bytes']:
        st.info("Audio record ho gaya hai. Ab process kiya ja raha hai...")
        
        # User ko sunane ke liye audio play karein - yahan .get('bytes') ka istemal karein
        st.audio(audio_info.get('bytes'), format="audio/wav")
        
        # Audio ko process karke text mein badlein
        recognizer = sr.Recognizer()
        try:
            # Audio bytes ko temporary file mein save karein
            with open("audio.wav", "wb") as f:
                # File mein likhne ke liye bhi .get('bytes') ka istemal karein
                f.write(audio_info.get('bytes'))
            
            # Temporary file ko audio source ke roop mein istemal karein
            with sr.AudioFile("audio.wav") as source:
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
            if os.path.exists("audio.wav"):
                os.remove("audio.wav")

# --- Text Input Logic ---
else:
    user_story = st.text_area(
        "Apni kahani yahan likhiye:", 
        placeholder="Example: Hum 3 dost, main, Rohit aur Suman, Goa gaye. Maine hotel ke 6000 diye, Rohit ne khaane ke 3000 kharch kiye."
    )

# ---------------------------
# Process Query and Display Results (Naye RAG system ke saath)
# ---------------------------
if user_story:
    st.divider()
    st.subheader("📊 Aapka Detailed Hisaab")
    
    # Environment variable se API key lein
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("❌ GOOGLE_API_KEY set nahi hai. Kripya environment variable set karein.")
    else:
        # Spinner dikhayein jab tak process ho raha hai
        with st.spinner('Smart RAG system hisaab laga raha hai...'):
            try:
                # `st.write_stream` ka istemal karein streaming response ke liye
                response_generator = main.process_query_stream(api_key, user_story)
                detailed_text = st.write_stream(response_generator)
                
                # ---------------------------
                # Audio Summary Generation
                # ---------------------------
                if detailed_text and detailed_text.strip():
                    st.divider()
                    st.subheader("🔊 Audio Summary")
                    with st.spinner('Audio summary banaya ja raha hai...'):
                        # main module se audio function call karein
                        audio_file = main.generate_audio_summary(api_key, detailed_text, slow=False)
                        if audio_file and os.path.exists(audio_file):
                            st.audio(audio_file, format="audio/mp3")
                        else:
                            st.warning("Audio summary generate nahi ho paya.")

            except Exception as e:
                st.error(f"Hisaab process karte samay ek anjaan error aayi: {e}")

