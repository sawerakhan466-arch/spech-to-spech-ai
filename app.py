import os
import streamlit as st
from groq import Groq
import tempfile
import numpy as np
from pydub import AudioSegment
import soundfile as sf

# -----------------------------
# Groq Setup
# -----------------------------
os.environ['GROQ_API_KEY'] = " groq-api-key"
client = Groq(api_key=os.environ['GROQ_API_KEY'])

# Conversation memory
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# -----------------------------
# Helper: Convert audio to proper WAV
# -----------------------------
def convert_to_wav_from_bytes(audio_bytes, samplerate=16000):
    # Save temp WAV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_bytes, samplerate)

    # Convert to 16kHz mono WAV
    audio = AudioSegment.from_file(temp_file.name)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(wav_file.name, format="wav")
    return wav_file.name

# -----------------------------
# Streamlit App
# -----------------------------
st.title("100era's Baat GPT - Voice Assistant")
st.write("Upload a voice file or record your voice. The AI will reply in text and voice.")

# Upload voice file
uploaded_file = st.file_uploader("Upload your voice file", type=["wav","mp3","m4a"])

if uploaded_file is not None:
    # Load audio file
    audio_bytes = AudioSegment.from_file(uploaded_file)
    audio_bytes = audio_bytes.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio_bytes.get_array_of_samples(), dtype=np.float32)

    # Convert to WAV
    wav_file = convert_to_wav_from_bytes(samples, samplerate=16000)

    # 1️⃣ Whisper transcription
    with open(wav_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=(wav_file, f, "audio/wav")
        )
    user_text = transcription.text
    st.write("You said:", user_text)
    st.session_state.conversation.append({"role":"user","content":user_text})

    # 2️⃣ AI Reply
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=st.session_state.conversation,
        stream=False
    )
    msg_obj = response.choices[0].message
    reply = msg_obj.content if hasattr(msg_obj, "content") else str(msg_obj)
    st.session_state.conversation.append({"role":"assistant","content":reply})
    st.write("Assistant:", reply)

    # 3️⃣ Groq TTS
    speech_response = client.audio.speech.create(
        model="groq-tts",
        voice="alloy",
        input=reply
    )
    tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(tts_file.name, "wb") as f:
        f.write(speech_response.content)

    st.audio(tts_file.name)
