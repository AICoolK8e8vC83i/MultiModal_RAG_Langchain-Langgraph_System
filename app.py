# 🔮 Multimodal RAG + LangGraph Cognitive Pipeline (Streamlit Frontend)
# Backend handles: ASR, Text, Image, Video, Multilingual Translation, Memory, Routing
# Frontend handles: Uploads, Mic/Camera Streaming, File Routing via LangGraph

import streamlit as st
import tempfile
import os

# === IMPORT BACKEND PIPELINE ===
from multimodal_rag_langchain_pipeline import graph  # Assumes the compiled LangGraph is named `graph`

# === Streamlit UI ===
st.set_page_config(page_title="🧠 Multimodal AI Agent", layout="centered")
st.title("🧠 Multimodal Cognitive Agent")

uploaded_file = st.file_uploader("Upload a file (Text, Image, Audio, Video)", type=["txt", "md", "jpg", "png", "mp4", "wav", "mp3"])

mic_input = st.checkbox("🎤 Use Microphone Input")
cam_input = st.checkbox("📷 Use Webcam Input (Image)")

user_query = st.text_input("Optional Prompt/Command")

# === Save Uploaded File ===
def save_uploaded_file(file):
    suffix = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file.read())
        return temp_file.name

# === Handle Mic Input (Streamlit doesn't support native mic yet, workaround needed) ===
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av 
import wave 

if mic_input:
    class AudioRecorder(AudioProcessorBase):
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.frames.append(frame.to_ndarray().tobytes())
            return frame 
        
    # Display mic recorder widget
    webrtc_ctx = webrtc_streamer(key="speech", 
                                 mode=WebRtcMode.SENDRECV,
                                 audio_processor_factory=AudioRecorder,
                                 media_stream_constraints={"audio": True, "video": False})
    
    if webrtc_ctx and webrtc_ctx.audio_processor:
        st.success("🎙️ Recording... Speak now")
        if st.button("Stop & Transcribe"):
            # Save audio to temp WAV
            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name 
            with wave.open(audio_path, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(44100)
                f.writeframes(b''.join(webrtc_ctx.audio_processor.frames))

            result = graph.invoke({"input": audio_path})
            st.subheader("🧠 Agent Output")
            st.json(result)

# === Handle Webcam Input ===
import cv2 
from PIL import Image
import numpy as np

if cam_input:
    st.warning("📸 Take a webcam snapshot (for local Streamlit runs only)")
    capture_btn = st.button("📸 Capture from Webcam")

    if capture_btn:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            image_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name 
            cv2.imwrite(image_path, frame)
            st.image(frame, channels="BGR", caption="Captured Image")
            result = graph.invoke({"input": image_path})
            st.subheader("🧠 Agent Output")
            st.json(result)

# === Trigger LangGraph ===
if st.button("🚀 Run Multimodal Agent"):
    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        st.write(f"🧠 Processing: {file_path}")

        # Wrap in proper input dict for LangGraph
        state_input = {"input": file_path}
        result = graph.invoke(state_input)

        st.subheader("🧠 Agent Output")
        st.json(result)
    elif user_query:
        st.write("🧠 Processing text prompt...")
        result = graph.invoke({"input": user_query})
        st.subheader("🧠 Agent Output")
        st.json(result)
    else:
        st.warning("Upload a file or enter a text prompt to run the agent.")

# === Memory Sidebar ===
from multimodal_rag_langchain_pipeline import memory

st.sidebar.title("🧠 Memory")
if memory and memory["entries"]:
    for i, entry in enumerate(memory["entries"][-5:][::-1]):
        st.sidebar.markdown(f"**[{i+1}]** {entry}")