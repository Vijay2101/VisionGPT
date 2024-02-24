
import streamlit as st
import numpy as np
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import google.generativeai as genai
import os
from dotenv import load_dotenv
import requests

load_dotenv() 

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(img,input):
    model=genai.GenerativeModel('gemini-pro-vision')
    prompt = """You are bot a which answers the query of user based on the image provided, respond informally like how a normal persion would reply \n just answer the query nothing else"""
    response=model.generate_content([prompt, img,input])
    return response.text


TEXT_TO_AUDIO_API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
HUGGING_FACE_API_KEY=os.getenv("HUGGING_FACE_API_KEY")
headers = {"Authorization": HUGGING_FACE_API_KEY}

def text_to_audio(payload):
	response = requests.post(TEXT_TO_AUDIO_API_URL, headers=headers, json=payload)
	return response.content



def main():
    st.title("Realtime Object Detection")

    col1, col2 = st.columns([3, 1])

    with col2:
        # Recording audio
        audio_bytes = audio_recorder(pause_threshold=1.5, sample_rate=41_000)
        
        # Display the recorded audio if available
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

    with col1:
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            img = Image.open(img_file_buffer)
            st.image(img, caption="Your Image Caption")

if __name__ == "__main__":
    main()

