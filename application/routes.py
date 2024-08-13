from application import app
from flask import render_template, request, jsonify, flash,Response,url_for,redirect

import cv2
import requests
import google.generativeai as genai
import os
from PIL import Image
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(image_path,input):
    img = Image.open(image_path)
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


AUDIO_TO_TEXT = "https://api-inference.huggingface.co/models/openai/whisper-medium"

def audio_to_text(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(AUDIO_TO_TEXT, headers=headers, data=data)
    return response.json()



camera=cv2.VideoCapture(-1)

def generate_frames():
    while True:
        global last_frame
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            # Convert the frame into bytes
            frame_bytes = buffer.tobytes()
            last_frame = frame.copy()  # Save the last frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            

# @app.route('/')
# def index():
#     return render_template('index.html')
global res
global a_to_t

@app.route('/', methods=['GET', 'POST'])
def index():
    global last_frame


    
    if request.method == 'POST':
        if last_frame is not None:
            
            filename = f'static/capture_img.jpg'
            filepath = os.path.join(app.root_path, filename)
            cv2.imwrite(filepath, last_frame)
            static_dir = os.path.join(app.root_path, 'static')
            filename = os.path.join(static_dir, 'recorded_audio.wav')
            print(filename)
            a_to_t = audio_to_text(filename)['text']
            print(a_to_t)
            res = get_gemini_response(filepath,a_to_t)
            print(res)
            if res != '':
                return render_template('index.html',ques=a_to_t,res=res)
            return render_template('index.html')
        else:
            return jsonify({"error": "No frame captured"}), 500

    return render_template('index.html')

@app.route("/redirec/<string:ques>/<string:res>")
def redirec(ques, res):
    return render_template('index.html',ques= ques,res=res )


@app.route('/video')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_data' in request.files:
        audio_file = request.files['audio_data']
        
        static_dir = os.path.join(app.root_path, 'static')
        filename = os.path.join(static_dir, 'recorded_audio.wav')
        audio_file.save(os.path.join(filename))
        return 'File uploaded successfully', 200
    return 'No file uploaded', 400