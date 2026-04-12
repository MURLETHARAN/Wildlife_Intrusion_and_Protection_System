import cv2
import numpy as np
import time
import threading
from flask import Flask, jsonify
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from playsound import playsound

# ------------------ LOAD MODEL ------------------
print("Loading ML model...")
model = MobileNetV2(weights='imagenet')
print("Model loaded")

# ------------------ GLOBAL STATUS ------------------
status = "Monitoring..."

# ------------------ FLASK ------------------
app = Flask(__name__)

@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>Wildlife Monitor</title>
    </head>
    <body>
        <h1>Wildlife Monitoring System</h1>
        <h2 id="status">Loading...</h2>

        <script>
        function updateStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(data => {
                document.getElementById('status').innerText = data.status;
            });
        }

        setInterval(updateStatus, 1000);
        updateStatus();
        </script>
    </body>
    </html>
    """

@app.route('/status')
def get_status():
    return jsonify({"status": status})

def run_server():
    app.run(host='0.0.0.0', port=5000)

# ------------------ SOUND ------------------
def play_alert():
    try:
        playsound(r"C:\Users\PREMI\OneDrive\Desktop\ghcyg\alert.mp3")  # change if needed
    except Exception as e:
        print("Sound error:", e)

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()

# ------------------ FACE DETECTOR ------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ------------------ MOTION SETUP ------------------
ret, frame1 = cap.read()
ret, frame2 = cap.read()

last_trigger = 0
cooldown = 5

# ------------------ START SERVER ------------------
threading.Thread(target=run_server, daemon=True).start()

print("System running (event-driven + live UI)...")

# ------------------ MAIN LOOP ------------------
while True:

    # Motion detection
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            motion_detected = True
            break

    if motion_detected:
        print("Motion detected!")

        cv2.imwrite("capture.jpg", frame1)

        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

        if len(faces) == 0:

            img = image.load_img("capture.jpg", target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            preds = model.predict(img_array)
            results = decode_predictions(preds, top=3)[0]

            human_keywords = ["person", "man", "woman", "shirt", "face"]
            is_human = any(any(h in label.lower() for h in human_keywords) for _, label, _ in results)

            if not is_human:
                current_time = time.time()

                if current_time - last_trigger > cooldown:
                    print("Animal detected!")

                    status = "Animal Detected! ALERT!"

                    threading.Thread(target=play_alert).start()
                    last_trigger = current_time
            else:
                status = "Human detected"
        else:
            status = "Human detected (face)"

    else:
        status = "Monitoring..."

    frame1 = frame2
    ret, frame2 = cap.read()

    time.sleep(0.3)
