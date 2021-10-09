from flask import Flask,Response,render_template
import cv2
import mediapipe as mp
import time

app=Flask(__name__)
camera = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
Hands = mpHands.Hands(min_detection_confidence=0.75,min_tracking_confidence=0.75,max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

def generate_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = Hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run()