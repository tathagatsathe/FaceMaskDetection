from flask import Flask, render_template, Response
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
# from datetime import datetime

model = load_model('./facemask_keras_model.h5')

app = Flask(__name__)

label = ''

def generate_frames():
    camera = cv2.VideoCapture(cv2.CAP_V4L2)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            grscl_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(grscl_frame, (128,128))
            frame_arr = img_to_array(resized_frame)*.1/255

            # now = datetime.now()
            predictions = model.predict(frame_arr.reshape((1, 128, 128, 1)))
            predictions = (predictions>0.5).astype('int32')
            label = 'Face Mask detected' if predictions[0][0]==0 else 'Face Mask Not detected'
            # print(now)
            # print('Face Mask detected' if predictions[0][0]==0 else 'Face Mask Not detected') 

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (150, 50), font, 1.0, (255,255,255),1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)