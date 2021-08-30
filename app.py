# IMPORTS
from flask import Flask, render_template, Response, request
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import cvlib as cv
import os
import base64

# INITIALIZATION
model = load_model('genderDetection.model')
app = Flask(__name__)
classes = ['Female', 'Male']

# GENDER DETECTION USING IMAGE
def usingImage(img_path, model):
    img = cv2.imread(img_path)
    face, confidence = cv.detect_face(img)
    if(len(face) == 0):
        return("No Face Detected")
    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        cv2.rectangle(img, (startX, startY),
                      (endX, endY), (0, 255, 0), 2)
        crop = np.copy(img[startY:endY, startX:endX])
        crop = cv2.resize(crop, (96, 96))
        crop = crop.astype("float") / 255.0
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)
        conf = model.predict(crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(img, label, (startX, Y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer)
    return(jpg_as_text)

# GENDER DETECTION USING WEBCAM
def usingWebcam():
    camera = cv2.VideoCapture(0)
    while(True):
        status, frame = camera.read()
        if(status == False):
            break
        else:
            face, confidence = cv.detect_face(frame)
            for idx, f in enumerate(face):
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)
                face_crop = np.copy(frame[startY:endY, startX:endX])
                if(face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue
                face_crop = cv2.resize(face_crop, (96, 96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)
                conf = model.predict(face_crop)[0]
                idx = np.argmax(conf)
                label = classes[idx]
                label = "{}: {:.2f}%".format(label, conf[idx] * 100)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, Y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ENDPOINTS
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/webcam')
def webcam():
    return(render_template('webcam.html'))


@app.route('/video_capture')
def video_capture():
    return(Response(usingWebcam(), mimetype='multipart/x-mixed-replace; boundary=frame'))


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'upload', secure_filename(f.filename))
        f.save(file_path)
        preds = usingImage(file_path, model)
        return preds
    return None


# MAIN
if __name__ == "__main__":
    app.run(debug=True)
